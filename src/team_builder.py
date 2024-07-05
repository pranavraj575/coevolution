import torch
from torch import nn
from torch.nn import Transformer
from torch.nn import Embedding
from torch.utils.data import DataLoader
from src.positional_encoder import PositionalEncoding


class TeamBuilder:
    def __init__(self, berteam=None, num_agents=None):
        if berteam is None:
            berteam = BERTeam(num_agents=num_agents)
        self.berteam = berteam
        self.optim = torch.optim.Adam(berteam.parameters())
        self.MASK = berteam.MASK

    def preprocess_generators(self, input_embedding_gen, team_gen):
        """
        preprocesses these into tensors to feed into learning step
        Args:
            input_embedding_gen: iterable of input embeddings, the ith one is size (1, S_i,E), or None if no input
            team_gen: iterable of corresponding winning, each sized (T)
        Returns:
            input embedding (N,S,E), teams (N,T), and input mask (N, S)
            or None, teams, None if there is no input embedding
        """
        if not torch.is_tensor(team_gen):
            out_teams = torch.stack([team for team in team_gen], dim=0)
        else:
            out_teams = team_gen
        N, T = out_teams.shape

        input_embedding_gen = list(input_embedding_gen)
        S = max(0 if t is None else t.shape[1] for t in input_embedding_gen)
        if S == 0:
            return None, out_teams, None
        input_embedding = torch.zeros((N, S, self.berteam.embedding_dim))
        input_mask = torch.ones((N, S), dtype=torch.bool)
        for i, embedding in enumerate(input_embedding_gen):
            Si = 0 if embedding is None else embedding.shape[1]
            if embedding is not None:
                input_embedding[i, :Si] = embedding
            input_mask[i, :Si] = 0
        return input_embedding, out_teams, input_mask

    def epoch(self,
              loader: DataLoader,
              mask_probs=None,
              replacement_probs=(.8, .1, .1),
              minibatch=True,
              ):
        """

        Args:
            input_embedding_gen:
            winning_team_gen:
            data_length:
            batch_size:
            mask_probs:
            replacement_probs:
            minibatch: whether to take a step after each batch

        Returns:

        """
        if not minibatch:
            self.optim.zero_grad()
        all_losses = []
        for item in loader:
            if type(item) == list:
                input_embedding, teams, input_mask = item
            else:
                teams = item
                input_embedding = None
                input_mask = None
            if minibatch:
                self.optim.zero_grad()

            losses = self.mask_and_learn(input_embedding=input_embedding,
                                         teams=teams,
                                         input_mask=input_mask,
                                         mask_probs=mask_probs,
                                         replacement_probs=replacement_probs,
                                         )
            if minibatch:
                self.optim.step()
            all_losses.append(losses)
        if not minibatch:
            self.optim.step()
        return all_losses

    def mask_and_learn(self, input_embedding, teams, input_mask, mask_probs=None, replacement_probs=(.8, .1, .1)):
        """
        runs learn step on a lot of mask_probabilities
            need to test with a wide range of mask probabilites because for generation, we may start with a lot of masks
        Args:
            input_embedding: tensor (N, S, E) of input embeddings
            teams: tensor (N, T) of teams
            input_mask: boolean tensor (N, S) of whether to pad each input
            mask_probs: list of proabilities of masking to use
                if None, uses (1/team_size, 2/team_size, ..., 1)
            replacement_probs: proportion of ([MASK], random element, same element) to replace masked elements with
                default (.8, .1, .1) because BERT
        Returns:
            list of crossentropy loss
        """
        if mask_probs is None:
            N, T = teams.shape
            mask_probs = torch.arange(0, T + 1)/T
        losses = []
        for mask_prob in mask_probs:
            loss = self.learn_step(input_embedding=input_embedding,
                                   teams=teams,
                                   input_mask=input_mask,
                                   mask_prob=mask_prob,
                                   replacement_probs=replacement_probs,
                                   )
            losses.append(loss)
        return losses

    def learn_step(self, input_embedding, teams, input_mask, mask_prob=.5, replacement_probs=(.8, .1, .1)):
        """
        randomly masks winning team members, runs the transformer token prediction model, and gets crossentropy loss
            of predicting the masked tokens
        Args:
            input_embedding: tensor (N, S, E) of input embeddings
            teams: tensor (N, T) of teams
            input_mask: boolean tensor (N, S) of whether to pad each input
            mask_prob: proportion of elements to mask (note that we will always mask at least one per batch
            replacement_probs: proportion of ([MASK], random element, same element) to replace masked elements with
                default (.8, .1, .1) because BERT
        Returns:
            crossentropy loss of prediction
        """

        in_teams, mask, mask_indices = self._randomly_mask_teams(teams=teams,
                                                                 mask_prob=mask_prob,
                                                                 replacement_props=replacement_probs,
                                                                 )
        logits = self.berteam.forward(input_embedding=input_embedding,
                                      target_team=in_teams,
                                      input_mask=input_mask,
                                      output_probs=True,
                                      pre_softmax=True,
                                      )
        criterion = nn.CrossEntropyLoss()

        loss = criterion(logits[mask_indices], teams[mask_indices])
        loss.backward()
        return loss

    def _randomly_mask_teams(self, teams, mask_prob, replacement_props):
        """
        Args:
            teams: size (N,T) tensor to randomly mask
            mask_prob: proportion of elements to mask (note that we will always mask at least one per batch)
            replacement_props: proportion of ([MASK], random element, same element) to replace masked elements with
                tuple of three floats
        """
        masked_teams = teams.clone()
        N, T = teams.shape

        which_to_mask = torch.bernoulli(torch.ones_like(teams)*mask_prob)
        forced_mask = torch.randint(0, T, (N,))
        which_to_mask[torch.arange(N), forced_mask] = 1

        mask_indices = torch.where(which_to_mask == 1)
        num_masks = len(mask_indices[0])
        what_to_replace_with = torch.multinomial(torch.tensor([replacement_props for _ in range(num_masks)]),
                                                 1).flatten()
        # this will be a tensor of 0s, 1s, and 2s, indicating whether to replace each mask with
        #   [MASK], a random element, or not to replace it

        # map this to -1,-2,-3 respectively
        what_to_replace_with = -1 - what_to_replace_with
        # now map to the actual values to replace with in the array
        what_to_replace_with = torch.masked_fill(what_to_replace_with,
                                                 mask=(what_to_replace_with == -1),
                                                 value=self.MASK,
                                                 )
        rand_rep = torch.where(what_to_replace_with == -2)
        what_to_replace_with[rand_rep] = self.berteam.random_member_tensor((len(rand_rep[0]),))

        unchange = torch.where(what_to_replace_with == -3)
        # grab the correct team members from the original array
        what_to_replace_with[unchange] = masked_teams[[dim_idx[unchange] for dim_idx in mask_indices]]

        masked_teams[mask_indices] = what_to_replace_with
        return masked_teams, which_to_mask, mask_indices

    def create_masked_teams(self, T, N=1):
        """
        Args:
            T: team size
            N: batch size
        Returns:
            an (N,T) tensor of all masked members
        """
        return torch.fill(torch.zeros((N, T), dtype=torch.long), self.berteam.MASK)

    def create_nose_model_towards_uniform(self, t):
        """
        creates a noise model that takes in a distribution and does a weighted average with a uniform dist
        Args:
            t: weight to give uniform dist (1.0 replaces the distribution, 0.0 leaves dist unchanged)
        Returns:
            distribution -> distribution map, which are sized (N,K) for an N batch of a K-multinomial
        """
        return (lambda dist: (1 - t)*dist + (t)*torch.ones_like(dist)/dist.shape[1])

    def mutate_add_member(self, initial_team, indices=None, input_embedding=None, input_mask=None, noise_model=None, ):
        """
        Args:
            initial_team: initial torch array of team members, shape (N,T)
            indices: indices to replace (if None, picks one masked index at random)
                should be in pytorch format, (list of dim 0 indices, list of dim 1 indices)
            input_embedding: input to give team builder, size (N,S,embedding_dim)
                if None, gives dummy input and masks it
            input_mask: size (N,S) boolean array of whether to mask each input embedding
            noise_model: noise to add to probability distribution, if None, doesnt add noise
        Returns:
            team with updates
        """
        N, T = initial_team.shape
        if indices is None:
            indices = [[], []]
            for i in range(N):
                potential = torch.where(initial_team[i] == self.MASK)[0]
                if len(potential) == 0:
                    print("warning: tried to add to full team")
                else:
                    indices[0].append(i)
                    indices[1].append(potential[torch.randint(0, len(potential), (1,))])

        if input_embedding is None:
            input_embedding = torch.zeros((N, 1, self.berteam.embedding_dim))
            input_mask = torch.ones((N, 1), dtype=torch.bool)
        output = self.berteam.forward(input_embedding=input_embedding,
                                      target_team=initial_team,
                                      input_mask=input_mask,
                                      output_probs=True,
                                      )
        dist = output[indices]  # (|indices|,num_agents) multinomial distribution for each index to update
        if noise_model is not None:
            # add noise if this is a thing
            dist = noise_model(dist)
        # torch.multinomial samples each

        initial_team[indices] = torch.multinomial(dist, 1).flatten()
        return initial_team


class BERTeam(nn.Module):
    def __init__(self,
                 num_agents,
                 embedding_dim=512,
                 nhead=8,
                 num_encoder_layers=1,
                 num_decoder_layers=12,
                 dim_feedforward=None,
                 dropout=.1
                 ):
        super().__init__()
        self.embedding_dim = embedding_dim
        if dim_feedforward is None:
            dim_feedforward = self.embedding_dim*4
        self.num_agents = num_agents
        self.num_tokens = num_agents + 3
        # adding [MASK], [CLS], and [CLS2] tokens ([CLS] is for target (team) sequences, [CLS2] is for inputs)
        self.CLS = num_agents
        self.CLS2 = num_agents + 1
        self.MASK = num_agents + 2
        self.agent_embedding = Embedding(num_embeddings=self.num_tokens,
                                         embedding_dim=embedding_dim,
                                         )
        self.pos_encoder = PositionalEncoding(d_model=embedding_dim,
                                              dropout=dropout,
                                              )
        self.transform = Transformer(d_model=embedding_dim,
                                     nhead=nhead,
                                     num_encoder_layers=num_encoder_layers,
                                     num_decoder_layers=num_decoder_layers,
                                     dim_feedforward=dim_feedforward,
                                     batch_first=True,  # takes in (N,S,E) input where N is batch and S is seq length
                                     dropout=dropout,
                                     )
        self.output_layer = nn.Linear(embedding_dim, num_agents)
        self.softmax = nn.Softmax(dim=-1)

    def add_cls_tokens(self, target_team):
        """
        adds [CLS] tokens to target teams
        Args:
            target_team: (N, T) vector
        Returns:
            (N, T+1) vector where self.CLS is added to the end of every team
        """
        (N, T) = target_team.shape
        return torch.cat((target_team, torch.ones((N, 1), dtype=target_team.dtype)*self.CLS), dim=1)

    def forward(self, input_embedding, target_team, input_mask, output_probs=True, pre_softmax=False):
        """
        Args:
            input_embedding: (N, S, E) shape tensor of input, or None if no input
                S should be very small, probably the output of embeddings with a more efficient method like LSTM
            target_team: (N, T) shape tensor of team members
                EACH TEAM SHOULD END WITH A [CLS] token
            input_mask: (N, S) tensor of booleans on whether to mask each input
            output_probs: whether to output the probability of each team member
                otherwise just outputs the final embedding
            pre_softmax: if True, does not apply softmax to logits
        Returns:
            if output_probs, (N, T, num_agents) probability distribution for each position
            otherwise, (N, T, embedding_dim) output of transformer model
        """
        N, T = target_team.shape

        # creates a sequence of size S+1
        # dimensions (N, S+1, E) where we add embeddings of [CLS2] tokens for the last values
        if input_embedding is None:
            source = self.agent_embedding(torch.fill(torch.zeros((N, 1), dtype=target_team.dtype), self.CLS2))
            S = 0
        else:
            source = torch.cat((
                input_embedding,
                self.agent_embedding(torch.fill(torch.zeros((N, 1), dtype=target_team.dtype), self.CLS2)),), dim=1)
            N, S, _ = input_embedding.shape
        if input_mask is None:
            input_mask = torch.zeros((N, S), dtype=torch.bool)
        input_mask = torch.cat((input_mask, torch.zeros((N, 1), dtype=torch.bool)), dim=1)

        target = self.agent_embedding(target_team)
        pos_enc_target = self.pos_encoder(target)

        output = self.transform.forward(src=source,
                                        tgt=pos_enc_target,
                                        src_key_padding_mask=input_mask,
                                        memory_key_padding_mask=input_mask,
                                        )
        if output_probs:
            output = self.output_layer(output)
            if not pre_softmax:
                output = self.softmax(output)

        return output

    def random_member_tensor(self, shape):
        return torch.randint(0, self.num_agents, shape)


if __name__ == '__main__':
    N = 64
    S = 4
    T = 3
    epochs = 20
    torch.random.manual_seed(69)

    test = TeamBuilder(num_agents=69)
    E = test.berteam.embedding_dim
    init_team = torch.arange(T, dtype=torch.long)


    def random_shuffle():
        return init_team[torch.randperm(len(init_team))]


    for _ in range(epochs):
        # input_embedding_gen = (torch.rand((1, i%S, E)) if i > 0 else None for i in range(N))
        input_embedding_gen = (None for i in range(N))
        winning_team_gen = (random_shuffle() for _ in range(N))
        input_embedding, out_teams, input_mask = test.preprocess_generators(input_embedding_gen=input_embedding_gen,
                                                                            team_gen=winning_team_gen)
        if input_embedding is None:
            data = list(out_teams)
        else:
            data = list(zip(input_embedding, out_teams, input_mask))
        loader = DataLoader(data,
                            shuffle=True,
                            batch_size=64,
                            )
        losses = test.epoch(
            loader=loader,
            minibatch=True
        )
        del loader

        for loss_set in losses:
            print(loss_set)
        print()

    # print(test.mask_and_learn(input_embedding_gen=input_embeddings,
    #                          winning_team_gen=(init_team.clone() for _ in range(N))))
    num_teams = 8
    init_teams = test.create_masked_teams(T=T, N=num_teams)
    print(init_teams)
    for _ in range(T):
        test.mutate_add_member(initial_team=init_teams, )
        print(init_teams)

    test = Transformer(d_model=16, batch_first=True)
    s = torch.rand((1, 5, 16))
    s[0, 0, :] = 100.

    t = torch.rand((1, 2, 16))
    (test.forward(src=s, tgt=t))

    # quit()
    N = 10
    test = BERTeam(num_agents=69, embedding_dim=16, dropout=0)
    E = test.embedding_dim
    team = torch.randint(0, test.num_agents, (N, T))
    team = test.add_cls_tokens(team)

    input_embedding = torch.rand((N, S, E))
    input_mask = torch.ones((N, S), dtype=torch.bool)
    input_mask[:, 0] = False

    a = test.forward(input_embedding=input_embedding,
                     target_team=team,
                     input_mask=input_mask
                     )
    input_embedding[:, 1] *= 10
    b = test.forward(input_embedding=input_embedding,
                     target_team=team,
                     input_mask=input_mask,
                     )

    assert torch.all(a == b)
