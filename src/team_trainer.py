import torch
from torch import nn
from torch.utils.data import DataLoader

from networks.team_builder import TeamBuilder


class TeamTrainer:
    def __init__(self, team_builder: TeamBuilder):
        """
        Args:
        """
        self.team_builder = team_builder
        self.MASK = team_builder.berteam.MASK
        self.num_agents = team_builder.berteam.num_agents

        self.optim = torch.optim.Adam(team_builder.parameters())

    def epoch(self,
              loader: DataLoader,
              mask_probs=None,
              replacement_probs=(.8, .1, .1),
              minibatch=True,
              ):
        """
        Args:
            minibatch: whether to take a step after each batch

        Returns:

        """
        if not minibatch:
            self.optim.zero_grad()
        all_losses = []
        for item in loader:
            if type(item) == list:
                input_preembedding, teams, input_mask = item
            else:
                teams = item
                input_preembedding = None
                input_mask = None
            if minibatch:
                self.optim.zero_grad()

            losses = self.mask_and_learn(input_preembedding=input_preembedding,
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

    def mask_and_learn(self, input_preembedding, teams, input_mask, mask_probs=None, replacement_probs=(.8, .1, .1)):
        """
        runs learn step on a lot of mask_probabilities
            need to test with a wide range of mask probabilites because for generation, we may start with a lot of masks
        Args:
            input_preembedding: tensor (N, S, *) of input preembeddings, or None if no preembeddings
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
            loss = self.learn_step(input_preembedding=input_preembedding,
                                   teams=teams,
                                   input_mask=input_mask,
                                   mask_prob=mask_prob,
                                   replacement_probs=replacement_probs,
                                   )
            losses.append(loss)
        return losses

    def learn_step(self, input_preembedding, teams, input_mask, mask_prob=.5, replacement_probs=(.8, .1, .1)):
        """
        randomly masks winning team members, runs the transformer token prediction model, and gets crossentropy loss
            of predicting the masked tokens
        Args:
            input_preembedding: tensor (N, S, *) of input preembeddings, or None if no preembeddings
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
        logits = self.team_builder.forward(input_preembedding=input_preembedding,
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
        what_to_replace_with[rand_rep] = self.uniform_random_team((len(rand_rep[0]),))

        unchange = torch.where(what_to_replace_with == -3)
        # grab the correct team members from the original array
        what_to_replace_with[unchange] = masked_teams[[dim_idx[unchange] for dim_idx in mask_indices]]

        masked_teams[mask_indices] = what_to_replace_with
        return masked_teams, which_to_mask, mask_indices

    def uniform_random_team(self, shape):
        return torch.randint(0, self.num_agents, shape)

    def create_masked_teams(self, T, N=1):
        """
        Args:
            T: team size
            N: batch size
        Returns:
            an (N,T) tensor of all masked members
        """
        return torch.fill(torch.zeros((N, T), dtype=torch.long), self.MASK)

    def create_nose_model_towards_uniform(self, t):
        """
        creates a noise model that takes in a distribution and does a weighted average with a uniform dist
        Args:
            t: weight to give uniform dist (1.0 replaces the distribution, 0.0 leaves dist unchanged)
        Returns:
            distribution -> distribution map, which are sized (N,K) for an N batch of a K-multinomial
        """
        return (lambda dist: (1 - t)*dist + (t)*torch.ones_like(dist)/dist.shape[1])

    def fill_in_teams(self,
                      initial_teams,
                      input_preembedding=None,
                      input_mask=None,
                      noise_model=None,
                      num_masks=None,
                      ):
        """
        replaces all [MASK] with team members
            repeatedly calls mutate_add_member
        Args:
            initial_teams: initial torch array of team members, shape (N,T)
            input_preembedding: input to give input embedder, size (N,S,*)
                None if no input
            input_mask: size (N,S) boolean array of whether to mask each input embedding
            noise_model: noise to add to probability distribution, if None, doesnt add noise
            num_masks: specify the largest number of [MASK] tokens in any row
                if None, just calls mutate_add_member (T) times
        Returns:
            filled in teams of size (N,T)
        """
        N, T = initial_teams.shape
        if num_masks is None:
            num_masks = T
        for _ in range(num_masks):
            initial_teams = self.mutate_add_member(initial_teams=initial_teams,
                                                   indices=None,
                                                   input_preembedding=input_preembedding,
                                                   input_mask=input_mask,
                                                   noise_model=noise_model
                                                   )
        return initial_teams

    def mutate_add_member(self,
                          initial_teams,
                          indices=None,
                          input_preembedding=None,
                          input_mask=None,
                          noise_model=None,
                          ):
        """
        updates initial teams by updating specified indices with samples from the probability distribution
            if indices is None, chooses indices that correspond with [MASK] tokens (one for each row)
        Args:
            initial_teams: initial torch array of team members, shape (N,T)
            indices: indices to replace (if None, picks one masked index at random)
                should be in pytorch format, (list of dim 0 indices, list of dim 1 indices)
            input_preembedding: input to give input embedder, size (N,S,*)
                None if no input
            input_mask: size (N,S) boolean array of whether to mask each input embedding
            noise_model: noise to add to probability distribution, if None, doesnt add noise
        Returns:
            team with updates
        """
        N, T = initial_teams.shape
        if indices is None:
            indices = [[], []]
            for i in range(N):
                potential = torch.where(initial_teams[i] == self.MASK)[0]
                if len(potential) > 0:
                    indices[0].append(i)
                    indices[1].append(potential[torch.randint(0, len(potential), (1,))])
                # else:
                #    print("warning: tried to add to full team")
        if len(indices[0]) == 0:
            # no [MASK] tokens exist
            return initial_teams
        output = self.team_builder.forward(input_preembedding=input_preembedding,
                                           target_team=initial_teams,
                                           input_mask=input_mask,
                                           output_probs=True,
                                           pre_softmax=False,
                                           )
        dist = output[indices]  # (|indices|,num_agents) multinomial distribution for each index to update
        if noise_model is not None:
            # add noise if this is a thing
            dist = noise_model(dist)
        # torch.multinomial samples each

        initial_teams[indices] = torch.multinomial(dist, 1).flatten()
        return initial_teams


if __name__ == '__main__':
    N = 64
    S = 4
    T = 3
    epochs = 20
    torch.random.manual_seed(69)

    E = 512

    num_inputs = 4

    test = TeamTrainer(
        team_builder=
        TeamBuilder(
            num_agents=69,
            input_embedder=
            torch.nn.Embedding(num_embeddings=num_inputs,
                               embedding_dim=E)),
    )

    init_team = torch.arange(T, dtype=torch.long)


    def random_shuffle():
        return init_team[torch.randperm(len(init_team))]


    for _ in range(epochs):
        # input_preembedding_gen = (torch.randint(0, num_inputs, (i%S,)) if i > 0 else None for i in range(N))
        # input_embedding_gen = (None for i in range(N))
        # input_preembedding = None
        # input_mask = None
        input_preembedding = torch.randint(0, num_inputs, (N, S))
        input_mask = torch.zeros((N, S), dtype=torch.bool)
        out_teams = torch.stack([random_shuffle() for _ in range(N)], dim=0)

        if input_preembedding is None:
            data = list(out_teams)
        else:
            data = list(zip(input_preembedding, out_teams, input_mask))
        loader = DataLoader(data,
                            shuffle=True,
                            batch_size=64,
                            )
        losses = test.epoch(
            loader=loader,
            minibatch=True
        )

        for loss_set in losses:
            print(loss_set)
        print()

    # print(test.mask_and_learn(input_embedding_gen=input_embeddings,
    #                          winning_team_gen=(init_team.clone() for _ in range(N))))
    num_teams = 8
    init_teams = test.create_masked_teams(T=T, N=num_teams)
    print(init_teams)
    for _ in range(T):
        test.mutate_add_member(initial_teams=init_teams, )
        print(init_teams)
