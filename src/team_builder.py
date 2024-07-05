import torch
from torch import nn
from torch.nn import Transformer
from torch.nn import Embedding


class TeamBuilder:
    def __init__(self, berteam=None, num_agents=None):
        if berteam is None:
            berteam = BERTeam(num_agents=num_agents)
        self.berteam = berteam
        self.optim = torch.optim.Adam(berteam.parameters())
        self.MASK = berteam.MASK

    def create_masked_teams(self, T, N=1):
        """
        Args:
            T: team size
            N: batch size
        Returns:
            an (N,T) tensor of all masked members
        """
        return torch.fill(torch.zeros((N, T), dtype=torch.long), self.berteam.MASK)

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
            potential = torch.where(initial_team == self.MASK)
            if len(potential[0]) == 0:
                print("warning: tried to add to full team")
                return initial_team
            idx = torch.randint(0, len(potential[0]), (1,))
            indices = [[t[idx]] for t in potential]

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
        initial_team[indices] = torch.multinomial(dist, 1)
        return initial_team


class BERTeam(nn.Module):
    def __init__(self,
                 num_agents,
                 embedding_dim=512,
                 nhead=8,
                 num_encoder_layers=1,
                 num_decoder_layers=12,
                 dim_feedforward=None,
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
        self.transform = Transformer(d_model=embedding_dim,
                                     nhead=nhead,
                                     num_encoder_layers=num_encoder_layers,
                                     num_decoder_layers=num_decoder_layers,
                                     dim_feedforward=dim_feedforward,
                                     batch_first=True,  # takes in (N,S,E) input where N is batch and S is seq length
                                     dropout=0.,
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

    def forward(self, input_embedding, target_team, input_mask, output_probs=True):
        """
        Args:
            input_embedding: (N, S, E) shape tensor of input
                S should be very small, probably the output of embeddings with a more efficient method like LSTM
            target_team: (N, T) shape tensor of team members
                EACH TEAM SHOULD END WITH A [CLS] token
            input_mask: (N, S) tensor of booleans on whether to mask each input
            output_probs: whether to output the probability of each team member
                otherwise just outputs the final embedding
        Returns:
            if output_probs, (N, T, num_agents) probability distribution for each position
            otherwise, (N, T, embedding_dim) output of transformer model
        """
        # N, T = target_team.shape
        N, S, _ = input_embedding.shape

        if input_mask is None:
            input_mask = torch.zeros((N, S), dtype=torch.bool)

        # creates a sequence of size S+1
        # dimensions (N, S+1, E) where we add embeddings of [CLS2] tokens for the last values
        source = torch.cat((
            input_embedding,
            self.agent_embedding(torch.fill(torch.zeros((N, 1), dtype=target_team.dtype), self.CLS2)),
        ), dim=1)

        input_mask = torch.cat((input_mask, torch.zeros((N, 1), dtype=torch.bool)), dim=1)

        target = self.agent_embedding(target_team)

        output = self.transform.forward(src=source,
                                        tgt=target,
                                        src_key_padding_mask=input_mask,
                                        memory_key_padding_mask=input_mask,
                                        )
        if output_probs:
            output = self.softmax(self.output_layer(output))

        return output


if __name__ == '__main__':
    N = 1
    S = 10
    T = 3
    torch.random.manual_seed(69)

    test = TeamBuilder(num_agents=69)
    init_team = test.create_masked_teams(T=T)
    print('initial', init_team)
    for _ in range(3):
        test.mutate_add_member(init_team)
        print(init_team)

    test = Transformer(d_model=16, batch_first=True)
    s = torch.rand((1, 5, 16))
    s[0, 0, :] = 100.

    t = torch.rand((1, 2, 16))
    (test.forward(src=s, tgt=t))

    # quit()

    test = BERTeam(num_agents=69, embedding_dim=16)
    E = test.embedding_dim
    team = torch.randint(0, test.num_agents, (N, T))
    team = test.add_cls_tokens(team)

    input_embedding = torch.ones((N, S, E))
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
