import torch
from torch import nn
from torch.nn import Transformer
from torch.nn import Embedding


class TeamBuilder(nn.Module):
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
        # adding [MSK], [CLS], and [CLS2] tokens ([CLS] is for target (team) sequences, [CLS2] is for inputs)
        self.CLS = num_agents
        self.CLS2 = num_agents + 1
        self.MSK = num_agents + 1
        self.agent_embedding = Embedding(num_embeddings=self.num_tokens,
                                         embedding_dim=embedding_dim,
                                         )
        self.transform = Transformer(d_model=embedding_dim,
                                     nhead=nhead,
                                     num_encoder_layers=num_encoder_layers,
                                     num_decoder_layers=num_decoder_layers,
                                     dim_feedforward=dim_feedforward,
                                     batch_first=True,  # takes in (N,S,E) input where N is batch and S is seq length
                                     )

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

    def forward(self, input_embedding, target_team, input_mask, target_mask):
        """
        Args:
            input_embedding: (N, S, E) shape tensor of input
                S should be very small, probably the output of embeddings with a more efficient method like LSTM
            target_team: (N, T) shape tensor of team members
                EACH TEAM SHOULD END WITH A [CLS] token
            input_mask: (N, S) tensor of booleans on whether to mask each input
            target_mask: (N, T) tensor of which team members are masked
        Returns:
        """
        N, T = target_team.shape
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

        # size (N, T, T), target i cannot see target j
        tgt_mask = torch.zeros((N, T, T))  # should prevent everyone from seeing the masked team members
        tgt_mask[torch.where(target_mask)] = -torch.inf

        # size (N, S+1, S+1), source i cannot see source j
        # src_mask = torch.zeros((N, S + 1, S + 1))  # should mask the maked elements of input embedding
        # src_mask[torch.where(input_mask)]=-

        # size (N, T, S+1), target i cannot see source j
        memory_mask = torch.zeros((T, S + 1))  # shoult mask hidden input embeddings from the target
        memory_mask = torch.fill(memory_mask, -torch.inf)

        output = self.transform.forward(src=source,
                                        tgt=target,
                                        # tgt_mask=tgt_mask,
                                        src_key_padding_mask=input_mask,
                                        # memory_key_padding_mask=input_mask,
                                        )
        return output


if __name__ == '__main__':
    torch.random.manual_seed(69)
    test = Transformer(d_model=16, batch_first=True)
    s = torch.rand((1, 5, 16))
    s[0, 0, :] = 100.

    t = torch.rand((1, 2, 16))
    (test.forward(src=s, tgt=t))

    # quit()

    test = TeamBuilder(num_agents=69, embedding_dim=16, num_encoder_layers=10)
    N = 3
    S = 10
    T = 5
    E = test.embedding_dim
    team = torch.randint(0, test.num_agents, (N, T))
    team = test.add_cls_tokens(team)

    target_mask = torch.zeros((N, T), dtype=torch.bool)

    input_embedding = torch.ones((N, S, E))*999
    print(test.forward(input_embedding=input_embedding,
                       target_team=team,
                       input_mask=torch.ones((N, S), dtype=torch.bool),
                       target_mask=target_mask
                       ))
