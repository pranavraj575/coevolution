import torch
from torch import nn
from torch.nn import Transformer
from torch.nn import Embedding


class TeamBuilder(nn.Module):
    def __init__(self,
                 num_agents,
                 embedding_dim=512,
                 nhead=8,
                 num_encoder_layers=12,
                 num_decoder_layers=12,
                 dim_feedforward=2048,
                 ):
        super().__init__()

        self.num_agents = num_agents
        self.num_tokens = num_agents + 2
        # adding [MSK] token and [CLS] token
        self.CLS = num_agents
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
            (N, T+1) vector where self.CLS is added to the front of every team
        """
        (N, T) = target_team.shape
        return torch.cat((torch.ones((N, 1))*self.CLS, target_team), dim=1)

    def forward(self, input_embedding, target_team, input_mask, target_mask):
        """
        Args:
            input_embedding: (N, S, E) shape tensor of input
                S should be very small, probably do the embedding with a more efficient method like LSTM
            target_team: (N, T) shape tensor of team members
                EACH TEAM SHOULD START WITH A self.CLS token
            input_mask: (N,) tensor of booleans on whether to mask each input
            target_mask: (N, T) tensor of which team members are masked
        Returns:

        """
        target = self.agent_embedding(target_team)
        self.transform()
        pass


if __name__ == '__main__':
    torch.random.manual_seed(69)
    embedder = Embedding(num_embeddings=40, embedding_dim=5)
    print(embedder(torch.tensor([2, 3, 39, 39])))
