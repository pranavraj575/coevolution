import torch
from torch import nn
from torch.nn import Transformer
from torch.nn import Embedding
from src.positional_encoder import PositionalEncoding



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



class TeamBuilder(nn.Module):
    def __init__(self, input_embedder, berteam:BERTeam=None, num_agents=None):
        """
        one of berteam or num_agents must be defined
        Args:
            input_embedder:
            berteam:
            num_agents:
        """
        super().__init__()

        if berteam is None:
            berteam = BERTeam(num_agents=num_agents)
        self.input_embedder = input_embedder
        self.berteam = berteam

    def forward(self, input_preembedding, target_team, input_mask, output_probs=True, pre_softmax=False):
        """
        Args:
            input_preembedding: (N, S, *) shape tensor of input, or None if no input
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
        if input_preembedding is None:
            input_embedding = None
        else:
            input_embedding = self.input_embedder(input_preembedding)
        return self.berteam.forward(input_embedding=input_embedding,
                                    target_team=target_team,
                                    input_mask=input_mask,
                                    output_probs=output_probs,
                                    pre_softmax=pre_softmax,
                                    )

if __name__ == '__main__':
    N = 10
    S = 4
    T = 3

    test = Transformer(d_model=16, batch_first=True)
    s = torch.rand((1, 5, 16))
    s[0, 0, :] = 100.

    t = torch.rand((1, 2, 16))
    (test.forward(src=s, tgt=t))

    test = BERTeam(num_agents=69, embedding_dim=16, dropout=0)
    E = test.embedding_dim
    team = torch.randint(0, test.num_agents, (N, T))
    team = test.add_cls_tokens(team)

    input_preembedding = torch.rand((N, S, E))
    input_mask = torch.ones((N, S), dtype=torch.bool)
    input_mask[:, 0] = False

    a = test.forward(input_embedding=input_preembedding,
                     target_team=team,
                     input_mask=input_mask
                     )
    input_preembedding[:, 1] *= 10
    b = test.forward(input_embedding=input_preembedding,
                     target_team=team,
                     input_mask=input_mask,
                     )

    assert torch.all(a == b)
