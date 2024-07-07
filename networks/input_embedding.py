import torch
from torch import nn
from networks.positional_encoder import PositionalEncoding


class InputEmbedder(nn.Module):
    def forward(self, pre_embedding, preembed_mask):
        """
        Args:
            pre_embedding: (N, S, *) observation preembedding
            mask: (N, S) boolean mask, or None
        Returns:
            (embedding (N, S', E), mask (N, S'))
        """
        raise NotImplementedError


class DiscreteInputEmbedder(InputEmbedder):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )

    def forward(self, pre_embedding, preembed_mask):
        return self.embed(pre_embedding), preembed_mask


class DiscreteInputPosEmbedder(InputEmbedder):
    def __init__(self, num_embeddings, embedding_dim, dropout=.1):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )
        self.pos_enc = PositionalEncoding(d_model=embedding_dim, dropout=dropout)

    def forward(self, pre_embedding, preembed_mask):
        return self.pos_enc(self.embed(pre_embedding)), preembed_mask


class DiscreteInputPosAppender(InputEmbedder):
    """
    instead of adding positional encoding, just appends it, and also has a linear layer to match dimensions
    """

    def __init__(self, num_embeddings, embedding_dim, dropout=.1):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )
        self.pos_enc = PositionalEncoding(d_model=embedding_dim, dropout=dropout)
        self.linear = nn.Linear(2*embedding_dim, embedding_dim)

    def forward(self, pre_embedding, preembed_mask):
        embedding = self.embed(pre_embedding)
        positional = self.pos_enc(torch.zeros_like(embedding))
        cat = torch.cat((embedding, positional), dim=-1)
        return self.linear(cat), preembed_mask
