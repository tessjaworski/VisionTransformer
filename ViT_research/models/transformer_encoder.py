
# Transformer encoder processes patches to understand relationships.
# Multi head attention: computes relationship and relevancy between patches
# Feed-Forward network: refines patch embeddings
# Layer Norm: stabilizes the network

import torch
import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, activation=nn.GELU):

        # Processes input embeddings using self-attention and a feed-forward network.

        # embed_dim: the dimensionality of the input embeddings
        # num_heads: number of attention heads in the multi-head attention layer
        # ff_dim: hidden layer size in feed forward network
        # dropout: dropout probability for regularization
        # activation: defaults to GELU for smooth activation

        super(TransformerEncoderBlock, self).__init__()

        # multi-head self-attention layer
        # allows model to focus on different parts of input sequence at the same time
        # parallel attention
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # feed-forward network (FFN)
        # temporarily expands to a larger space so it can extract more subtle features and then returns to original dimensions
        # this extracts richer features
        # two layer connected network that refines embeddings
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            activation(),
            nn.Linear(ff_dim, embed_dim),
        )

        # layer normalization applied after each layer for stabilization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # dropout layers after each layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # multi head attention layer
        attn_output, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # feed forward layer
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x