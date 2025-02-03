
# positional encoding assigns information about the order or position of tokens
# adds positional encodings to the input embeddings
# helps the model understand relative positions

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, seq_len):

        # seq_len: number of input tokens (image patches)
        # embed_dim: dimensions of the embeddings
        # initialized randomly
        # model learns optimal positional encoding values during training
        super(PositionalEncoding, self).__init__()

        # Compute sinusoidal positional encoding
        pos = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))

        pe = torch.zeros(seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(pos * div_term)  # Sine for even indices
        pe[:, 1::2] = torch.cos(pos * div_term)  # Cosine for odd indices

        self.positional_encoding = nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x):
        # adds positional encoding to each embedding
        x = x + self.positional_encoding
        return x