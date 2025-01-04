
# positional encoding assigns information about the order or position of tokens
# adds positional encodings to the input embeddings
# helps the model understand relative positions
# used learnable approach mentioned in Dosovitskiy paper

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, seq_len):

        # seq_len: number of input tokens (image patches)
        # embed_dim: dimensions of the embeddings
        # initialized randomly
        # model learns optimal positional encoding values during training
        super(PositionalEncoding, self).__init__()
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, embed_dim))

    def forward(self, x):
        # adds positional encoding to each embedding
        x = x + self.positional_encoding
        return x