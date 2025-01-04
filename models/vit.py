import torch
import torch.nn as nn
from models.patch_embedding import PatchEmbedding
from models.positional_encoding import PositionalEncoding
from models.transformer_encoder import TransformerEncoderBlock

# Combines patch embedding, positional encoding, and transformer encoder

class VisionTransformer(nn.Module):
   def __init__(self, img_size, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1, depth_size=None, time_size=None):
        super(VisionTransformer, self).__init__()

        self.depth_size = depth_size
        self.time_size = time_size

        # Patch embedding layer
        # Handles 2D, 3D, and 4D inputs dynamically
        self.patch_embedding = PatchEmbedding(
            in_channels=3,
            embed_dim=embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            depth_size=depth_size,
            time_size=time_size
        )

        # Positional encoding layer placeholder
        # Dynamically initialized in forward pass
        self.positional_encoding = None

        # Stack of transformer encoder blocks
        self.encoder = nn.Sequential(
            *[TransformerEncoderBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        )

        # Output projection for 2D input
        self.output_projection_2d = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=3,
            kernel_size=1
        )

        # Output projection for 3D input
        self.output_projection_3d = nn.Conv3d(
            in_channels=embed_dim,
            out_channels=3,
            kernel_size=1
        )

        # Output projection for 4D input
        self.output_projection_4d = nn.Conv3d(
            in_channels=embed_dim,
            out_channels=3,
            kernel_size=1
        )

   def forward(self, x):
       print(f"Input shape: {x.shape}")

       # Patch embedding
       x = self.patch_embedding(x)
       print(f"After Patch Embedding: {x.shape}")


       batch_size = x.size(0)
       spatial_dims = x.size()[2:]  # Spatial dim are remaining dimensions after batch and channel

       # Handles 4D input
       if self.depth_size and self.time_size and len(spatial_dims) == 3:
           # Correct spatial_dims to include depth and time
           spatial_dims = (self.depth_size, self.time_size, spatial_dims[-2], spatial_dims[-1])
           print(f"Spatial dimensions inferred (4D corrected): {spatial_dims}, len(spatial_dims): {len(spatial_dims)}")
       else:
           print(f"Spatial dimensions inferred: {spatial_dims}, len(spatial_dims): {len(spatial_dims)}")

       seq_len = x.numel() // (batch_size * x.size(1))  # Number of patches
       x = x.view(batch_size, seq_len, -1)  # Reshape to (batch_size, seq_len, embed_dim)

       # Dynamically initialize positional encoding based on seq_len
       if self.positional_encoding is None:
           self.positional_encoding = PositionalEncoding(embed_dim=x.size(-1), seq_len=seq_len)

       # Apply positional encoding
       x = self.positional_encoding(x)
       print(f"After Positional Encoding: {x.shape}")

       # Transformer encoder
       x = self.encoder(x)
       print(f"After Transformer Encoder: {x.shape}")

       # Reshape back to original dimensions for final output
       if len(spatial_dims) == 4:  # For 4D input
           depth, time, height, width = spatial_dims
           x = x.reshape(batch_size, -1, depth * time, height, width)  # Combine depth and time for Conv3D

           # Apply 4D projection
           x = self.output_projection_4d(x)  # Output has shape (batch_size, channels, depth * time, height, width)

           # Reshape back to separate depth and time dimensions
           x = x.reshape(batch_size, depth, time, height, width, -1).permute(0, 5, 1, 3, 4, 2)
       elif len(spatial_dims) == 3:  # For 3D input
           x = x.view(batch_size, spatial_dims[0], spatial_dims[1], spatial_dims[2], -1).permute(0, 4, 1, 2, 3)
           x = self.output_projection_3d(x)
       else:  # For 2D input
           x = x.view(batch_size, spatial_dims[0], spatial_dims[1], -1).permute(0, 3, 1, 2)
           x = self.output_projection_2d(x)

       return x