import torch
import torch.nn as nn
from models.patch_embedding import PatchEmbedding
from models.positional_encoding import PositionalEncoding
from models.transformer_encoder import TransformerEncoderBlock

# Combines patch embedding, positional encoding, and transformer encoder

class VisionTransformer(nn.Module):
   def __init__(self, img_size, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1, depth_size=None, time_size=None, in_channels=3):
        super(VisionTransformer, self).__init__()

        self.depth_size = depth_size
        self.time_size = time_size

        # Patch embedding layer
        # Handles 2D, 3D, and 4D inputs dynamically
        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            depth_size=depth_size,
            time_size=time_size
        )

        # Initialize positional encoding in __init__
        self.num_patches = (img_size[0] // 1) * (img_size[1] // 1)  # Update patch size logic
        self.positional_encoding = PositionalEncoding(embed_dim=embed_dim, seq_len=self.num_patches)

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

       # Patch embedding
       x = self.patch_embedding(x)

       # Handles 4D data by checking for time and if tensor is 5D
       if self.time_size is not None and x.ndim == 5:
            bt = x.size(0)  # this is the batch x time dimension
            B = bt // self.time_size  # computes original batch size by dividing (batch x time) by time
            x = x.view(B, self.time_size, x.size(1), x.size(2), x.size(3), x.size(4))  # unmerges batch and time

            # Reorder to (B, embed_dim, D, T, H, W) so length of spatial dims after skipping first two dimensions is 4
            x = x.permute(0, 2, 3, 1, 4, 5)

       batch_size = x.size(0)
       spatial_dims = x.size()[2:] # Spatial dim are remaining dimensions after batch and channel

       seq_len = x.numel() // (batch_size * x.size(1))  # Number of patches
       x = x.reshape(batch_size, seq_len, -1)  # Reshape to (batch_size, seq_len, embed_dim)

       # Apply positional encoding
       x = self.positional_encoding(x)

       # Transformer encoder
       try:
           x = self.encoder(x)
       except Exception as e:
           print(f"Error during Transformer Encoder: {e}")
           raise

       # Reshape back to original dimensions for final output
       if len(spatial_dims) == 4:  # For 4D input
           depth, time, height, width = spatial_dims
           x = x.view(batch_size, -1, depth, time, height, width)

           x = x.permute(0, 3, 1, 2, 4, 5)  # moves time next to batch
           b, t, e, d, h, w = x.shape
           x = x.reshape(b * t, e, d, h, w)  # Combine batch and time for Conv3D

           # Apply 4D projection
           x = self.output_projection_4d(x)

           out_channels = x.size(1)
           x = x.view(b, t, out_channels, d, h, w)  # separate batch and time

           # Permute to final shape (B, out_channels, D, T, H, W)
           x = x.permute(0, 2, 3, 4, 5, 1)

       elif len(spatial_dims) == 3:  # For 3D input
           x = x.view(batch_size, spatial_dims[0], spatial_dims[1], spatial_dims[2], -1).permute(0, 4, 1, 2, 3)
           x = self.output_projection_3d(x)

       else:  # For 2D input
           x = x.view(batch_size, spatial_dims[0], spatial_dims[1], -1).permute(0, 3, 1, 2)
           x = self.output_projection_2d(x)
       return x