import torch
from models.vit import VisionTransformer

# Test parameters
embed_dim = 64
num_heads = 4
ff_dim = 128
num_layers = 2
dropout = 0.1

# Initialize Vision Transformers for 2D, 3D, and 4D inputs
vit_2d = VisionTransformer(img_size=(28, 28), embed_dim=embed_dim, num_heads=num_heads,
                           ff_dim=ff_dim, num_layers=num_layers, dropout=dropout)
vit_3d = VisionTransformer(img_size=(28, 28, 4), embed_dim=embed_dim, num_heads=num_heads,
                           ff_dim=ff_dim, num_layers=num_layers, dropout=dropout, depth_size=4)
vit_4d = VisionTransformer(img_size=(14, 14, 2, 2), embed_dim=embed_dim, num_heads=num_heads,
                           ff_dim=ff_dim, num_layers=num_layers, dropout=dropout, depth_size=2, time_size=2)

# Define test inputs
input_2d = torch.randn(4, 3, 27, 27)  # (batch_size, channels, height, width)
input_3d = torch.randn(4, 3, 4, 28, 28)  # (batch_size, channels, depth, height, width)
input_4d = torch.randn(4, 3, 2, 14, 14, 2)  # (batch_size, channels, depth, height, width, time)

# Test Vision Transformer with 2D input
print("\nTesting Vision Transformer with 2D input...")
output_2d = vit_2d(input_2d)
print(f"2D Input Shape: {input_2d.shape}, Output Shape: {output_2d.shape}")

# Test Vision Transformer with 3D input
print("\nTesting Vision Transformer with 3D input...")
output_3d = vit_3d(input_3d)
print(f"3D Input Shape: {input_3d.shape}, Output Shape: {output_3d.shape}")

# Test Vision Transformer with 4D input
print("\nTesting Vision Transformer with 4D input...")
output_4d = vit_4d(input_4d)
print(f"4D Input Shape: {input_4d.shape}, Output Shape: {output_4d.shape}")