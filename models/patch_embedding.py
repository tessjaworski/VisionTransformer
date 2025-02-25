
# Patch embedding converts images into patch representations.
# Uses Conv2D and Conv3D to extract features from patches.
# Flattens and transposes so that patches are in a sequence.
# This class creates embeddings from patches extracted from the input

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, kernel_size=16, stride=16, padding=0, depth_size=None, time_size=None):

        # kernel size and stride should be equal so patches do not overlap and all patches are 16x16
        # ^^ from "Dive into Deep Learning" textbook

        # in_channels: number of input channels. ex: RGB has 3 input channels
        # embed_dim: dimension of the patch embeddings. every patch is transformed into a list with "embed_dim" amount of numbers
        # embedding is like a description of each patch and embed_dim is how long/detailed the description is
        # the bigger the embed_dim the more detailed the embedding/description of the patch is
        # depth_size: used for 3d data like volumetric images that have height, width, and depth
        # time_size: used for video data that have height, width, depth, and time

        super(PatchEmbedding, self).__init__()   # initializes parent class

        # A kernel/filter is used to look at smaller parts of the image at once. slides to diff parts of the image
        # I used a fixed kernel size of 3 for simplicity.

        # Stride defines how far the kernel slides after each operation
        # I used a fixed stride of 1 to ensure no patches are skipped.

        # Padding is additional rows/columns around edges to ensure correct dimensions after operation
        # This ensures that input size is the same as output size
        # Used a fixed padding of 1 for simlplicity.

        # Conv2D and Conv3D are used to extract features through convolution
        # The output of convolution is a feature map that highlights patterns
        if time_size:  # For 4D input
            self.conv = nn.Conv3d(
                in_channels, embed_dim,
                kernel_size=(kernel_size, kernel_size, kernel_size),
                stride=(stride, stride, stride),
                padding=(padding, padding, padding)
            )
            self.is_4d = True
        elif depth_size:  # For 3D input
            self.conv = nn.Conv3d(
                in_channels, embed_dim,
                kernel_size=(kernel_size, kernel_size, kernel_size),
                stride=(stride, stride, stride),
                padding=(padding, padding, padding)
            )
            self.is_4d = False
        else:  # For 2D input
            self.conv = nn.Conv2d(
                in_channels, embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
            self.is_4d = False

    def forward(self, x):

        if self.is_4d:  # Combine batch and time for 4D input before convolution
            batch_size, channels, depth, height, width, time = x.shape
            x = x.view(batch_size * time, channels, depth, height, width)
            print(f"Combined batch and time: {x.shape}")

        x = self.conv(x)
        return x