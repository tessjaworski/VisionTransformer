# Vision Transformer (ViT)

This repository demonstrates a Vision Transformer capable of dynamically handling 2D, 3D, and 4D input data. This implementation emphasizes a flexible approach to computer vision tasks by carefully adapting the pipeline to adapt to various inputs. The inputs can range from static images to spatiotemporal datasets.

The core of my implementation focuses on preserving spatial, temporal, and depth-related information across different inputs. By leveraging a combination of Conv2D and Conv3D layers for patch embedding, the model converts raw input data into fixed-length token representations for transformer processing.

When handling 4D inputs, like videos with both depth and time dimensions, the pipeline merges batch and time dimensions to reduce redundant computations. 

## Features
- **Dynamic Input Hnadling**:
  - **2D inputs**: Processes images with spatial dimensions (height x width) .
  - **3D inputs**: Adapts to volumetric data (height x width x depth).
  - **4D inputs**: Incorporates a time dimension alongside spatial and depth dimensions (height x width x depth x time). My implementation merges batch size and time.
- **Core Componenets**:
   - **Patch Embedding**:
       - Converts raw inputs into fixed-size patches and embeds them into tokens suitable for transformer processing.
       - Utilizes **Conv2D** and **Conv3D** for efficient extraction of patches.
       - For 4D inputs, combines batch and time to handle spatial and temporal information without redundant computations.
   - **Positional Encoding**:
       - Adds spatial and temporal context to tokens via learnable positional encodings.
       - Enhances the model's spatial and temporal awareness.
   - **Transformer Encoder**:
       - Uses **multi-head self-attention** to allow the model to attend to different regions in parallel.
       - Utilizes **feed-forward layers** to expand the feature space and extract richer features, then reduces back to original features.
       - The model can establish relationships between patches despite their spatial and temporal distance.
   - **Output projection**:
       - Converts processed tokens back to their original input shape.
    
## How It Works

1. **Patch Embedding**
     - Input data is divided into smaller patches using Conv2D (2D input) or Conv3D (3D input and 4D input).
     - These patches are flattened and transformed into token embeddings.
2. **Positional Encoding**
     - A learnable embedding is added to each token to keep information about its position and ordering.
3. **Transformer Encoder**
     - Multi-head self-attention layers allow the model to attend to different parts of the input simultaneously.
     - Feed-forward layers allow the model to capture richer context by expanding to a higher dimension and then returning to the original dimensions.
4. **Output Projection**
     - After processing, the tokens are reshaped back to the original input format. This allows the model to return outputs that have the same shape as the input data. Returning the same output shape makes tasks like segmentation easier to facilitate.
