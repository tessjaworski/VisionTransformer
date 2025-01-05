# Vision Transformer (ViT)

The Vision Transformer (ViT) demonstrates the application of a transformer-based architecture specifically for computer vision tasks. My implementation stands out by dynamically processing 2D, 3D, and 4D input data with an adaptable pipeline. This dynamic handling ensures compatibility with different types of input, ranging from static images to complex multi-dimensional data, such as videos with a time dimension.

The core of my implementation focuses on preserving spatial, temporal, and depth-related information across these data types. By leveraging a combination of Conv2D and Conv3D layers for patch embedding, the model effectively transforms raw input data into fixed-length token representations for any input. This flexibility is important for ensuring accurate and efficient processing of different input formats within the same model.

By reorganizing the pipeline to merge batch size and time dimensions specifically for 4D input, I reduced redundant computations. This implementation highlights a strong emphasis on modularity which allows it to adapt to new input formats with minimal reconfiguration.

## Features
- **Dynamic Input Hnadling**:
  - Processes 2D inputs like images (height x width) .
  - Handles 3D inputs like images with a depth dimension (height x width x depth).
  - Adapts to 4D inputs, combining time and channels (height x width x depth x time).
- **Core Componenets**:
   - **Patch Embedding**: Converts raw inputs into fixed-size patches and embeds them into tokens suitable for trnasformer processing
       - Utilizes Conv2D and Conv3D for efficient extraction of patches.
       - For 4D inputs, combines batch size and time during patch extraction to handle both spatial and temporal information.
   - **Positional Encoding**: Adds spatial and temporal context to tokens.
       - Utilized learnable positional encodings that adapt during training.
       - Provides richer spatial and temporal context.
   - **Transformer Encoder**:Allows the model establish relationships between all parts of the input data despite their spatial or temporal distance.
       - Multi-head self-attention: Allows model to focus on different parts of input sequence at the same time through parallel attention.
       - Feed-forward layer: temporarily expands to a larger space so it can extract richer features and then returns to original dimensions.
   - **Output projection**: Converts processed tokens back to their original input shape.
