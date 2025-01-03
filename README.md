# Vision Transformer (ViT)

The Vision Transformer (ViT) demonstrates the implementation of a transformer task for computer vision tasks. My implementation dynamically processes 2D, 3D, and 4D input data.

## Features
- **Dynamic Input Hnadling**:
  - Processes 2D inputs like images.
  - Handles 3D inputs such as videos with depth frames.
  - Adapts to 4D inputs, combining time and channels.
- **Core Componenets**:
   - **Patch Embedding**: Converts raw inputs into fixed-size patches and embeds them into tokens suitable for trnasformer processing
       - Utilizes Conv2D and Conv3D for efficient extraction of patches.
       - For 4D inputs, combines channels and time during patch extraction to handle both spatial and temporal information.
   - **Positional Encoding**: Adds spatial and temporal context to tokens.
       - Utilized learnable positional encodings that adapt during training.
       - Provides richer spatial and temporal context.
   - **Transformer Encoder**:Allows the model establish relationships between all parts of the input data despite their spatial or temporal distance.
       - Multi-head self-attention: Allows model to focus on different parts of input sequence at the same time through parallel attention.
       - Feed-forward layer: temporarily expands to a larger space so it can extract richer features and then returns to original dimensions.
   - **Output projection**: Converts processed tokens back to their original input shape.
