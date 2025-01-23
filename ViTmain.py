import torch
import os
import torch.nn as nn
import torch.optim as optim
from models.vit import VisionTransformer
from preprocessing import load_pth_files, process_consecutive_files, list_pth_files, split_file_paths_into_train_val_test


def chunked_training_simple(
    folder_2d="sample_data/2D",
    num_time_steps=6,
    chunk_size=10,
    batch_size=8,
    num_epochs=5,
    reduced_size=20,
    checkpoint_path="vit_model.pth"
):

    # Splits files into (train, val, test) chronologically.
    # For each epoch, loads training data in chunks of 'chunk_size' files.
    # Converts each chunk to tensors, does a manual training loop with 'batch_size' mini-batches.
    # Repeats for val and test at the end.
    # Replicates the target to 3 channels, matching a model output of (B, 3, H, W).


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    all_files = list_pth_files(folder_2d)  # lists all files in my 2D folder
    num_files = len(all_files)
    print(f"Total .pth files found: {num_files}")

    # Splits the data into training, validating, and testing chronologically with 80/10/10 ratio
    train_end = int(0.8 * num_files)
    val_end = int(0.9 * num_files)

    train_files, val_files, test_files = split_file_paths_into_train_val_test(
        all_files,
        train_end=train_end,
        val_end=val_end
    )

    print(f"Train files: {len(train_files)}  |  Val files: {len(val_files)}  |  Test files: {len(test_files)}")

    # Initialize vision transformer model
    vit = VisionTransformer(
        img_size=(reduced_size, reduced_size),
        embed_dim=128,
        num_heads=8,
        ff_dim=512,
        num_layers=6,
        in_channels=num_time_steps
    ).to(device)

    loss_fn = nn.MSELoss()

    # Dummy Forward Pass
    # Dummy forward pass is necessary because we need to ensure that the positional encoding
    # parameter is registered before loading the checkpoint.
    # Triggering .forward() once ensures that self.positional_encoding is created inside VisionTransformer.
    dummy_batch_size = 1
    dummy_height = reduced_size
    dummy_width = reduced_size
    dummy_time = num_time_steps

    # shape: [B, H, W, T]
    dummy_input = torch.randn(dummy_batch_size, dummy_height, dummy_width, dummy_time).to(device)

    # Permute to (B, T, H, W) for VisionTransformer
    dummy_input = dummy_input.permute(0, 3, 1, 2)

    # Forward pass
    _ = vit(dummy_input)  # This line creates positional_encoding submodule in the first pass

    # Now positional_encoding is no longer None, and the model has
    # a parameter "positional_encoding.positional_encoding" registered.


    # Load Checkpoint
    # This step is optional but helpful when I want to resume training
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)  # loads saved state dictionary
        vit.load_state_dict(state_dict)  # Assigns loaded weights to model's parameters
        print("Checkpoint loaded successfully.")
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting fresh.")

    # Optionally, adjust your optimizer if resuming
    optimizer = optim.Adam(vit.parameters(), lr=1e-5)


    # Chunked training loop
    # Loading all 100 .pth files at once uses too much RAM
    # Instead, I load 10 files at a time, process them, and then discard them
    # This helps with memory usage

    num_chunks = (len(train_files) + chunk_size - 1) // chunk_size
    print(f"chunk_size={chunk_size}, so we'll have {num_chunks} chunks in the train set.")

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        vit.train()
        epoch_loss = 0.0
        epoch_samples = 0

        # Iterate over chunks
        for c in range(num_chunks):
            chunk_start = c * chunk_size
            chunk_end = min((c + 1) * chunk_size, len(train_files))
            chunk_paths = train_files[chunk_start:chunk_end]

            # Loads each .pth file in that chunk and returns a list of tensors
            data_list = load_pth_files(chunk_paths)
            # Converts data_list to (inputs, outputs)
            inputs_list, outputs_list = process_consecutive_files(data_list, num_time_steps)
            # Convert to PyTorch tensors
            # .stack() combines a list of PyTorch tensors into one big tensor
            # .float() ensures 32-bit floating point dtype for training
            # N samples
            # each sample is a2D grid(HxW) over T time steps
            inputs_tensor = torch.stack(inputs_list).float()   # shape [N, H, W, T]
            targets_tensor = torch.stack(outputs_list).float()  # shape [N, H, W]
            N = inputs_tensor.size(0)

            # Manual mini-batch loop within this chunk
            # Slices the chunk sensor into smaller batches of size batch_size
            for b in range(0, N, batch_size):
                batch_in = inputs_tensor[b : b + batch_size]   # (B, H, W, T)
                batch_tgt = targets_tensor[b : b + batch_size]  # (B, H, W)

                batch_in = batch_in.to(device)
                batch_tgt = batch_tgt.to(device)

                # Slice to reduced size
                # Crops 900x600 to 20x20 for memory reasons
                batch_in = batch_in[:, :reduced_size, :reduced_size, :]
                batch_tgt = batch_tgt[:, :reduced_size, :reduced_size]

                # Permute input to (B, T, H, W)
                batch_in = batch_in.permute(0, 3, 1, 2)

                # Replicate the target to 3 channels (B, 3, H, W)
                batch_tgt = batch_tgt.unsqueeze(1).repeat(1, 3, 1, 1)

                optimizer.zero_grad()
                predictions = vit(batch_in)   # forward pass
                loss = loss_fn(predictions, batch_tgt)  # finds MSE between predictions and batch_tgt
                loss.backward()  # compute gradients
                optimizer.step()  # update weights in direction that reduces the loss

                # computes total loss
                epoch_loss += loss.item() * batch_in.size(0)
                epoch_samples += batch_in.size(0)

            # Free up memory from chunk
            del data_list, inputs_list, outputs_list, inputs_tensor, targets_tensor
            torch.cuda.empty_cache()

        # End for chunk in num_chunks
        avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0
        print(f"Training Loss (epoch {epoch+1}): {avg_loss:.4f}")


        # Validation loop
        # Validation set is not used to update model weights
        # Validation set is used to gauge performance on unseen data
        vit.eval()  # sets the model to evaluation mode
        with torch.no_grad():
            # If val set is small, it's okay to load it all at once
            val_data_list = load_pth_files(val_files)
            val_in_list, val_tgt_list = process_consecutive_files(val_data_list, num_time_steps)

            val_in_tensor = torch.stack(val_in_list).float()
            val_tgt_tensor= torch.stack(val_tgt_list).float()
            valN = val_in_tensor.size(0)

            val_loss_sum = 0.0

            for b in range(0, valN, batch_size):
                batch_in = val_in_tensor[b : b + batch_size]   # (B, H, W, T)
                batch_tgt = val_tgt_tensor[b : b + batch_size]  # (B, H, W)

                batch_in = batch_in.to(device)
                batch_tgt = batch_tgt.to(device)

                batch_in = batch_in[:, :reduced_size, :reduced_size, :]
                batch_tgt = batch_tgt[:, :reduced_size, :reduced_size]

                batch_in = batch_in.permute(0, 3, 1, 2)

                # Replicate target to (B, 3, H, W)
                batch_tgt = batch_tgt.unsqueeze(1).repeat(1, 3, 1, 1)

                preds = vit(batch_in)
                loss = loss_fn(preds, batch_tgt)
                val_loss_sum += loss.item() * batch_in.size(0)

            avg_val_loss = val_loss_sum / valN if valN > 0 else 0
            print(f"Validation Loss (epoch {epoch+1}): {avg_val_loss:.4f}")

            del val_data_list, val_in_list, val_tgt_list, val_in_tensor, val_tgt_tensor
            torch.cuda.empty_cache()

    # end of all epochs

    # Testing loop
    # Testing is only run once at the very end
    # Commented out for now while I am training
    # Not used for training decisions

    vit.eval()
    with torch.no_grad():
        test_data_list = load_pth_files(test_files)
        test_in_list, test_tgt_list = process_consecutive_files(test_data_list, num_time_steps)

        test_in_tensor = torch.stack(test_in_list).float()
        test_tgt_tensor= torch.stack(test_tgt_list).float()
        testN = test_in_tensor.size(0)

        test_loss_sum = 0.0

        for b in range(0, testN, batch_size):
            batch_in = test_in_tensor[b : b + batch_size]   # (B, H, W, T)
            batch_tgt = test_tgt_tensor[b : b + batch_size]  # (B, H, W)

            batch_in = batch_in.to(device)
            batch_tgt = batch_tgt.to(device)

            batch_in = batch_in[:, :reduced_size, :reduced_size, :]
            batch_tgt = batch_tgt[:, :reduced_size, :reduced_size]

            batch_in = batch_in.permute(0, 3, 1, 2)

            # Replicate target here too
            batch_tgt = batch_tgt.unsqueeze(1).repeat(1, 3, 1, 1)

            preds = vit(batch_in)
            loss = loss_fn(preds, batch_tgt)
            test_loss_sum += loss.item() * batch_in.size(0)

        avg_test_loss = test_loss_sum / testN if testN > 0 else 0
        print(f"\nFinal Test Loss: {avg_test_loss:.4f}")



    # 5) Saves Model
    torch.save(vit.state_dict(), "vit_model.pth")
    print("Saved model to vit_model.pth")


def main():
    chunked_training_simple(
        folder_2d="sample_data/2D",
        num_time_steps=6,
        chunk_size=10,     # load 10 files at a time
        batch_size=8,
        num_epochs=5,  # 5 epochs for training
        reduced_size=20
    )

if __name__ == "__main__":
    main()
