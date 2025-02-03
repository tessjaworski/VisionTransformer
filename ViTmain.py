import torch
import os
import torch.nn as nn
import torch.optim as optim
from models.vit import VisionTransformer
from torch.utils.data import DataLoader
from preprocessing import list_pth_files, WeatherDataset, list_pth_files, split_file_paths_into_train_val_test


def chunked_training_simple(
    folder_2d="sample_data/2D",
    num_time_steps=6,
    chunk_size=10,
    batch_size=8,
    num_epochs=20,
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

    # Split into train, val, test
    train_files, val_files, test_files = split_file_paths_into_train_val_test(all_files)
    # Create Datasets
    train_dataset = WeatherDataset(train_files, num_time_steps, reduced_size)
    val_dataset = WeatherDataset(val_files, num_time_steps, reduced_size)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize ViT model
    vit = VisionTransformer(
        img_size=(reduced_size, reduced_size),
        embed_dim=64,
        num_heads=4,
        ff_dim=256,
        num_layers=3,
        in_channels=num_time_steps,
        dropout=0.3
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(vit.parameters(), lr=5e-6,weight_decay=1e-3)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        state_dict.pop("positional_encoding.positional_encoding", None)  # Remove mismatched keys
        vit.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully.")
    else:
        print(f"No checkpoint found, starting fresh.")

    best_val_loss = float("inf")
    patience = 7  # Stop if validation loss doesn't improve for 5 epochs
    counter = 0

    # Training Loop
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        vit.train()
        total_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.permute(0, 1, 2, 3)  # Should be (B, T, H, W)

            optimizer.zero_grad()
            predictions = vit(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vit.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Training Loss (epoch {epoch + 1}): {avg_train_loss:.4f}")

        # Validation
        vit.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:

                inputs, targets = inputs.to(device), targets.to(device)

                if targets.ndim == 3:  # Add batch dimension if missing
                    targets = targets.unsqueeze(0)

                predictions = vit(inputs)

                targets = targets.expand(inputs.shape[0], -1, -1, -1)
                loss = loss_fn(predictions, targets)
                total_val_loss += loss.item()


        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Validation Loss (epoch {epoch + 1}): {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(vit.state_dict(), checkpoint_path)  # Save only best model
            counter = 0  # Reset counter
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        # Adjust learning rate based on validation loss
        scheduler.step(avg_val_loss)

    # Save Model
    torch.save(vit.state_dict(), checkpoint_path)
    print("Saved model to vit_model.pth")


def main():
    chunked_training_simple(
        folder_2d="sample_data/2D",
        num_time_steps=6,
        batch_size=8,
        num_epochs=20,
        reduced_size=20
    )


if __name__ == "__main__":
    main()