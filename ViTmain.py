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
    batch_size=16,
    num_epochs=20,
    reduced_size=20,
    checkpoint_path="vit_model.pth"
):

    # Splits files into (train, val, test) chronologically.
    # For each epoch, loads training data in chunks of 'chunk_size' files.
    # Converts each chunk to tensors, does a training loop with 'batch_size' mini-batches.
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

    # Create Dataloaders
    # Dataloaders take the dataset and load it in batches
    # They shuffle the batches to improve generalization
    # The batches are shuffled but the individual sequences are in tact
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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

    # Mean squared error loss function
    # Calculates average squared difference between predicted and actual values
    loss_fn = nn.MSELoss()

    # Defines optimizer for training
    # Optimizer updates model weights during training
    # Tells optimizer to update ViT model's weights
    # Weight decay helps prevent overfitting and memorization of the data
    optimizer = optim.AdamW(vit.parameters(), lr=1e-4, weight_decay=1e-3)

    # Reduces learning rate when the model stops improving
    # Monitors validation loss and reduces learning rate when it stops decreasing
    # When triggered the LR is multiplied by 0.5
    # If validation loss does not improve for 3 epochs, it reduces the learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

   # Loads checkpoint if exists
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=device)
        state_dict.pop("positional_encoding.positional_encoding", None)  # Remove mismatched keys
        vit.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully.")
    else:
       print(f"No checkpoint found, starting fresh.")


    best_val_loss = float("inf")
    patience = 7  # Stops training completely if validation loss doesn't improve for 7 epochs
    counter = 0   # Keeps track of how many epochs have passed without an improvement

    # Training Loop
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        vit.train()   # Sets model to training mode
        total_loss = 0  # Initialize total loss for epoch

        for inputs, targets in train_loader:  # Loads mini batches from the dataset
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.permute(0, 1, 2, 3)  # Should be (B, T, H, W)

            optimizer.zero_grad()  # Clears previous gradients
            predictions = vit(inputs)  # Gets models predictions
            loss = loss_fn(predictions, targets)  # Computes loss between predictions and targets using MSE
            loss.backward()  # Computes gradients
            torch.nn.utils.clip_grad_norm_(vit.parameters(), max_norm=1.0)  # Prevents exploding gradients
            optimizer.step()  # Updates the model's weights

            total_loss += loss.item()  # Track the loss

        avg_train_loss = total_loss / len(train_loader)  # Computes average training loss
        print(f"Training Loss (epoch {epoch + 1}): {avg_train_loss:.4f}")

        # Validation
        vit.eval()  # Sets model to evaluation mode
        total_val_loss = 0  # Initializes validation loss
        with torch.no_grad():  # Disables gradients to speed up validation
            for inputs, targets in val_loader:  # Iterates over mini batches

                inputs, targets = inputs.to(device), targets.to(device)

                predictions = vit(inputs)  # Gets models predictions

                loss = loss_fn(predictions, targets)  # Computes loss
                total_val_loss += loss.item()  # Accumulates loss


        avg_val_loss = total_val_loss / len(val_loader)  # Computes average validation loss
        print(f"Validation Loss (epoch {epoch + 1}): {avg_val_loss:.4f}")


        # If validation loss improves, save model
        # If it doesn't improve, increase counter
        # Stop training if no improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(vit.state_dict(), checkpoint_path)  # Save only best model
            counter = 0  # Reset counter
        else:
            counter += 1  # Increase patience counter

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
        batch_size=16,
        num_epochs=20,
        reduced_size=20
    )


if __name__ == "__main__":
    main()