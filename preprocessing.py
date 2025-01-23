import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split

# prepares the raw dataset for my ViT model
# data slicing: input and output pairs (ex: use the first 6 time steps to predict the 7th time step)
# data splitting:
# patch splitting: divide each spatial grid (900 x 600) into smaller patches (30 x 30)


# Gathers all .pth files and sorts in ascending order
def list_pth_files(folder_path):
    all_files = sorted([
        os.path.join(folder_path, f)  # lists all files in folder
        for f in os.listdir(folder_path)  # ensures the file ends in .pth
        if f.endswith('.pth')
    ])
    return all_files


# Splits the data into training, validating, and testing chronologically
    # 80/10/10 ratio:
    #  - first 80% train
    #  - next 10% val
    #  - last 10% test
def split_file_paths_into_train_val_test(all_files, train_end=80, val_end=90):
    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]
    return train_files, val_files, test_files


# Loads each .pth file into a list of tensors
def load_pth_files(file_paths):
    data_list = []
    for fp in file_paths:
        tensor = torch.load(fp)  # loads the file as a tensor
        data_list.append(tensor)  # appends that tensor to a list
    return data_list


# Iterates over list of tensors and pairs each tensor with the next one in the list
# Ex: Use the first 6 time steps to predict the 7th time step
def process_consecutive_files(data_list, num_time_steps):

    inputs = []
    outputs = []

    for i in range(len(data_list) - 1):
        current_data = data_list[i]
        next_data = data_list[i + 1]

        # Slide over time dimension
        for t in range(current_data.shape[3] - num_time_steps):
            input_seq = current_data[0, :, :, t:t + num_time_steps]
            target = next_data[0, :, :, t]

            inputs.append(input_seq)
            outputs.append(target)

    return inputs, outputs

# Function to split data into training, validation, and testing sets
# Training: model learns from the data by adjusting its parameters to minimize the loss
# Validation: fine tunes the model and evaluates its performance on unseen data to prevent overfitting
# Testing: assesses the final performance of the trained model on completely unseen data.

# This was for my initial load everything at once approach
# Not needed for my chunked approach

# def create_dataset_splits(folder_path, num_time_steps=6):
#
#     # List all pth files
#     all_files = list_pth_files(folder_path)
#
#     # Chronological split (example: 80/10/10)
#     train_files, val_files, test_files = split_file_paths_into_train_val_test(
#         all_files, train_end=80, val_end=90
#     )
#
#     # Load data
#     train_data = load_pth_files(train_files)
#     val_data = load_pth_files(val_files)
#     test_data = load_pth_files(test_files)
#
#     # Convert to (inputs, outputs)
#     inputs_train, outputs_train = process_consecutive_files(train_data, num_time_steps)
#     inputs_val, outputs_val = process_consecutive_files(val_data,   num_time_steps)
#     inputs_test, outputs_test = process_consecutive_files(test_data,  num_time_steps)
#
#     return (inputs_train, outputs_train), (inputs_val, outputs_val), (inputs_test, outputs_test)

