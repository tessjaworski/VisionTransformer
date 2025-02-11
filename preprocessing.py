import os
import torch
import numpy as np
from torch.utils.data import Dataset

# prepares the raw dataset for my ViT model


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


class WeatherDataset(Dataset):
    def __init__(self, file_paths, num_time_steps=6, reduced_size=20):
        self.file_paths = file_paths
        self.num_time_steps = num_time_steps
        self.reduced_size = reduced_size
        self.data_list = self.load_data()

    def load_data(self):
        data_list = []
        for file_path in self.file_paths:
            tensor = torch.load(file_path)  # Load .pth file as tensor
            data_list.append(tensor)
        return data_list

    def __len__(self):
        return len(self.data_list) - 1  # Need consecutive files

    def __getitem__(self, index):
        current_data = self.data_list[index]
        next_data = self.data_list[index + 1]

        assert current_data.shape == next_data.shape, "Mismatched tensor shapes!"

        # Random time step selection
        t = np.random.randint(0, current_data.shape[3] - self.num_time_steps)

        # Extract input sequence and target frame
        input_seq = current_data[0, :self.reduced_size, :self.reduced_size, t:t + self.num_time_steps]
        target = next_data[0, :self.reduced_size, :self.reduced_size, t]

        # Adjust shape to match model input
        input_seq = input_seq.permute(2, 0, 1)  # (T, H, W)
        target = target.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)

        return input_seq.float(), target.float()