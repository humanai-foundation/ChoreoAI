import torch


class DancerDatasetAugmented(torch.utils.data.Dataset):
    def __init__(self, data1, data2, seq_len, augmentation_factor=2, mean=0, std=0.01):
        self.data1 = data1
        self.data2 = data2
        self.seq_len = seq_len
        self.dataset_length = len(self.data1) - self.seq_len - 1
        self.augmentation_factor = augmentation_factor
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.dataset_length * self.augmentation_factor

    def __getitem__(self, index):
        if index < self.dataset_length:
            dancer1_data = self.data1[index: index + self.seq_len]
            dancer2_data = self.data2[index: index + self.seq_len]
        
        else:
            dancer1_data = self.add_noise(self.data1[index: index + self.seq_len])
            dancer2_data = self.add_noise(self.data2[index: index + self.seq_len])

        return {
            'dancer1': dancer1_data,
            'dancer2': dancer2_data,
            # label
            'dancer1_next_timestamp': self.data1[index + 1: index + self.seq_len + 1],
            'dancer2_next_timestamp': self.data2[index + 1: index + self.seq_len + 1],
        }

    def add_noise(self, data):
        noise = torch.randn_like(data) * self.std + self.mean
        return data + noise
