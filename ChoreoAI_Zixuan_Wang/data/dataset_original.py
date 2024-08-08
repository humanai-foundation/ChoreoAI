import torch


class DancerDatasetOriginal(torch.utils.data.Dataset):
    def __init__(self, data1, data2, seq_len):
        self.data1 = data1
        self.data2 = data2
        self.seq_len = seq_len
        self.dataset_length = len(self.data1) - self.seq_len - 1

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        return {
            'dancer1': self.data1[index: index + self.seq_len],
            'dancer2': self.data2[index: index + self.seq_len],
            'dancer1_next_timestamp': self.data1[index + 1: index + self.seq_len + 1],
            'dancer2_next_timestamp': self.data2[index + 1: index + self.seq_len + 1],
        }
