import numpy as np
import torch
from torch.utils.data import Subset

from model.transformer import DancerTransformer
from data.dataset_original import DancerDatasetOriginal


def preprocess_dataset(dancer_np):
    dancer1_np = dancer_np[::2, :, :]
    dancer2_np = dancer_np[1::2, :, :]
    return dancer1_np, dancer2_np


def create_test_dataset(dataset_dir):
    dancer_np = np.load('dataset/' + dataset_dir)
    dancer1_np, dancer2_np = preprocess_dataset(dancer_np)
    dataset = DancerDatasetOriginal(torch.from_numpy(dancer1_np), torch.from_numpy(dancer2_np), 64)

    train_size = int(0.9 * len(dataset))
    test_dataset = Subset(dataset, range(train_size, len(dataset)))

    return test_dataset


def generate_dancer_data(test_dataset, test_set_idx, seq_len, net: DancerTransformer, device, generate_d1=True):
    test_dict_data = test_dataset[test_set_idx]
    test_dict_data_next_timestamap = test_dataset[test_set_idx + seq_len]
    dancer1_data = test_dict_data['dancer1'].to(device)
    dancer2_data = test_dict_data['dancer2'].to(device)

    dancer1_data_next_timestamp = test_dict_data_next_timestamap['dancer1'].to(device)
    dancer2_data_next_timestamp = test_dict_data_next_timestamap['dancer2'].to(device)

    dancer1_data_all = torch.cat((dancer1_data, dancer1_data_next_timestamp), dim=0)
    dancer2_data_all = torch.cat((dancer2_data, dancer2_data_next_timestamp), dim=0)

    d1_all = dancer1_data_all[None, :].clone().detach().float()
    d2_all = dancer2_data_all[None, :].clone().detach().float()

    # 64 frames
    d1 = dancer1_data[None, :].clone().detach().float()
    d2 = dancer2_data[None, :].clone().detach().float()

    with torch.no_grad():
        vae_1, vae_2, vae_duet, transformer_decoder_1, transformer_decoder_2 = net.vae_1, net.vae_2, net.vae_duet, net.transformer_decoder_1, net.transformer_decoder_2

        for i in range(64):
            # [1, 64, 29, 3]
            if generate_d1:
                combined = torch.stack((d1[:, i: i + seq_len, :, :], d2_all[:, i: i + seq_len, :, :]), dim=-1)
            else:
                combined = torch.stack((d1_all[:, i: i + seq_len, :, :], d2[:, i: i + seq_len, :, :]), dim=-1)

            mean = combined.mean(dim=(2, 3, 4), keepdim=True)
            std = combined.std(dim=(2, 3, 4), keepdim=True)
            combined_normalized = (combined - mean) / (std + 1e-6)

            d1_normalized = combined_normalized[..., 0]
            d2_normalized = combined_normalized[..., 1]

            out_1, mean_1, log_var_1 = vae_1(d1_normalized)
            out_2, mean_2, log_var_2 = vae_2(d2_normalized)
            out_duet, mean_duet, log_var_duet = vae_duet(d1_normalized, d2_normalized)

            # [batch_size, seq_len, 29 * 3]
            memory_1 = out_1 + out_duet
            memory_2 = out_2 + out_duet

            # transformer decoder [1, 64, 29, 3]
            if generate_d1:
                pred_1 = transformer_decoder_2(d1_normalized, memory_2)
                last_dim = pred_1[:, -1, :, :].unsqueeze(1)
                d1 = torch.cat((d1, last_dim), dim=1)
            else:
                pred_2 = transformer_decoder_1(d2_normalized, memory_1)
                last_dim = pred_2[:, -1, :, :].unsqueeze(1)
                d2 = torch.cat((d2, last_dim), dim=1)


    seq1_original = dancer1_data_all[..., [2, 0, 1]]
    seq1_original[..., 2] = -seq1_original[..., 2]
    seq1_original = seq1_original.cpu().numpy()

    seq2_original = dancer2_data_all[..., [2, 0, 1]]
    seq2_original[..., 2] = -seq2_original[..., 2]
    seq2_original = seq2_original.cpu().numpy()

    seq1_next_ts = None
    seq2_next_ts = None

    if generate_d1:
        seq1_next_ts = d1[0][..., [2, 0, 1]]
        seq1_next_ts[..., 2] = -seq1_next_ts[..., 2]
        seq1_next_ts = seq1_next_ts.cpu().numpy()
    else:
        seq2_next_ts = d2[0][..., [2, 0, 1]]
        seq2_next_ts[..., 2] = -seq2_next_ts[..., 2]
        seq2_next_ts = seq2_next_ts.cpu().numpy()
    
    return seq1_original, seq2_original, seq1_next_ts, seq2_next_ts
