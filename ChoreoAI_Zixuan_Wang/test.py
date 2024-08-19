import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
import random

from data.dataset_original import DancerDatasetOriginal
from model.model_pipeline import Pipeline
from model.transformer import DancerTransformer

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


def test():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    seq_length = 64
    test_set_idx = 1000

    # dataset_names = ['pose_extraction_img_9085.npy', 'pose_extraction_ilya_hannah_dyads.npy', 'pose_extraction_hannah_cassie.npy', 'pose_extraction_dyads_rehearsal_leah.npy']
    
    test_dataset = create_test_dataset('pose_extraction_img_9085.npy')
    print(len(test_dataset))
    test_dict_data = test_dataset[test_set_idx]
    test_dict_data_next_timestamap = test_dataset[test_set_idx + seq_length]

    # [seq_len, 29, 3]
    dancer1_data = test_dict_data['dancer1'].to(device)
    dancer2_data = test_dict_data['dancer2'].to(device)

    print(dancer1_data.shape)

    dancer1_data_next_timestamp = test_dict_data_next_timestamap['dancer1'].to(device)
    dancer2_data_next_timestamp = test_dict_data_next_timestamap['dancer2'].to(device)

    # 128 frames
    dancer1_data_all = torch.cat((dancer1_data, dancer1_data_next_timestamp), dim=0)
    dancer2_data_all = torch.cat((dancer2_data, dancer2_data_next_timestamp), dim=0)

    print(dancer2_data_all.shape)

    model = Pipeline()
    net = DancerTransformer(64, 8, 32, 32, 64).to(device)
    model.load_network(net, "result/best_model_0811.pth")

    # test
    # 128 frames
    d1 = torch.tensor(dancer1_data_all[None, :], dtype=torch.float32)
    # 64 frames
    d2 = torch.tensor(dancer2_data[None, :], dtype=torch.float32)

    with torch.no_grad():
        vae_1, vae_2, vae_duet, transformer_decoder_1, transformer_decoder_2 = net.vae_1, net.vae_2, net.vae_duet, net.transformer_decoder_1, net.transformer_decoder_2

        for i in range(64):
            # [1, 64, 29, 3]
            combined = torch.stack((d1[:, i: i + seq_length, :, :], d2[:, i: i + seq_length, :, :]), dim=-1)
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
            pred_2 = transformer_decoder_1(d2_normalized, memory_1)

            last_dim = pred_2[:, -1, :, :].unsqueeze(1)
            d2 = torch.cat((d2, last_dim), dim=1)

        np.save("seq1_original_3.npy", dancer1_data_all.detach().cpu().numpy())
        np.save("seq2_original_3.npy", dancer2_data_all.detach().cpu().numpy())
        np.save("seq2_next_ts_3.npy", d2.squeeze(0).detach().cpu().numpy())


if __name__ == '__main__':
    test()
