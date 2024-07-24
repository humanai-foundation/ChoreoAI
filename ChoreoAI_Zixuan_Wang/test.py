import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
import random

from data.dataset import DancerDataset
from model.model_pipeline import Pipeline
from model.transformer import DanceTransformer

def preprocess_dataset(dancer_np):
    dancer1_np = dancer_np[::2, :, :]
    dancer2_np = dancer_np[1::2, :, :]
    return dancer1_np, dancer2_np


def create_test_dataset(dataset_dir):
    dancer_np = np.load('dataset/' + dataset_dir)
    dancer1_np, dancer2_np = preprocess_dataset(dancer_np)
    dataset = DancerDataset(torch.from_numpy(dancer1_np), torch.from_numpy(dancer2_np), 64)

    train_size = int(0.9 * len(dataset))
    test_dataset = Subset(dataset, range(train_size, len(dataset)))

    return test_dataset


def test():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # dataset_names = ['pose_extraction_img_9085.npy', 'pose_extraction_ilya_hannah_dyads.npy', 'pose_extraction_hannah_cassie.npy', 'pose_extraction_dyads_rehearsal_leah.npy']
    
    test_dataset = create_test_dataset('pose_extraction_img_9085.npy')
    test_dict_data = test_dataset[random.randint(0, len(test_dataset) - 1)]

    # [seq_len, 29, 3]
    dancer1_data = test_dict_data['dancer1'].to(device)
    dancer2_data = test_dict_data['dancer2'].to(device)

    model = Pipeline()
    net = DanceTransformer(16, 16, 16, 16, 8, 4, 64).to(device)
    model.load_network(net, "result/best_model.pth")

    # test
    dancer1_data = torch.tensor(dancer1_data[None, :], dtype=torch.float32)
    dancer2_data = torch.tensor(dancer2_data[None, :], dtype=torch.float32)

    with torch.no_grad():
        linear_1, multihead_attention_1, spatial_temporal_module, fc_1 = net.linear_1, net.multihead_attention_1, net.spatial_temporal_module, net.fc_1
        decoder_2 = net.decoder_2
        
        seq1_fea = linear_1(dancer1_data.view(1, 64, -1))
        seq1_fea, _ = multihead_attention_1(seq1_fea, seq1_fea, seq1_fea)
        seq1_fea, hidden_1 = spatial_temporal_module(seq1_fea)
        aux_seq_1 = fc_1(seq1_fea)
        aux_seq_1 = aux_seq_1.view(1, 64, 29, 3)

        aux_seq_2 = torch.zeros(1, 64, 29, 3, dtype=torch.float32).to(device)

        # full
        latent_space_2 = torch.tensor(np.random.normal(0, 0.1, (1, 16)), dtype=torch.float32).to(device)
        pred_2_full = decoder_2(latent_space_2, aux_seq_1 + aux_seq_2)

        print(pred_2_full.shape)

        # incremental
        pred_2_incremental = torch.zeros(1, 64, 29, 3, dtype=torch.float32).to(device)
        # auto-regressive
        for i in range(64):
            # every time only generate 1 frame
            # [1, 64, 29, 3]
            tmp = decoder_2(latent_space_2, aux_seq_1 + pred_2_incremental)
            pred_2_incremental[:, i, :, :] = tmp[:, i, :, :]

        np.save("seq1.npy", dancer1_data.squeeze(0).detach().cpu().numpy())
        np.save("seq2_full.npy", pred_2_full.squeeze(0).detach().cpu().numpy())
        np.save("seq2_incremental.npy", pred_2_incremental.squeeze(0).detach().cpu().numpy())

if __name__ == '__main__':
    test()
