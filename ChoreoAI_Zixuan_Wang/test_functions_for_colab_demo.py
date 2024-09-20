import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import Subset

import math
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import juggle_axes
from IPython.display import display, HTML


point_labels = [
    'pelvis', 'left_hip', 'right_hip',      # 2
    'spine1', 'left_knee', 'right_knee',    # 5
    'spine2', 'left_ankle', 'right_ankle',  # 8
    'spine3', 'left_foot', 'right_foot',    # 11
    'neck', 'left_collar', 'right_collar',  # 14
    'jaw',                                  # 15
    'left_shoulder', 'right_shoulder',      # 17
    'left_elbow', 'right_elbow',            # 19
    'left_wrist', 'right_wrist',            # 21
    'left_thumb', 'right_thumb',
    'head', 'left_middle', 'right_middle',  # 26
    'left_bigtoe', 'right_bigtoe'
]

skeleton_lines = [
    #  ( (start group), (end group) ),
    (('pelvis',), ('left_hip',)),
    (('pelvis',), ('right_hip',)),
    (('left_hip',), ('left_knee',)),
    (('right_hip',), ('right_knee',)),
    (('left_knee',), ('left_ankle',)),
    (('right_knee',), ('right_ankle',)),
    (('left_ankle',), ('left_foot',)),
    (('right_ankle',), ('right_foot',)),
    (('pelvis',), ('spine1',)),
    (('spine1',), ('spine2',)),
    (('spine2',), ('spine3',)),
    (('spine3',), ('neck',)),
    (('spine3',), ('left_collar',)),
    (('spine3',), ('right_collar',)),
    (('neck',), ('jaw',)),
    (('left_collar',), ('left_shoulder',)),
    (('right_collar',), ('right_shoulder',)),
    (('left_shoulder',), ('left_elbow',)),
    (('right_shoulder',), ('right_elbow',)),
    (('left_elbow',), ('left_wrist',)),
    (('right_elbow',), ('right_wrist',)),
    (('left_wrist',), ('left_thumb',)),
    (('right_wrist',), ('right_thumb',)),
    (('neck',), ('head',)),

    # (('left_shoulder',), ('left_middle',)),
    # (('right_shoulder',), ('right_middle',)),
    (('left_ankle',), ('left_bigtoe',)),
    (('right_ankle',), ('right_bigtoe',)),
]


skeleton_idxs = []
for g1, g2 in skeleton_lines:
    entry = []
    entry.append([point_labels.index(l) for l in g1])
    entry.append([point_labels.index(l) for l in g2])
    skeleton_idxs.append(entry)

# Cloud of every point connected:
cloud_idxs = []
for i in range(29):
    for j in range(29):
        entry = []
        entry.append([i])
        entry.append([j])
        cloud_idxs.append(entry)

all_idxs = skeleton_idxs + cloud_idxs

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_line_segments(seq, zcolor=None, cmap=None, cloud=False, edge_types=None, edge_class=None):
    xline = np.zeros((seq.shape[0], len(all_idxs), 3, 2))
    if cmap:
        colors = np.zeros((len(all_idxs), 4))
    for edge, (joint1, joint2) in enumerate(all_idxs):
        xline[:, edge, :,0] = np.mean(seq[:, joint1], axis=1)
        xline[:, edge, :,1] = np.mean(seq[:, joint2], axis=1)
        if cmap:
            if edge_types is not None:
                if edge >= len(skeleton_idxs): # cloud edges
                    if edge_types[edge - len(skeleton_idxs), edge_class] == 1:
                        colors[edge] = cmap(1)
                    else:
                        colors[edge] = cmap(0)
            else:
                colors[edge] = cmap(0)
    if cmap:
        return xline, colors
    else:
        return xline

# put line segments on the given axis, with given colors
def put_lines(ax, segments, color=None, lw=2.5, alpha=None, skeleton=True, skeleton_alpha=0.3, cloud=False, cloud_alpha=0.03, threshold=0, edge_types=None, edge_opacities=None, edge_class=None):
    lines = []
    ### Main skeleton
    for i in tqdm(range(len(skeleton_idxs)), desc="Skeleton lines"):
        if isinstance(color, (list, tuple, np.ndarray)):
            c = color[i]
        else:
            c = color

        if skeleton: alpha = skeleton_alpha
        else: alpha = 0

        ### THESE LINES PLOT THE MAIN SKELETON
        l = ax.plot(np.linspace(segments[i, 0, 0],segments[i, 0, 1], 2),
                np.linspace(segments[i, 1, 0], segments[i, 1, 1], 2),
                np.linspace(segments[i, 2, 0], segments[i, 2, 1], 2),
                color=c,
                alpha=alpha,
                lw=lw)[0]
        lines.append(l)

    if cloud:
        ### Cloud of all-connected joints
        for i in tqdm(range(len(cloud_idxs)), desc="Cloud lines"):
            if isinstance(color, (list, tuple, np.ndarray)):
                c = color[i]
            else:
                c = color

            l = ax.plot(
                np.linspace(segments[i, 0, 0], segments[i, 0, 1], 2),
                np.linspace(segments[i, 1, 0], segments[i, 1, 1], 2),
                np.linspace(segments[i, 2, 0], segments[i, 2, 1], 2),
                color=c,
                alpha=cloud_alpha,
                lw=lw)[0]
            lines.append(l)
    return lines

# animate a video of the stick figure.
# `ghost` may be a second sequence, which will be superimposed on the primary sequence.
# If ghost_shift is given, the primary and ghost sequence will be separated laterally by that amount.
# `zcolor` may be an N-length array, where N is the number of vertices in seq, and will be used to color the vertices. Typically this is set to the avg. z-value of each vtx.
def animate_stick(seq, ghost=None, ghost_shift=0, threshold=0, figsize=None, zcolor=None, pointer=None, ax_lims=(-0.4, 0.4), ay_lims=(-1, 1), az_lims=(-1, 1), speed=45, dot_size=20, dot_alpha=0.5, lw=2.5, cmap='cool_r', pointer_color='black', cloud=False, cloud_alpha=0.03, skeleton=True, skeleton_alpha=0.3, dpi=50):
    if zcolor is None:
        zcolor = np.zeros(seq.shape[1])

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.axes(projection='3d')

    # The following lines eliminate background lines/axes:
    ax.axis('off')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    # ax.set_frame_on(True)

    # set figure background opacity (alpha) to 0:
    fig.patch.set_alpha(0.)

    if ghost_shift and ghost is not None:
        seq = seq.copy()
        ghost = ghost.copy()
        seq[:, :, 0] -= ghost_shift
        ghost[:, :, 0] += ghost_shift

    cm = matplotlib.colormaps[cmap]

    dot_color = "black"
    pts = ax.scatter(seq[0, :, 0], seq[0, :, 1], seq[0, :, 2], c=dot_color, s=dot_size, alpha=dot_alpha)
    ghost_color = 'blue'

    if ghost is not None:
        pts_g = ax.scatter(ghost[0, :, 0], ghost[0, :, 1], ghost[0, :, 2], c=ghost_color, s=dot_size, alpha=dot_alpha)

    if ax_lims:
        ax.set_xlim(*ax_lims)
        ax.set_ylim(*ay_lims)
        ax.set_zlim(*az_lims)

    plt.close(fig)
    xline, colors = get_line_segments(seq, zcolor, cm)
    lines = put_lines(ax, xline[0], color=colors, lw=lw, alpha=0.9, cloud=cloud, cloud_alpha=cloud_alpha, threshold=threshold, skeleton=skeleton, skeleton_alpha=skeleton_alpha)

    if ghost is not None:
        xline_g = get_line_segments(ghost)
        lines_g = put_lines(ax, xline_g[0], ghost_color, lw=lw, alpha=1.0, cloud=cloud, cloud_alpha=cloud_alpha, skeleton=skeleton, skeleton_alpha=skeleton_alpha)

    if pointer is not None:
        vR = 0.15
        dX, dY = vR * np.cos(pointer), vR * np.sin(pointer)
        zidx = point_labels.index('CLAV')
        X = seq[:, zidx, 0]
        Y = seq[:, zidx, 1]
        Z = seq[:, zidx, 2]
        quiv = ax.quiver(X[0], Y[0], Z[0], dX[0], dY[0], 0, color=pointer_color)
        ax.quiv = quiv

    def update(t):
        pts._offsets3d = juggle_axes(seq[t, :, 0], seq[t, :, 1], seq[t, :, 2], 'z')
        for i,l in enumerate(lines):
            if l is not None:
                l.set_data(xline[t, i, :2])
                l.set_3d_properties(xline[t, i, 2])

        if ghost is not None:
            pts_g._offsets3d = juggle_axes(ghost[t, :, 0], ghost[t, :, 1], ghost[t, :, 2], 'z')
            for i, l in enumerate(lines_g):
                l.set_data(xline_g[t, i, :2])
                l.set_3d_properties(xline_g[t, i, 2])

        if pointer is not None:
            ax.quiv.remove()
            ax.quiv = ax.quiver(X[t], Y[t], Z[t], dX[t], dY[t], 0, color=pointer_color)

    return animation.FuncAnimation(
        fig,
        update,
        len(seq),
        interval=speed,
        blit=False,
    )


class VAEForSingleDancerEncoder(nn.Module):
    def __init__(self, linear_num_features, n_head, latent_dim):
        super(VAEForSingleDancerEncoder, self).__init__()
        self.linear = nn.Linear(29 * 3, linear_num_features)
        self.pos_encoding = PositionalEncoding(linear_num_features)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=linear_num_features, num_heads=n_head, batch_first=True)
        self.lstm = nn.LSTM(input_size=linear_num_features, hidden_size=linear_num_features, num_layers=2, batch_first=True)
        self.mean = nn.Linear(in_features=linear_num_features, out_features=latent_dim)
        self.log_var = nn.Linear(in_features=linear_num_features, out_features=latent_dim)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.pos_encoding(self.linear(x))
        attn_output, _ = self.multihead_attention(x, x, x)
        _, (hidden, _) = self.lstm(attn_output)
        z_mean = self.mean(hidden[-1])
        z_log_var = self.log_var(hidden[-1])
        return z_mean, z_log_var


class VAEForSingleDancerDecoder(nn.Module):
    def __init__(self, latent_dim, n_units, seq_len):
        super(VAEForSingleDancerDecoder, self).__init__()
        self.linear = nn.Linear(latent_dim, n_units)
        self.lstm = nn.LSTM(input_size=n_units, hidden_size=n_units, num_layers=2, batch_first=True)
        self.out = nn.Conv1d(in_channels=n_units, out_channels=29 * 3, kernel_size=3, padding=1)
        self.seq_len = seq_len

    def forward(self, x):
        x = F.relu(self.linear(x))
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.reshape(-1, lstm_out.shape[2], lstm_out.shape[1])
        lstm_out = self.out(lstm_out)
        lstm_out = lstm_out.reshape(-1, lstm_out.shape[2], 29 * 3)
        return lstm_out


class VAEForSingleDancer(nn.Module):
    def __init__(self, linear_num_features, n_head, latent_dim, n_units, seq_len, default_log_var, device='cuda'):
        super(VAEForSingleDancer, self).__init__()
        self.encoder = VAEForSingleDancerEncoder(linear_num_features, n_head, latent_dim)
        self.decoder = VAEForSingleDancerDecoder(latent_dim, n_units, seq_len)
        self.device = device
        self.latent_dim = latent_dim
        self.default_log_var = default_log_var

    def sample_z(self, mean, log_var):
        batch, dim = mean.shape
        epsilon = torch.randn(batch, dim).to(self.device)
        return mean + torch.exp(0.5 * log_var) * epsilon

    def forward(self, x, no_input=False):
        batch = x.size()[0]
        if no_input:
            mean, log_var = torch.zeros(batch, self.latent_dim), torch.ones(batch, self.latent_dim) * self.default_log_var
            mean = mean.to(self.device)
            log_var = log_var.to(self.device)
        else:
            mean, log_var = self.encoder(x)
        z = self.sample_z(mean, log_var)
        x = self.decoder(z)
        return x, mean, log_var


class VAEForDuet(nn.Module):
    def __init__(self, linear_num_features, n_head, latent_dim, n_units, seq_len, device='cuda'):
        super(VAEForDuet, self).__init__()
        self.encoder = VAEForSingleDancerEncoder(linear_num_features, n_head, latent_dim)
        self.decoder = VAEForSingleDancerDecoder(latent_dim, n_units, seq_len)
        self.device = device

    def sample_z(self, mean, log_var):
        batch, dim = mean.shape
        epsilon = torch.randn(batch, dim).to(self.device)
        return mean + torch.exp(0.5 * log_var) * epsilon

    def forward(self, d1, d2):
        # proximity
        proximity = torch.abs(d1 - d2)
        mean, log_var = self.encoder(proximity)
        z = self.sample_z(mean, log_var)
        x = self.decoder(z)
        return x, mean, log_var


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :, :x.size(2)]
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead=8, num_layers=2, dim_feedforward=256):
        super(TransformerDecoder, self).__init__()
        self.linear = nn.Linear(29 * 3, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 29 * 3)

    def forward(self, tgt, memory):
        # tgt: [29, 3], memory: [29 * 3]
        batch_size, seq_len, num_joints, joint_dim = tgt.shape
        tgt = tgt.view(batch_size, seq_len, num_joints * joint_dim)

        tgt = self.linear(tgt)
        memory = self.linear(memory)

        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(tgt, memory)
        output = self.fc_out(output)
        output = output.view(batch_size, seq_len, num_joints, joint_dim)
        return output


class DancerTransformer(nn.Module):
    def __init__(self, linear_num_features, n_head, latent_dim, n_units, seq_len, no_input_prob, default_log_var=0.5):
        super(DancerTransformer, self).__init__()
        self.vae_1 = VAEForSingleDancer(linear_num_features, n_head, latent_dim, n_units, seq_len, default_log_var, device=device)
        self.vae_2 = VAEForSingleDancer(linear_num_features, n_head, latent_dim, n_units, seq_len, default_log_var, device=device)
        self.vae_duet = VAEForDuet(linear_num_features, n_head, latent_dim, n_units, seq_len, device=device)
        self.transformer_decoder_1 = TransformerDecoder(linear_num_features)
        self.transformer_decoder_2 = TransformerDecoder(linear_num_features)
        self.no_input_prob = no_input_prob

    def forward(self, d1, d2, is_inference=False):
        rdm_val = torch.randn(1)
        is_simplified_model = False

        combined = torch.stack((d1, d2), dim=-1)
        mean = combined.mean(dim=(2, 3, 4), keepdim=True)
        std = combined.std(dim=(2, 3, 4), keepdim=True)
        combined_normalized = (combined - mean) / (std + 1e-6)

        d1_normalized = combined_normalized[..., 0]
        d2_normalized = combined_normalized[..., 1]

        if not is_inference and rdm_val < self.no_input_prob:
            is_simplified_model = True
            # only focus on one VAE model
            out_1, mean_1, log_var_1 = self.vae_1(d1_normalized)
            out_2, mean_2, log_var_2 = self.vae_2(d2_normalized)
            batch_size, seq_len, _ = out_1.shape
            out_1 = out_1.view(batch_size, seq_len, 29, 3)
            out_2 = out_2.view(batch_size, seq_len, 29, 3)
            pred_1 = None
            pred_2 = None
            mean_duet = None
            log_var_duet = None

        else:
            out_1, mean_1, log_var_1 = self.vae_1(d1_normalized)
            out_2, mean_2, log_var_2 = self.vae_2(d2_normalized)
            out_duet, mean_duet, log_var_duet = self.vae_duet(d1_normalized, d2_normalized)

            # [batch_size, seq_len, 29 * 3]
            memory_1 = out_1 + out_duet
            memory_2 = out_2 + out_duet

            # transformer decoder
            pred_2 = self.transformer_decoder_1(d2_normalized, memory_1)
            pred_1 = self.transformer_decoder_2(d1_normalized, memory_2)

        return pred_1, pred_2, mean_1, log_var_1, mean_2, log_var_2, mean_duet, log_var_duet, is_simplified_model, out_1, out_2
    

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


def load_network(net, load_path, strict=True, param_key='params'):
    if isinstance(net, (DataParallel, DistributedDataParallel)):
        net = net.module
    load_net = torch.load(
        load_path, map_location=lambda storage, loc: storage, weights_only=True)
    if param_key is not None:
        load_net = load_net[param_key]
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    net.load_state_dict(load_net, strict=strict)


def preprocess_dataset(dancer_np):
    dancer1_np = dancer_np[::2, :, :]
    dancer2_np = dancer_np[1::2, :, :]
    return dancer1_np, dancer2_np


def create_test_dataset(dataset_dir):
    dancer_np = np.load(dataset_dir, allow_pickle=True)
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
