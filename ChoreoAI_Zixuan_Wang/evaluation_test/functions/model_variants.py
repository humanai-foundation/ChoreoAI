import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RotationLayer(nn.Module):
    def __init__(self, theta, device='cpu'):
        super(RotationLayer, self).__init__()
        self.R = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 0]]) * torch.cos(theta) + torch.Tensor([[0, -1, 0], [1, 0, 0], [0, 0, 0]]) * torch.sin(theta) + torch.Tensor([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        self.R = self.R.to(device)

    def forward(self, x):
        return torch.matmul(x, self.R)

class Encoder(nn.Module):
    def __init__(self, n_vtx, latent_dim, n_units, device='cpu'):
        super(Encoder, self).__init__()
        # self.rotation = RotationLayer(theta=theta, device=device)
        self.lstm = nn.LSTM(input_size=n_vtx * 3, hidden_size=n_units, batch_first=True, num_layers=2)
        self.mean = nn.Linear(in_features=n_units, out_features=latent_dim)
        self.log_var = nn.Linear(in_features=n_units, out_features=latent_dim)
    
    def forward(self, x):
        # if is_train:
        #     x = self.rotation(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        _, (h_n, _) = self.lstm(x)
        z_mean = self.mean(h_n[-1])
        z_log_var = self.log_var(h_n[-1])
        return z_mean, z_log_var


class Decoder(nn.Module):
    def __init__(self, seq_len, n_vtx, latent_dim=32, n_units=32, device='cpu'):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.n_vtx = n_vtx
        # self.rotation = RotationLayer(theta=theta, device=device)
        self.linear = nn.Linear(latent_dim, n_units)
        self.lstm = nn.LSTM(input_size=n_units, hidden_size=n_units, batch_first=True, num_layers=2)
        self.out = nn.Conv1d(in_channels=n_units, out_channels=self.n_vtx * 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.linear(x))
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.reshape(-1, lstm_out.shape[2], lstm_out.shape[1])
        lstm_out = self.out(lstm_out)
        lstm_out = lstm_out.reshape(-1, lstm_out.shape[2], self.n_vtx, 3)
        # if is_train:
        #     lstm_out = self.rotation(lstm_out)
        return lstm_out


class VAELSTM(nn.Module):
    def __init__(self, seq_len, latent_dim, n_units, reduced_joints=False, device='cuda'):
        super(VAELSTM, self).__init__()
        self.num_joints = 18 if reduced_joints else 53
        self.device = device

        # theta = 2 * np.pi * torch.rand(1) 

        self.encoder = Encoder(n_vtx=self.num_joints, latent_dim=latent_dim, n_units=n_units, device=device)
        self.decoder = Decoder(seq_len=seq_len, n_vtx=self.num_joints, device=device, latent_dim=latent_dim, n_units=n_units)

    def sample_z(self, mean, log_var):
        batch, dim = mean.shape
        epsilon = torch.randn(batch, dim).to(self.device)
        return mean + torch.exp(0.5 * log_var) * epsilon

    def forward(self, x, is_train=True):
        # input size: [batch_size, seq_len, joints, 3]

        mean, log_var = self.encoder(x)
        z = self.sample_z(mean, log_var)
        x = self.decoder(z)
        return x, mean, log_var
