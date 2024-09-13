import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
        self.vae_1 = VAEForSingleDancer(linear_num_features, n_head, latent_dim, n_units, seq_len, default_log_var)
        self.vae_2 = VAEForSingleDancer(linear_num_features, n_head, latent_dim, n_units, seq_len, default_log_var)
        self.vae_duet = VAEForDuet(linear_num_features, n_head, latent_dim, n_units, seq_len)
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


if __name__ == '__main__':
    model = DancerTransformer(64, 8, 32, 32, 64).to('cuda')
    print(model)
    input_1 = torch.rand(8, 64, 29, 3).to('cuda')
    input_2 = torch.rand(8, 64, 29, 3).to('cuda')

    out_1, out_2, _, _, _, _, _, _ = model(input_1, input_2)
    print(out_1.shape)
