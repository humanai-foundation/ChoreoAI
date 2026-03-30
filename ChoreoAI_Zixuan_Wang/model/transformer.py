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
        # Fix: index by sequence length (dim 1), not batch (dim 0)
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


class VAEForDuetEncoder(nn.Module):
    """
    Variant of VAEForSingleDancerEncoder whose input linear accepts
    29 * 9 features instead of 29 * 3, to accommodate the richer
    proximity representation used by VAEForDuet.
    Everything else (positional encoding, attention, LSTM, mean/log_var
    heads) is identical to the single-dancer encoder.
    """
    def __init__(self, linear_num_features, n_head, latent_dim,
                 proximity_dim: int = 29 * 9):
        super(VAEForDuetEncoder, self).__init__()
        self.linear = nn.Linear(proximity_dim, linear_num_features)
        self.pos_encoding = PositionalEncoding(linear_num_features)
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=linear_num_features, num_heads=n_head, batch_first=True
        )
        self.lstm = nn.LSTM(
            input_size=linear_num_features,
            hidden_size=linear_num_features,
            num_layers=2,
            batch_first=True,
        )
        self.mean = nn.Linear(linear_num_features, latent_dim)
        self.log_var = nn.Linear(linear_num_features, latent_dim)

    def forward(self, x):
        # x: [B, T, 29*9] — already flattened by VAEForDuet.forward
        x = self.pos_encoding(self.linear(x))
        attn_output, _ = self.multihead_attention(x, x, x)
        _, (hidden, _) = self.lstm(attn_output)
        return self.mean(hidden[-1]), self.log_var(hidden[-1])


class VAEForDuet(nn.Module):
    """
    Duet VAE with enriched proximity encoding.

    Instead of the single unsigned distance  abs(d1 - d2)  (29*3 features),
    the proximity tensor now stacks three complementary signals per joint:

        unsigned_dist  = |d1 - d2|          direction-agnostic magnitude
        signed_diff    = d1 - d2            who is ahead/behind on each axis
        contact_flag   = (|d1-d2| < 0.1)   binary near-contact indicator

    Concatenated along the joint-feature axis this gives 29 * 9 features.
    The richer signal lets the duet encoder distinguish mirroring from
    following, and detect near-contact moments that strongly shape
    partner choreography.

    contact_threshold controls the Euclidean-per-axis distance below which
    two joints are considered "in contact". Default 0.1 assumes unit-scale
    joint positions; tune to your dataset's coordinate range.
    """
    def __init__(self, linear_num_features, n_head, latent_dim, n_units,
                 seq_len, device='cuda', contact_threshold: float = 0.1):
        super(VAEForDuet, self).__init__()
        self.encoder = VAEForDuetEncoder(linear_num_features, n_head, latent_dim)
        self.decoder = VAEForSingleDancerDecoder(latent_dim, n_units, seq_len)
        self.device = device
        self.contact_threshold = contact_threshold

    def sample_z(self, mean, log_var):
        batch, dim = mean.shape
        epsilon = torch.randn(batch, dim).to(self.device)
        return mean + torch.exp(0.5 * log_var) * epsilon

    def forward(self, d1, d2):
        # d1, d2: [B, T, 29, 3]
        unsigned_dist = torch.abs(d1 - d2)                           # [B, T, 29, 3]
        signed_diff   = d1 - d2                                      # [B, T, 29, 3]
        contact_flag  = (unsigned_dist < self.contact_threshold).float()  # [B, T, 29, 3]

        # Stack along the feature axis then flatten joints+features together
        proximity = torch.cat([unsigned_dist, signed_diff, contact_flag], dim=-1)  # [B, T, 29, 9]
        proximity = proximity.reshape(proximity.shape[0], proximity.shape[1], -1)  # [B, T, 261]

        mean, log_var = self.encoder(proximity)
        z = self.sample_z(mean, log_var)
        x = self.decoder(z)
        return x, mean, log_var


class DuetFusionAttention(nn.Module):
    """
    Cross-attention fusion module.

    Replaces the naive elementwise addition  `memory = out_i + out_duet`
    with a learned attention mechanism:

        memory_i = LayerNorm(out_i + CrossAttn(Q=out_i, K=out_duet, V=out_duet))

    This allows the model to selectively pull relevant parts of the shared
    duet representation into each dancer's motion stream, rather than
    blending every dimension equally regardless of relevance.

    Because the VAE decoder outputs 29*3=87 features (not divisible by
    typical head counts), the module projects inputs into an internal
    attention dimension (attn_dim, default 64) that IS head-divisible,
    runs attention there, and projects back to d_model. The residual is
    added in the original d_model space so no information is lost.

    Args:
        d_model:   Feature dimension of the VAE decoder outputs (29*3).
        n_head:    Number of attention heads. Must divide attn_dim evenly.
        attn_dim:  Internal dimension for the attention computation.
                   Defaults to d_model rounded down to the nearest multiple
                   of n_head, so you never have to set this manually.
        dropout:   Dropout applied inside the attention layer.
    """
    def __init__(self, d_model: int, n_head: int, attn_dim: int = None, dropout: float = 0.1):
        super(DuetFusionAttention, self).__init__()
        if attn_dim is None:
            # Round down to nearest n_head multiple
            attn_dim = (d_model // n_head) * n_head
        self.proj_in  = nn.Linear(d_model, attn_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True,
        )
        self.proj_out = nn.Linear(attn_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dancer_out: torch.Tensor, duet_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dancer_out: [batch, seq_len, d_model]  — individual VAE decoder output
            duet_out:   [batch, seq_len, d_model]  — duet VAE decoder output

        Returns:
            memory:     [batch, seq_len, d_model]  — fused representation
        """
        q = self.proj_in(dancer_out)   # project to attn_dim
        k = self.proj_in(duet_out)
        v = self.proj_in(duet_out)

        attn_out, _ = self.cross_attn(query=q, key=k, value=v)
        attn_out = self.proj_out(attn_out)   # project back to d_model

        # Residual connection + layer norm for training stability
        memory = self.norm(dancer_out + self.dropout(attn_out))
        return memory


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
        # Fix: use x.size(1) for sequence length when batch_first=True
        x = x + self.pe[:x.size(1), :, :x.size(2)].transpose(0, 1)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead=8, num_layers=2, dim_feedforward=256, memory_dim=None):
        super(TransformerDecoder, self).__init__()
        # tgt always comes from raw joint positions (29*3)
        self.tgt_linear = nn.Linear(29 * 3, d_model)
        # memory comes from DuetFusionAttention whose output is n_units (= d_model here),
        # but we keep a separate projection so the two paths are decoupled and
        # memory_dim can differ from 29*3 without touching tgt_linear.
        self.memory_linear = nn.Linear(memory_dim if memory_dim is not None else d_model, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 29 * 3)

    def forward(self, tgt, memory):
        batch_size, seq_len, num_joints, joint_dim = tgt.shape
        tgt = tgt.view(batch_size, seq_len, num_joints * joint_dim)

        tgt    = self.tgt_linear(tgt)        # [B, T, d_model]
        memory = self.memory_linear(memory)  # [B, T, d_model]

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

        # VAE decoders output [B, T, 29*3] — fusion d_model must match
        self.fusion_1 = DuetFusionAttention(d_model=29 * 3, n_head=n_head)
        self.fusion_2 = DuetFusionAttention(d_model=29 * 3, n_head=n_head)

        self.transformer_decoder_1 = TransformerDecoder(linear_num_features, memory_dim=29 * 3)
        self.transformer_decoder_2 = TransformerDecoder(linear_num_features, memory_dim=29 * 3)
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

            # Cross-attention fusion:
            #   dancer 1 queries the duet stream to build its context memory
            #   dancer 2 queries the duet stream independently
            # Shape: [batch, seq_len, n_units] throughout
            memory_1 = self.fusion_1(dancer_out=out_1, duet_out=out_duet)
            memory_2 = self.fusion_2(dancer_out=out_2, duet_out=out_duet)

            # Transformer decoder uses fused memory to predict the other dancer
            pred_2 = self.transformer_decoder_1(d2_normalized, memory_1)
            pred_1 = self.transformer_decoder_2(d1_normalized, memory_2)

        return pred_1, pred_2, mean_1, log_var_1, mean_2, log_var_2, mean_duet, log_var_duet, is_simplified_model, out_1, out_2


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DancerTransformer(
        linear_num_features=64,
        n_head=8,
        latent_dim=32,
        n_units=64,
        seq_len=64,
        no_input_prob=0.2,
    ).to(device)

    input_1 = torch.rand(8, 64, 29, 3).to(device)
    input_2 = torch.rand(8, 64, 29, 3).to(device)

    out = model(input_1, input_2)
    pred_1, pred_2 = out[0], out[1]
    print("pred_1 shape:", pred_1.shape)  # expect [8, 64, 29, 3]
    print("pred_2 shape:", pred_2.shape)
