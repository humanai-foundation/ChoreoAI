import torch_geometric.nn as geo_nn
from torch_geometric.nn import GCNConv
from torch.nn import Linear, BatchNorm1d, Dropout

from utils import *

# Implementing LSTM variant with GCN layers
class gcn_lstm_cell(nn.Module):
    def __init__(self, n_in, n_out):
        super(gcn_lstm_cell, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        
        # Rebuilding LSTM cell with GCN layers
        self.gcn_i = GCNConv(n_in + n_out, n_out)
        self.gcn_f = GCNConv(n_in + n_out, n_out)
        self.gcn_o = GCNConv(n_in + n_out, n_out)
        self.gcn_g = GCNConv(n_in + n_out, n_out)

    def forward(self, x, h, c, edge_index):
        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=-1)
        
        # Compute gates
        i = torch.sigmoid(self.gcn_i(combined, edge_index))
        f = torch.sigmoid(self.gcn_f(combined, edge_index))
        o = torch.sigmoid(self.gcn_o(combined, edge_index))
        g = torch.tanh(self.gcn_g(combined, edge_index))
        
        # Compute new cell and hidden states
        c_new = f*c + i*g
        h_new = o*torch.tanh(c_new)
        
        return h_new, c_new

# Defining NRI encoder
class nri_encoder(nn.Module):
    def __init__(self, device, n_joints, edge_index_t, n_in, n_hid, n_out, do_prob=0., compact=False):
        super(nri_encoder, self).__init__()

        # Computing edge index given transposed edge index
        self.edge_index = torch.Tensor(edge_index_t).t().long().to(device)
        self.compact = compact

        # Computing the message passing matrices
        self.m_in, self.m_out = message_passing_matrices(n_joints, self.edge_index)
        self.m_in = self.m_in.to(device)
        self.m_out = self.m_out.to(device)

        # Defining the network itself interleaving GCN and MLP layers
        self.conv1 = GCNConv(n_in, n_hid).to(device)
        
        self.mlp1 = Linear(n_hid*2, n_hid).to(device)
        self.bnorm1 = BatchNorm1d(n_hid).to(device)
        self.dropout1 = Dropout(do_prob).to(device)
        
        self.conv2 = GCNConv(n_hid, n_hid).to(device)
        
        if compact:
            self.mlp2 = Linear(n_hid*2, n_hid).to(device)
        else:
            self.mlp2 = Linear(n_hid*3, n_hid).to(device)

        self.bnorm2 = BatchNorm1d(n_hid).to(device)
        self.fc_out = Linear(n_hid, n_out).to(device)
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.1)

            elif isinstance(m, GCNConv):
                nn.init.xavier_normal_(m.lin.weight)
                m.bias.data.fill_(0.1)

    def forward(self, x):
        # Rearranging shapes: [num_seqs, num_timesteps, num_atoms, num_dims] -> [num_seqs, num_atoms, num_timesteps*num_dims]
        x = x.view(x.size(0), x.size(2), -1)

        # Forward pass interleaving GCN layers, operations to switch from nodes to edges or vice-versa, and MLP layers
        x = self.conv1(x, self.edge_index)
        x = F.relu(x)

        if not self.compact:
            edge_x = [node2edge(x_samp, self.m_in, self.m_out) for x_samp in x]
            x = torch.stack(edge_x)
            
            x = self.mlp1(x)
            x = F.relu(x)

            x = x.permute(0, 2, 1)
            x = self.bnorm1(x)
            x = x.permute(0, 2, 1)
            
            x = self.dropout1(x)

            # Skip connection
            x_skip = x.clone()

            node_x = [edge2node(x_samp, self.m_in) for x_samp in x]
            x = torch.stack(node_x)
        
        x = self.conv2(x, self.edge_index)
        x = F.relu(x)
        
        edge_x = [node2edge(x_samp, self.m_in, self.m_out) for x_samp in x]
        x = torch.stack(edge_x)
        
        if not self.compact:
            x = torch.cat((x, x_skip), dim=2)
        
        x = self.mlp2(x)
        x = F.relu(x)

        x = x.permute(0, 2, 1)
        x = self.bnorm2(x)
        x = x.permute(0, 2, 1)

        return self.fc_out(x)
    

# NRI recurrent encoder
class nri_rec_encoder(nn.Module):
    def __init__(self, device, n_joints, edge_index_t, n_in, n_hid, n_out):
        super(nri_rec_encoder, self).__init__()
        self.device = device

        # Computing edge index given transposed edge index
        self.edge_index = torch.Tensor(edge_index_t).t().long().to(device)

        # Computing the message passing matrices
        self.m_in, self.m_out = message_passing_matrices(n_joints, self.edge_index)
        self.m_in = self.m_in.to(device)
        self.m_out = self.m_out.to(device)

        # Defining the network itself starting with GRNN and then MLP layers
        self.grnn = gcn_lstm_cell(n_in, n_hid).to(device)
        
        self.mlp1 = Linear(n_hid*2, n_hid).to(device)
        self.fc_out = Linear(n_hid, n_out).to(device)
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.1)

            elif isinstance(m, GCNConv):
                nn.init.xavier_normal_(m.lin.weight)
                m.bias.data.fill_(0.1)

    def forward(self, x):
        # Rearranging shapes: [num_seqs, num_timesteps, num_atoms, num_dims]
        num_seqs, num_timesteps, num_atoms, num_dims = x.shape

        # Iterating through samples in the batch
        h_batch = []
        for x_b in x:
            # Initializing cell and hidden states
            h = torch.zeros(num_atoms, self.grnn.n_out).to(self.device)
            c = torch.zeros(num_atoms, self.grnn.n_out).to(self.device)
            
            # Iterating through GRNN
            for x_t in x_b:
                h, c = self.grnn(x_t, h, c, self.edge_index)

            h_batch.append(h)
        h = torch.stack(h_batch)
        
        # Forward pass with an operation to switch from nodes to edges and MLP layers
        edge_x = [node2edge(h_samp, self.m_in, self.m_out) for h_samp in h]
        x = torch.stack(edge_x)
        
        x = self.mlp1(x)
        x = F.relu(x)

        return self.fc_out(x)


# Defining NRI decoder
class nri_decoder(nn.Module):
    def __init__(self, device, n_in, n_hid, n_out, do_prob=0.):
        super(nri_decoder, self).__init__()

        # Defining the network itself interleaving GCN and MLP layers
        self.conv1 = GCNConv(n_in, n_hid).to(device)
        
        self.mlp1 = Linear(n_hid*2, n_hid).to(device)
        self.dropout1 = Dropout(do_prob).to(device)
        
        self.conv2 = GCNConv(n_hid, n_out).to(device)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.1)

            elif isinstance(m, GCNConv):
                nn.init.xavier_normal_(m.lin.weight)
                m.bias.data.fill_(0.1)

    def forward(self, x, edge_index, m_in, m_out):
        # Rearranging shapes: [num_seqs, num_timesteps, num_atoms, num_dims] -> [num_seqs, num_atoms, num_timesteps*num_dims]
        x = x.view(x.size(0), x.size(2), -1)

        # Forward pass interleaving GCN layers, operations to switch from nodes to edges or vice-versa, and MLP layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        edge_x = [node2edge(x_samp, m_in, m_out) for x_samp in x]
        x = torch.stack(edge_x)
        
        x = self.mlp1(x)
        x = F.relu(x)

        x = self.dropout1(x)

        node_x = [edge2node(x_samp, m_in) for x_samp in x]
        x = torch.stack(node_x)
        
        x = self.conv2(x, edge_index)

        return x
    

# NRI recurrent decoder
class nri_rec_decoder(nn.Module):
    def __init__(self, device, n_in, n_hid, n_out, do_prob=0.):
        super(nri_rec_decoder, self).__init__()
        self.device = device
        
        # Defining the network itself starting with GRNN and then interleaving MLP and GCN layers
        self.grnn = gcn_lstm_cell(n_in, n_hid).to(device)
        
        self.mlp1 = Linear(n_hid*2, n_hid).to(device)
        self.dropout1 = Dropout(do_prob).to(device)
        
        self.conv1 = GCNConv(n_hid, n_out).to(device)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.1)

            elif isinstance(m, GCNConv):
                nn.init.xavier_normal_(m.lin.weight)
                m.bias.data.fill_(0.1)

    def forward(self, x, edge_index, m_in, m_out):
        # [num_timesteps, num_atoms, num_dims]
        num_timesteps, num_atoms, num_dims = x.shape
        
        # Initializing cell and hidden states
        h = torch.zeros(num_atoms, self.grnn.n_out).to(self.device)
        c = torch.zeros(num_atoms, self.grnn.n_out).to(self.device)
        
        # Iterating through GRNN
        for x_t in x:
            h, c = self.grnn(x_t, h, c, edge_index)

        # Forward pass interleaving GCN layers, operations to switch from nodes to edges or vice-versa, and MLP layers
        x = node2edge(h, m_in, m_out)
        
        x = self.mlp1(x)
        x = F.relu(x)

        x = self.dropout1(x)

        x = edge2node(x, m_in)
        
        x = self.conv1(x, edge_index)

        return x


# Defining NRI VAE
class nri_vae(nn.Module):
    def __init__(self, device, n_joints, edge_index_t, n_in, n_hid, edge_types, n_out, tau, hard, do_prob=0., n_dims=6, compact_enc=False, rec_enc=False, rec_dec=True):
        super(nri_vae, self).__init__()

        # Initializing encoder and decoder
        if rec_enc:
            self.encoder = nri_rec_encoder(device, n_joints, edge_index_t, n_dims, n_hid, edge_types)
        else:
            self.encoder = nri_encoder(device, n_joints, edge_index_t, n_in, n_hid, edge_types, do_prob, compact_enc)

        if rec_dec:
            self.decoder = nri_rec_decoder(device, n_dims, n_hid, n_out, do_prob)
        else:
            self.decoder = nri_decoder(device, n_in, n_hid, n_out, do_prob)

        # Saving variables that will be used by the forward pass
        self.device = device
        self.n_joints = n_joints
        
        self.tau = tau
        self.hard = hard

        self.edge_index_t = torch.Tensor(edge_index_t).to(device)
    
    def forward(self, x):
        # Computing logits for edges with encoder
        logits = self.encoder(x)

        # Sampling edge index classes using Gumbel-Softmax
        y = gumbel_softmax_sample(logits, self.tau, self.hard)

        # Getting sampled edges for every element in the batch
        edge_index_dict = {i: [] for i in range(logits.size(0))}
        edge_index_classes = torch.nonzero(y[:, :, -1])
        for batch_element, edge in edge_index_classes:
            edge_index_dict[batch_element.item()].append(edge.item())

        recon_output = []
        for k, v in edge_index_dict.items():
            # Building edge_index for sampled edges
            edge_index_samp = self.edge_index_t[v].t().long()

            # Creating message passing matrices for decoder newly sampled edge index
            decoder_m_in, decoder_m_out = message_passing_matrices(self.n_joints, edge_index_samp)
            decoder_m_in = decoder_m_in.to(self.device)
            decoder_m_out = decoder_m_out.to(self.device)

            # Reconstructing sequences using decoder
            recon_output.append(self.decoder(x[k], edge_index_samp, decoder_m_in, decoder_m_out))
        
        recon_output = torch.stack(recon_output)

        return logits, recon_output