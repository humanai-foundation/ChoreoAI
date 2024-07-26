import torch_geometric.nn as geo_nn
from torch_geometric.nn import GCNConv
from torch.nn import Linear, BatchNorm1d, Dropout

from utils import *

# Defining NRI encoder
class nri_encoder(nn.Module):
    def __init__(self, device, n_joints, edge_index_t, n_in, n_hid, n_out, do_prob=0.):
        super(nri_encoder, self).__init__()

        # Computing edge index given transposed edge index
        self.edge_index = torch.Tensor(edge_index_t).t().long().to(device)

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
        
        self.mlp2 = Linear(n_hid*3, n_hid).to(device)
        self.bnorm2 = BatchNorm1d(n_hid).to(device)
        
        self.fc_out = Linear(n_hid, n_out).to(device)
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.1)

    def forward(self, x):
        # Rearranging shapes: [num_seqs, num_timesteps, num_atoms, num_dims] -> [num_seqs, num_atoms, num_timesteps*num_dims]
        x = x.view(x.size(0), x.size(2), -1)

        # Forward pass interleaving GCN layers, operations to switch from nodes to edges or vice-versa, and MLP layers
        x = self.conv1(x, self.edge_index)
        x = F.relu(x)

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
        
        x = torch.cat((x, x_skip), dim=2)
        x = self.mlp2(x)
        x = F.relu(x)

        x = x.permute(0, 2, 1)
        x = self.bnorm2(x)
        x = x.permute(0, 2, 1)

        return self.fc_out(x)
    

# Defining NRI decoder
class nri_decoder(nn.Module):
    def __init__(self, device, n_in, n_hid, n_out, do_prob=0.):
        super(nri_decoder, self).__init__()

        # Defining the network itself interleaving GCN and MLP layers
        self.conv1 = GCNConv(n_in, n_hid).to(device)
        
        self.mlp1 = Linear(n_hid*2, n_hid).to(device)
        self.bnorm1 = BatchNorm1d(n_hid).to(device)
        self.dropout1 = Dropout(do_prob).to(device)
        
        self.conv2 = GCNConv(n_hid, n_out).to(device)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.xavier_normal_(m.weight)
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

        x = x.permute(0, 2, 1)
        x = self.bnorm1(x)
        x = x.permute(0, 2, 1)
        
        x = self.dropout1(x)

        node_x = [edge2node(x_samp, m_in) for x_samp in x]
        x = torch.stack(node_x)
        
        x = self.conv2(x, edge_index)

        return x
    

# Defining NRI VAE
class nri_vae(nn.Module):
    def __init__(self, device, n_joints, edge_index_t, n_in, n_hid, edge_types, n_out, tau, hard, do_prob=0.):
        super(nri_vae, self).__init__()

        # Initializing encoder and decoder
        self.encoder = nri_encoder(device, n_joints, edge_index_t, n_in, n_hid, edge_types, do_prob)
        self.decoder = nri_decoder(device, n_in, n_hid, n_out, do_prob)

        # Saving variables that will be used by the forward pass
        self.device = device
        self.n_joints = n_joints
        self.edge_index_t = edge_index_t
        self.tau = tau
        self.hard = hard
    
    def forward(self, x):
        # Computing logits for edges with encoder
        logits = self.encoder(x)

        # Sampling edge index classes using Gumbel-Softmax. Since we are using only two types of edges at the moment,
        # existent or non-existent, we create a newly sampled edge index for the decoder to use
        edge_index_classes = gumbel_softmax_sample(logits, self.tau, self.hard)
        edge_index_samp = torch.Tensor(self.edge_index_t).to(x.device)[torch.where(edge_index_classes[:, 1])[0]].t().long()

        # Creating message passing matrices for decoder newly sampled edge index
        decoder_m_in, decoder_m_out = message_passing_matrices(self.n_joints, edge_index_samp)
        decoder_m_in = decoder_m_in.to(self.device)
        decoder_m_out = decoder_m_out.to(self.device)

        # Reconstructing sequences using decoder
        recon_output = self.decoder(x, edge_index_samp, decoder_m_in, decoder_m_out)

        return logits, recon_output