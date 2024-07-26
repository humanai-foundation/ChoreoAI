import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#### General auxiliar functions ####
# Animation function
def animation(sequence, skeleton, n_joints, interval=100):
    fig = plt.figure(figsize=(16, 12))
    
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ms = 70
    scatt1 = ax.scatter([], [], [], color='red', s=ms)
    scatt2 = ax.scatter([], [], [], color='blue', s=ms)

    lw = 4
    lines = [ax.plot([], [], [], 'gray', linewidth=lw)[0] for _ in skeleton]

    sequence_x = sequence[:, :, 2]
    sequence_y = sequence[:, :, 0]
    sequence_z = -sequence[:, :, 1]
    
    def update(frame):
        
        scatt1._offsets3d = (sequence_x[frame, :n_joints], sequence_y[frame, :n_joints], sequence_z[frame, :n_joints])
        scatt2._offsets3d = (sequence_x[frame, n_joints:], sequence_y[frame, n_joints:], sequence_z[frame, n_joints:])
    
        for line, (start, end) in zip(lines, skeleton):
            line.set_data([sequence_x[frame, start], sequence_x[frame, end]], [sequence_y[frame, start], sequence_y[frame, end]])
            line.set_3d_properties([sequence_z[frame, start], sequence_z[frame, end]])
        
        return scatt1, scatt2, *lines

    plt.close(fig)
    return FuncAnimation(fig, update, frames=range(len(sequence_x)), interval=interval, blit=False)


#### NRI auxiliar functions ####
# Creating message passing matrices for receivers and senders - shape R^(E x N)
def message_passing_matrices(n_joints, edge_index):
    message_passing_in = torch.zeros((edge_index.size(1), n_joints))
    message_passing_out = torch.zeros((edge_index.size(1), n_joints))

    for j in range(edge_index.size(1)):
        message_passing_out[j, int(edge_index[0, j])] = 1.
        message_passing_in[j, int(edge_index[1, j])] = 1.

    return message_passing_in, message_passing_out


# NRI VAE auxiliar functions to change between nodes and edges
def node2edge(x, m_in, m_out):    
    receivers = torch.matmul(m_in, x)
    senders = torch.matmul(m_out, x)
    edges = torch.cat([senders, receivers], dim=1)
    
    return edges


def edge2node(x, m_in):
    incoming = torch.matmul(m_in.t(), x)
    
    return incoming / incoming.size(0)


# Gumbel-Softmax sampling function to allow for backpropagation with categorical distributions
def gumbel_softmax_sample(logits, temp, hard=False):
    y = F.gumbel_softmax(logits, tau=temp, hard=hard)
    
    return y


# Computing KL Divergence for categorical distribution
def gumbel_softmax_kl_divergence(logits, log_prior, batch_size):
    q_y = F.softmax(logits, dim=-1)
    kl_div = q_y * (F.log_softmax(logits, dim=-1) - log_prior)

    # Normalizing by the batch size and number of edges
    return kl_div.sum() / (batch_size * logits.size(0))


# Gaussian NLL loss
def nll_gaussian_loss():
    return nn.GaussianNLLLoss(reduction='sum')

# MSE loss
def mse_loss():
    return nn.MSELoss(reduction='sum')