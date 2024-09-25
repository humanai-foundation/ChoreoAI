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


# Defining velocity function
def compute_velocities(data, frame_gap=2):
    velocities = data[frame_gap:] - data[:-frame_gap]

    # Repeating velocity for final frames
    padding = velocities[-1, :, :].repeat(frame_gap, 1, 1)

    velocities = torch.cat((velocities, padding), dim=0)
    
    # Fixing velocity configuration
    velocities = velocities[:, :, [2, 0, 1]]
    velocities[:, :, 2] = -velocities[:, :, 2]
    
    return velocities


########## REMOVED PIECE OF CODE. NOW WE IMPLEMENT ROTATION WITHIN BATCH PROCESSING AND ONLY ALONG Z-AXIS ##########
# # Rotation along X-axis function
# def rotation_matrix_x(angle):
#     c, s = np.cos(angle), np.sin(angle)
    
#     return np.array([[1, 0, 0],
#                      [0, c, -s],
#                      [0, s, c]])

# # Rotation along Y-axis function
# def rotation_matrix_y(angle):
#     c, s = np.cos(angle), np.sin(angle)
    
#     return np.array([[c, 0, s],
#                      [0, 1, 0],
#                      [-s, 0, c]])

# # Computing final rotation matrix
# def rotate_points(points, angle_x, angle_y, angle_z):
#     Rx = rotation_matrix_x(angle_x)
#     Ry = rotation_matrix_y(angle_y)
#     Rz = rotation_matrix_z(angle_z)
    
#     rotation_matrix = Rz @ Ry @ Rx
#     rotated_points = points @ rotation_matrix.T
    
#     return rotated_points

# Rotation along Z-axis function
def rotation_matrix_z(angle):
    c, s = torch.cos(angle), torch.sin(angle)
    
    return torch.tensor([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]], dtype=torch.float32)


#### NRI auxiliar functions ####
# Creating message passing matrices for receivers and senders - shape R^(E x N)
def message_passing_matrices(n_joints, edge_index):
    message_passing_in = torch.zeros((edge_index.size(1), n_joints))
    message_passing_out = torch.zeros((edge_index.size(1), n_joints))

    # Vectorizing message_passing matrices creation
    edge_indices = torch.arange(edge_index.size(1))
    message_passing_out[edge_indices, edge_index[0]] = 1.
    message_passing_in[edge_indices, edge_index[1]] = 1.

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
def nll_gaussian_loss(reduction='mean'):
    return nn.GaussianNLLLoss(reduction=reduction)

# MSE loss
def mse_loss(reduction='mean'):
    return nn.MSELoss(reduction=reduction)