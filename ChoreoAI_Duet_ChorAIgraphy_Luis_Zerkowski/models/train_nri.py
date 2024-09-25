import time
from tqdm import tqdm
import argparse

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from prepare_data import *
from modules import *
from utils import *

# Model iteration function
def model_iteration(model, optimizer, scheduler, batches, batches_cumsum, beta, frame_gap, dataset_multiplier, \
                    mode='train', recon_mode='nll', save_path='best_weights/nri_parameters.pt'):
    t = time.time()
    
    kl_aux = []
    recon_aux = []
    loss_aux = []

    if mode == 'train':
        model.train()
    elif mode == 'val':
        model.eval()
    
    for dsm in range(dataset_multiplier):
        choreo_cumsum_idx = 0
        for idx in range(batches_cumsum[-1]):
            # Skipping last batch of a video, since the next batch belongs to the next video, not a delta_t of the movement
            if idx == batches_cumsum[choreo_cumsum_idx]:
                choreo_cumsum_idx += 1
                continue

            if mode == 'train':
                optimizer.zero_grad()
            
            batch = batches[idx*batch_size:(idx+1)*batch_size]

            rotation = rotation_matrix_z(torch.tensor(2*np.random.rand(1)[0]*np.pi)).to(device)
            rotated_batches_pos = torch.einsum('...ij,jk->...ik', batch[:, :, :, :3], rotation)
            rotated_batches_vel = torch.einsum('...ij,jk->...ik', batch[:, :, :, 3:], rotation)
            batch = torch.cat([rotated_batches_pos, rotated_batches_vel], dim=-1)

            # Non-overlapping sequences
            # next_batch = batches[idx*batch_size+1:(idx+1)*batch_size+1]

            # Overlapping sequences
            next_batch = batches[(idx+1)*batch_size:(idx+2)*batch_size]

            rotated_batches_pos = torch.einsum('...ij,jk->...ik', next_batch[:, :, :, :3], rotation)
            rotated_batches_vel = torch.einsum('...ij,jk->...ik', next_batch[:, :, :, 3:], rotation)
            next_batch = torch.cat([rotated_batches_pos, rotated_batches_vel], dim=-1)
            
            logits, recon_output = model(batch)

            kl_loss = gumbel_softmax_kl_divergence(logits, log_prior, batch_size)
            
            recon_output = recon_output.view(batch_size, seq_len_out, n_joints, int(dims/2))

            if recon_mode == 'nll':
                nll_gaussian = nll_gaussian_loss()
                var_tensor = torch.full(recon_output.shape, out_var, device=device)
                recon_loss = nll_gaussian(recon_output, next_batch[:, \
                                        frame_gap-1:frame_gap-1+seq_len_out, :, :int(dims/2)], var_tensor)    
            elif recon_mode == 'mse':
                mse = mse_loss()
                recon_loss = mse(recon_output, next_batch[:, \
                                frame_gap-1:frame_gap-1+seq_len_out, :, :int(dims/2)])
                
            recon_loss = recon_loss / (recon_output.size(0) * recon_output.size(1) * recon_output.size(2))

            if recon_mode == 'nll':
                recon_coef = 0.0001
            elif recon_mode == 'mse':
                recon_coef = 1
                
            loss = beta*0.01*kl_loss + recon_coef*recon_loss
            # loss = recon_coef*recon_loss

            if mode == 'train':
                loss.backward()
                optimizer.step()

            kl_aux.append(0.01*kl_loss.item())
            recon_aux.append(recon_coef*recon_loss.item())    
            loss_aux.append(loss.data.item())

        del batch, next_batch, logits, kl_loss, recon_loss

    kl_aux = torch.Tensor(kl_aux)
    recon_aux = torch.Tensor(recon_aux)
    loss_aux = torch.Tensor(loss_aux)
    tqdm.write(f'Epoch: {epoch + 1:04d}, '
               f'KL Loss ({mode}): {torch.mean(kl_aux):.4f}, '
               f'Reconstruction Loss ({mode}): {torch.mean(recon_aux):.4f}, '
               f'Combined Loss ({mode}): {torch.mean(loss_aux):.4f}, '
               f'time: {time.time() - t:.4f}s')

    if mode == 'train':
        scheduler.step()

    if mode == 'val':
        global best_val_loss
        global best_epoch
        
        if best_val_loss is torch.inf or torch.mean(loss_aux) < best_val_loss:    
            best_val_loss = torch.mean(loss_aux)
            best_epoch = epoch
            
            torch.save(model.state_dict(), save_path)
            tqdm.write(f'Epoch: {epoch + 1:04d}, Saving best parameters!')

    if recon_mode == 'nll':
        del var_tensor

    torch.cuda.empty_cache()

    return kl_aux, recon_aux, loss_aux

# Parsing arguments to choose general options and hyperparameter values
parser = argparse.ArgumentParser()

# parser.add_argument('--seed', type=int, default=42, help='Random seed.')

# parser.add_argument('--batch-size', type=int, default=128, help='Number of samples per batch.')

parser.add_argument('--n-sample-dancers', type=int, default=3, help='Number of joints to sample for dancers.')
parser.add_argument('--frame-gap', type=int, default=1, help='Number of frames to skip for velocity computation.')
parser.add_argument('--dataset-multiplier', type=int, default=5, help='Number of times to augment the dataset with rotations.')

parser.add_argument('--compact-encoder', action='store_true', default=False, help='Whether to use the compact version of the encoder.')
parser.add_argument('--recurrent-encoder', action='store_true', default=False, help='Whether to use the recurrent encoder.')
parser.add_argument('--recurrent-decoder', action='store_false', default=True, help='Whether to use the recurrent decoder.')

parser.add_argument('--seq-len-in', type=int, default=8, help='Number of input time steps per sample.')
parser.add_argument('--seq-len-out', type=int, default=1, help='Number of steps to predict.')

parser.add_argument('--hidden-dim', type=int, default=64, help='Number of hidden units for both encoder and decoder.')
parser.add_argument('--edge-types', type=int, default=4, help='The number of edge types to infer.')

parser.add_argument('--tau', type=float, default=0.5, help='Temperature for Gumbel-Softmax.')
parser.add_argument('--hard', action='store_false', default=True, help='Whether to use discrete samples in training forward pass.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
parser.add_argument('--var', type=float, default=5e-5, help='Output variance used for the NLL reconstruction loss.')

parser.add_argument('--prior', action='store_false', default=True, help='Whether to use sparsity prior.')

parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0003, help='Initial learning rate.')
parser.add_argument('--lr-cycle', type=int, default=10, help='After how many epochs to vary LR in the cycle.')
parser.add_argument('--gamma', type=float, default=0.5, help='LR decay factor.')

parser.add_argument('--loss-mode', type=str, default='nll', help='Type of reconstruction loss to used')
parser.add_argument('--save-path', type=str, default='best_weights/nri_parameters.pt', help='Where to save the trained model.')
parser.add_argument('--load-path', type=str, default='best_weights/nri_parameters.pt', help='Where to load the trained model from.')

parser.add_argument('--param-count', action='store_true', default=False, help='Wheter to count the number of trainable parameters in the model.')

parser.add_argument('--visual', type=str, default='assets/loss_graphs.png', help='Where to save training graphs.')

args = parser.parse_args()

# Acquiring data
joint_poses, edge_index_t, batches, choreo_lens, train_batches, train_split, val_batches, val_split = load_data(args.seq_len_in, \
                                                                                                                args.frame_gap, \
                                                                                                                args.n_sample_dancers)

# Initializing all the hyperparameters and moving the required ones to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_batches = train_batches.to(device)
val_batches = val_batches.to(device)

batch_size = args.seq_len_in
train_batches_cumsum = np.cumsum(np.array(train_split) // batch_size)
val_batches_cumsum = np.cumsum(np.array(val_split) // batch_size)

seq_len_in = args.seq_len_in
seq_len_out = args.seq_len_out
n_joints = batches.size(2)
dims = batches.size(3)

hidden_dim = args.hidden_dim
edge_types = args.edge_types

tau = args.tau
hard = args.hard
dropout = args.dropout
out_var = args.var

if args.prior:
    if args.edge_types == 2:
        prior = [0.7, 0.3]
    elif args.edge_types == 3:
        prior = [0.4, 0.3, 0.3]
    elif args.edge_types == 4:
        prior = [0.35, 0.25, 0.15, 0.25]
    elif args.edge_types == 5:
        prior = [0.3, 0.2, 0.2, 0.1, 0.2]

    log_prior = torch.FloatTensor(np.log(prior)).unsqueeze(0).unsqueeze(0).to(device)

epochs = args.epochs
lr = args.lr
lr_cycle = args.lr_cycle
gamma = args.gamma

loss_mode = args.loss_mode
save_path = args.save_path
load_path = args.load_path

model = nri_vae(device, n_joints, edge_index_t, seq_len_in*dims, hidden_dim, edge_types, seq_len_out*int(dims/2), \
                tau, hard, dropout, dims, args.compact_encoder, args.recurrent_encoder, args.recurrent_decoder)

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_cycle, gamma=args.gamma)
# scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=lr/5, max_lr=lr, step_size_up=lr_cycle, mode='triangular', cycle_momentum=False)

if args.param_count:
    # Counting number of trainable parameters to compare to the dataset size
    print('######## Total number of trainable parameters: {} ########\n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    for n, p in model.named_parameters():
        if p.requires_grad:
            print('Layer {} has {} trainbale parameters'.format(n, p.numel()))

# Initializing lists to save losses across iterations
kl_train = []
recon_train = []
loss_train = []
kl_val = []
recon_val = []
loss_val = []

# Initializing variables to save best model
best_val_loss = torch.inf
best_epoch = 0

# Training loop
for epoch in tqdm(range(epochs), desc='Training Epochs'):
    # Beta coefficient to handle KL-Divergence vanishing gradients and balance reconstruction loss
    beta = epoch % int(epochs*0.2) / (epochs*0.2)
    
    kl_aux, recon_aux, loss_aux = model_iteration(model, optimizer, scheduler, train_batches, \
                                                  train_batches_cumsum, beta, args.frame_gap, args.dataset_multiplier, \
                                                  'train', loss_mode)
    
    kl_train.append(torch.mean(kl_aux))
    recon_train.append(torch.mean(recon_aux))
    loss_train.append(torch.mean(loss_aux))

    kl_aux, recon_aux, loss_aux = model_iteration(model, optimizer, scheduler, val_batches, \
                                                  val_batches_cumsum, beta, args.frame_gap, args.dataset_multiplier, \
                                                  'val', loss_mode, save_path)
    
    kl_val.append(torch.mean(kl_aux))
    recon_val.append(torch.mean(recon_aux))
    loss_val.append(torch.mean(loss_aux))

print("Optimization Finished!")

if len(args.visual) > 0:
    fig = plt.figure(figsize=(12, 6))

    # Plotting training losses
    ax1 = fig.add_subplot(231)
    ax1.plot(kl_train, color='blue')

    ax1 = fig.add_subplot(232)
    ax1.plot(recon_train, color='blue')

    ax1 = fig.add_subplot(233)
    ax1.plot(loss_train, color='blue')

    # Plotting validation losses
    ax1 = fig.add_subplot(234)
    ax1.plot(kl_val, color='orange')

    ax1 = fig.add_subplot(235)
    ax1.plot(recon_val, color='orange')

    ax1 = fig.add_subplot(236)
    ax1.plot(loss_val, color='orange')

    # Adding column labels
    fig.text(0.22, 0.96, 'KL Divergence', ha='center', fontsize=14)
    fig.text(0.53, 0.96, 'Reconstruction Loss', ha='center', fontsize=14)
    fig.text(0.85, 0.96, 'Combined Loss', ha='center', fontsize=14)

    # Adding row labels
    fig.text(0.02, 0.73, 'Training', va='center', rotation='vertical', fontsize=14)
    fig.text(0.02, 0.30, 'Validation', va='center', rotation='vertical', fontsize=14)

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.savefig(args.visual)