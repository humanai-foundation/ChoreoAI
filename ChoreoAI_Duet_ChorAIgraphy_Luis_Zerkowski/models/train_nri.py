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
def model_iteration(model, optimizer, scheduler, batches, batches_cumsum, beta, mode='train', \
                    recon_mode='nll', save_path='best_weights/nri_parameters.pt'):
    t = time.time()
    
    kl_aux = []
    recon_aux = []
    loss_aux = []

    if mode == 'train':
        model.train()
    elif mode == 'val':
        model.eval()
    
    choreo_cumsum_idx = 0
    for idx in range(batches_cumsum[-1]):
        # Skipping last batch of a video, since the next batch belongs to the next video, not a delta_t of the movement
        if idx == batches_cumsum[choreo_cumsum_idx]:
            choreo_cumsum_idx += 1
            continue

        if mode == 'train':
            optimizer.zero_grad()
        
        batch = batches[idx*batch_size:(idx+1)*batch_size]
        logits, recon_output = model(batch)

        kl_loss = gumbel_softmax_kl_divergence(logits, log_prior, batch_size)
        
        recon_output = recon_output.view(batch_size, seq_len_out, n_joints, dims)

        if recon_mode == 'nll':
            nll_gaussian = nll_gaussian_loss()
            var_tensor = torch.full(recon_output.shape, out_var, device=device)
            recon_loss = nll_gaussian(recon_output, batches[(idx+1)*batch_size:(idx+2)*batch_size, :seq_len_out, :, :], var_tensor)    
        elif recon_mode == 'mse':
            mse = mse_loss()
            recon_loss = mse(recon_output, batches[(idx+1)*batch_size:(idx+2)*batch_size, :seq_len_out, :, :])
            
        recon_loss = recon_loss / (recon_output.size(0) * recon_output.size(1) * recon_output.size(2))

        if recon_mode == 'nll':
            recon_coef = 0.0001
        elif recon_mode == 'mse':
            recon_coef = 1
            
        # loss = beta*0.01*kl_loss + recon_coef*recon_loss
        loss = recon_coef*recon_loss

        if mode == 'train':
            loss.backward()
            optimizer.step()

        kl_aux.append(0.01*kl_loss.item())
        recon_aux.append(recon_coef*recon_loss.item())    
        loss_aux.append(loss.data.item())

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

    del var_tensor
    torch.cuda.empty_cache()

    return kl_aux, recon_aux, loss_aux

# Parsing arguments to choose general options and hyperparameter values
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42, help='Random seed.')

parser.add_argument('--batch-size', type=int, default=128, help='Number of samples per batch.')

parser.add_argument('--seq-len-in', type=int, default=32, help='Number of input time steps per sample.')
parser.add_argument('--seq-len-out', type=int, default=4, help='Number of steps to predict.')

parser.add_argument('--hidden-dim', type=int, default=256, help='Number of hidden units for both encoder and decoder.')
parser.add_argument('--edge-types', type=int, default=2, help='The number of edge types to infer.')

parser.add_argument('--tau', type=float, default=0.5, help='Temperature for Gumbel-Softmax.')
parser.add_argument('--hard', action='store_false', default=True, help='Wether to use discrete samples in training forward pass.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
parser.add_argument('--var', type=float, default=5e-5, help='Output variance used for the reconstruction loss.')

parser.add_argument('--prior', action='store_false', default=True, help='Whether to use sparsity prior.')

parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')
parser.add_argument('--lr-cycle', type=int, default=10, help='After how many epochs to vary LR in the cycle.')
# parser.add_argument('--gamma', type=float, default=0.5, help='LR decay factor.')

parser.add_argument('--loss-mode', type=str, default='nll', help='Type of reconstruction loss to used')
parser.add_argument('--save-path', type=str, default='best_weights/nri_parameters.pt', help='Where to save the trained model.')
parser.add_argument('--load-path', type=str, default='best_weights/nri_parameters.pt', help='Where to load the trained model from.')

parser.add_argument('--param-count', action='store_true', default=False, help='Wheter to count the number of trainable parameters in the model.')

parser.add_argument('--visual', action='store_false', default=True, help='Wheter to visualize training graphs.')

args = parser.parse_args()

# Acquiring data
joint_poses, edge_index_t, batches, choreo_lens, train_batches, train_split, val_batches, val_split = load_data(args.seq_len_in)

# Initializing all the hyperparameters and moving the required ones to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_batches = train_batches.to(device)
val_batches = val_batches.to(device)

batch_size = args.batch_size
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
    prior = [0.95, 0.05]
    log_prior = torch.FloatTensor(np.log(prior)).unsqueeze(0).unsqueeze(0).to(device)

epochs = args.epochs
lr = args.lr
lr_cycle = args.lr_cycle
# gamma = args.gamma

loss_mode = args.loss_mode
save_path = args.save_path
load_path = args.load_path

# Initializing model, optimizer and scheduler
model = nri_vae(device, n_joints, edge_index_t, seq_len_in*dims, hidden_dim, edge_types, seq_len_out*dims, tau, hard, dropout)

if args.param_count:
    # Counting number of trainable parameters to compare to the dataset size
    print('######## Total number of trainable parameters: {} ########\n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    for n, p in model.named_parameters():
        if p.requires_grad:
            print('Layer {} has {} trainbale parameters'.format(n, p.numel()))

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=lr/5, max_lr=lr, step_size_up=lr_cycle, mode='triangular', cycle_momentum=False)

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
                                                  train_batches_cumsum, beta, 'train', loss_mode)
    
    kl_train.append(torch.mean(kl_aux))
    recon_train.append(torch.mean(recon_aux))
    loss_train.append(torch.mean(loss_aux))

    kl_aux, recon_aux, loss_aux = model_iteration(model, optimizer, scheduler, val_batches, \
                                                  val_batches_cumsum, beta, 'val', loss_mode, save_path)
    
    kl_val.append(torch.mean(kl_aux))
    recon_val.append(torch.mean(recon_aux))
    loss_val.append(torch.mean(loss_aux))

print("Optimization Finished!")

if args.visual:
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
    fig.text(0.22, 0.96, 'KL Loss', ha='center', fontsize=14)
    fig.text(0.53, 0.96, 'Reconstruction Loss', ha='center', fontsize=14)
    fig.text(0.85, 0.96, 'Combined Loss', ha='center', fontsize=14)

    # Adding row labels
    fig.text(0.02, 0.73, 'Training', va='center', rotation='vertical', fontsize=14)
    fig.text(0.02, 0.30, 'Validation', va='center', rotation='vertical', fontsize=14)

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.savefig('assets/loss_graphs.png')