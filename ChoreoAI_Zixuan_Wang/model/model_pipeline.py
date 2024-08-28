import torch
import os
import logging
from copy import deepcopy

from model.transformer import DancerTransformer
from torch.nn.parallel import DataParallel, DistributedDataParallel
from loss.reconstruction_loss import ReconstructionLoss
from utils.misc import get_time_str

logger = logging.getLogger('ai_choreo')


class Pipeline:
    def __init__(self, linear_num_features, n_head, latent_dim, n_units, seq_len, no_input_prob, velocity_loss_weight, kl_loss_weight, mse_loss_weight=0.5, epochs=100):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.linear_num_features = linear_num_features
        self.n_head = n_head
        self.latent_dim = latent_dim
        self.n_units = n_units
        self.seq_len = seq_len
        self.no_input_prob = no_input_prob
        self.velocity_loss_weight = velocity_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.mse_loss_weight = mse_loss_weight
        self.epochs = epochs
        self.network = DancerTransformer(linear_num_features, n_head, latent_dim, n_units, seq_len, no_input_prob).to(self.device)
        self.loss = 0
        self.criterion = ReconstructionLoss(self.velocity_loss_weight, self.kl_loss_weight, self.mse_loss_weight)

        self.init_training_settings(epochs)

    def init_training_settings(self, epochs):
        self.network.train()
        self.setup_optimizers()
        self.setup_schedulers(epochs)

    def setup_optimizers(self):
        optim_params = []
        for k, v in self.network.named_parameters():
            if v.requires_grad:
                optim_params.append(v)

        self.optimizer = torch.optim.Adam(optim_params)

    def setup_schedulers(self, epochs):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs)

    def feed_data(self, data):
        self.dancer1_data = torch.tensor(data['dancer1'], dtype=torch.float32).to(self.device)
        self.dancer2_data = torch.tensor(data['dancer2'], dtype=torch.float32).to(self.device)
        self.dancer1_next_timestamp = torch.tensor(data['dancer1_next_timestamp'], dtype=torch.float32).to(self.device)
        self.dancer2_next_timestamp = torch.tensor(data['dancer2_next_timestamp'], dtype=torch.float32).to(self.device)

    def compute_loss(self, pred_data_1, pred_data_2, mean_1, log_var_1, mean_2, log_var_2, mean_duet, log_var_duet):
        return self.criterion(pred_data_1, pred_data_2, self.dancer1_next_timestamp, self.dancer2_next_timestamp, mean_1, log_var_1, mean_2, log_var_2, mean_duet, log_var_duet)

    def optimize_parameters(self):
        self.network.zero_grad()
        pred_data1, pred_data2, mean_1, log_var_1, mean_2, log_var_2, mean_duet, log_var_duet = self.network(self.dancer1_data, self.dancer2_data)

        self.loss = self.compute_loss(pred_data1, pred_data2, mean_1, log_var_1, mean_2, log_var_2, mean_duet, log_var_duet)

        self.loss.backward()
        self.optimizer.step()

        return self.loss

    def load_network(self, net, load_path, strict=True, param_key='params'):
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        logger.info(
            f'Loading {net.__class__.__name__} model from {load_path}.')
        load_net = torch.load(
            load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            load_net = load_net[param_key]
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        net.load_state_dict(load_net, strict=strict)

    def update_learning_rate(self):
        self.scheduler.step()

    def save_network(self, param_key='params'):
        save_filename = "best_model" + "_fea_" + str(self.linear_num_features) + "_head_" + str(self.n_head) + "_latent_" + str(self.latent_dim) + "_units_" + str(self.n_units) + "_seq_len_" + str(self.seq_len) + "_prob_" + str(self.no_input_prob) + "_velo_" + str(self.velocity_loss_weight) + "_kl_" + str(self.kl_loss_weight) + "_mse_" + str(self.mse_loss_weight) + ".pth"
        save_path = os.path.join("result", save_filename)

        param_key = param_key if isinstance(param_key, list) else [param_key]

        save_dict = {}
        net_ = self.network
        param_key_ = param_key[0]

        if isinstance(net_, (DataParallel, DistributedDataParallel)):
            net_ = net_.module
        state_dict = net_.state_dict()
        for key, param in state_dict.items():
            if key.startswith('module.'):
                key = key[7:]
            state_dict[key] = param.cpu()
        save_dict[param_key_] = state_dict

        torch.save(save_dict, save_path)

    def test(self, test_loader):
        self.network.eval()
        cur_loss = 0
        with torch.no_grad():
            for test_data in test_loader:
                self.feed_data(test_data)
                pred_data1, pred_data2, mean_1, log_var_1, mean_2, log_var_2, mean_duet, log_var_duet = self.network(self.dancer1_data, self.dancer2_data)
                cur_loss += self.compute_loss(pred_data1, pred_data2, mean_1, log_var_1, mean_2, log_var_2, mean_duet, log_var_duet)

            cur_loss /= len(test_loader)
        self.network.train()
        return cur_loss
