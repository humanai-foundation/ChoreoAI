import torch
import os
import logging
from copy import deepcopy

from model.transformer import DancerTransformer
# from model.transformer_prob import DancerTransformerWithNoInputProb
from torch.nn.parallel import DataParallel, DistributedDataParallel
from loss.reconstruction_loss import ReconstructionLoss

logger = logging.getLogger('ai_choreo')


class Pipeline:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = DancerTransformer(64, 8, 32, 32, 64).to(self.device)
        self.loss = 0
        self.criterion = ReconstructionLoss()

        self.init_training_settings()

    def init_training_settings(self):
        self.network.train()
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        optim_params = []
        for k, v in self.network.named_parameters():
            if v.requires_grad:
                optim_params.append(v)

        self.optimizer = torch.optim.Adam(optim_params)

    def setup_schedulers(self):
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, 40, 60, 80])

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
        save_filename = "best_model.pth"
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
