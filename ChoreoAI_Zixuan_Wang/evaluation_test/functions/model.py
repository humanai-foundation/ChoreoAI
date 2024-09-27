import torch
import logging
import importlib
from torch.nn.parallel import DataParallel, DistributedDataParallel
from functions.model_variants import VAELSTM
from functions.model_variants_static import VAELSTM_Static

logger = logging.getLogger('ai_enabled_choreography')


class GenerativeModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.is_train = cfg['is_train']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kl_weight = 0.0001

        self.network = VAELSTM(seq_len=cfg['train']['seq_len'],
                               latent_dim=cfg['train']['vae']['latent_dim'], 
                               n_units=cfg['train']['vae']['n_units'], 
                               reduced_joints=cfg['train']['reduced_joints'],
                               device=self.device).to(self.device)

        self.loss = -1

        self.print_network(self.network)

        self.setup_optimizers()
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, **self.cfg['train']['scheduler'])

        if torch.cuda.device_count() > 1:
            self.network = DataParallel(self.network)
        
        if self.is_train:
            self.init_training_settings()
        
    def init_training_settings(self):
        pass

    def setup_optimizers(self):
        optim_params = []
        for k, v in self.network.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
        self.optimizer = torch.optim.Adam(optim_params, **self.cfg['train']['optim'])

    def print_network(self, net):
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net_cls_str = (f'{net.__class__.__name__} - '
                           f'{net.module.__class__.__name__}')
        else:
            net_cls_str = f'{net.__class__.__name__}'

        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module

        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        logger.info(f'Network: {net_cls_str}, with parameters: {net_params:,d}')
        logger.info(net_str)
    
    def feed_data(self, data):
        self.model_input = data['seq'][:, :, :, :3].to(self.device)

    def compute_loss(self, pred, mean, log_var):
        reconstruction_loss = torch.nn.MSELoss()
        l_reconstruction = reconstruction_loss(pred, self.model_input[:, :, :, :3]) * 2
        # 0.5 * torch.mean(torch.sum((self.model_input[:, :, :, :3] - pred) ** 2, dim=-1))
        l_kl = -0.5 * torch.mean(torch.sum(1 + log_var - mean ** 2 - torch.exp(log_var), dim=-1), dim=0) * self.kl_weight
        l_total = l_reconstruction + l_kl
        return l_total

    def optimize_parameters(self):
        self.optimizer.zero_grad()

        pred, mean, log_var = self.network(self.model_input, is_train=True)

        self.loss = self.compute_loss(pred, mean, log_var)

        self.loss.backward()
        self.optimizer.step()

    def update_learning_rate(self):
        self.scheduler.step()

    def validate(self, dataloader_val):
        self.network.eval()
        val_loss = 0
        with torch.no_grad():
            for data in dataloader_val:
                self.feed_data(data)
                pred, mean, log_var = self.network(self.model_input, is_train=False)
                val_loss += self.compute_loss(pred, mean, log_var)

        val_loss /= len(dataloader_val)
        self.network.train()
        return val_loss

    def test(self, dataloader_test):
        self.network.eval()
        test_loss = 0
        with torch.no_grad():
            for data in dataloader_test:
                self.feed_data(data)
                pred, mean, log_var = self.network(self.model_input, is_train=False)
                test_loss += self.compute_loss(pred, mean, log_var)

        test_loss /= len(dataloader_test)
        self.network.train()
        return test_loss

    def save_network(self, epoch, strict=True, param_key='params'):
        save_filename = f'model_{epoch}.pth'
        net = self.network if isinstance(self.network, list) else [self.network]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            if isinstance(net_, (DataParallel, DistributedDataParallel)):
                net_ = net_.module
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        torch.save(save_dict, save_filename)

