import argparse
import numpy as np
import torch
import os
from os import path as osp
import logging
import wandb

from torch_geometric import loader

from functions.utils import parse, get_time_str, get_root_logger, dict2str
from functions.dataset import MarielDataset
from functions.model import GenerativeModel

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='train_config.yml', help='Path to configuration YAML file.')
    args = parser.parse_args()
    cfg = parse(args.config)

    return cfg

def init_loggers(cfg):
    log_file = osp.join("logs", f"train_{cfg['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='ai_enabled_choreography', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(cfg))
    return logger


def create_dataset_loader(cfg, logger):
    if os.path.exists("train_set.pt") and os.path.exists("val_set.pt") and os.path.exists("test_set.pt"):
        train_set = torch.load("train_set.pt")
        val_set = torch.load("val_set.pt")
        test_set = torch.load("test_set.pt")
    else:
        data = MarielDataset(seq_len=cfg['train']['seq_len'], reduced_joints=cfg['train']['reduced_joints'], predicted_timesteps=cfg['train']['predicted_timesteps'], no_overlap=cfg['train']['no_overlap'], use_graph=cfg['use_graph'])

        train_indices = np.arange(int(0.7 * len(data))) # 70% split for training data
        val_indices = np.arange(int(0.7 * len(data)), int(0.85 * len(data))) # next 15% on validation
        test_indices = np.arange(int(0.85 * len(data)), len(data)) # last 15% on test

        train_set = torch.utils.data.Subset(data, train_indices)
        val_set = torch.utils.data.Subset(data, val_indices)
        test_set = torch.utils.data.Subset(data, test_indices)

        # torch.save(train_set, "train_set.pt")
        # torch.save(val_set, "val_set.pt")
        # torch.save(test_set, "test_set.pt")

    if not cfg['use_graph']:
        dataloader_train = torch.utils.data.DataLoader(train_set, batch_size=cfg['train']['batch_size'], shuffle=True, drop_last=True)
        dataloader_val = torch.utils.data.DataLoader(val_set, batch_size=cfg['train']['batch_size'], shuffle=True, drop_last=True)
        dataloader_test = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, drop_last=True)
    else:
        dataloader_train = loader.DataLoader(train_set, batch_size=cfg['train']['batch_size'], shuffle=True, drop_last=True)
        dataloader_val = loader.DataLoader(val_set, batch_size=cfg['train']['batch_size'], shuffle=True, drop_last=True)
        dataloader_test = loader.DataLoader(test_set, batch_size=1, shuffle=True, drop_last=True)

    logger.info("Dataset created.")
    logger.info(f"Total iterations per epoch: {int(len(train_set) / cfg['train']['batch_size'])}")

    return dataloader_train, dataloader_val, dataloader_test


def main():

    wandb.init(project='ai_enabled_choreography')

    cfg = parse_config()
    logger = init_loggers(cfg)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)

    dataloader_train, dataloader_val, dataloader_test = create_dataset_loader(cfg, logger)

    model = GenerativeModel(cfg)

    current_iter = 0
    logger.info(f'Start training from epoch: 0, iter: {current_iter}')

    best_loss = 1

    # train
    for epoch in range(cfg['train']['epochs']):
        # err = 0
        loss = 0

        for train_data in dataloader_train:
            current_iter += 1
            model.feed_data(train_data)
            model.optimize_parameters()

            loss += model.loss
            # err += model.error

            wandb.log({'train/loss': model.loss})

            if current_iter % cfg['logger']['print_freq'] == 0:
                logger.info(f'Epoch: {epoch}, iter: {current_iter}, loss: {model.loss}')
        
        model.update_learning_rate()
        loss /= len(dataloader_train)
        wandb.log({'train/epoch_loss': loss})

        logger.info(f'Epoch average training loss: {loss}')
        logger.info(f'Validating...')
        val_loss = model.validate(dataloader_val)
        wandb.log({'eval/loss': val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            # best_model = model
            model.save_network(epoch)

        test_loss = model.test(dataloader_test)
        wandb.log({'test/loss': test_loss})


if __name__ == '__main__':
    main()
