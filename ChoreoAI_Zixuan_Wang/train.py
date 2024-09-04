import torch
from os import path as osp
import logging
import numpy as np
from torch.utils.data import Subset, DataLoader
import wandb
from itertools import product

from utils.logger import get_root_logger
from utils.misc import get_time_str
from data.dataset_original import DancerDatasetOriginal
from model.model_pipeline import Pipeline

import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

def init_logger():
    log_file = osp.join("log", f"train_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='ai_choreo', log_level=logging.INFO, log_file=log_file)

    return logger


def preprocess_dataset(dancer_np):
    dancer1_np = dancer_np[::2, :, :]
    dancer2_np = dancer_np[1::2, :, :]
    return dancer1_np, dancer2_np


def create_dataset_loader(dataset_dir):
    dancer_np = np.load('dataset/' + dataset_dir)
    dancer1_np, dancer2_np = preprocess_dataset(dancer_np)
    dataset = DancerDatasetOriginal(torch.from_numpy(dancer1_np), torch.from_numpy(dancer2_np), 64)

    train_size = int(0.7 * len(dataset))
    validation_size = int(0.2 * len(dataset))

    train_dataset = Subset(dataset, range(train_size))
    validation_dataset = Subset(dataset, range(train_size, train_size + validation_size))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    return train_loader, validation_loader


def main():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    logger = init_logger()

    dataset_names = ['pose_extraction_img_9085.npy', 'pose_extraction_ilya_hannah_dyads.npy', 'pose_extraction_hannah_cassie.npy', 'pose_extraction_dyads_rehearsal_leah.npy']

    train_loaders = []
    validation_loaders = []

    train_data_length = 0

    wandb.login(key="f73ba40c8f47503a7c50b110fb21cb8f740a59dc")

    for name in dataset_names:
        train_loader, validation_loader = create_dataset_loader(name)
        train_loaders.append(train_loader)
        train_data_length += len(train_loader)
        validation_loaders.append(validation_loader)

    epochs = 100

    logger.info("Started training.")

    linear_num_features = 64
    n_head = 8
    latent_dim_all = [32, 64]
    n_units_all = [32, 64]
    seq_len = 64
    no_input_prob = 0.1
    velocity_loss_weight_all = [0.05, 0.1, 0.2]
    kl_loss_weight_all = [0.00005, 0.0001]
    mse_loss_weight = 0.5

    hyperparameters_combinations = list(product(latent_dim_all, n_units_all, velocity_loss_weight_all, kl_loss_weight_all))

    for latent_dim, n_units, velocity_loss_weight, kl_loss_weight in hyperparameters_combinations:

        logger.info(
            f'linear_num_features: {linear_num_features}\n'
            f'n_head: {n_head}\n'
            f'latent_dim: {latent_dim}\n'
            f'n_units: {n_units}\n'
            f'seq_len: {seq_len}\n'
            f'no_input_prob: {no_input_prob}\n'
            f'velocity_loss_weight: {velocity_loss_weight}\n'
            f'kl_loss_weight: {kl_loss_weight}\n'
            f'mse_loss_weight: {mse_loss_weight}'
        )

        model = Pipeline(linear_num_features, n_head, latent_dim, n_units, seq_len, no_input_prob, velocity_loss_weight, kl_loss_weight, mse_loss_weight)

        prev_best_validation_loss = -1
        
        wandb.init(project='duet_hyperparameter_tuning')

        for epoch in range(epochs):
            logger.info(f"Epoch: {epoch}")
            train_loss = 0

            for train_loader in train_loaders:
                logger.info(f"train loader size: {len(train_loader)}")
                for train_data in train_loader:
                    model.feed_data(train_data)
                    cur_loss = model.optimize_parameters()
                    train_loss += cur_loss.item()
                    logger.info(f"one train data training loss: {cur_loss}")

            model.update_learning_rate()
            wandb.log({'train/loss': train_loss / train_data_length})
            logger.info(f"total training loss: {train_loss / train_data_length}")

            validation_loss = 0

            for validation_loader in validation_loaders:
                validation_loss += model.test(validation_loader)
            
            validation_loss /= len(validation_loaders)
            logger.info(f"validation loss: {validation_loss}")
            wandb.log({'validation/loss': validation_loss.item()})

            if prev_best_validation_loss == -1 or validation_loss < prev_best_validation_loss:
                prev_best_validation_loss = validation_loss
                logger.info("save network...")
                model.save_network()


if __name__ == '__main__':
    main()
