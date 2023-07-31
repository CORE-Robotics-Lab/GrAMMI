
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from datasets.load_datasets import load_datasets
from models.configure_model import configure_model
import math
import os

import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import math
import random
from utils import set_seeds
import shutil
import argparse

from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils.rnn import pack_padded_sequence
import yaml

from sklearn.cluster import KMeans

from torch.distributions import Normal
from models.utils import log_prob

def train(seed, 
          device, 
          train_dataloader, 
          test_dataloader, 
          batch_size,
          model,
          learning_rate,
          n_epochs,
          l2_lambda,
          log_dir, 
          config,
          wandb_flag = False):
    set_seeds(seed)
    optimizer = torch.optim.AdamW([{'params': model.parameters()}], lr=learning_rate)

    # scheduler = MultiStepLR(optimizer, milestones=[10,80], gamma=0.1)
    losses = []
    train_loss, prob_true_acts = 0, 0
    best_test_loss = np.inf
    weight_regularization = 1

    # Initialize for writing on tensorboard
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(log_dir, str(time))

    # Create directories for saving stuff
    os.makedirs(os.path.join(log_dir, 'traj'))
    os.makedirs(os.path.join(log_dir, 'models'))


    summary_dir = os.path.join(log_dir, 'summary')
    writer = SummaryWriter(log_dir=summary_dir)
    # copy config to log dir
    with open(os.path.join(log_dir, "config.yaml"), 'w') as file:
        yaml.dump(config, file)

    model.writer = writer

    i = 0
    for epoch in tqdm(range(1, n_epochs+1)):
        batch_loss = 0
        num_batches = 0

        for tup in train_dataloader:
            model.train()
            if config["datasets"]["dataset_type"] == "padded":
                x, x_lens, y_train = tup
                x_train = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)

            elif config["datasets"]["dataset_type"] == "padded_vae":
                x, prev_red_state, x_lens, y_train = tup
                x_train = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
                prev_red_state = torch.stack(prev_red_state).to(device).float()

            else:
                x_train, y_train = tup
                x_train = x_train.to(device).float()

            y_train = y_train.to(device).float()

            train_loss = 0
            i += 1
            num_batches += 1

            if "vae" in config["model"]["model_type"]:
                if config["datasets"]["dataset_type"] == "padded_vae":
                    loss_one = model.compute_loss((x_train, prev_red_state), y_train)
                else:
                    loss_one = model.compute_loss((x_train, y_train), y_train)
            else:
                loss_one = model.compute_loss(x_train, y_train)
            train_loss = loss_one
            batch_loss += train_loss.item()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            writer.add_scalar('loss/train/neg_logp_train', loss_one.item(), i)

        losses.append(batch_loss)
        writer.add_scalar('loss/train/overall_loss', batch_loss / (num_batches), epoch)

        # After every n epochs evaluate
        if epoch % 2 == 0:
            batch_test = 0
            num_batches_test = 0 
            for tup_test in test_dataloader:
                if config["datasets"]["dataset_type"] == "padded":
                    x, x_lens, y_test = tup_test
                    x_test = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
                elif config["datasets"]["dataset_type"] == "padded_vae":
                    x, prev_red_state, x_lens, y_test = tup_test
                    x_test = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
                    prev_red_state = torch.stack(prev_red_state).to(device).float()
                else:
                    x_test, y_test = tup_test
                    x_test = x_test.to(device).float()

                y_test = y_test.to(device).float()

                num_batches_test += 1
                if "vae" in config["model"]["model_type"]:
                    if config["datasets"]["dataset_type"] == "padded_vae":
                        test_loss = model.compute_loss((x_test, prev_red_state), y_test)
                    else:
                        test_loss = model.compute_loss((x_test, y_test), y_test)
                else:
                    test_loss = model.compute_loss(x_test, y_test)
                # test_loss = loss_fn(out, y_test)
                batch_test += test_loss.item()

            writer.add_scalar('loss/test/overall_loss', batch_test / (num_batches_test), epoch)

            if wandb_flag:
                wandb.log({'epoch': epoch,
                        'train_loss': batch_loss / (num_batches),
                        'test_loss': batch_test / (num_batches_test)})

            model.train()
            if log_dir:
                if batch_test < best_test_loss:
                    best_test_loss = batch_test
                    print(f"Saving Best Model... {batch_test / num_batches_test}")
                    torch.save(model.state_dict(), os.path.join(log_dir,  "best.pth"))


            torch.save(model.state_dict(), os.path.join(log_dir,  f"models/{epoch}.pth"))

def get_configs():
    """
    Parse command line arguments and return the resulting args namespace
    """
    parser = argparse.ArgumentParser("Train Filtering Modules")
    parser.add_argument("--config", type=str, required=True, help="Path to .yaml config file")
    # make --wandb a true or false flag
    parser.add_argument("--wandb", type=bool, default=False, help="Use wandb")
    args = parser.parse_args()
    
    with open(args.config, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    return data_loaded, args.config, args.wandb

def main(wandb_flag):
    print("Loading config file ", sys.argv[1])
    # Load configs
    config, config_path, _ = get_configs()
    if wandb_flag:
        run = wandb.init()
        # batch_size = config["batch_size"]
        
        learning_rate = wandb.config.lr
        epochs = wandb.config.epochs
        batch_size = wandb.config.batch_size

        config["training"]["learning_rate"] = wandb.config.lr
        config["training"]["epochs"] = wandb.config.epochs
        config["batch_size"] = wandb.config.batch_size

        config["model"]["h1"] = wandb.config.h1
        config["model"]["h2"] = wandb.config.h2
    else:
        batch_size = config["batch_size"]
        learning_rate = config["training"]["learning_rate"]
        epochs = config["training"]["epochs"]

    # Configure dataloaders
    train_dataloader, test_dataloader = load_datasets(config["datasets"], batch_size)
    device = config["device"]

    # Load model
    model = configure_model(config).to(device)
    # model = nn.DataParallel(model).to(device)

    if config["model"]["load_pth"] is not None:
        model.load_state_dict(torch.load(config["model"]["load_pth"]))

    train_configs = config["training"]
    seed = train_configs["seed"]
    # learning_rate = train_configs["learning_rate"]
    # epochs = train_configs["epochs"]

    log_dir = train_configs["log_dir"]
    l2_lambda = train_configs["l2_lambda"]

    train(seed,
        device,
        train_dataloader,
        test_dataloader,
        batch_size,
        model,
        learning_rate = learning_rate,
        n_epochs = epochs,
        l2_lambda = l2_lambda,
        log_dir = log_dir,
        config = config)


def main_config_reg(config):
    """ Pass in config dictionary to train model """
    batch_size = config["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    epochs = config["training"]["epochs"]

    # Configure dataloaders
    train_dataloader, test_dataloader = load_datasets(config["datasets"], batch_size)
    device = config["device"]

    # Load model
    model = configure_model(config).to(device)
    # model = nn.DataParallel(model).to(device)

    if config["model"]["load_pth"] is not None:
        model.load_state_dict(torch.load(config["model"]["load_pth"]))

    train_configs = config["training"]
    seed = train_configs["seed"]
    # learning_rate = train_configs["learning_rate"]
    # epochs = train_configs["epochs"]

    log_dir = train_configs["log_dir"]
    l2_lambda = train_configs["l2_lambda"]

    train(seed,
        device,
        train_dataloader,
        test_dataloader,
        batch_size,
        model,
        learning_rate = learning_rate,
        n_epochs = epochs,
        l2_lambda = l2_lambda,
        log_dir = log_dir,
        config = config)

if __name__ == "__main__":
    config, config_path, wandb_flag = get_configs()

    if wandb_flag:
        import wandb
        # Define sweep config
        sweep_configuration = {
            'method': 'random',
            'name': 'sweep',
            'metric': {'goal': 'minimize', 'name': 'test_loss'},
            'parameters': 
            {
                'optimizer':{
                    'values':['adam']
                },
                'batch_size': {
                    'values': [32, 64, 128, 256]
                    },
                'h1': {
                    'values': [8, 16, 32]
                },
                'h2': {
                    'values': [8, 16, 32]
                },
                'epochs': {'values': [80, 100]},
                'lr': {'max': 0.01, 'min': 0.0001, 'distribution': 'uniform'}
            }
        }

        # func = lambda: main(config)
        # Initialize sweep by passing in config. (Optional) Provide a name of the project.
        sweep_id = wandb.sweep(sweep=sweep_configuration, project='fugitive')
        wandb.agent(sweep_id, function=lambda: main(wandb_flag), count=50)
    else:
        print("No WANDB")
        main(wandb_flag)
        