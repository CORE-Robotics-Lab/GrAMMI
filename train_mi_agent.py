"""
Copyright (2022)
Georgia Tech Research Corporation (Sean Ye, Manisha Natarajan, Rohan Paleja, Letian Chen, Matthew Gombolay)
All rights reserved.
"""
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
from models.mi.mi_blue import mdn_negative_log_likelihood_loss

class PosteriorMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(PosteriorMLP, self).__init__() 
        self.h1 = nn.Linear(in_dim, hidden_dim)
        self.h2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)
        self.leaky_relu = nn.LeakyReLU()

        # initialize these layers, zero bias as well
        nn.init.xavier_uniform_(self.h1.weight)
        nn.init.xavier_uniform_(self.h2.weight)
        nn.init.xavier_uniform_(self.out.weight)

        nn.init.zeros_(self.h1.bias)
        nn.init.zeros_(self.h2.bias)
        nn.init.zeros_(self.out.bias)

    def forward(self, state, next_state):
        x = torch.cat([state, next_state], dim=1)
        x = self.leaky_relu(self.h1(x))
        x = self.leaky_relu(self.h2(x))
        x = self.out(x)
        x = self.softmax(x)
        return x

def compute_loss(output, red_locs):
    mean, std = output
    distribution = Normal(mean, std)
    logprob = log_prob(distribution, red_locs)
    decoder_loss = -logprob.mean()
    return decoder_loss

def nll_categorical(probs, labels):
    probs = torch.clip(probs, 1e-8, 1.0 - 1e-8)
    return (-torch.log(probs) * labels).sum(dim=-1)

# def initialize_categorical_embeddings(num_categories, num_embeddings, device):
#     cats = torch.eye(num_categories)
#     params = [nn.Parameter(torch.tensor(cats[np.random.choice(num_categories)]).to(device).long()) for _ in range(num_embeddings)]
#     return nn.ParameterList(params)

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
          config_path, 
          config):
    set_seeds(seed)

    # embeddings_dim = config["model"]["embedding_dim"]
    num_gaussians = config["model"]["number_gaussians"]
    alpha = config["training"]["alpha"]

    # trajectory_specific_embeddings = initialize_categorical_embeddings(num_gaussians, 300, device)
    # test_specific_embeddings = initialize_categorical_embeddings(num_gaussians, 100, device)

    # trajectory_specific_embeddings = nn.ParameterList([nn.Parameter(torch.zeros(embeddings_dim)) for _ in range(300)]).to(device).long()
    # test_specific_embeddings = nn.ParameterList([nn.Parameter(torch.zeros(embeddings_dim)) for _ in range(100)]).to(device).long()

    posterior = PosteriorMLP(in_dim=37, hidden_dim=16, out_dim=4).to(device)


    # include model parameters 
    # optimizer = torch.optim.RMSprop([{'params': model.parameters()}, {'params': posterior.parameters()}], lr=learning_rate, weight_decay=1e-5)

    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': posterior.parameters()}], lr=learning_rate, weight_decay=1e-5)

    # optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': posterior.parameters()}], lr=learning_rate)

    scheduler = MultiStepLR(optimizer, milestones=[12,80], gamma=0.1)
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
    os.makedirs(os.path.join(log_dir, 'train_embeds'))
    os.makedirs(os.path.join(log_dir, 'test_embeds'))
    os.makedirs(os.path.join(log_dir, 'posterior'))

    summary_dir = os.path.join(log_dir, 'summary')
    writer = SummaryWriter(log_dir=summary_dir)

    # copy config to log dir
    # shutil.copy(config_path, os.path.join(log_dir, "config.yaml"))

    with open(os.path.join(log_dir, "config.yaml"), 'w') as file:
        yaml.dump(config, file)

    # loss_fn = nn.MSELoss()
    loss_fn = compute_loss
    model.writer = writer

    clip_value = 0.1

    i = 0
    for epoch in tqdm(range(1, n_epochs+1)):
        batch_loss = 0
        num_batches = 0

        for tup in train_dataloader:
            model.train()

            x, x_lens, agent_obs, agent_lens, n_agents, y_train = tup
            x_train = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
            agent_packed = pack_padded_sequence(agent_obs, agent_lens, batch_first=True, enforce_sorted=False)

            y_train = y_train.to(device).float()

            train_loss = 0
            i += 1
            num_batches += 1

            out, posterior_input, pi_cats = model.forward_encoder((x_train, agent_packed, n_agents))            
            post = posterior(*posterior_input)


            # loss_one = loss_fn(out, y_train) # this is the loss for the real next state
            loss_one = mdn_negative_log_likelihood_loss(out, y_train)
            loss_two = nll_categorical(post, pi_cats).mean()

            train_loss = loss_one + loss_two * alpha
            # print('NLL loss: ' + str(loss_one.item()))
            # print('MI Loss: ' + str(max(loss_two))

            batch_loss += train_loss.item()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
            torch.nn.utils.clip_grad_value_(posterior.parameters(), clip_value)
            writer.add_scalar('loss/train/neg_logp_train', loss_one.item(), i)
            writer.add_scalar('loss/train/mi_loss', loss_two.item(), i)
        scheduler.step()
        losses.append(batch_loss)
        writer.add_scalar('loss/train/overall_loss', batch_loss / (num_batches), epoch)

        # After every n epochs evaluate
        if epoch % 2 == 0:
            batch_test = 0
            num_batches_test = 0 
            nll_test = 0
            mi_test = 0
            for tup_test in test_dataloader:
                x, x_lens, agent_obs, agent_lens, n_agents, y_test = tup_test
                x_test = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
                agent_packed = pack_padded_sequence(agent_obs, agent_lens, batch_first=True, enforce_sorted=False)

                y_test = y_test.to(device).float()

                num_batches_test += 1
                out, posterior_input, pi_cats = model.forward_encoder((x_test, agent_packed, n_agents))
                post = posterior(*posterior_input)

                
                # test_loss = loss_fn(out, y_test)
                test_loss = mdn_negative_log_likelihood_loss(out, y_test)
                test_loss_two = nll_categorical(post, pi_cats).mean()

                batch_test += test_loss.item() + test_loss_two.item() * alpha
                nll_test += test_loss.item()
                mi_test += test_loss_two.item()
                

            writer.add_scalar('loss/test/overall_loss', batch_test / (num_batches_test), epoch)
            writer.add_scalar('loss/test/neg_logp_test', nll_test / (num_batches_test), epoch)
            writer.add_scalar('loss/test/mi_loss', mi_test / (num_batches_test), epoch)

            if wandb_flag:
                wandb.log({'epoch': epoch,
                        'train_loss': batch_loss / (num_batches),
                        'test_loss': batch_test / (num_batches_test),
                        'neg_logp_test': nll_test / (num_batches_test)})

            model.train()
            if log_dir:
                if batch_test < best_test_loss:
                    best_test_loss = batch_test
                    print(f"Saving Best Model... {batch_test / num_batches_test}")
                    torch.save(model.state_dict(), os.path.join(log_dir,  "best.pth"))
                    # torch.save(test_specific_embeddings, os.path.join(log_dir, "embeddings_test.pth"))
                    # torch.save(trajectory_specific_embeddings, os.path.join(log_dir, "embeddings_train.pth"))

        if epoch % 2 == 0:

            # plot the embeddings
            # embeds = torch.stack([trajectory_specific_embeddings[idx] for idx in range(300)]).cpu().detach().numpy()
            # kmeans = KMeans(n_clusters=2, random_state=0).fit(embeds)
            # labels = kmeans.labels_
            # colors = ['red', 'green', 'blue', 'orange', 'purple']
            # for j, traj in enumerate(train_dataloader.dataset.red_locs_per_traj):
            #     color = colors[labels[j]]
            #     plt.plot(traj[:, 0], traj[:, 1], c=color)
            #     plt.xlim(0,1)
            #     plt.ylim(0,1)
            #     plt.savefig(os.path.join(log_dir, f'traj/{epoch}.png'))

            torch.save(model.state_dict(), os.path.join(log_dir,  f"models/{epoch}.pth"))
            # torch.save(test_specific_embeddings, os.path.join(log_dir, f"test_embeds/{epoch}.pth"))
            # torch.save(trajectory_specific_embeddings, os.path.join(log_dir, f"train_embeds/{epoch}.pth"))

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

        config["model"]["h1"] = wandb.config.h1
        config["model"]["h2"] = wandb.config.h2
        config["model"]["alpha"] = wandb.config.alpha
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
        config_path = config_path,
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
                    'values': [32]
                },
                'h2': {
                    'values': [32]
                },
                'alpha': {'max': 10, 'min': 1, 'distribution': 'uniform'},
                'epochs': {'values': [60]},
                'lr': {'max': 0.001, 'min': 0.0001, 'distribution': 'uniform'}
            }
        }

        # func = lambda: main(config)
        # Initialize sweep by passing in config. (Optional) Provide a name of the project.
        sweep_id = wandb.sweep(sweep=sweep_configuration, project='fugitive')
        wandb.agent(sweep_id, function=lambda: main(wandb_flag), count=50)
    else:
        print("No WANDB")
        main(wandb_flag)