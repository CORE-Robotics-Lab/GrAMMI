import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
from functools import wraps
import torch.optim as optim
import sys, os
sys.path.append(os.getcwd())
import numpy as np
import torch
import random
import shutil

from models.encoders import EncoderRNN

import pickle
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
# from multi_step_prediction.dataset import PrisonerLocationDataset, create_dataset
# from multi_step_prediction.utils_msp import proba_distribution, calculate_loss, plot_gaussian_heatmap
from torch.utils.data import Dataset, DataLoader


class VAE_Opponent(nn.Module):
    def __init__(self, input_dim, enc_hidden_dim, latent_dim, num_layers=1,
                 beta=4):
        super(VAE_Opponent, self).__init__()
        self.input_dim = input_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.num_layers = num_layers

        self.rnn = EncoderRNN(input_dim, enc_hidden_dim, num_layers)
        self.vae_input_dim = enc_hidden_dim + 2  # Concat the hidden vector of blue obs with the red state
        self.vae_latent_dim = latent_dim
        self.beta = beta

        self.fc_mu = nn.Linear(self.vae_input_dim, self.vae_latent_dim)
        self.fc_var = nn.Linear(self.vae_input_dim, self.vae_latent_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, x):
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return (mu, log_var)

    # def decode(self, x):
    #     concat_embed, enc_mean, enc_log_var = x
    #
    #     x = self.fc(concat_embed)
    #     x = self.dropout(x)
    #
    #     batch_size = x.size(0)
    #     # Predict the mixture of gaussians around the fugitive
    #     pi = self.pi(x)
    #     pi = pi.view(batch_size, self.num_heads, self.num_gaussians)
    #     pi = self.softmax(pi)
    #
    #     sigma = torch.exp(self.sigma(x))
    #     sigma = sigma.view(batch_size, self.num_heads, self.num_gaussians, self.output_dim)
    #
    #     mu = self.mu(x)
    #     mu = mu.view(batch_size, self.num_heads, self.num_gaussians, self.output_dim)
    #
    #     sigma = nn.ELU()(sigma) + 1e-15
    #     # sigma = torch.clamp(mu, min=0.00001)
    #     # sigma = self.relu(sigma)
    #     return pi, mu, sigma

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from Gaussian (mu, var)
        :param mu:
        :param log_var:
        :return:
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        """Need both x and y for training variational autoencoder"""
        obs, y = x
        hidden_blue = self.rnn(obs)

        # Concatenate and pass through VAE
        mu, log_var = self.encode(torch.cat((hidden_blue, y), dim=-1))
        std = torch.exp(0.5 * log_var)

        # Reparametrization trick
        z = self.reparameterize(mu, log_var)

        res = torch.cat((hidden_blue, z), dim=-1)  # Concatenate z with hidden blue obs (instead of red state)

        return (res, mu, std)