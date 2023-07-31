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


class CategoricalVAE(nn.Module):
    def __init__(self, input_dim, enc_hidden_dim, latent_dim, num_layers=1,
                 categorical_dim=4, beta=4, temperature=0.5):
        super(CategoricalVAE, self).__init__()
        self.input_dim = input_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.num_layers = num_layers

        self.rnn = EncoderRNN(input_dim, enc_hidden_dim, num_layers)
        self.vae_input_dim = enc_hidden_dim + 2  # Concat the hidden vector of blue obs with the red state
        self.vae_latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.beta = beta
        self.temp = temperature

        self.fc_z = nn.Linear(self.vae_input_dim, self.vae_latent_dim * self.categorical_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, x):
        z = self.fc_z(x)
        z = z.view(-1, self.vae_latent_dim, self.categorical_dim)
        return z

    def reparameterize(self, z, eps: float = 1e-7):
        """
        Gumbel-softmax trick to sample from Categorical Distribution
        :param z: (Tensor) Latent Codes [B x D x Q]
        :return: (Tensor) [B x D]
        """
        # Sample from Gumbel
        u = torch.rand_like(z)
        g = - torch.log(- torch.log(u + eps) + eps)

        # Gumbel-Softmax sample
        s = F.softmax((z + g) / self.temp, dim=-1)
        s = s.view(-1, self.vae_latent_dim * self.categorical_dim)
        return s

    def forward(self, x):
        """Need both x and y for training variational autoencoder"""
        obs, y = x
        hidden_blue = self.rnn(obs)

        # Concatenate and pass through VAE
        q = self.encode(torch.cat((hidden_blue, y), dim=-1))

        # Reparametrization trick with Gumbel
        z = self.reparameterize(q)

        res = torch.cat((hidden_blue, z), dim=-1)  # Concatenate z with hidden blue obs (instead of red state)

        return (res, z)