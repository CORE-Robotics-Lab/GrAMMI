import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models.

Source Code adapted from: https://github.com/emited/VariationalRecurrentNeuralNetwork/blob/master/model.py
"""


class VariationalRNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, num_layers=1, bias=False, padded_input=False):

        super(VariationalRNNEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.num_layers = num_layers

        self.padded_input = padded_input

        # Feature extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU()
        )

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.enc_mean = nn.Linear(hidden_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(hidden_dim, z_dim),
            nn.Softplus()
        )

        # Prior
        self.prior = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.prior_mean = nn.Linear(hidden_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(hidden_dim, z_dim),
            nn.Softplus()
        )

        # recurrence
        self.rnn = nn.GRU(hidden_dim + hidden_dim, hidden_dim, num_layers, bias, batch_first=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, hidden_state=None):
        all_enc_mean, all_enc_std, all_prior_mean, all_prior_std = [], [], [], []
        # print(x)
        x_lengths = []
        if self.padded_input:
            x, x_lengths = x
            x_lengths = torch.Tensor(x_lengths).to(x.device).float()

        batch_size = x.size(0)
        seq_len = x.shape[1]
        x = x.to(self.device).float()

        # if self.padded_input:
        #     assert len(x_lengths) != 0
        #     x_lengths = torch.Tensor(x_lengths).to(x.device).float()

        if hidden_state is None:
            # initializing hidden state to zeros
            hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_().to(self.device)

        if self.padded_input:
            assert len(x_lengths) != 0
            for t in range(seq_len):
                phi_x_t = self.phi_x(x[:, t])
                # mask to remove interference of padded inputs
                phi_x_mask = (t < x_lengths).float().unsqueeze(1).expand_as(phi_x_t)

                if t == 0:
                    prev_phi_x_t = phi_x_t
                else:
                    phi_x_t = phi_x_t*phi_x_mask + prev_phi_x_t * (1-phi_x_mask)
                    prev_phi_x_t = phi_x_t

                # Encoder
                enc_t = self.enc(torch.cat((phi_x_t, hidden_state[-1]), dim=-1))
                enc_mean_t = self.enc_mean(enc_t)
                enc_std_t = self.enc_std(enc_t)

                # Prior
                prior_t = self.prior(hidden_state[-1])
                prior_mean_t = self.prior_mean(prior_t)
                prior_std_t = self.prior_std(prior_t)

                # Sampling and reparameterization
                z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
                phi_z_t = self.phi_z(z_t)

                # Recurrence
                _, hidden_state = self.rnn(torch.cat((phi_x_t, phi_z_t), dim=-1).unsqueeze(0), hidden_state)
                # mask to remove interference of padded inputs
                hidden_mask = (t < x_lengths).float().unsqueeze(1).expand_as(hidden_state)
                if t == 0:
                    prev_hidden_state = hidden_state
                else:
                    hidden_state = hidden_state*hidden_mask + prev_hidden_state * (1-hidden_mask)
                    prev_hidden_state = hidden_state

                # Append mean and std from encoder and prior dist to calculate loss
                all_enc_mean.append(enc_mean_t)
                all_enc_std.append(enc_std_t)
                all_prior_mean.append(prior_mean_t)
                all_prior_std.append(prior_std_t)
        else:
            for t in range(seq_len):
                phi_x_t = self.phi_x(x[:, t])

                # Encoder
                enc_t = self.enc(torch.cat((phi_x_t, hidden_state[-1]), dim=-1))
                enc_mean_t = self.enc_mean(enc_t)
                enc_std_t = self.enc_std(enc_t)

                # Prior
                prior_t = self.prior(hidden_state[-1])
                prior_mean_t = self.prior_mean(prior_t)
                prior_std_t = self.prior_std(prior_t)

                # Sampling and reparameterization
                z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
                phi_z_t = self.phi_z(z_t)

                # Recurrence
                _, hidden_state = self.rnn(torch.cat((phi_x_t, phi_z_t), dim=-1).unsqueeze(0), hidden_state)

                # Append mean and std from encoder and prior dist to calculate loss
                all_enc_mean.append(enc_mean_t)
                all_enc_std.append(enc_std_t)
                all_prior_mean.append(prior_mean_t)
                all_prior_std.append(prior_std_t)

        # Return hidden state from last timestep to decoder
        hn = hidden_state.view(-1, hidden_state.shape[-1])
        res = torch.cat((phi_z_t, hn), dim=-1)
        if self.padded_input:
            return res, all_enc_mean, all_enc_std, all_prior_mean, all_prior_std, x_lengths
        return res, all_enc_mean, all_enc_std, all_prior_mean, all_prior_std

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.empty(size=std.size(), device=self.device, dtype=torch.float).normal_()
        return eps.mul(std).add_(mean)