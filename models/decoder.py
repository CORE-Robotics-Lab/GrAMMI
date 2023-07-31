import sys, os

sys.path.append(os.getcwd())
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from tqdm import tqdm
from datetime import datetime
from torch.distributions import Normal
import math
import random

from torch import Tensor
from typing import Callable, Dict, List, Tuple, TypeVar, Union
import torch.distributions as D
# from shared_latent.functions import logavgexp, flatten_batch, unflatten_batch, insert_dim, NoNorm


import math
import torch
from torch.distributions import Categorical

from models.utils import log_prob


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
torch.cuda.empty_cache()

from torch.autograd import Variable

class SingleGaussianDecoderStd(nn.Module):
    def __init__(self, input_dim, output_dim):
        """This Decoder consists of a single Gaussian where the std is a learned parameter

        Args:
            input_dim (_type_): Input dimension to the fully connected layer
            output_dim (_type_): Output dimension, in prisoner scenario this is 2 (x, y)
        """
        super().__init__()
        # Predict Fugitive Position (x, y) from the encoded embedding
        
        self.fc = nn.Linear(input_dim, output_dim*2)
        self.output_dim = output_dim
        # self.log_std = nn.Parameter(torch.ones(output_dim) * log_std_init, requires_grad=True)
        self.sigmoid = nn.Sigmoid()

    # def forward(self, x):
    #     # Use Linear Layer to predict fugitive location (not DeConv)
    #     x = self.fc(x)
    #     return x

    def forward(self, x):
        x = self.fc(x)
        mean = x[:, :self.output_dim]
        logstd = x[:, self.output_dim:]
        return mean, torch.ones_like(mean) * logstd.exp()

    def compute_loss(self, output, red_locs):
        mean, std = output
        distribution = Normal(mean, std)
        logprob = log_prob(distribution, red_locs)

        # prob_true_act = torch.exp(logprob).mean()
        # logprob = logprob.mean()
        decoder_loss = -logprob.mean()
        return decoder_loss

    def get_stats(self, output, red_locs):
        mean, std = output
        distribution = Normal(mean, std)
        logprob = distribution.log_prob(red_locs)
        logprob = logprob.view(logprob.shape[0], 13, 2)
        # print(logprob)
        # take mean across a dimension
        logprob = logprob.sum(dim=2)
        # logprob_stack.append(logprob)
        return logprob

class SingleGaussianDecoderStdParameter(nn.Module):
    def __init__(self, input_dim, output_dim, log_std_init=0.0):
        """This Decoder consists of a single Gaussian where the std is set as a parameter

        Args:
            input_dim (_type_): Input dimension to the fully connected layer
            output_dim (_type_): Output dimension, in prisoner scenario this is 2 (x, y)
            log_std_init (float, optional): _description_. Defaults to 0.0.
        """
        super().__init__()
        # Predict Fugitive Position (x, y) from the encoded embedding
        self.fc = nn.Linear(input_dim, output_dim)
        self.log_std = nn.Parameter(torch.ones(output_dim) * log_std_init, requires_grad=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Use Linear Layer to predict fugitive location (not DeConv)
        x = self.fc(x)
        return x

    def get_mean_std(self, x):
        x = self.forward(x)
        return x, torch.ones_like(x) * self.log_std.exp()

    def compute_loss(self, features, red_locs):
        decoder_output = self.forward(features)
        std = torch.ones_like(decoder_output) * self.log_std.exp()
        distribution = Normal(decoder_output, std)
        logprob = log_prob(distribution, red_locs)

        # prob_true_act = torch.exp(logprob).mean()
        # logprob = logprob.mean()
        decoder_loss = -logprob.mean()
        return decoder_loss

class MixtureDensityDecoder(nn.Module):
    def __init__(self, input_dim, output_dim=2, num_gaussians=2):
        super(MixtureDensityDecoder, self).__init__()
        """ This Decoder consists of a mixture of gaussians """
        # self.batch_size = batch_size
        self.input_dim = input_dim
        self.num_gaussians = num_gaussians
        self.output_dim = output_dim

        # Predict Mixture of gaussians from input
        self.pi = nn.Sequential(
            nn.Linear(input_dim, num_gaussians),
            nn.Softmax(dim=1)
        )

        self.sigma = nn.Linear(input_dim, output_dim * num_gaussians)
        nn.init.normal_(self.sigma.weight)
        
        self.mu = nn.Linear(input_dim, output_dim * num_gaussians)
        nn.init.normal_(self.mu.weight)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.elu = nn.ELU()

    def forward(self, x):
        # Predict the mixture of gaussians 
        pi = self.pi(x)
        sigma = torch.exp(self.sigma(x))
        sigma = sigma.view(-1, self.num_gaussians, self.output_dim)
        mu = self.mu(x)
        mu = mu.view(-1, self.num_gaussians, self.output_dim)
        return pi, mu, sigma

    def compute_loss(self, nn_output, red_locs):
        # nn_output = self.forward(features)
        return mdn_negative_log_likelihood_loss(nn_output, red_locs)

    def get_stats(self, nn_output, target):
        pi, mu, sigma = nn_output
        log_likelihoods = []
        for i in range(0,self.output_dim,2):
            mu_i = mu[:, :, i:i+2]
            sigma_i = sigma[:, :, i:i+2]
            red_locs = target[:, i:i+2]
            log_likelihoods.append(-1 * mdn_negative_log_likelihood(pi, mu_i, sigma_i, red_locs))
        return torch.stack(log_likelihoods, dim=1)

def mdn_negative_log_likelihood(pi, mu, sigma, target):
    """ Use torch.logsumexp for more stable training 
    
    This is equivalent to the mdn_loss but computed in a numerically stable way

    """
    # target = target.unsqueeze(1).expand_as(sigma)
    neg_logprob = -torch.log(sigma) - (math.log(2 * math.pi) / 2) - \
        ((target - mu) / sigma)**2 / 2
    
    inner = torch.log(pi) + torch.sum(neg_logprob, 2) # Sum the log probabilities of (x, y) for each 2D Gaussian
    return -torch.logsumexp(inner, dim=1)

def mdn_negative_log_likelihood_loss(nn_output, target):
    """
    Compute the negative log likelihood loss for a MoG model.
    """
    pi, mu, sigma = nn_output
    return mdn_negative_log_likelihood(pi, mu, sigma, target).mean()


def sample(pi, mu, sigma):
    """Draw samples from a MoG.
    """
    # Choose which gaussian we'll sample from
    pis = Categorical(pi).sample().view(pi.size(0), 1, 1)
    # Choose a random sample, one randn for batch X output dims
    # Do a (output dims)X(batch size) tensor here, so the broadcast works in
    # the next step, but we have to transpose back.
    gaussian_noise = torch.randn(
        (sigma.size(2), sigma.size(0)), requires_grad=False)
    variance_samples = sigma.gather(1, pis).detach().squeeze()
    mean_samples = mu.detach().gather(1, pis).squeeze()
    return (gaussian_noise * variance_samples + mean_samples).transpose(0, 1)


class MixtureDensityMLP(nn.Module):
    def __init__(self, input_dim, hidden=16, output_dim=2, num_gaussians=2, non_linear=nn.ReLU()):
        super(MixtureDensityMLP, self).__init__()
        """ This Decoder consists of a mixture of gaussians """
        # self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden = hidden
        self.num_gaussians = num_gaussians
        self.output_dim = output_dim
        self.non_linear = non_linear

        # Predict Mixture of gaussians from input
        # Predict Mixture of gaussians from encoded embedding
        self.pi = nn.Linear(input_dim, num_gaussians)
        nn.init.xavier_uniform_(self.pi.weight)
        nn.init.zeros_(self.pi.bias)
        self.softmax = nn.Softmax(dim=-1)

        self.sigma = nn.Linear(input_dim, output_dim * num_gaussians)
        nn.init.xavier_normal_(self.sigma.weight)
        nn.init.zeros_(self.sigma.bias)

        self.mu = nn.Linear(input_dim, output_dim * num_gaussians)
        nn.init.xavier_normal_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)


        # self.pi = nn.Sequential(
        #     nn.Linear(input_dim, hidden),
        #     self.non_linear,
        #     nn.Linear(hidden, hidden),
        #     self.non_linear,
        #     nn.Linear(hidden, num_gaussians),
        #     nn.Softmax(dim=1)
        # )
        #
        # self.sigma = nn.Sequential(
        #     nn.Linear(input_dim, hidden),
        #     self.non_linear,
        #     nn.Linear(hidden, hidden),
        #     self.non_linear,
        #     nn.Linear(hidden, output_dim * num_gaussians),
        # )
        # # nn.init.zeros_(self.sigma.weight)
        #
        # self.mu = nn.Sequential(
        #     nn.Linear(input_dim, hidden),
        #     self.non_linear,
        #     nn.Linear(hidden, hidden),
        #     self.non_linear,
        #     nn.Linear(hidden, output_dim * num_gaussians),
        # )

        # nn.init.normal_(self.mu.weight)

        # self.sigmoid = nn.Sigmoid()
        # self.dropout = nn.Dropout(0.5)
        # self.elu = nn.ELU()

    def forward(self, x):
        # Predict the mixture of gaussians
        pi = self.softmax(self.pi(x))
        sigma = torch.exp(self.sigma(x))
        sigma = sigma.view(-1, self.num_gaussians, self.output_dim)
        # mu = self.sigmoid(self.mu(x))
        mu = self.mu(x)
        mu = mu.view(-1, self.num_gaussians, self.output_dim)
        return pi, mu, sigma

    def compute_loss(self, out, red_locs):
        target = red_locs.to(self.device).float()
        # out = self.forward(obs.to(self.device))
        return mdn_negative_log_likelihood_loss(out, target)

    def get_stats(self, nn_output, target):
        pi, mu, sigma = nn_output
        log_likelihoods = []
        for i in range(0, self.output_dim, 2):
            mu_i = mu[:, :, i:i + 2]
            sigma_i = sigma[:, :, i:i + 2]
            red_locs = target[:, i:i + 2]
            log_likelihoods.append(-1 * mdn_negative_log_likelihood(pi, mu_i, sigma_i, red_locs))
        return torch.stack(log_likelihoods, dim=1)

    @property
    def device(self):
        return next(self.parameters()).device

class DenseNormalDecoder(nn.Module):

    def __init__(self, in_dim, out_dim=1, hidden_dim=400, hidden_layers=2, layer_norm=True, std=0.3989422804):
        super().__init__()
        self.model = MLP(in_dim, out_dim, hidden_dim, hidden_layers, layer_norm)
        self.std = std
        self.out_dim = out_dim

    def forward(self, features: Tensor) -> D.Distribution:
        y = self.model.forward(features)
        p = D.Normal(loc=y, scale=torch.ones_like(y) * self.std)
        if self.out_dim > 1:
            p = D.independent.Independent(p, 1)  # Makes p.logprob() sum over last dim
        return p

    def loss(self, output: D.Distribution, target: Tensor) -> Tensor:
        var = self.std ** 2  # var cancels denominator, which makes loss = 0.5 (target-output)^2
        return -output.log_prob(target) * var

    def training_step(self, features: Tensor, target: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # assert len(features.shape) == 4
        # I = features.shape[2]
        # target = insert_dim(target, 2, I)  # Expand target with iwae_samples dim, because features have it

        decoded = self.forward(features)
        loss_tbi = self.loss(decoded, target)
        loss_tb = -logavgexp(-loss_tbi, dim=0)  # TBI => TB
        # decoded = decoded.mean.mean(dim=2)

        # assert len(loss_tbi.shape) == 3
        # assert len(loss_tb.shape) == 2
        # assert len(decoded.shape) == (2 if self.out_dim == 1 else 3)
        return loss_tbi, loss_tb, decoded

    def compute_loss(self, features, target):
        loss_tbi, loss_tb, decoded = self.training_step(features, target)
        return loss_tbi.mean()

class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, hidden_layers, layer_norm, activation=nn.ELU):
        super().__init__()
        self.out_dim = out_dim
        norm = nn.LayerNorm if layer_norm else NoNorm
        layers = []
        for i in range(hidden_layers):
            layers += [
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                norm(hidden_dim, eps=1e-3),
                activation()
            ]
        layers += [
            nn.Linear(hidden_dim, out_dim),
        ]
        if out_dim == 1:
            layers += [
                nn.Flatten(0),
            ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x, bd = flatten_batch(x)
        y = self.model(x)
        y = unflatten_batch(y, bd)
        return y

class MixureDecoder(nn.Module):
    """
    """

    def __init__(self, input_dim, output_dim=2, num_gaussians=2, log_std_init=0.0):
        super(MixureDecoder, self).__init__()

        # self.batch_size = batch_size
        self.num_gaussians = num_gaussians
        self.output_dim = output_dim
        hidden_dim = input_dim
        # hidden_dim = 32

        # self.fc = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)

        # Predict Mixture of gaussians from encoded embedding
        self.pi = nn.Sequential(
            nn.Linear(hidden_dim, num_gaussians),
            nn.Softmax(dim=1)
        )

        self.sigma = nn.Linear(hidden_dim, output_dim * num_gaussians)
        nn.init.normal_(self.sigma.weight)
        
        self.mu = nn.Linear(hidden_dim, output_dim * num_gaussians)
        nn.init.normal_(self.mu.weight)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = self.fc(x)
        # x = self.dropout(x)

        # Predict the mixture of gaussians around the fugitive
        pi = self.pi(x)
        sigma = torch.exp(self.sigma(x))
        sigma = sigma.view(-1, self.num_gaussians, self.output_dim)
        mu = self.mu(x)
        mu = mu.view(-1, self.num_gaussians, self.output_dim)

        sigma = torch.clamp(mu, min=0.00001)
        return pi, sigma, mu


    def mdn_negative_log_likelihood(self, pi, sigma, mu, target):
        """ Use torch.logsumexp for more stable training 
        
        This is equivalent to the mdn_loss but computed in a numerically stable way

        """
        target = target.unsqueeze(1).expand_as(sigma)
        neg_logprob = -torch.log(sigma) - (math.log(2 * math.pi) / 2) - \
            ((target - mu) / sigma)**2 / 2
        
        inner = torch.log(pi) + torch.sum(neg_logprob, 2) # Sum the log probabilities of (x, y) for each 2D Gaussian
        return -torch.logsumexp(inner, dim=1)

    def mdn_negative_log_likelihood_loss(self, nn_output, target):
        """
        Compute the negative log likelihood loss for a MoG model.
        """
        pi, sigma, mu = nn_output
        return self.mdn_negative_log_likelihood(pi, sigma, mu, target)