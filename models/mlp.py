import torch
import math
import torch.nn as nn

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

        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

        # Predict Mixture of gaussians from input
        self.pi_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden),
            self.non_linear,
            nn.Linear(hidden, hidden),
            )

        self.pi = nn.Sequential(
            self.non_linear,
            nn.Linear(hidden, num_gaussians),
            nn.Softmax(dim=1)
            )

        self.sigma_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden),
            self.non_linear,
            nn.Linear(hidden, hidden),
        )

        self.sigma = nn.Sequential(
            self.non_linear,
            nn.Linear(hidden, output_dim * num_gaussians),
            )
        
        self.mu_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden),
            self.non_linear,
            nn.Linear(hidden, hidden),
        )

        self.mu = nn.Sequential(
            self.non_linear,
            nn.Linear(hidden, output_dim * num_gaussians),
            )
        
    def forward(self, x):
        # Predict the mixture of gaussians 
        pi_hidden = self.pi_embedding(x)
        pi = self.pi(pi_hidden) + 1e-4
        sigma_hidden = self.sigma_embedding(x)
        sigma = torch.exp(self.sigma(sigma_hidden)).view(-1, self.num_gaussians, self.output_dim) + 1e-4
        mu_hidden = self.mu_embedding(x)
        mu = self.mu(mu_hidden).view(-1, self.num_gaussians, self.output_dim)
        return pi, mu, sigma

    def generate_embeddings(self, x):
        pi_hidden = self.pi_embedding(x)
        sigma_hidden = self.sigma_embedding(x)
        mu_hidden = self.mu_embedding(x)
        return pi_hidden, mu_hidden, sigma_hidden

    def compute_loss(self, obs, red_locs):
        target = red_locs.to(self.device).float()
        out = self.forward(obs.to(self.device))
        neg_log_likelihood_loss = mdn_negative_log_likelihood_loss(out, target)
        loss = neg_log_likelihood_loss # + 5 * dynamic_loss
        return loss

    def get_stats(self, nn_output, target):
        pi, mu, sigma = nn_output
        log_likelihoods = []
        for i in range(0,self.output_dim,2):
            mu_i = mu[:, :, i:i+2]
            sigma_i = sigma[:, :, i:i+2]
            red_locs = target[:, i:i+2]
            log_likelihoods.append(-1 * mdn_negative_log_likelihood(pi, mu_i, sigma_i, red_locs))
        return torch.stack(log_likelihoods, dim=1)

    @property
    def device(self):
        return next(self.parameters()).device

def mdn_negative_log_likelihood(pi, mu, sigma, target):
    """ Use torch.logsumexp for more stable training 
    
    This is equivalent to the mdn_loss but computed in a numerically stable way
    """
    target = target.unsqueeze(1).expand_as(sigma)
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