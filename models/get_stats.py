import torch
import numpy as np
from scipy.stats import multivariate_normal
import math

def get_stats(nn_output, red_locs, mode='mixture'):

    if mode == 'mixture':
        dist_func = distance_threshold_metric
    else:
        dist_func = distance_threshold_metric_with_single_mean

    log_prob = -mdn_negative_log_likelihood_loss(nn_output, red_locs)
    # red_locs = red_locs.view(-1, num_heads, output_dim)
    red_locs = red_locs.unsqueeze(1)

    pi, mu, sigma = nn_output
    mu = mu.unsqueeze(1)
    sigma = sigma.unsqueeze(1)
    pi = pi.unsqueeze(1)

    nn_output = (pi, mu, sigma)

    if mode == 'mixture':
        ade = average_displacement_error_from_mixture(nn_output, red_locs)
    else:
        ade = average_displacement_error_with_single_mean(nn_output, red_locs)
    dist_thresh_prob, one_sigma, two_sigma, three_sigma, num_steps = dist_func(nn_output, red_locs)
    return log_prob, ade, dist_thresh_prob, one_sigma, two_sigma, three_sigma, num_steps

def average_displacement_error_with_single_mean(nn_output, target):
    pi, mu, sigma = nn_output
    num_mixtures = pi.shape[-1]
    # Get the means and sigmas of the gaussian with the highest mixing coeff
    idx = torch.argmax(pi, dim=-1, keepdim=True)
    mu_0 = torch.gather(mu[:, :, :, 0], 2, idx)
    mu_1 = torch.gather(mu[:, :, :, 1], 2, idx)
    mu = torch.cat((mu_0, mu_1), dim=-1)
    mse_error = torch.linalg.norm((mu - target), dim=-1)
    return mse_error

def average_displacement_error_from_mixture(nn_output, target):
    pi, mu, sigma = nn_output
    num_mixtures = pi.shape[-1]
    target = target.unsqueeze(dim=2).repeat(1, 1, num_mixtures, 1)
    mse_error = torch.linalg.norm((mu - target), dim=-1)
    mse_error = torch.sum(pi * mse_error, axis=-1)  # Calculate weighted average of mse errors
    return mse_error

def distance_threshold_metric(nn_output, target, dist_threshold=0.05, device='cpu'):
    """
    Given mean, and logstd predicted from the filtering module, calculate the likelihood
    of the fugitive's ground truth location from the predicted distribution
    :param mean: (np.array) Mean of the predicted distribution from the filtering module
    :param logstd: (np.array) Logstd of the predicted distribution from the filtering module
    :param true_location: (np.array) Ground Truth location of the fugitive (x,y)
    :return:
        prob: The probability of the fugitive being at the ground truth location
                as predicted by the filtering module's distribution
    """
    pi, mu, sigma = nn_output
    bs, seq_len, num_mixtures, n_dim = sigma.shape
    # var = np.exp(logstd) ** 2
    var = torch.clamp(sigma ** 2, min=1e-5, max=1)

    var = var.repeat(1, 1, 1, n_dim)
    var = var.reshape(bs, seq_len, num_mixtures, n_dim, n_dim)

    cov = torch.ones(bs, seq_len, num_mixtures, n_dim, n_dim).to(device)
    probs = torch.empty(pi.shape).to('cpu')
    mahalanobis_dist = torch.empty(pi.shape).to('cpu').detach().numpy()

    num_steps_conf_thresh = np.zeros((bs, seq_len))

    target = target.unsqueeze(dim=2).repeat(1, 1, num_mixtures, 1)

    # Torch does not have CDF for multi-variate normal.
    # Hence converting to numpy and using scipy
    target = target.to('cpu').detach().numpy()
    pi = pi.to('cpu').detach()
    mu = mu.to('cpu').detach().numpy()
    cov = cov.to('cpu').detach().numpy()
    var = var.to('cpu').detach().numpy()

    # TODO: @Manisha get rid of the for loops
    for b in range(bs):
        for s in range(seq_len):
            cov[b, s] = cov[b, s] * np.repeat(np.eye(n_dim)[np.newaxis, :, :], num_mixtures, axis=0)
            cov[b, s] = cov[b, s] * var[b, s]

            # CDF calculates from -infty to the upper limit.
            # Therefore subtracting the lower limit to calculate the cdf between lower limit to upper limit instead of -infty to upper limit
            for i in range(num_mixtures):
                probs[b, s, i] = multivariate_normal.cdf(target[b, s, i] + dist_threshold, mean=mu[b, s, i], cov=cov[b, s, i]) - \
                                    multivariate_normal.cdf(target[b, s, i] - dist_threshold, mean=mu[b, s, i], cov=cov[b, s, i])
                x_mean = target[b, s, i] - mu[b, s, i]

                mahalanobis_dist[b, s, i] = np.sqrt(x_mean.T.dot(np.linalg.inv(cov[b, s, i])).dot(x_mean))

    # total_prob = torch.sum(pi * probs, axis=-1)
    total_prob = torch.max(probs, axis=-1)[0]

    # Calculate 1-sigma counts
    k = np.zeros_like(mahalanobis_dist)
    k[mahalanobis_dist < 1] = 1
    one_sigma_count = torch.sum(pi * k, axis=-1)

    # Calculate 2-sigma counts
    k = np.zeros_like(mahalanobis_dist)
    k[mahalanobis_dist < 2] = 1
    two_sigma_count = torch.sum(pi * k, axis=-1)

    # Calculate 3-sigma counts
    k = np.zeros_like(mahalanobis_dist)
    k[mahalanobis_dist < 3] = 1
    three_sigma_count = torch.sum(pi * k, axis=-1)

    # Distance threshold metric
    k = np.zeros_like(probs)
    k[probs >= 0.5] = 1
    num_steps_conf_thresh = torch.sum(pi * k, axis=-1)

    return total_prob, one_sigma_count, two_sigma_count, three_sigma_count, num_steps_conf_thresh

def distance_threshold_metric_with_single_mean(nn_output, target, dist_threshold=0.05, device='cpu'):
    pi, mu, sigma = nn_output
    # bs, seq_len, n_dim = sigma.shape
    bs, seq_len, num_mixtures, n_dim  = sigma.shape

    # Get the means and sigmas of the gaussian with the highest mixing coeff
    idx = torch.argmax(pi, dim=-1, keepdim=True)
    mu_0 = torch.gather(mu[:, :, :, 0], 2, idx)
    mu_1 = torch.gather(mu[:, :, :, 1], 2, idx)
    mu = torch.cat((mu_0, mu_1), dim=-1)

    sigma_0 = torch.gather(sigma[:, :, :, 0], 2, idx)
    sigma_1 = torch.gather(sigma[:, :, :, 1], 2, idx)
    sigma = torch.cat((sigma_0, sigma_1), dim=-1)

    var = torch.clamp(sigma ** 2, min=1e-5, max=1)

    var = var.repeat(1, 1, n_dim)
    var = var.reshape(bs, seq_len, n_dim, n_dim)

    cov = torch.ones(bs, seq_len, n_dim, n_dim).to(device)
    probs = torch.empty((bs, seq_len)).to('cpu')
    mahalanobis_dist = torch.empty((bs, seq_len)).to('cpu').detach().numpy()

    # Torch does not have CDF for multi-variate normal.
    # Hence converting to numpy and using scipy
    target = target.to('cpu').detach().numpy()
    pi = pi.to('cpu').detach()
    mu = mu.to('cpu').detach().numpy()
    cov = cov.to('cpu').detach().numpy()
    var = var.to('cpu').detach().numpy()

    for b in range(bs):
        for s in range(seq_len):
            cov[b, s] = cov[b, s] * np.eye(n_dim)
            cov[b, s] = cov[b, s] * var[b, s]

            probs[b, s] = multivariate_normal.cdf(target[b, s] + dist_threshold, mean=mu[b, s],
                                                        cov=cov[b, s]) - \
                                multivariate_normal.cdf(target[b, s] - dist_threshold, mean=mu[b, s],
                                                        cov=cov[b, s])
            x_mean = target[b, s] - mu[b, s]

            mahalanobis_dist[b, s] = np.sqrt(x_mean.T.dot(np.linalg.inv(cov[b, s])).dot(x_mean))


    # Calculate 1-sigma counts
    one_sigma_count = np.zeros_like(mahalanobis_dist)
    one_sigma_count[mahalanobis_dist < 1] = 1
    # one_sigma_count = torch.Tensor(one_sigma_count).to(self.device)

    # Calculate 2-sigma counts
    two_sigma_count = np.zeros_like(mahalanobis_dist)
    two_sigma_count[mahalanobis_dist < 2] = 1
    # two_sigma_count = torch.Tensor(two_sigma_count).to(self.device)

    # Calculate 3-sigma counts
    three_sigma_count = np.zeros_like(mahalanobis_dist)
    three_sigma_count[mahalanobis_dist < 3] = 1
    # three_sigma_count = torch.Tensor(three_sigma_count).to(self.device)

    # Distance threshold metric
    num_steps_conf_thresh = np.zeros_like(probs)
    num_steps_conf_thresh[probs >= 0.5] = 1

    return probs, one_sigma_count, two_sigma_count, three_sigma_count, num_steps_conf_thresh


def mdn_negative_log_likelihood(pi, mu, sigma, target):
    """ Use torch.logsumexp for more stable training 
    
    This is equivalent to the mdn_loss but computed in a numerically stable way

    """
    target = target.unsqueeze(1).expand_as(sigma)
    # target = target.unsqueeze(2).expand_as(sigma)
    neg_logprob = -torch.log(sigma) - (math.log(2 * math.pi) / 2) - \
        ((target - mu) / sigma)**2 / 2
    
    inner = torch.log(pi) + torch.sum(neg_logprob, 2) # Sum the log probabilities of (x, y) for each 2D Gaussian
    return -torch.logsumexp(inner, dim=1)

def mdn_negative_log_likelihood_loss(nn_output, target):
    """
    Compute the negative log likelihood loss for a MoG model.
    """
    pi, mu, sigma = nn_output
    return mdn_negative_log_likelihood(pi, mu, sigma, target)

# def mdn_negative_log_likelihood(pi, mu, sigma, target):
#     """ Use torch.logsumexp for more stable training

#     This is equivalent to the mdn_loss but computed in a numerically stable way

#     """
#     target = target.unsqueeze(1).expand_as(sigma)
#     neg_logprob = -torch.log(sigma) - (math.log(2 * math.pi) / 2) - \
#                     ((target - mu) / sigma) ** 2 / 2

#     # (B, num_heads, num_gaussians)
#     inner = torch.log(pi) + torch.sum(neg_logprob, 3)  # Sum the log probabilities of (x, y) for each 2D Gaussian
#     return -torch.logsumexp(inner, dim=2)

# def mdn_negative_log_likelihood_loss(nn_output, target):
#     """
#     Compute the negative log likelihood loss for a MoG model.
#     """
#     pi, mu, sigma = nn_output
#     return mdn_negative_log_likelihood(pi, mu, sigma, target)

# def compute_kl_loss(self):
#     """
#     Compute KL Divergence for ELBO for VRNN between the prior distribution and the
#     latent distribution from the encoder
#     :return:
#     """
#     self._kld_loss = 0
#     EPS = torch.finfo(torch.float).eps  # numerical logs

#     for t in range(len(self.all_prior_mean)):
#         # KL Divergence between two gaussians with std, taken from :
#         # https://github.com/emited/VariationalRecurrentNeuralNetwork/blob/0f23c87d11597ecf50ecbbf1dd37429861fd7aca/model.py#L175
#         kld_element = (2 * torch.log(self.all_prior_std[t] + EPS) - 2 * torch.log(self.all_enc_std[t] + EPS) +
#                         (self.all_enc_std[t].pow(2) + (self.all_enc_mean[t] - self.all_prior_mean[t]).pow(2)) /
#                         self.all_prior_std[t].pow(2) - 1)
#         self._kld_loss += 0.5 * torch.sum(kld_element)

#     self._kld_loss = self.kl_loss_wt * self._kld_loss

#     return self._kld_loss

# def get_kld_loss(self):
#     return self._kld_loss