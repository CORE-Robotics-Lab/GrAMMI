import sys, os
sys.path.append(os.getcwd())
import numpy as np
import math
from matplotlib import pyplot as plt
import torch
from torch.distributions import Normal
from scipy.stats import multivariate_normal


# ------------------------- Metrics for evaluating with respect to the Gaussian Mixture ----------------------------- #
def mdn_negative_log_likelihood(nn_output, target):
    """
    Calculate log likelihood of the target only with respect to the Gaussian Mixture for all prediction horizons
    (Not the entire mixture)
    Use torch.logsumexp for more stable training
    This is equivalent to the mdn_loss but computed in a numerically stable way
    All our models output a mixture of bi-variate Gaussians for predicting the opponent's position (num_dim=2)
    :param nn_output: Tuple of (pi, mu, sigma)
                     pi: [batch_size, num_heads, num_mixtures] mixing coefficients of the GMM
                     mu: [batch_size, num_heads, num_mixtures, num_dim] mean of each gaussian in the GMM
                     sigma: [batch_size, num_heads, num_mixtures, num_dim] sigma of each gaussian in the GMM
                     (We assume that the covariance matrix is a diagonal matrix)
    :param target: [batch_size, num_heads] Ground truth location of the opponent
    :return: [batch_size, num_heads] negative log-likelihood for each prediction horizon t (=num_heads) for the opponent
            evaluated with respect to the predicted GMM at that horizon
    """
    pi, mu, sigma = nn_output
    target = target.unsqueeze(2).expand_as(sigma)

    # (batch_size, num_heads, num_mixtures, num_dim)
    neg_logprob = -torch.log(sigma) - (math.log(2 * math.pi) / 2) - \
                  ((target - mu) / sigma) ** 2 / 2

    # (batch_size, num_heads, num_mixtures)
    inner = torch.log(pi) + torch.sum(neg_logprob, 3)  # Sum the log probabilities of (x, y) for each 2D Gaussian

    return -torch.logsumexp(inner, dim=2)   # [batch_size, num_heads]


def distance_threshold_metric(nn_output, target, dist_threshold=0.05):
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
    device = nn_output[0].device
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
    one_sigma_count = torch.sum(pi * k, axis=-1).to('cpu').detach().numpy()

    # Calculate 2-sigma counts
    k = np.zeros_like(mahalanobis_dist)
    k[mahalanobis_dist < 2] = 1
    two_sigma_count = torch.sum(pi * k, axis=-1).to('cpu').detach().numpy()

    # Calculate 3-sigma counts
    k = np.zeros_like(mahalanobis_dist)
    k[mahalanobis_dist < 3] = 1
    three_sigma_count = torch.sum(pi * k, axis=-1).to('cpu').detach().numpy()

    # Distance threshold metric
    k = np.zeros_like(probs)
    k[probs >= 0.5] = 1
    num_steps_conf_thresh = torch.sum(pi * k, axis=-1).to('cpu').detach().numpy()

    return total_prob, one_sigma_count, two_sigma_count, three_sigma_count, num_steps_conf_thresh


def average_displacement_error(nn_output, target):
    pi, mu, sigma = nn_output
    num_mixtures = pi.shape[-1]

    target = target.unsqueeze(dim=2).repeat(1, 1, num_mixtures, 1)
    mse_error = torch.linalg.norm((mu - target), dim=-1)
    mse_error = torch.sum(pi * mse_error, axis=-1)  # Calculate weighted average of mse error

    return mse_error


# -------------------- Metrics for evaluating with respect to the Highest weight Gaussian in GMM --------------------- #
def mdn_negative_log_likelihood_with_single_mean(nn_output, target):
    """
    Calculate log likelihood of the target only with respect to the Gaussian with the highest mixing co-eff
    (Not the entire mixture)
    All our models output a mixture of bi-variate Gaussians for predicting the opponent's position (num_dim=2)
    :param nn_output: Tuple of (pi, mu, sigma)
                     pi: [batch_size, num_heads, num_mixtures] mixing coefficients of the GMM
                     mu: [batch_size, num_heads, num_mixtures, num_dim] mean of each gaussian in the GMM
                     sigma: [batch_size, num_heads, num_mixtures, num_dim] sigma of each gaussian in the GMM
                     (We assume that the covariance matrix is a diagonal matrix)
    :param target: [batch_size, num_heads] Ground truth location of the opponent
    :return: [batch_size, num_heads] negative log-likelihood for each prediction horizon t (=num_heads) for the opponent
            evaluated with respect to the predicted GMM at that horizon
    """
    device = nn_output[0].device
    pi, mu, sigma = nn_output
    # Get the means and sigmas of the gaussian with the highest mixing coeff
    idx = torch.argmax(pi, dim=-1, keepdim=True)
    mu_0 = torch.gather(mu[:, :, :, 0], 2, idx)
    mu_1 = torch.gather(mu[:, :, :, 1], 2, idx)
    mu = torch.cat((mu_0, mu_1), dim=-1)

    sigma_0 = torch.gather(sigma[:, :, :, 0], 2, idx)
    sigma_1 = torch.gather(sigma[:, :, :, 1], 2, idx)
    sigma = torch.cat((sigma_0, sigma_1), dim=-1)

    neg_logprob = -torch.log(sigma) - (math.log(2 * math.pi) / 2) - \
                  ((target - mu) / sigma) ** 2 / 2
    
    neg_logprob = -torch.sum(neg_logprob, 2)
    neg_logprob = torch.clamp(neg_logprob, min=-10, max=10)

    # bs, seq_len, n_dim = sigma.shape

    # var = torch.clamp(sigma ** 2, min=1e-5, max=1)

    # var = var.repeat(1, 1, n_dim)
    # var = var.reshape(bs, seq_len, n_dim, n_dim)

    # cov = torch.ones(bs, seq_len, n_dim, n_dim).to(device)

    # target = target.to('cpu').detach()
    # pi = pi.to('cpu').detach()
    # mu = mu.to('cpu').detach()
    # cov = cov.to('cpu').detach().numpy()
    # var = var.to('cpu').detach().numpy()

    # for b in range(bs):
    #     for s in range(seq_len):
    #         cov[b, s] = cov[b, s] * np.eye(n_dim)
    #         cov[b, s] = cov[b, s] * var[b, s]

    # cov = torch.Tensor(cov).to('cpu')
    # m = torch.distributions.multivariate_normal.MultivariateNormal(loc=mu, covariance_matrix=torch.Tensor(cov))
    # neg_logprob = -m.log_prob(target)

    return neg_logprob


def distance_threshold_metric_with_single_mean(nn_output, target, dist_threshold=0.05):
    device = nn_output[0].device
    pi, mu, sigma = nn_output
    bs, seq_len, num_mixtures, n_dim = sigma.shape

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