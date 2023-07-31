"""
This file includes evaluation metrics for filtering, prediction and the particle filter
-- Additionally has a function to estimate mean, std from a mixture of gaussians
-- Metrics:
    -- KL Divergence, RMSE, NLL, delta_likelihood
    Binary value for predicting true fugitive's location (only for Gaussian Distribution -- Single or Mixture)
"""
import sys, os
sys.path.append(os.getcwd())
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.distributions import Normal
from scipy.stats import multivariate_normal
from simulator.utils import distance
from datasets.load_datasets import load_datasets
import random
from collections import defaultdict
from past_files.filtering.models.particle_filter import ParticleFilter
import cv2
from blue_policies.heuristic import BlueHeuristic
from past_files.filtering.utils.plot_fugitive_likelihood_from_gaussian import plot_gaussian_heatmap
from heatmap import generate_heatmap_img
from utils import save_video
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from utils import get_configs, set_seeds
from models.configure_model import configure_model
from simulator.prisoner_env_variations import initialize_prisoner_environment
import pickle


def calculate_params_from_mixture(mixture_probs, means, stds):
    """
    Calculate weighted average of means and stds from a mixture of gaussians
    :param mixture_probs (tensor (1, k)): Mixing co-effs of the GMM
    :param means (tensor (1, k, 2)): Means of each gaussian in the GMM
    :param stds (tensor (1, k, 2)): Stds of each gaussian in the GMM
    :return:
    """
    # assert mixture_probs.sum == 1
    # Mixture mean is simply weighted average of the means of the distribution (weighted by mixing co-effs)
    mixture_mean = torch.einsum('ij, ijk -> ik', mixture_probs, means)
    # mixture_mean = mixture_mean.unsqueeze(0)

    # Mixture variance is calculated from law of total variance and law of total expectations
    # See here: https://math.stackexchange.com/questions/195911/calculation-of-the-covariance-of-gaussian-mixtures
    mixture_cov = torch.zeros((2, 2)).to(device) # Covariance matrix of mixture gaussian will be (2, 2) since we are looking at 2D Gaussians
    var_means = torch.zeros((2, 2)).to(device)
    for i, p in enumerate(mixture_probs[0]):
        mixture_cov += p * torch.eye(2).to(device) * stds[:, i]**2
        diff_means = means[:, i] - mixture_mean
        var_means += p * torch.einsum('ij, ik -> jk', diff_means, diff_means)

    mixture_cov = mixture_cov + var_means
    mixture_std = torch.sqrt(torch.tensor([mixture_cov[0][0], mixture_cov[1][1]])).to(device)
    mixture_std = mixture_std.unsqueeze(0)

    return mixture_mean, mixture_std


def get_ground_truth_likelihood(mean, logstd, true_location, dist_threshold=0.05):
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
    n_dims = 2  # Location: (x,y)
    var = np.exp(logstd) ** 2
    cov = np.eye(n_dims) * var

    # CDF calculates from -infty to the upper limit.
    # Therefore subtracting the lower limit to calculate the cdf between lower limit to upper limit instead of -infty to upper limit
    prob = multivariate_normal.cdf(true_location + dist_threshold, mean=mean, cov=cov) - multivariate_normal.cdf(
        true_location - dist_threshold, mean=mean, cov=cov)
    return prob


def binary_prob_metric(mean, logstd, true_location, threshold=0.5):
    """
    Provides a binary score based on the probability distribution output from the filtering model.
    returns 1 if P(true_location) >= threshold else returns 0

    :param mean: (np.array) Mean of the predicted distribution from the filtering module
    :param logstd: (np.array) Logstd of the predicted distribution from the filtering module
    :param true_location: (np.array) Ground truth location of the fugitive at the current timestep
    :param threshold: probability threshold
    :return:
        binary value based on the probability threshold
    """
    true_prob = get_ground_truth_likelihood(mean, logstd, true_location)
    # print(true_prob)
    if true_prob >= threshold:
        return 1
    else:
        return 0


def nll(mean, std, true_location):
    mean = torch.from_numpy(mean).float()
    std = torch.from_numpy(std).float()
    true_location = torch.from_numpy(true_location).float()
    distribution = Normal(mean, std)
    logprob = distribution.log_prob(true_location)
    res = -logprob.sum().item()
    if res > 10:
        print(mean, std, true_location, res)
    return res


def rmse_from_mode(mean, true_location, scale=1):
    """
    Calculates the Root Mean Squared Error between the predicted mean (or mode) from the filtering
    module and the ground truth location of the fugitive.
    :param mean: (np.array) Mean of the predicted distribution from the filtering module (0-1)
    :param true_location: (np.array) Normalized Ground truth location of the fugitive at the current timestep (0-1)
    :param scale: At what scale the MSE should be computed. By default, the value is 1 which means that we will
                  compute the MSE in the value range 0-1. If we set to the env dim, then we calculate the MSE w.r.t. the
                  actual positions on the map
    :return:
        mse_error: Mean squared Difference between the predicted mean (or mode) from the filtering
                   module and the ground truth location of the fugitive.
    """

    mean = mean * scale
    true_location = true_location * scale

    mse_error = np.linalg.norm(mean - true_location)
    return mse_error


def kl_divergence(predicted_mean, predicted_logstd, true_location):
    """
    Calculates the KL Divergence between the predicted distribution from the filtering location
    and the true distribution (centered on the true location)

    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| +
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )

         where m0, S0 are the mean and covariance of the true distribution and
               m1, S1 are the mean and covariance from the predicted distribution

    Code Reference: https://stackoverflow.com/questions/44549369/kullback-leibler-divergence-from-gaussian-pm-pv-to-gaussian-qm-qv

    :param predicted_mean: (np.array) Mean of the predicted distribution from the filtering module
    :param predicted_logstd: (np.array) Logstd of the predicted distribution from the filtering module
    :param true_location: (np.array) Ground truth location of the fugitive at the current timestep
    :return:
        KL Divergence between the two distributions
    """
    n_dims = 2  # x, y dimension for prisoner location
    predicted_var = np.exp(predicted_logstd) ** 2
    predicted_cov = np.eye(n_dims) * predicted_var

    true_mean = true_location
    true_cov = np.eye(n_dims) * 0.001  # Assume very small covariance for the true distribution

    # Calculate KL Divergence
    inverse_pred_cov = np.linalg.inv(predicted_cov)
    diff = predicted_mean - true_mean

    #  KL Divergence has three terms
    trace_term = np.trace(inverse_pred_cov @ true_cov)
    determinant_term = np.log(
        np.linalg.det(predicted_cov) / np.linalg.det(true_cov))  # np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ inverse_pred_cov @ diff  # np.sum( (diff*diff) * iS1, axis=1)

    KL_div = 0.5 * (trace_term + determinant_term + quad_term - n_dims)
    return KL_div


def get_ground_truth_likelihood_particle_filter(all_particles, true_location, dist_threshold=0.05, top_particles=100):
    dist_threshold = 2428 * dist_threshold
    # first get top weighted particles
    particle_with_weight = []
    for particle in all_particles:
        particle_with_weight.append((particle.weight, particle))
    particle_with_weight_sorted = sorted(particle_with_weight, key=lambda x: x[0], reverse=True)

    total_weights = 0
    closer_than_dist_threshold_weights = 0
    for weight, particle in particle_with_weight_sorted[:top_particles]:
        total_weights += weight
        if distance(true_location, particle.location) < dist_threshold:
            closer_than_dist_threshold_weights += weight
    return closer_than_dist_threshold_weights / total_weights


def binary_prob_metric_particle_filter(all_particles, true_location, dist_threshold=0.05, top_particles=100,
                                       threshold=0.5):
    true_prob = get_ground_truth_likelihood_particle_filter(all_particles, true_location, dist_threshold, top_particles)
    # print("threshold_prob", true_prob)
    if true_prob >= threshold:
        return 1
    else:
        return 0


def rmse_particle_filter(all_particles, true_location, top_particles=100):
    particle_with_weight = []
    for particle in all_particles:
        particle_with_weight.append((particle.weight, particle))
    particle_with_weight_sorted = sorted(particle_with_weight, key=lambda x: x[0], reverse=True)

    total_weights = 0
    weighted_average_location = np.array([0.0, 0.0])
    for weight, particle in particle_with_weight_sorted[:top_particles]:
        total_weights += weight
        weighted_average_location += particle.location * weight
    weighted_average_location /= total_weights
    # print("mean location", weighted_average_location)
    # print("true location", true_location)
    distance_between = distance(weighted_average_location, true_location)
    # print("distance", distance_between)
    return distance_between


def kl_divergence_particle_filter(all_particles, true_location, top_particles=100):
    particle_with_weight = []
    for particle in all_particles:
        particle_with_weight.append((particle.weight, particle))
    particle_with_weight_sorted = sorted(particle_with_weight, key=lambda x: x[0], reverse=True)

    total_weights = 0
    weighted_average_location = np.array([0.0, 0.0])
    for weight, particle in particle_with_weight_sorted[:top_particles]:
        total_weights += weight
        weighted_average_location += particle.location * weight
    weighted_average_location /= total_weights

    total_weights = 0
    weighted_average_location_std = np.array([0.0, 0.0])
    for weight, particle in particle_with_weight_sorted[:top_particles]:
        total_weights += weight
        weighted_average_location_std += (particle.location - weighted_average_location) ** 2 * weight
    weighted_average_location_std /= total_weights
    # # TODO: Manisha check this... My neural net typically outputs low logstd of -4.5 --> 0.011 std --> 0.011 * 2428 = 27
    weighted_average_location_std = np.maximum(np.sqrt(weighted_average_location_std),
                                               np.ones_like(weighted_average_location_std) * 30)
    # weighted_average_location_std = np.sqrt(weighted_average_location_std) / 2428
    # print("mean location", weighted_average_location, "std location", weighted_average_location_std)
    # print("true location", true_location)
    return kl_divergence(weighted_average_location, np.log(weighted_average_location_std), true_location)


def nll_particle_filter(all_particles, true_location, normalization_constant=2428, top_particles=100):
    """ calculate the negative log likelihood of the particle filter output assuming a Gaussian centered
        around the particle's weighted mean and std

        Normalization constant used to normalize the locations onto 2428 grid

    """
    particle_with_weight = []
    for particle in all_particles:
        particle_with_weight.append((particle.weight, particle))
    particle_with_weight_sorted = sorted(particle_with_weight, key=lambda x: x[0], reverse=True)

    total_weights = 0
    weighted_average_location = np.array([0.0, 0.0])
    for weight, particle in particle_with_weight_sorted[:top_particles]:
        total_weights += weight
        weighted_average_location += particle.location / normalization_constant * weight
    weighted_average_location /= total_weights

    total_weights = 0
    weighted_average_location_std = np.array([0.0, 0.0])
    for weight, particle in particle_with_weight_sorted[:top_particles]:
        total_weights += weight
        weighted_average_location_std += (
                                                     particle.location / normalization_constant - weighted_average_location) ** 2 * weight
    weighted_average_location_std /= total_weights
    # # TODO: Manisha check this... My neural net typically outputs low logstd of -4.5 --> 0.011 std --> 0.011 * 2428 = 27
    # weighted_average_location_std = np.maximum(np.sqrt(weighted_average_location_std), np.ones_like(weighted_average_location_std)*30)
    weighted_average_location_std = np.sqrt(weighted_average_location_std)
    return nll(weighted_average_location, weighted_average_location_std, true_location / normalization_constant)


def generate_delta_plot(avg_ground_truth_likelihoods, pf_ground_truth_likelihoods, deltas):
    """
    Generate the plot for comparing the probability of ground truth with respect to varying
    distance thresholds (delta)
    :param avg_ground_truth_likelihoods: (np array of shape: n_rollouts x n_deltas) Ground truth likelihoods averaged across all timesteps for each delta
    :param deltas: Delta values for which we are computing the ground truth likelihoods
    :return:
        None. Saves a matplotlib figure
    """
    fig, ax = plt.subplots()
    likelihood_means = np.mean(avg_ground_truth_likelihoods, axis=0)
    likelihood_stds = np.std(avg_ground_truth_likelihoods, axis=0)

    pf_likelihood_means = np.mean(pf_ground_truth_likelihoods, axis=0)
    pf_likelihood_stds = np.std(pf_ground_truth_likelihoods, axis=0)

    plt.plot(deltas, likelihood_means, c="tab:blue", label='filtering')
    plt.fill_between(deltas, likelihood_means - likelihood_stds,
                     likelihood_means + likelihood_stds, alpha=0.3, color="tab:blue")

    plt.plot(deltas, pf_likelihood_means, c="tab:orange", label='particle_filter')
    plt.fill_between(deltas, pf_likelihood_means - pf_likelihood_stds,
                     pf_likelihood_means + pf_likelihood_stds, alpha=0.3, color="tab:orange")
    ax.legend()
    ax.set_ylabel(r'$P(Likelihood \leq \delta$)')
    ax.set_xlabel(r'$\delta$')
    plt.savefig('tmp/delta_all.png')


def plot_for_time_between_detections(all_metrics, pf_metrics, metric, num_rollouts):
    # Plot across all rollouts
    x, y, pf_y = [], [], []
    assert metric in ['RMSE', 'KL', 'binary_count']
    for k in range(num_rollouts):
        x.extend(all_metrics[k]['time_between_detections'])
        y.extend(all_metrics[k]['average_{}_between_detections'.format(metric)])
        pf_y.extend(pf_metrics[k]['average_{}_between_detections'.format(metric)])

    fig, ax = plt.subplots()
    ax.scatter(x, y, alpha=0.5, color='tab:blue', label='filtering')
    ax.scatter(x, pf_y, alpha=0.5, color='tab:orange', label='particle_filter')
    ax.legend()
    plt.xlabel('time between detections across rollouts')
    plt.ylabel("Average {} between detections".format(metric))
    plt.title('LSTM - Current Timestep')
    plt.savefig('tmp/{}_time_bw_detections_std.png'.format(metric))


def plot_metrics(all_metrics, pf_metrics, metric, num_rollouts):
    """
    Plot a specific metric (rmse, Kl, etc.) mean and std across multiple rollouts
    :param all_metrics: dictionary of dictionaries indexed by rollout id and the metric corresponding to that rollout
    :param metric: (str) 'KL', 'RMSE', 'binary_count'
    :param num_rollouts: Number of rollouts in the all_metrics dictionary whose metrics have been evaluated
    :return:
        None
    """
    metric_means, pf_metric_means = [], []
    metric_std, pf_metric_std = [], []
    assert metric in all_metrics[0].keys()
    plot_x1, plot_x2, plot_x3 = [], [], []
    for k in range(num_rollouts):
        metric_means.append(np.mean(all_metrics[k][metric]))
        metric_std.append(np.std(all_metrics[k][metric]))
        pf_metric_means.append(np.mean(pf_metrics[k][metric]))
        pf_metric_std.append(np.std(pf_metrics[k][metric]))
        plot_x1.append(all_metrics[k]['detection_ratio'])
        plot_x2.append(all_metrics[k]['detected_timesteps_std'])
        # plot_x3.append(all_metrics[k]['time_between_detections'])

    # Plot with respect to proportion of timesteps detected
    fig, ax = plt.subplots()
    ax.scatter(plot_x1, metric_means, alpha=0.5, c="tab:blue", label='filtering')
    ax.scatter(plot_x1, pf_metric_means, alpha=0.5, c="tab:orange", label='particle_filter')

    if metric == "binary_count_ratio":
        ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    ax.legend()
    plt.xlabel('Ratio of timesteps detected across rollouts')
    plt.ylabel(metric)
    plt.title('LSTM - Current Timestep')
    plt.savefig('tmp/{}_detection_ratio.png'.format(metric))

    # Plot with respect to proportion of timesteps detected
    fig, ax = plt.subplots()
    ax.scatter(plot_x2, metric_means, alpha=0.5, c="tab:blue", label='filtering')
    ax.scatter(plot_x2, pf_metric_means, alpha=0.5, c="tab:orange", label='particle_filter')
    ax.legend()
    plt.xlabel('Std: detected timesteps across rollouts')
    plt.ylabel(metric)
    plt.title('LSTM - Current Timestep')
    plt.savefig('tmp/{}_detected_timestep_std.png'.format(metric))

    # # Plot with respect to proportion of timesteps detected
    # fig, ax = plt.subplots()
    # ax.scatter(plot_x3, metric_means, alpha=0.5)
    # plt.xlabel('time between detections across rollouts')
    # plt.ylabel(metric)
    # plt.title('LSTM - Current Timestep')
    # plt.savefig('tmp/{}_time_bw_detections_std.png'.format(metric))


if __name__ == '__main__':
    config, config_path = get_configs()
    device = config["device"]

    model = configure_model(config["model"], config["datasets"]["num_heads"]).to(device)
    policy_path = "/nethome/mnatarajan30/codes/PrisonerEscape/logs/gnn/filtering/20220613-1821/best.pth"
    policy = torch.load(policy_path, map_location=device)
    model.load_state_dict(policy)
    model.eval()

    seed = 0
    # env = initialize_prisoner_environment(variation=0,
    #                                       epsilon=0.1,
    #                                       observation_step_type="Blue",
    #                                       seed=seed)
    set_seeds(seed)

    # blue_heuristic = BlueHeuristic(env, debug=False)
    # blue_heuristic.reset()
    # blue_heuristic.init_behavior()
    #
    # env.reset()

    num_rollouts = 20

    # TODO: Write a wrapper for getting gnn based observations from the env for eval

    # For now I'll use dummy inputs to the neural network
    # Configure dataloaders
    batch_size = 1
    train_dataloader, test_dataloader = load_datasets(config["datasets"], batch_size)
    for x_train, y_train in train_dataloader:
        x = [x_train[0], x_train[1], x_train[2]]
        pi, mu, sigma = model(x)
        print(pi.shape, mu.shape, sigma.shape)
        mixture_mean, mixture_std = calculate_params_from_mixture(pi, mu, sigma)
        break