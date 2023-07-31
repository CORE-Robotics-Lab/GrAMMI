import sys, os

sys.path.append(os.getcwd())

import numpy as np
from models.configure_model import configure_model
import yaml
import os
import torch
import pandas as pd
from datasets.dataset import VectorPrisonerDataset, GNNPrisonerDataset
# from datasets.old_gnn_dataset import LSTMGNNSequence
from torch.utils.data import DataLoader
from datasets.load_datasets import load_dataset
from collections import defaultdict
from utils import get_configs, set_seeds
import tqdm
import csv


def evaluate_model(model_folder_path):
    config_path = os.path.join(model_folder_path, "config.yaml")
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    batch_size = config["batch_size"]

    # Configure dataloaders
    test_dataset = load_dataset(config["datasets"], "test_path")
    print("Using test dataset: ", config["datasets"]["test_path"])
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    num_heads = config["datasets"]["num_heads"]
    device = config["device"]
    # Load model
    model = configure_model(config).to(device)

    model.load_state_dict(torch.load(os.path.join(model_folder_path, "best.pth")))
    logprob_stack = []
    ade_stack = []
    dist_thresh_stack = []
    one_sigma_vals, two_sigma_vals, three_sigma_vals, num_steps_vals = [], [], [], []

    i = 0
    for x_test, y_test in tqdm.tqdm(test_dataloader):
        logprob, ade, dist_thresh, one_sigma, two_sigma, three_sigma, num_steps = model.get_stats(x_test, y_test)

        logprob = logprob.detach().cpu().numpy()
        logprob_stack.append(logprob)
        del logprob

        ade = ade.detach().cpu().numpy()
        ade_stack.append(ade)
        del ade

        dist_thresh = dist_thresh.detach().cpu().numpy()
        dist_thresh_stack.append(dist_thresh)
        del dist_thresh

        # one_sigma = one_sigma.detach().cpu().numpy()
        one_sigma_vals.append(one_sigma)
        del one_sigma

        # two_sigma = two_sigma.detach().cpu().numpy()
        two_sigma_vals.append(two_sigma)
        del two_sigma

        # three_sigma = three_sigma.detach().cpu().numpy()
        three_sigma_vals.append(three_sigma)
        del three_sigma

        # num_steps = num_steps.detach().cpu().numpy()
        num_steps_vals.append(num_steps)
        del num_steps
        i += 1

    # logprob_stack = torch.cat(logprob_stack, dim=0)
    logprob_stack = np.concatenate(logprob_stack, axis=0)
    ade_stack = np.concatenate(ade_stack, axis=0)
    dist_thresh_stack = np.concatenate(dist_thresh_stack, axis=0)
    one_sigma_vals = np.concatenate(one_sigma_vals, axis=0)
    two_sigma_vals = np.concatenate(two_sigma_vals, axis=0)
    three_sigma_vals = np.concatenate(three_sigma_vals, axis=0)
    num_steps_vals = np.concatenate(num_steps_vals, axis=0)

    metrics = np.concatenate((logprob_stack, ade_stack, dist_thresh_stack, one_sigma_vals, two_sigma_vals,
                              three_sigma_vals, num_steps_vals), axis=1)

    print("Log probs:")
    ll_means = np.mean(logprob_stack, axis=0).tolist()
    print(",".join(map(str, ll_means)))
    print("----------------------------------------------------------------------")
    print("ADE:")
    ade_means = np.mean(ade_stack, axis=0).tolist()
    print(",".join(map(str, ade_means)))
    print("----------------------------------------------------------------------")
    print("Dist thresh:")
    dist_thresh_means = np.mean(dist_thresh_stack, axis=0).tolist()
    print(",".join(map(str, dist_thresh_means)))
    print("----------------------------------------------------------------------")

    # d = defaultdict()
    # # d['steps'] = np.arange(i+1)
    # for i in range(num_heads+1):
    #     d['ll_timestep_{}'.format(i*5)] = logprob_stack[:, i]
    #     d['ade_timestep_{}'.format(i*5)] = ade_stack[:, i]
    #     d['dist_thresh_timestep_{}'.format(i*5)] = dist_thresh_stack[:, i]
    #     d['one_sigma_timestep_{}'.format(i*5)] = one_sigma_vals[:, i]
    #     d['two_sigma_timestep_{}'.format(i*5)] = two_sigma_vals[:, i]
    #     d['three_sigma_timestep_{}'.format(i*5)] = three_sigma_vals[:, i]
    # df = pd.DataFrame(d)
    #
    # # save to log directory
    # save_path = os.path.join(model_folder_path, 'eval_metrics.csv')
    # df.to_csv(save_path)

    return metrics


def build_header(total_timesteps):
    header = []
    for name in ["logprob", "ade", "dist_thresh", "one_sigma", "two_sigma", "three_sigma", "num_steps"]:
        for timestep in range(0, total_timesteps + 5, 5):
            header.append(f"{name}_{timestep}")
    return header


def build_std_header(total_timesteps):
    header = []
    for name in ["logprob", "ade", "dist_thresh", "one_sigma", "two_sigma", "three_sigma", "num_steps"]:
        for timestep in range(0, total_timesteps + 5, 5):
            header.append(f"{name}_{timestep}_mean")
            header.append(f"{name}_{timestep}_std")
    return header


if __name__ == "__main__":

    name = "simclr_retrained_vals_mean"

    f = open(f'evaluate/{name}.csv', 'w')
    writer = csv.writer(f)

    f_std = open(f'evaluate/{name}_std.csv', 'w')
    writer_std = csv.writer(f_std)

    header = ['model'] + build_header(60)
    header_std = ['model'] + build_std_header(60)
    writer.writerow(header)
    writer_std.writerow(header_std)

    # read paths from text file
    with open('evaluate/evaluate.txt', 'r') as f:
        model_paths = f.readlines()
    model_paths = [line.strip() for line in model_paths]
    print(model_paths)

    for model_path in model_paths:
        metrics = evaluate_model(model_path)
        metric_means = np.mean(metrics, axis=0)
        metric_std = np.std(metrics, axis=0)

        interleave = np.empty((metric_means.size + metric_std.size,), dtype=metric_means.dtype)
        interleave[0::2] = metric_means
        interleave[1::2] = metric_std

        writer.writerow([model_path] + metric_means.tolist())
        writer_std.writerow([model_path] + interleave.tolist())

    f.close()
    f_std.close()
