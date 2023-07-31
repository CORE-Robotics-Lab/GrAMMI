
import sys, os
sys.path.append(os.getcwd())
import argparse
import tqdm
import shutil, yaml, csv
import torch
from torch.utils.data import DataLoader
from datasets.load_datasets import load_dataset, load_dataset_with_config_and_file_path
from evaluate.metrics import *
from models.get_stats import get_stats
from models.configure_model import configure_model
from datasets.padded_dataset import PaddedDataset, PaddedDatasetAgent, pad_collate, pad_collate_agent_gnn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

def evaluate_mixture_metrics(nn_output, targets):
    log_prob = - mdn_negative_log_likelihood(nn_output, targets)
    ade = average_displacement_error(nn_output, targets)
    dist_thresh_prob, one_sigma, two_sigma, three_sigma, num_steps = distance_threshold_metric(nn_output, targets)
    return log_prob, ade, dist_thresh_prob, one_sigma, two_sigma, three_sigma, num_steps


def evaluate_mode_metrics(nn_output, targets):
    log_prob = - mdn_negative_log_likelihood_with_single_mean(nn_output, targets)
    ade = average_displacement_error_with_single_mean(nn_output, targets)
    dist_thresh_prob, one_sigma, two_sigma, three_sigma, num_steps = distance_threshold_metric_with_single_mean(nn_output, targets)
    return log_prob, ade, dist_thresh_prob, one_sigma, two_sigma, three_sigma, num_steps


def evaluate_model(model_folder_path, test_path, mode):
    config_path = os.path.join(model_folder_path, "config.yaml")
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    batch_size = config["batch_size"]

    # Evaluate with validation set...
    # test_dataset = load_dataset(config["datasets"], "test_path")

    # Evaluate with test set...
    test_dataset = load_dataset_with_config_and_file_path(config["datasets"], test_path)
    print("Using test dataset: ", test_path)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # test_dataloader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle=True, collate_fn=pad_collate_with_traj)

    if type(test_dataset) == PaddedDatasetAgent:
        test_dataloader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle=True, collate_fn=pad_collate_agent_gnn)
    else:
        test_dataloader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle=True, collate_fn=pad_collate)

    num_heads = config["datasets"]["num_heads"]
    device = config["device"]
    # Load model
    model = configure_model(config).to(device)

    if timestep == 60 or timestep == 30:
        if "_mi" in model_folder_path:
            print("using models/2.pth")
            model.load_state_dict(torch.load(os.path.join(model_folder_path, "models/2.pth")))
        else:
            model.load_state_dict(torch.load(os.path.join(model_folder_path, "best.pth")))
    else:
        model.load_state_dict(torch.load(os.path.join(model_folder_path, "best.pth")))

    logprob_stack = []
    ade_stack = []
    dist_thresh_stack = []
    one_sigma_vals, two_sigma_vals, three_sigma_vals, num_steps_vals = [], [], [], []

    i = 0
    for tup in tqdm.tqdm(test_dataloader):
        if type(test_dataset) == PaddedDatasetAgent:
            x, x_lens, agent_obs, agent_lens, n_agents, y_test = tup
            x_test = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
            agent_packed = pack_padded_sequence(agent_obs, agent_lens, batch_first=True, enforce_sorted=False)

            x_test = x_test, agent_packed, n_agents
        else:
            x_test, x_lens, y_test = tup
            x_test = pack_padded_sequence(x_test, x_lens, batch_first=True, enforce_sorted=False)

        y_test = y_test.to(device)
        nn_output = model.forward(x_test)

        nn_output = [i.squeeze() for i in nn_output] # remove head dimension on some models

        # logprob, ade, dist_thresh, one_sigma, two_sigma, three_sigma, num_steps = model.get_stats(x_test, y_test)
        logprob, ade, dist_thresh, one_sigma, two_sigma, three_sigma, num_steps = get_stats(nn_output, y_test, mode)

        logprob = logprob.detach().cpu().unsqueeze(1).numpy()
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

    print("One sigma:")
    one_sigma_means = np.mean(one_sigma_vals, axis=0).tolist()
    print(",".join(map(str, one_sigma_means)))

    print("Three sigma:")
    three_sigma_means = np.mean(three_sigma_vals, axis=0).tolist()
    print(",".join(map(str, three_sigma_means)))

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
    for name in ["model_type", "logprob", "ade", "dist_thresh", "one_sigma", "two_sigma", "three_sigma", "num_steps"]:
        for timestep in range(0, total_timesteps + 5, 5):
            header.append(f"{name}_{timestep}")
    return header


def build_std_header(total_timesteps):
    header = []
    for name in ["model_type", "logprob", "ade", "dist_thresh", "one_sigma", "two_sigma", "three_sigma", "num_steps"]:
        for timestep in range(0, total_timesteps + 5, 5):
            header.append(f"{name}_{timestep}_mean")
            header.append(f"{name}_{timestep}_std")
    return header


def write_metric_logs(model_paths, writer, eval_mode):
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

def evaluate_IROS(log_directory, dataset_root_dir, writer, writer_std, writer_cumulative, mode, env):
    if env == "prisoner":
        dataset = ['3_detect', '4_detect', '7_detect']
    else:
        dataset = ['smuggler_paper_2_helo_40', 'smuggler_paper_3_helo_40']
        # dataset = ['smuggler_paper_2_helo_40']

    for data in dataset:
        log = os.path.join(log_directory, data)
        # list all model types in the log directory

        # all_model_types = ['regular_mixture', 'categorical', 'categorical_mi', 'agent_gnn', 'agent_gnn_mi']
        # all_model_types = ['categorical_mi', 'agent_gnn_mi']
        all_model_types = ['agent_gnn_mi']
        model_types = os.listdir(log)
        print(model_types)

        test_dataset_dir = os.path.join(dataset_root_dir, data + '/test_2')

        for model_type in all_model_types:
            if model_type not in model_types:
                continue
            model_type_path = os.path.join(log, model_type)
            model_paths = os.listdir(model_type_path)

            metrics_cumulative = []

            for model_path in model_paths:
                model_path = os.path.join(model_type_path, model_path)
                metrics = evaluate_model(model_path, test_dataset_dir, mode)

                metric_means = np.mean(metrics, axis=0)
                metric_std = np.std(metrics, axis=0)

                interleave = np.empty((metric_means.size + metric_std.size,), dtype=metric_means.dtype)
                interleave[0::2] = metric_means
                interleave[1::2] = metric_std

                writer.writerow([model_path] + metric_means.tolist())
                writer_std.writerow([model_path] + interleave.tolist())

                metrics_cumulative.append(metric_means)

            metrics_cumulative = np.stack(metrics_cumulative, axis=0)
            metrics_cumulative = np.mean(metrics_cumulative, axis=0)
            writer_cumulative.writerow([data, model_type] + metrics_cumulative.tolist())

def evaluate_logs(timestep, mode, env, name):

    if env == "prisoner":
        dataset_root_dir = '/grammi_datasets/prisoner_datasets/'
    elif env == "smuggler":
        dataset_root_dir = '/grammi_datasets/smuggler_datasets/'

    log_directory = f"logs/{timestep}"

    if not os.path.exists(f'evaluate/IROS/{env}_{timestep}'):
        os.makedirs(f'evaluate/IROS/{env}_{timestep}')

    if not os.path.exists(f'evaluate/IROS/{env}_{timestep}/{mode}'):
        os.makedirs(f'evaluate/IROS/{env}_{timestep}/{mode}')

    f = open(f'evaluate/IROS/{env}_{timestep}/{mode}/individual_{name}.csv', 'w')
    writer = csv.writer(f)

    f_cumulative = open(f'evaluate/IROS/{env}_{timestep}/{mode}/cumulative_{name}.csv', 'w')
    writer_cumulative = csv.writer(f_cumulative)

    f_std = open(f'evaluate/IROS/{env}_{timestep}/{mode}/std_{name}.csv', 'w')
    writer_std = csv.writer(f_std)

    header = ['model'] + build_header(0)
    header_std = ['model'] + build_std_header(0)
    writer.writerow(header)
    writer_std.writerow(header_std)
    writer_cumulative.writerow(header)

    evaluate_IROS(log_directory, dataset_root_dir, writer, writer_std, writer_cumulative, mode, env)

    f.close()
    f_std.close()
    f_cumulative.close()

if __name__ == '__main__':
    timestep = 0
    mode = "mixture"
    name = "categorical"
    # evaluate_logs(timestep, mode, "prisoner", name)
    evaluate_logs(timestep, mode, "smuggler", name)

    