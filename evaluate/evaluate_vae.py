"""
Different VAE Configurations experimented:
- Regular vs Autoregressive
- Continuous latent (to MoG) vs. Categorical latent (to single Gaussian)
"""
import sys, os
sys.path.append(os.getcwd())
import argparse
from models.configure_model import configure_model
import tqdm
import shutil, yaml, csv
from torch.utils.data import DataLoader
from datasets.load_datasets import load_datasets
from evaluate.metrics import *
from torch.nn.utils.rnn import pack_padded_sequence


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


def evaluate_model(model_folder_path, test_path=None):
    config_path = os.path.join(model_folder_path, "config.yaml")
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    batch_size = config["batch_size"]

    # Configure dataloaders

    # Evaluate with validation set...
    # test_dataset = load_datasets(config["datasets"], "test_path")

    # Evaluate with test set...
    # test_dataset = load_dataset_with_config_and_file_path(config["datasets"], test_path)
    # print("Using test dataset: ", test_path)
    test_dataloader, _ = load_datasets(config["datasets"], batch_size)

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
    for tup_test in test_dataloader:
        if config["datasets"]["dataset_type"] == "padded_vae":
            x, x_prev_red, x_lens, y_test = tup_test
            x_prev_red = torch.stack(x_prev_red).to(device).float()
        else:
            x, x_lens, y_test = tup_test  # Just padded (not autoregressive)

        x_test = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)

        y_test = y_test.to(device).float()

        if config["datasets"]["dataset_type"] == "padded_vae":
            logprob, ade, dist_thresh, one_sigma, two_sigma, three_sigma, num_steps = model.get_stats((x_test, x_prev_red), y_test)

        else:
            logprob, ade, dist_thresh, one_sigma, two_sigma, three_sigma, num_steps = model.get_stats(
                (x_test, y_test), y_test)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_from_file", help="set to true if model logs are loaded from evaluate.txt",
                        type=bool, default=False)
    parser.add_argument("--eval_mode", help="Set it as mixture or single to evaluate models wrt GMM or a single Gaussian",
                        default="mean")

    args = parser.parse_args()
    load_logs_from_file = args.load_from_file
    eval_mode = args.eval_mode

    name = "opponent_vae"

    f = open(f'evaluate/{name}_{eval_mode}.csv', 'w')
    writer = csv.writer(f)

    f_std = open(f'evaluate/{name}_{eval_mode}_std.csv', 'w')
    writer_std = csv.writer(f_std)

    header = ['model'] + build_header(0)
    header_std = ['model'] + build_std_header(0)
    writer.writerow(header)
    writer_std.writerow(header_std)

    if load_logs_from_file:
        # read paths from text file
        with open('evaluate/evaluate_opp_vae.txt', 'r') as f:
            model_paths = f.readlines()
        model_paths = [line.strip() for line in model_paths]

    else:
        # Load all logs from the data folder
        model_paths = []

        seeds = [1, 2, 3, 4, 5]

        f = open(f'evaluate/{name}.csv', 'w')
        writer = csv.writer(f)

        f_std = open(f'evaluate/{name}_std.csv', 'w')
        writer_std = csv.writer(f_std)

        header = ['model'] + build_header(60)
        header_std = ['model'] + build_std_header(60)
        writer.writerow(header)
        writer_std.writerow(header_std)

        datasets = ['3_detects/', '4_detects/', '7_detects/']
        models = ['contrastive_gnn/cpc/long_horizon/', 'lstm/', 'gnn/', 'vrnn_vector/']

        prisoner_test_path = '/data/prisoner_datasets/october_datasets/'
        all_test_dataset_paths = [prisoner_test_path + '3_detect_october_50/',
                                  prisoner_test_path + '4_detect_october_50/',
                                  prisoner_test_path + '7_detect_october_50/']

        for idx, dataset in enumerate(datasets):
            test_data_path = all_test_dataset_paths[idx]
            print(f"loading logs for dataset: {dataset}")
            print('########################################')
            for model in models:
                for seed in seeds:
                    logdir_path = f'/data/manisha/prisoner_logs/seed_{seed}/'
                    directories = os.listdir(logdir_path + dataset + model)
                    for dir in directories:
                        if dir == "20221023-1016":
                            continue
                        config_path = logdir_path + dataset + model + dir
                        model_paths.append(config_path)

    print(f"Evaluate the following models for {eval_mode} Gaussian")
    print(model_paths)

    write_metric_logs(model_paths, writer, eval_mode)



