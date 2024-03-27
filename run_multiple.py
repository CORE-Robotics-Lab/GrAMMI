""" File to run multiple datasets without having to create a config file for everything """

import os, sys
sys.path.append(os.getcwd())
from train_wandb import main_config_reg
from train_agent import main_config_agent
from train_mi_gaussian_posterior_categorical import main_config_mutual_info
import yaml


def run_reg(timestep, dataset_paths, base_log_directory):
    # this is base file
    config_file = "configs/baselines/regular_mixture.yaml"
    model_type = "regular_mixture"
    run(config_file, base_log_directory, dataset_paths, model_type, main_config_reg, timestep)

def run_agent(timestep, dataset_paths, base_log_directory):
    # this is base file
    config_file = "configs/baselines/agent_gnn.yaml"

    model_type = "agent_gnn"
    main_func = main_config_agent
    run(config_file, base_log_directory, dataset_paths, model_type, main_func, timestep)

def run_categorical(timestep, dataset_paths, base_log_directory):
    # this is base file
    config_file = "configs/baselines/categorical.yaml"

    model_type = "categorical"
    main_func = main_config_reg
    run(config_file, base_log_directory, dataset_paths, model_type, main_func, timestep)

def run_categorical_mutual_info(timestep, dataset_paths, base_log_directory):
    # this is base file
    config_file = "configs/baselines/categorical_mi.yaml"
    model_type = "categorical_mi"

    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    for (dataset_path, env_name, trained_folder) in dataset_paths:
        dataset_name = os.path.basename(dataset_path)
        log_directory = os.path.join(base_log_directory, str(timestep), dataset_name)
        config["datasets"]["train_path"] = os.path.join(dataset_path, "train")
        print(config["datasets"]["train_path"])
        config["datasets"]["test_path"] = os.path.join(dataset_path, "test")
        config["datasets"]["env"] = env_name
        config["datasets"]["step_length"] = timestep
        config["training"]["log_dir"] = os.path.join(log_directory, model_type)

        # get all folders in trained folder
        trained_folders = [f for f in os.listdir(trained_folder) if os.path.isdir(os.path.join(trained_folder, f))]

        # print(trained_folders)

        seeds = list(range(101, 104))
        # trained_paths = [os.path.join(trained_folder, f, "best.pth") for f in trained_folders]

        for seed, train_folder in zip(seeds, trained_folders):
            config_trained_path = os.path.join(trained_folder, train_folder, "config.yaml")
            train_file = os.path.join(trained_folder, train_folder, "best.pth")

            print(config_trained_path)

            with open(config_trained_path, 'r') as stream:
                trained_config = yaml.safe_load(stream)

            if trained_config["datasets"]["step_length"] != timestep:
                print("Skipping timestep: ", timestep)
                continue
            
            config["training"]["seed"] = seed
            config["model"]["load_pth"] = train_file

            print("Training model: ", model_type)
            print("Dataset: ", dataset_name)
            print(seed, train_file)
            # print(config)
            print("\n")
            main_config_mutual_info(config)


def run_categorical_mutual_info_agent(timestep, dataset_paths, base_log_directory):
    # this is base file
    config_file = "configs/baselines/agent_gnn_mi.yaml"
    model_type = "agent_gnn_mi"

    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    for (dataset_path, env_name, trained_folder) in dataset_paths:
        dataset_name = os.path.basename(dataset_path)
        log_directory = os.path.join(base_log_directory, str(timestep), dataset_name)
        config["datasets"]["train_path"] = os.path.join(dataset_path, "train")
        config["datasets"]["step_length"] = timestep
        print(config["datasets"]["train_path"])
        config["datasets"]["test_path"] = os.path.join(dataset_path, "test")
        config["datasets"]["env"] = env_name
        config["training"]["log_dir"] = os.path.join(log_directory, model_type)
        config["datasets"]["step_length"] = timestep

        # get all folders in trained folder
        trained_folders = [f for f in os.listdir(trained_folder) if os.path.isdir(os.path.join(trained_folder, f))]
        # print(trained_folders)

        seeds = list(range(101, 104))

        trained_folders = [f for f in os.listdir(trained_folder) if os.path.isdir(os.path.join(trained_folder, f))]

        for seed, train_folder in zip(seeds, trained_folders):
            config_trained_path = os.path.join(trained_folder, train_folder, "config.yaml")
            train_file = os.path.join(trained_folder, train_folder, "best.pth")

            print(config_trained_path)

            with open(config_trained_path, 'r') as stream:
                trained_config = yaml.safe_load(stream)

            if trained_config["datasets"]["step_length"] != timestep:
                print("Skipping timestep: ", timestep)
                continue
            config["training"]["seed"] = seed
            config["model"]["load_pth"] = train_file

            print("Training model: ", model_type)
            print("Dataset: ", dataset_name)
            print(seed, train_file)
            print("\n")
            main_config_mutual_info(config)

def run(config_file, base_log_directory, dataset_paths, model_type, main_func, timestep):
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    for (dataset_path, env_name) in dataset_paths:
        dataset_name = os.path.basename(dataset_path)
        log_directory = os.path.join(base_log_directory, str(timestep), dataset_name)
        config["datasets"]["train_path"] = os.path.join(dataset_path, "train")
        config["datasets"]["test_path"] = os.path.join(dataset_path, "test")
        config["datasets"]["step_length"] = timestep
        config["datasets"]["env"] = env_name
        config["training"]["log_dir"] = os.path.join(log_directory, model_type)

        for seed in range(100, 103):
            config["training"]["seed"] = seed
            print(config)
            print("\n")
            main_func(config)

if __name__ == "__main__":

    # location of saved datasets
    ## Prisoner:
    # 3-detect corresponds to low detection rate
    # 4-detect corresponds to medium detection rate
    # 7-detect corresponds to high detection rate

    ## Smuggler
    # paper_2_helo_40 corresponds to low detection rate
    # paper_3_helo_40 corresponds to high detection rate

    # This file is used to train multiple seeds for the models
    base_log_directory = "logs"

    # Run baseline models
    timestep = 0 # in the paper, we train for 0, 30, and 60 timesteps

    # The below for training non-gnn models
    ######################################################################################

    dataset_paths = [("grammi_datasets/prisoner_datasets/3_detect", "prisoner"),
                     ("grammi_datasets/prisoner_datasets/4_detect", "prisoner"),
                     ("grammi_datasets/prisoner_datasets/7_detect", "prisoner"),
                     ("grammi_datasets/smuggler_datasets/smuggler_paper_2_helo_40", "smuggler"),
                     ("grammi_datasets/smuggler_datasets/smuggler_paper_3_helo_40", "smuggler"),]

    # Uncomment below to train baseline models
    # run_reg(timestep, dataset_paths, base_log_directory)

    # train categorical (just \omega model) with no mutual information
    run_categorical(timestep, dataset_paths, base_log_directory)

    # #### Mutual Information models
    # fine tune the previous models to include mutual information
    dataset_paths = [("grammi_datasets/prisoner_datasets/3_detect", "prisoner", f"{base_log_directory}/{timestep}/3_detect/categorical"),
                     ("grammi_datasets/prisoner_datasets/4_detect", "prisoner",  f"{base_log_directory}/{timestep}/4_detect/categorical"),
                     ("grammi_datasets/prisoner_datasets/7_detect", "prisoner",  f"{base_log_directory}/{timestep}/7_detect/categorical"),
                     ("grammi_datasets/smuggler_datasets/smuggler_paper_2_helo_40", "smuggler", f"{base_log_directory}/{timestep}/smuggler_paper_2_helo_40/categorical"),
                     ("grammi_datasets/smuggler_datasets/smuggler_paper_3_helo_40", "smuggler", f"{base_log_directory}/{timestep}/smuggler_paper_3_helo_40/categorical")]

    run_categorical_mutual_info(timestep, dataset_paths, base_log_directory)
    
    # The below is for training gnn models
    ######################################################################################
    dataset_paths = [("grammi_datasets/prisoner_datasets/3_detect", "prisoner"),
                     ("grammi_datasets/prisoner_datasets/4_detect", "prisoner"),
                     ("grammi_datasets/prisoner_datasets/7_detect", "prisoner"),
                     ("grammi_datasets/smuggler_datasets/smuggler_paper_2_helo_40", "smuggler"),
                     ("grammi_datasets/smuggler_datasets/smuggler_paper_3_helo_40", "smuggler"),]

    # train categorical with agent GNN encoding
    run_agent(timestep, dataset_paths, base_log_directory)

    dataset_paths = [("grammi_datasets/prisoner_datasets/3_detect", "prisoner", f"{base_log_directory}/{timestep}/3_detect/agent_gnn"),
                     ("grammi_datasets/prisoner_datasets/4_detect", "prisoner",  f"{base_log_directory}/{timestep}/4_detect/agent_gnn"),
                     ("grammi_datasets/prisoner_datasets/7_detect", "prisoner",  f"{base_log_directory}/{timestep}/7_detect/agent_gnn"),
                     ("grammi_datasets/smuggler_datasets/smuggler_paper_2_helo_40", "smuggler", f"{base_log_directory}/{timestep}/smuggler_paper_2_helo_40/agent_gnn"),
                     ("grammi_datasets/smuggler_datasets/smuggler_paper_3_helo_40", "smuggler", f"{base_log_directory}/{timestep}/smuggler_paper_3_helo_40/agent_gnn")]

    # train with mutual information (fine-tuned from last model). 
    run_categorical_mutual_info_agent(timestep, dataset_paths, base_log_directory)