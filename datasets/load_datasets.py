import numpy as np
# from datasets.dataset_old import RedBlueDataset, RedBlueSequence, RedBlueSequenceOriginal, RedBlueSequenceSkip
# from datasets.multi_head_dataset_old import MultiHeadDataset
from torch.utils.data import DataLoader
from datasets.dataset import *
from datasets.contrastive_datasets import *
from datasets.padded_dataset import PaddedDataset, PaddedDatasetAgent, PaddedDatasetVAE, \
    pad_collate, pad_collate_agent_gnn, pad_collate_vae
from datasets.padded_dataset_particle import PaddedDatasetParticle
from datasets.mi_dataset import MIDataset, pad_collate_with_traj
from datasets.mi_dataset_hideouts import MIDatasetHideouts, pad_collate_with_traj_hideouts


def load_dataset(config, path_type):
    file_path = config[path_type]
    return load_dataset_with_config_and_file_path(config, file_path)


def load_datasets(config, batch_size):
    """
    Load datasets from config
    """
    train_dataset = load_dataset(config, "train_path")
    test_dataset = load_dataset(config, "test_path")

    if config["dataset_type"] == "mi_hideouts":
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                      collate_fn=pad_collate_with_traj_hideouts)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,
                                     collate_fn=pad_collate_with_traj_hideouts)
    elif config["dataset_type"] == "padded":
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                      collate_fn=pad_collate)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    elif config["dataset_type"] == "padded_vae":
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                      collate_fn=pad_collate_vae)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,
                                     collate_fn=pad_collate_vae)
    elif config["dataset_type"] == "padded_agent_gnn":
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                      collate_fn=pad_collate_agent_gnn)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,
                                     collate_fn=pad_collate_agent_gnn)
    elif config["dataset_type"] == "mi" and config["input_type"] == "blue":
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                      collate_fn=pad_collate_with_traj)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,
                                     collate_fn=pad_collate_with_traj)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader


def load_dataset_with_config_and_file_path(config, file_path):
    """ Utilize a configuration file but use a different model path """

    if config["dataset_type"] == "mi":
        return MIDataset(file_path, config["input_type"], config["output_type"])
    elif config["dataset_type"] == "mi_hideouts":
        return MIDatasetHideouts(file_path, config["input_type"], config["output_type"])

    seq_len = config["seq_len"]
    num_heads = config["num_heads"]
    step_length = config["step_length"]
    include_current = config["include_current"]
    view = config["view"]
    multi_head = config["multi_head"]

    autoregressive = False
    if "autoregressive" in config.keys():
        autoregressive = config["autoregressive"]

    # Check if multi_head and autoregressive are both set
    if multi_head and autoregressive:
        assert multi_head or autoregressive, "Either set decoder as multi-head or autoregressive"

    elif config["dataset_type"] == "gnn":
        dataset = GNNPrisonerDataset(file_path,
                                     seq_len,
                                     num_heads,
                                     step_length,
                                     include_current=include_current,
                                     multi_head=multi_head,
                                     one_hot=config["one_hot_agents"],
                                     timestep=config["timestep"],
                                     detected_location=config["detected_location"])

    elif config["dataset_type"] == "het_gnn":
        dataset = HeterogeneousGNNDataset(file_path,
                                          seq_len,
                                          num_heads,
                                          step_length,
                                          include_current=include_current,
                                          multi_head=multi_head,
                                          one_hot=config["one_hot_agents"],
                                          timestep=config["timestep"],
                                          detected_location=config["detected_location"])


    elif config["dataset_type"] == "contrastive_vector":
        dataset = ContrastiveVectorPrisonerDataset(file_path,
                                                   seq_len,
                                                   num_heads,
                                                   step_length,
                                                   include_current=include_current,
                                                   multi_head=multi_head,
                                                   view=view)

    elif config["dataset_type"] == "contrastive_gnn":
        num_negative_samples = 0
        if "num_negative_samples" in config:
            num_negative_samples = config["num_negative_samples"]
        if "contrastive_sample_type" in config:
            contrastive_sample_type = config["contrastive_sample_type"]
        else:
            contrastive_sample_type = "regular"

        dataset = ContrastiveGNNDataset(file_path,
                                        seq_len,
                                        num_heads,
                                        step_length,
                                        include_current=include_current,
                                        multi_head=multi_head,
                                        one_hot=config["one_hot_agents"],
                                        timestep=config["timestep"],
                                        detected_location=config["detected_location"],
                                        multiple_negative_samples=False,
                                        num_contrastive_samples=num_negative_samples,
                                        contrastive_sample_type=contrastive_sample_type)

    elif config["dataset_type"] == "simclr_gnn":
        modify_timestep_flag = False
        modify_camera_flag = False
        modify_dynamic_agent_flag = False

        if "modify_timestep_flag" in config.keys():
            modify_timestep_flag = config["modify_timestep_flag"]

        if "modify_camera_flag" in config.keys():
            modify_camera_flag = config["modify_camera_flag"]

        if "modify_dynamic_agent_flag" in config.keys():
            modify_dynamic_agent_flag = config["modify_dynamic_agent_flag"]

        dataset = SimCLRGNNDataset(file_path,
                                   seq_len,
                                   num_heads,
                                   step_length,
                                   include_current=include_current,
                                   multi_head=multi_head,
                                   one_hot=config["one_hot_agents"],
                                   timestep=config["timestep"],
                                   detected_location=config["detected_location"],
                                   modify_timestep_flag=modify_timestep_flag,
                                   modify_camera_flag=modify_camera_flag,
                                   modify_dynamic_agent_flag=modify_dynamic_agent_flag)

    elif config["dataset_type"] == "last_two_detects":
        dataset = LastTwoDetectsMLPDataset(file_path,
                                           seq_len,
                                           num_heads,
                                           step_length,
                                           include_current=include_current,
                                           multi_head=multi_head,
                                           view=view)

    elif config["dataset_type"] == "last_k_detects":
        if "produce_sequence_output" in config.keys():
            produce_sequence_output = config["produce_sequence_output"]
        else:
            produce_sequence_output = False

        if "env" in config:
            env_name = config["env"]
        else:
            env_name = "prisoner"  # By default use prisoner dataset

        dataset = LastKDetects(file_path,
                               seq_len,
                               num_heads,
                               step_length,
                               include_current=include_current,
                               multi_head=multi_head,
                               view=view,
                               produce_sequence_output=produce_sequence_output,
                               env_name=env_name)

    elif config["dataset_type"] == "padded":
        if config["env"] == "particle":
            dataset = PaddedDatasetParticle(file_path,
                                            num_heads,
                                            step_length,
                                            include_current=include_current,
                                            multi_head=multi_head)
        else:

            if "always_include_start" in config.keys():
                always_include_start = config["always_include_start"]
            else:
                always_include_start = False

            if "max_seq_len" in config.keys():
                max_seq_len = config["max_seq_len"]
            else:
                max_seq_len = 512

            dataset = PaddedDataset(file_path,
                                    num_heads,
                                    step_length,
                                    include_current=include_current,
                                    multi_head=multi_head,
                                    env_name=config["env"],
                                    max_seq_len=max_seq_len,
                                    always_include_start=always_include_start)
    elif config["dataset_type"] == "padded_agent_gnn":
        dataset = PaddedDatasetAgent(file_path,
                                     num_heads,
                                     step_length,
                                     include_current=include_current,
                                     multi_head=multi_head,
                                     env_name=config["env"],
                                     agent_length=config["agent_len"])

    elif config["dataset_type"] == "padded_vae":
        if "always_include_start" in config.keys():
            always_include_start = config["always_include_start"]
        else:
            always_include_start = False

        if "max_seq_len" in config.keys():
            max_seq_len = config["max_seq_len"]
        else:
            max_seq_len = 512

        dataset = PaddedDatasetVAE(file_path,
                                   num_heads,
                                   step_length,
                                   include_current=include_current,
                                   multi_head=multi_head,
                                   env_name=config["env"],
                                   max_seq_len=max_seq_len,
                                   always_include_start=always_include_start)

    else:
        dataset = VectorPrisonerDataset(file_path,
                                        seq_len,
                                        num_heads,
                                        step_length,
                                        include_current=include_current,
                                        multi_head=multi_head,
                                        view=view)
    return dataset
