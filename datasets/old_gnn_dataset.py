""" Dataset for the gnn models """
import torch
import numpy as np
import math
import random

from simulator import PrisonerEnv
import os
from torch.utils.data import DataLoader

class GNNDataset(torch.utils.data.Dataset):
    def __init__(self, agent_obs, hideouts, timesteps, red_locs, dones, max_env_timesteps):
        self.agent_obs = agent_obs
        self.red_locs_np = red_locs
        self.timesteps = timesteps
        self.hideouts = hideouts
        self.max_env_timesteps = max_env_timesteps

        # ensure that we have the same number of timesteps 
        assert len(self.agent_obs) == len(self.red_locs_np)
    
    def __len__(self):
        return len(self.agent_obs)

    def __getitem__(self, idx):
        # Generates one sample of data
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.agent_obs[idx], self.timesteps[idx], self.hideouts[idx], self.red_locs_np[idx]

class LSTMGNNSequence(torch.utils.data.Dataset):
    
    def __init__(self, agent_obs, hideouts, timesteps, red_locs, dones, sequence_length, future_step):
        """ Dataset for GNN where all the cameras are the same
        :param future_step: Number of timesteps to predict into the future, 0 is used for filtering
            For example, future_step = 10 indicates returning the prisoner location from 10 steps into the future
        """
        
        self.agent_obs = agent_obs
        self.hideouts = hideouts
        self.timesteps = timesteps

        self.red_locs_np = red_locs
        self.dones = dones
        self.sequence_length = sequence_length

        self.red_locs_shape = self.red_locs_np[0].shape
        self.dones_shape = self.dones[0].shape
        self.future_step = future_step

        # These mark the end of each episode
        self.done_locations = np.where(self.dones == True)[0]
    
    def __len__(self):
        return len(self.agent_obs)


    def process_start_observations(self, np_array, idx, episode_start_idx):
        """ If we're indexing at the start of an episode, need to pad the start with zeros"""
        last_obs = np_array[idx]
        shape = (self.sequence_length - (idx - episode_start_idx + 1),) + last_obs.shape
        empty_sequences = np.zeros(shape)
        sequence = np_array[episode_start_idx:idx+1]
        sequence = np.concatenate((empty_sequences, sequence), axis=0)
        return sequence

    def __getitem__(self, idx):
        
        # First episode does not have reset marker
        if idx < self.done_locations[0] + 1:
            episode_start_idx = 0
        else:
            # Get index of the episode's start
            episode_start_idx = self.done_locations[np.where(self.done_locations <= idx - 1)[0][-1]] + 1
        assert idx >= episode_start_idx

        obs = (self.agent_obs, self.hideouts, self.timesteps)

        if idx - episode_start_idx >= self.sequence_length:
            # sample = [i[idx - self.sequence_length: idx] for i in obs]
            sample = (self.agent_obs[idx - self.sequence_length: idx], self.hideouts[idx], self.timesteps[idx])
        else:
            # sample = [self.process_start_observations(i, idx, episode_start_idx) for i in obs]
            sample = (self.process_start_observations(self.agent_obs, idx, episode_start_idx),
                        self.hideouts[idx], self.timesteps[idx])

        dones_sequence = self.dones[idx - self.sequence_length:idx]

        # get index of next reset
        next_done = self.done_locations[np.where(self.done_locations >= idx)[0][0]]
        target_red_loc_idx = idx + self.future_step
        if target_red_loc_idx > next_done:
            red_loc = self.red_locs_np[next_done]
        else:
            red_loc = self.red_locs_np[target_red_loc_idx]
        
        return sample, red_loc

if __name__ == "__main__":
    np_file = np.load("/nethome/sye40/PrisonerEscape/datasets/gnn_map_0_run_100_eps_0.1_norm.npz", allow_pickle=True)
    seq_len = 4
    future_step = 0
    dataset = LSTMGNNSequence(
        np_file["agent_observations"], 
        np_file["hideout_observations"], 
        np_file["timestep_observations"],
        np_file["red_locations"], 
        np_file["dones"], 
        seq_len, future_step)

    train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    for x, y in train_dataloader:
        print(x, y)
        print(x[0].shape)
        break