# import pandas as pd
from dataset import BasePrisonerDataset
import numpy as np
import os

class LastKDetectsMLPDataset(BasePrisonerDataset):
    """
    Store last 4 detections, along with the dx, dy (from last 2 detections), dt (since the most recent detection)
    """
    # ['blue_observations', 'red_observations', 'agent_observations', 'hideout_observations', 'timestep_observations', 'detected_locations', 'red_locations', 'dones', 'agent_dict', 'detect', 'last_k_fugitive_detections', 'blue_obs_dict', 'prediction_obs_dict']

    def __init__(self, folder_path, sequence_length, num_heads, step_length, include_current, multi_head):
        
        self.graphs = []
        self.max_agent_size = 0
        self.process_first_graph = True

        super().__init__(folder_path, sequence_length, num_heads, step_length, include_current, multi_head)


    def _load_data(self, folder_path):
        np_files = []
        for file_name in sorted(os.listdir(folder_path)):
            np_file = np.load(os.path.join(folder_path, file_name), allow_pickle=True)
            # print(np_file)
            self.max_agent_size = max(self.max_agent_size, np.squeeze(np_file["agent_observations"]).shape[1])
            np_files.append(np_file)

        for np_file in np_files:
            self._load_graph(np_file)

    def _load_graph(self, file):
        agent_obs = file["agent_observations"]
        agent_obs = np.squeeze(agent_obs)
        num_agents = agent_obs.shape[1]

        detected_location = file["detected_locations"]
        timesteps = file["timestep_observations"]

        if self.process_first_graph:
            self.process_first_graph = False
            self.num_agents = [num_agents] * agent_obs.shape[0]
            self.hideouts = file["hideout_observations"]
            self.timesteps = timesteps
            self.dones = file["dones"]
            self.red_locs = file["red_locations"]/2428
            self.detected_location = detected_location
            self.last_k_fugitive_detections = file["last_k_fugitive_detections"]
        else:
            self.hideouts = np.append(self.hideouts, file["hideout_observations"], 0)
            self.red_locs = np.append(self.red_locs, file["red_locations"]/2428, 0)
            self.timesteps = np.append(self.timesteps, timesteps)
            self.dones = np.append(self.dones, file["dones"])
            self.last_k_fugitive_detections = np.append(self.last_k_fugitive_detections, file["last_k_fugitive_detections"], 0)

    def _produce_input(self, idx):
        # First episode does not have reset marker
        if idx < self.done_locations[0] + 1:
            episode_start_idx = 0
        else:
            # Get index of the episode's start
            episode_start_idx = self.done_locations[np.where(self.done_locations <= idx - 1)[0][-1]] + 1
        assert idx >= episode_start_idx

        # Last k Detections
        last_k_fugitive = self.last_k_fugitive_detections[idx]

        # Current timestep
        timestep = self.timesteps[idx]

        return np.concatenate((last_k_fugitive.flatten(), np.array([timestep])))


class LastKDetectsLSTMDataset(BasePrisonerDataset):
    def __init__(self, folder_path, sequence_length, num_heads, step_length, include_current, multi_head):
        super().__init__(folder_path, sequence_length, num_heads, step_length, include_current, multi_head)

    def _load_data(self, folder_path):
        data = pd.read_csv(folder_path+"_data.csv")

        # Red observation currently includes latitude, longitude, time since start,
        # course, speed, distance
        self.blue_obs_np = None  # Currently we do not have blue observations
        # self.red_obs_np = data.iloc[:, 0:6].to_numpy()
        self.red_obs_np = data[["latitude-3", "longitude-3", "detection-3",
                                "latitude-2", "longitude-2", "detection-2",
                                "latitude-1", "longitude-1", "detection-1",
                                "curr_latitude", "curr_longitude", "detection"]].to_numpy()

        self.dones = data["done"].to_numpy()
        # Get the next lat/long as the red location
        self.red_locs = data[["next_latitude", "next_longitude"]].to_numpy()
        self.detected_location = data[["observed_latitude", "observed_longitude"]].to_numpy()
        self.time_since_last_detection = data[["detection"]].to_numpy()
        self.dx = data[["vel_x"]].to_numpy()
        self.dy = data[["vel_y"]].to_numpy()

    def _produce_input(self, idx):
        # First episode does not have reset marker
        # if idx < self.done_locations[0] + 1:
        #     episode_start_idx = 0
        # else:
        #     # Get index of the episode's start
        #     episode_start_idx = self.done_locations[np.where(self.done_locations <= idx - 1)[0][-1]] + 1
        # assert idx >= episode_start_idx
        #
        # if idx - episode_start_idx >= self.sequence_length:
        #     red_sequence = self.red_obs_np[idx - self.sequence_length:idx]
        # else:
        #     red_sequence = self._process_start_observations(self.red_obs_np, idx, episode_start_idx)
        red_sequence = self.red_obs_np[idx]
        red_sequence = red_sequence.reshape(4, -1)  # Load last 4 detections

        return red_sequence

if __name__ == "__main__":
    # path = "/nethome/sye40/PrisonerEscape/datasets/test_same/gnn_map_0_run_100_eps_0.1_norm"
    path = "/data/manisha/datasets/fixed_policy/corner_start_fixed_hideouts/test"
    seq_len = 16
    step_length = 0
    num_heads = 1
    
    dataset = LastKDetectsMLPDataset(path, 
            seq_len,
            num_heads, 
            step_length,
            include_current=True, 
            multi_head = False)

    # print(np.where(dataset.dones == True))

    # 316, 592, 889
    print(dataset[610][0].shape)
    # maybe from last 2 detections, can predict velocity and evenly spaced points between now and then?