import numpy as np
import os
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
import copy

class MIDatasetHideouts(torch.utils.data.Dataset):
    """ We return the which demonstration it came from for mutual information maximization - currently just test with red state as input
    
    Add option for red actions as input to the model

    """
    def __init__(self, folder_path, input_type="blue", output_type="red_state"):
        self.dones = []
        self.red_locs = []
        self.process_first_graph = True
        self.red_locs_per_traj = []
        self._load_data(folder_path)
        self.dones_shape = self.dones[0].shape

        # These mark the end of each episode
        self.done_locations = np.where(self.dones == True)[0]
        
        self.input_type = input_type
        self.output_type = output_type

    def _load_data(self, folder_path):
        np_files = []
        for file_name in sorted(os.listdir(folder_path)):
            np_file = np.load(os.path.join(folder_path, file_name), allow_pickle=True)
            # print(np_file)
            # self.max_agent_size = max(self.max_agent_size, np.squeeze(np_file["agent_observations"]).shape[1])
            np_files.append(np_file)

        for i, np_file in enumerate(np_files):
            self._load_graph(np_file, i)

    def _load_graph(self, file, i):
        agent_obs = file["agent_observations"]
        agent_obs = np.squeeze(agent_obs)
        num_agents = agent_obs.shape[1]

        detected_location = file["detected_locations"]
        detected_location[0] = copy.deepcopy(file["red_locations"][0]) / 2428

        final_location = file["red_locations"][-1] / 2428

        timesteps = file["timestep_observations"]

        # Modify detected locations such that there is always a detection at the first timestep
        # This is so that we can use the first timestep as the input to the LSTM
        # detected_location[0] = copy.deepcopy(file["red_locations"][0]) / 2428

        self.red_locs_per_traj.append(file["red_locations"]/2428)
        # print(file["red_locations"].shape)

        last_timestep = file["timestep_observations"][-1]
        # repeat this array for entire episode
        last_timestep = np.array([last_timestep] * agent_obs.shape[0])

        if self.process_first_graph:
            self.process_first_graph = False
            self.num_agents = [num_agents] * agent_obs.shape[0]
            self.hideouts = file["hideout_observations"]
            self.timesteps = timesteps
            self.dones = file["dones"]
            self.red_locs = file["red_locations"]/2428
            self.last_k_fugitive_detections = file["last_k_fugitive_detections"]
            self.trajectory_num = np.array([i] * agent_obs.shape[0]) # track which demonstration it came from
            self.final_location = np.array([final_location] * agent_obs.shape[0])
            self.last_timestep = last_timestep
            # blue input
            self.detected_location = detected_location # for visualization
            self.detected_locations = self._preprocess_detections(detected_location, timesteps)
        else:
            self.hideouts = np.append(self.hideouts, file["hideout_observations"], 0)
            self.red_locs = np.append(self.red_locs, file["red_locations"]/2428, 0)
            self.timesteps = np.append(self.timesteps, timesteps)
            self.dones = np.append(self.dones, file["dones"])
            self.trajectory_num = np.append(self.trajectory_num, np.array([i] * agent_obs.shape[0]))
            self.final_location = np.append(self.final_location, np.array([final_location] * agent_obs.shape[0]), 0)
            self.detected_locations = self.detected_locations + self._preprocess_detections(detected_location, timesteps)
            self.last_timestep = np.append(self.last_timestep, last_timestep, 0)
            self.detected_location = np.append(self.detected_location, detected_location, 0) # for visualization


    def _preprocess_detections(self, detected_locs, timestamps):
        """ Given a numpy array of [T x 2] where if there is a detection, the value is (x, y) and if there is not, the value is (-1, -1)
        
        For each row in the array, return all previous detections before that row

        Also need to add the time difference between each step so we return a [dt, x, y] for each detection
        """
        processed_detections = []

        detected_locs = np.concatenate((detected_locs, timestamps), axis=1)
        for i in range(detected_locs.shape[0]):
            curr_detections = copy.deepcopy(detected_locs[:i+1])
            curr_detections = curr_detections[curr_detections[:, 0] != -1]
            curr_detections[:, 2] = detected_locs[i, 2] - curr_detections[:, 2]
            processed_detections.append(curr_detections)
        return processed_detections

    def _produce_input_red(self, idx):
        red_loc = self.red_locs[idx]
        timesteps = self.timesteps[idx]
        inputs = torch.from_numpy(np.concatenate((red_loc, [timesteps])))
        return inputs

    def _produce_input_blue(self, idx):
        return torch.from_numpy(self.detected_locations[idx]).float()

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        if self.input_type == "red":
            x = self._produce_input_red(idx)
        elif self.input_type == "blue":
            x = self._produce_input_blue(idx)
        else:
            raise ValueError("Invalid input type")

        final_location = self.final_location[idx]
        t = self.timesteps[idx:idx+1]
        final_t = self.last_timestep[idx]

        cat = np.concatenate((final_location, t, final_t))

        y = self._produce_output(idx)
        return x, cat, self.trajectory_num[idx], y

    def _produce_output(self, idx):
        # check if we are at the end of an episode
        if self.output_type == "red_state":
            return self._produce_output_red_state(idx)
        else:
            return self._produce_output_red_action(idx)
    
    def _produce_output_red_state(self, idx):
        # check if we are at the end of an episode
        if idx in self.done_locations:
            return torch.from_numpy(self.red_locs[idx]).float()
        else:
            return torch.from_numpy(self.red_locs[idx+1]).float()

    def _produce_output_red_action(self, idx):
        if idx in self.done_locations:
            theta = 0
            vel = 0 
        else:
            current_loc = self.red_locs[idx]
            next_loc = self.red_locs[idx+1]

            vel = np.linalg.norm(next_loc*2428 - current_loc*2428) / 15
            theta = np.arctan2(next_loc[1] - current_loc[1], next_loc[0] - current_loc[0])
        
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        return torch.from_numpy(np.array([sin_theta, cos_theta])).float()

def pad_collate_with_traj_hideouts(batch):
    (xx, x_final_location, trajectory_nums, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    x_final_location = torch.tensor(np.stack(x_final_location, axis=0))

    yy = torch.stack(yy, axis=0)
    return xx_pad, x_lens, x_final_location, trajectory_nums, yy

if __name__ == "__main__":
    # data_path = "/data/manisha/datasets/fixed_policy/corner_start_fixed_hideouts/test"
    data_path = "/data/prisoner_datasets/october_datasets/3_detect/train"
    dataset = MIDatasetHideouts(data_path, input_type = "blue", output_type = "red_state")

    # print(dataset[0])

    dataloader = DataLoader(dataset = dataset, batch_size = 100, shuffle=True, collate_fn=pad_collate_with_traj_hideouts)

    for tup in dataloader:
        x, x_lens, cat, trajectory_nums, y = tup

        print(x, cat)
        break
    # print(dataset[100])

    # from matplotlib import pyplot as plt
    # # from sklearn.datasets.samples_generator import make_blobs
    # # from sklearn.cluster import KMeans
    # # embeds = torch.stack([trajectory_specific_embeddings[idx] for idx in range(100)]).numpy()
    # # tmp_dist = embeds
    # # kmeans = KMeans(n_clusters=5, random_state=0).fit(tmp_dist)
    # colors = ['red', 'green', 'blue', 'orange']
    # for i, traj in enumerate(dataset.red_locs_per_traj[:20]):
    #     color = colors[i%4]
    #     plt.plot(traj[:, 0], traj[:, 1], c=color)
    #     plt.xlim(0,1)
    #     plt.ylim(0,1)
    #     plt.savefig('test.png')
