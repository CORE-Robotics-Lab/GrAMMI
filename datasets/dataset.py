""" Dataset for the gnn models """
import numpy as np
import os
from torch.utils.data import DataLoader
import torch
import copy


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


class BasePrisonerDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, sequence_length, num_heads, step_length, include_current, multi_head):
        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.step_length = step_length
        self.include_current = include_current
        self.multi_head = multi_head
        if self.multi_head:
            self.future_step = step_length * num_heads
        else:
            if self.num_heads > 1:
                self.future_step = step_length * num_heads  # for autoregressive
            else:
                self.future_step = self.step_length

        self.dones = []
        self.red_locs = []
        self._load_data(folder_path)
        self.dones_shape = self.dones[0].shape

        # These mark the end of each episode
        self.done_locations = np.where(self.dones == True)[0]

    def _load_data(self, folder_path):
        pass

    def _produce_input(self, idx):
        pass

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        x = self._produce_input(idx)
        y = self._produce_output(idx)
        # GNN obs [B x A x 3], Hideouts [B x 2], Timestep [B], Num Agents [B]
        return x, y

    def _produce_output(self, idx):
        if self.multi_head:
            return self._produce_multi_step_output(idx)
        else:
            if self.num_heads > 1:
                return self._produce_multi_step_output(idx)  # For autoregressive multi-step
            else:
                return self._produce_single_step_output(idx)

    def _produce_single_step_output(self, idx):
        next_done = self.done_locations[np.where(self.done_locations >= idx)[0][0]]
        target_red_loc_idx = idx + self.future_step
        if target_red_loc_idx > next_done:
            red_loc = self.red_locs[next_done]
        else:
            red_loc = self.red_locs[target_red_loc_idx]
        return red_loc

    def _produce_multi_step_output(self, idx):
        next_done = self.done_locations[np.where(self.done_locations >= idx)[0][0]]
        target_red_loc_idx = idx + self.future_step
        if target_red_loc_idx > next_done:
            # if we are at the end of the episode, just use the last location
            if idx + self.step_length > next_done:
                end_loc = np.expand_dims(self.red_locs[next_done], axis=0)
                red_loc = np.repeat(end_loc, self.num_heads, axis=0)
            else:
                # if there are steps between the current and end of the episode
                begin_locs = self.red_locs[idx + self.step_length:next_done:self.step_length]
                # unsqueeze a dimension in numpy
                end_loc = np.expand_dims(self.red_locs[next_done], axis=0)
                end_loc = np.repeat(end_loc, self.num_heads - len(begin_locs), axis=0)
                red_loc = np.concatenate((begin_locs, end_loc))
        else:
            red_loc = self.red_locs[idx + self.step_length:target_red_loc_idx + self.step_length:self.step_length]

        if self.include_current:
            # add the current location to the prediction
            red_loc = np.concatenate((np.expand_dims(self.red_locs[idx], 0), red_loc), axis=0)
        return red_loc

    def _process_start_observations(self, np_array, idx, episode_start_idx):
        """ If we're indexing at the start of an episode, need to pad the start with zeros"""
        last_obs = np_array[idx]
        shape = (self.sequence_length - (idx - episode_start_idx + 1),) + last_obs.shape
        empty_sequences = np.zeros(shape)
        sequence = np_array[episode_start_idx:idx + 1]
        sequence = np.concatenate((empty_sequences, sequence), axis=0)
        return sequence


class GNNPrisonerDataset(BasePrisonerDataset):
    def __init__(self, folder_path, sequence_length, num_heads, step_length, include_current, multi_head,
                 one_hot=False, timestep=False, detected_location=False, get_last_k_fugitive_detections=False):
        self.graphs = []
        self.max_agent_size = 0
        self.process_first_graph = True

        self.one_hot_bool = one_hot
        self.timestep_bool = timestep
        self.detected_location_bool = detected_location
        self.last_k_fugitive_detection_bool = get_last_k_fugitive_detections

        super().__init__(folder_path, sequence_length, num_heads, step_length, include_current, multi_head)

    def _load_data(self, folder_path):
        np_files = []
        for file_name in sorted(os.listdir(folder_path)):
            np_file = np.load(os.path.join(folder_path, file_name), allow_pickle=True)
            self.max_agent_size = max(self.max_agent_size, np.squeeze(np_file["agent_observations"]).shape[1])
            np_files.append(np_file)

        for np_file in np_files:
            self._load_graph(np_file)

    def _load_graph(self, file):
        agent_obs = file["agent_observations"]
        agent_obs = np.squeeze(agent_obs)
        # [timesteps, agents, 3]
        # print(agent_obs.shape)
        num_agents = agent_obs.shape[1]

        detected_location = file["detected_locations"]
        if self.detected_location_bool:
            detected_bools = agent_obs[:, :, 0]
            detected_agent_locs = np.einsum("ij,ik->ijk", detected_bools, detected_location)  # timesteps, agents, 2
            agent_obs = np.concatenate((agent_obs, detected_agent_locs), axis=2)

        timesteps = file["timestep_observations"]
        if self.timestep_bool:
            t = np.expand_dims(timesteps, axis=1)
            t = np.repeat(t, num_agents, axis=1)
            # print(t.shape)
            agent_obs = np.concatenate((agent_obs, t), axis=2)

        agent_obs = np.pad(agent_obs, ((0, 0), (0, self.max_agent_size - num_agents), (0, 0)), 'constant')

        agent_dict = file["agent_dict"].item()
        num_timesteps = agent_obs.shape[0]
        one_hots = self._create_one_hot_agents(agent_dict, num_timesteps)
        one_hots = np.pad(one_hots, ((0, 0), (0, self.max_agent_size - num_agents), (0, 0)), 'constant')

        if self.process_first_graph:
            self.num_agents = [num_agents] * agent_obs.shape[0]
            self.agent_obs = agent_obs
            self.hideouts = file["hideout_observations"]
            self.timesteps = timesteps
            self.dones = file["dones"]
            self.red_locs = file["red_locations"] / 2428
            self.detected_location = detected_location
            self.one_hots = one_hots
            self.process_first_graph = False
            if self.last_k_fugitive_detection_bool:
                self.last_k_fugitive_detections = file["last_k_fugitive_detections"]
        else:
            self.num_agents.extend([num_agents] * agent_obs.shape[0])
            self.agent_obs = np.concatenate((self.agent_obs, agent_obs), 0)
            self.hideouts = np.append(self.hideouts, file["hideout_observations"], 0)
            self.red_locs = np.append(self.red_locs, file["red_locations"] / 2428, 0)
            self.timesteps = np.append(self.timesteps, timesteps)
            self.detected_location = np.append(self.detected_location, detected_location)
            self.one_hots = np.concatenate((self.one_hots, one_hots), 0)
            self.dones = np.append(self.dones, file["dones"])
            if self.last_k_fugitive_detection_bool:
                self.last_k_fugitive_detections = np.append(self.last_k_fugitive_detections,
                                                            file["last_k_fugitive_detections"], 0)

    def _create_one_hot_agents(self, agent_dict, timesteps):
        one_hot_base = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        agents = [agent_dict["num_known_cameras"] + agent_dict["num_unknown_cameras"],
                  agent_dict["num_helicopters"], agent_dict["num_search_parties"]]
        a = np.repeat(one_hot_base, agents, axis=0)  # produce [num_agents, 3] (3 for each in one-hot)
        # one_hot = np.repeat(np.expand_dims(a, 0), self.sequence_length, axis=0) # produce [seq_len, num_agents, 3]
        one_hot = np.repeat(np.expand_dims(a, 0), timesteps, axis=0)  # produce [timesteps, num_agents, 3]
        return one_hot

    # def _produce_input(self, idx):
    #     # First episode does not have reset marker
    #     if idx < self.done_locations[0] + 1:
    #         episode_start_idx = 0
    #     else:
    #         # Get index of the episode's start
    #         episode_start_idx = self.done_locations[np.where(self.done_locations <= idx - 1)[0][-1]] + 1
    #     assert idx >= episode_start_idx

    #     if idx - episode_start_idx >= self.sequence_length:
    #         agent_obs = self.agent_obs[idx - self.sequence_length:idx]
    #         if self.one_hot_bool:
    #             # Produce for each agent in agent_obs, include one-hot of the agent type
    #             # Agents obs is [Seq_len x Num_agents x 3]
    #             # One hot should be [Seq_len x Num_agents x 3]
    #             # Agent obs is in order known cameras, unknown cameras, helicopters, search parties
    #             agent_obs = np.concatenate((agent_obs, self.one_hots[idx - self.sequence_length:idx]), axis=2)

    #         sample = (agent_obs,
    #                     # self.detected_location[idx - self.sequence_length: idx],
    #                     self.hideouts[idx],
    #                     np.expand_dims(self.timesteps[idx], 0),
    #                     np.expand_dims(self.num_agents[idx], 0),
    #                     self.last_k_fugitive_detections[idx])  # Append the last k fugitive detections
    #     else:
    #         agent_obs = self._process_start_observations(self.agent_obs, idx, episode_start_idx)
    #         if self.one_hot_bool:
    #             agent_obs = np.concatenate((agent_obs, self._process_start_observations(self.one_hots, idx, episode_start_idx)), axis=2)

    #         sample = (agent_obs,
    #                 #   self.process_start_observations(self.detected_location, idx, episode_start_idx),
    #                   self.hideouts[idx], np.expand_dims(self.timesteps[idx], 0), np.expand_dims(self.num_agents[idx], 0),
    #                   self.last_k_fugitive_detections[idx])
    #         # print(type(self.num_agents[idx]))
    #     return sample
    def _produce_input(self, idx):
        # First episode does not have reset marker
        if idx < self.done_locations[0] + 1:
            episode_start_idx = 0
        else:
            # Get index of the episode's start
            episode_start_idx = self.done_locations[np.where(self.done_locations <= idx - 1)[0][-1]] + 1
        assert idx >= episode_start_idx

        if idx - episode_start_idx >= self.sequence_length:
            agent_obs = self.agent_obs[idx - self.sequence_length:idx]
            if self.one_hot_bool:
                # Produce for each agent in agent_obs, include one-hot of the agent type
                # Agents obs is [Seq_len x Num_agents x 3]
                # One hot should be [Seq_len x Num_agents x 3]
                # Agent obs is in order known cameras, unknown cameras, helicopters, search parties
                agent_obs = np.concatenate((agent_obs, self.one_hots[idx - self.sequence_length:idx]), axis=2)

            sample = [agent_obs,
                      # self.detected_location[idx - self.sequence_length: idx],
                      self.hideouts[idx],
                      np.expand_dims(self.timesteps[idx], 0),
                      np.expand_dims(self.num_agents[idx], 0)]
        else:
            agent_obs = self._process_start_observations(self.agent_obs, idx, episode_start_idx)
            if self.one_hot_bool:
                agent_obs = np.concatenate(
                    (agent_obs, self._process_start_observations(self.one_hots, idx, episode_start_idx)), axis=2)

            sample = [agent_obs,
                      #   self.process_start_observations(self.detected_location, idx, episode_start_idx),
                      self.hideouts[idx], np.expand_dims(self.timesteps[idx], 0),
                      np.expand_dims(self.num_agents[idx], 0)]

        if self.last_k_fugitive_detection_bool:
            sample.append(self.last_k_fugitive_detections[idx])  # Append the last k fugitive detections

        # if self.get_start_location_bool:
        #     sample.append(self.start_locations[idx])

        return sample


class HeterogeneousGNNDataset(GNNPrisonerDataset):
    def __init__(self, folder_path, sequence_length, num_heads, step_length, include_current, multi_head,
                 one_hot=False, timestep=False, detected_location=False):

        super().__init__(folder_path, sequence_length, num_heads, step_length, include_current, multi_head,
                         one_hot, timestep, detected_location)

    def _produce_input(self, idx):
        # Output (agent_obs, hideouts, timesteps, num_agents)
        # However, unlike the gnn output, the timesteps will be a vector of t x 1 instead of a single timestep
        # First episode does not have reset marker
        if idx < self.done_locations[0] + 1:
            episode_start_idx = 0
        else:
            # Get index of the episode's start
            episode_start_idx = self.done_locations[np.where(self.done_locations <= idx - 1)[0][-1]] + 1
        assert idx >= episode_start_idx

        if idx - episode_start_idx >= self.sequence_length:
            agent_obs = self.agent_obs[idx - self.sequence_length:idx]
            if self.one_hot_bool:
                # Produce for each agent in agent_obs, include one-hot of the agent type
                # Agents obs is [Seq_len x Num_agents x 3]
                # One hot should be [Seq_len x Num_agents x 3]
                # Agent obs is in order known cameras, unknown cameras, helicopters, search parties
                agent_obs = np.concatenate((agent_obs, self.one_hots[idx - self.sequence_length:idx]), axis=2)

            sample = (agent_obs,
                      self.hideouts[idx],
                      self.timesteps[idx - self.sequence_length: idx],
                      np.expand_dims(self.num_agents[idx], 0))
        else:
            agent_obs = self._process_start_observations(self.agent_obs, idx, episode_start_idx)
            if self.one_hot_bool:
                agent_obs = np.concatenate(
                    (agent_obs, self._process_start_observations(self.one_hots, idx, episode_start_idx)), axis=2)

            sample = (agent_obs,
                      self.hideouts[idx],
                      self._process_start_observations(self.timesteps, idx, episode_start_idx),
                      np.expand_dims(self.num_agents[idx], 0))
        return sample


class LastTwoDetectsMLPDataset(BasePrisonerDataset):
    """
    Dataset Loader for storing just the last two detected locations of the fugitive
    along with dx/dt, dy/dt and the current timestamp. (Similar to Zixuan's approach)
    """

    def __init__(self, folder_path, sequence_length, num_heads, step_length, include_current, multi_head, view='blue',
                 get_start_location=False):
        self.view = view
        self.get_start_location = get_start_location
        print(self.view)
        super().__init__(folder_path, sequence_length, num_heads, step_length, include_current, multi_head)

    def _get_blue_obs(self, np_file):
        # Last_k_fugitive_detections has (t, x, y)
        # Get last 2 detected locations
        last_two_detected_locs = np_file["last_k_fugitive_detections"][:, -2:, 1:]
        last_two_detected_locs = last_two_detected_locs.reshape(-1, 4)

        # Get the timestamps at which the detections occured
        detected_timestamps = np_file["last_k_fugitive_detections"][:, -2:, 0]
        # Calculate time between detections and estimate dx/dt, dy/dt
        delta_t = detected_timestamps[:, 1] - detected_timestamps[:, 0]
        vel_x = ((last_two_detected_locs[:, 2] - last_two_detected_locs[:, 0]) / delta_t).reshape(-1, 1)
        vel_y = ((last_two_detected_locs[:, 3] - last_two_detected_locs[:, 1]) / delta_t).reshape(-1, 1)

        # If two detections are not present set dx/dt, dy/dt to 0
        vel_x[np.where(last_two_detected_locs[:, 0] == -1)] = 0
        vel_y[np.where(last_two_detected_locs[:, 0] == -1)] = 0
        curr_time = np_file["blue_observations"][:, 0].reshape(-1, 1)

        # Concat last 2 detected locations, dx/dt, dy/dt, episode timestamp
        blue_obs_np = np.concatenate((last_two_detected_locs, vel_x, vel_y, curr_time), axis=-1)

        return blue_obs_np

    def _load_data(self, folder_path):
        # Load files from a folder containing each episode
        for idx, file_name in enumerate(sorted(os.listdir(folder_path))):
            if file_name == "_graphs.npz":
                continue
            np_file = np.load(os.path.join(folder_path, file_name), allow_pickle=True)
            detected_location = np_file["detected_locations"]
            if idx == 0:
                if self.view == "blue":
                    self.detected_location = detected_location
                    self.blue_obs_np = self._get_blue_obs(np_file)

                elif self.view == "red":
                    self.red_obs_np = np_file["red_observations"]
                elif self.view == "both":
                    self.blue_obs_np = np_file["blue_observations"]
                    self.red_obs_np = np_file["red_observations"]
                else:
                    raise ValueError("View must be either red or blue")
                self.red_locs = np_file["red_locations"] / 2428
                self.dones = np_file["dones"]

            else:
                if self.view == "blue":
                    self.detected_location = np.append(self.detected_location, detected_location)
                    blue_obs_np = self._get_blue_obs(np_file)
                    self.blue_obs_np = np.concatenate((self.blue_obs_np, blue_obs_np), 0)

                elif self.view == "red":
                    self.red_obs_np = np.concatenate((self.red_obs_np, np_file["red_observations"]), 0)
                elif self.view == "both":
                    self.blue_obs_np = np.concatenate((self.blue_obs_np, np_file["blue_observations"]), 0)
                    self.red_obs_np = np.concatenate((self.red_obs_np, np_file["red_observations"]), 0)

                self.red_locs = np.append(self.red_locs, np_file["red_locations"] / 2428, 0)
                self.dones = np.append(self.dones, np_file["dones"])

    def _produce_input(self, idx):
        # First episode does not have reset marker
        if idx < self.done_locations[0] + 1:
            episode_start_idx = 0
        else:
            # Get index of the episode's start
            episode_start_idx = self.done_locations[np.where(self.done_locations <= idx - 1)[0][-1]] + 1
        assert idx >= episode_start_idx

        if idx - episode_start_idx >= self.sequence_length:
            if self.view == "red":
                red_sequence = self.red_obs_np[idx - self.sequence_length:idx]
            blue_sequence = self.blue_obs_np[idx - self.sequence_length:idx]
        else:
            if self.view == "red":
                red_sequence = self._process_start_observations(self.red_obs_np, idx, episode_start_idx)
            blue_sequence = self._process_start_observations(self.blue_obs_np, idx, episode_start_idx)

        if self.view == "red":
            return red_sequence
        elif self.view == "blue":
            return blue_sequence
        elif self.view == "both":
            return blue_sequence, red_sequence
        else:
            raise ValueError("View must be either red or blue")


class LastKDetects(BasePrisonerDataset):
    def __init__(self, folder_path, sequence_length, num_heads, step_length, include_current, multi_head, view='blue',
                 get_start_location=False, produce_sequence_output=True, env_name="prisoner"):
        self.view = view
        self.get_start_location = get_start_location
        print(self.view)
        if env_name == "prisoner":
            self.env_dim_x = 2428
            self.env_dim_y = 2428
        elif env_name == "smuggler":
            self.env_dim_x = 7884.466935319577
            self.env_dim_y = 3538.368

        self.always_include_start = False

        super().__init__(folder_path, sequence_length, num_heads, step_length, include_current, multi_head)

    def _load_data(self, folder_path):
        # Load files from a folder containing each episode
        for idx, file_name in enumerate(sorted(os.listdir(folder_path))):
            if file_name == "_graphs.npz":
                continue
            np_file = np.load(os.path.join(folder_path, file_name), allow_pickle=True)
            detected_location = np_file["detected_locations"]
            timesteps = np_file["timestep_observations"]

            # Modify detected locations such that there is always a detection at the first timestep
            # This is so that we can use the first timestep as the input to the LSTM
            detected_location[0] = copy.deepcopy(np_file["red_locations"][0]) / np.array(
                [[self.env_dim_x, self.env_dim_y]])

            if idx == 0:
                if self.view == "blue":
                    self.detected_location = detected_location
                    last_k_detects = np_file["last_k_fugitive_detections"]
                    curr_time = np_file["blue_observations"][:, 0].reshape(-1, 1)
                    curr_time = curr_time[:, np.newaxis, :]
                    curr_time = np.repeat(curr_time, 8, axis=1)
                    # self.blue_obs_np = np.concatenate((last_k_detects, curr_time), axis=-1)
                    self.blue_obs_np = self._preprocess_detections(detected_location, timesteps)
                elif self.view == "red":
                    self.red_obs_np = np_file["red_observations"]
                elif self.view == "both":
                    self.blue_obs_np = np_file["blue_observations"]
                    self.red_obs_np = np_file["red_observations"]
                else:
                    raise ValueError("View must be either red or blue")
                self.red_locs = np_file["red_locations"] / (np.array([[self.env_dim_x, self.env_dim_y]]))
                self.dones = np_file["dones"]

            else:
                if self.view == "blue":
                    self.detected_location = np.append(self.detected_location, detected_location)
                    curr_blue_obs = self._preprocess_detections(detected_location, timesteps)
                    curr_time = np_file["blue_observations"][:, 0].reshape(-1, 1)
                    curr_time = curr_time[:, np.newaxis, :]
                    curr_time = np.repeat(curr_time, 8, axis=1)
                    # curr_blue_obs = np.concatenate((curr_blue_obs, curr_time), axis=-1)
                    self.blue_obs_np = np.concatenate((self.blue_obs_np, curr_blue_obs), 0)
                elif self.view == "red":
                    self.red_obs_np = np.concatenate((self.red_obs_np, np_file["red_observations"]), 0)
                elif self.view == "both":
                    self.blue_obs_np = self.blue_obs_np + self._preprocess_detections(detected_location,
                                                                                      timesteps)
                    self.red_obs_np = np.concatenate((self.red_obs_np, np_file["red_observations"]), 0)

                self.red_locs = np.append(self.red_locs,
                                          np_file["red_locations"] / (np.array([[self.env_dim_x, self.env_dim_y]])), 0)
                self.dones = np.append(self.dones, np_file["dones"])

    def _preprocess_detections(self, detected_locs, timestamps):
        """ Given a numpy array of [T x 2] where if there is a detection, the value is (x, y) and if there is not, the value is (-1, -1)

        For each row in the array, return all previous detections before that row

        Also need to add the time difference between each step so we return a [dt, x, y] for each detection
        """
        processed_detections = []

        detected_locs = np.concatenate((detected_locs, timestamps), axis=1)
        for i in range(detected_locs.shape[0]):
            curr_detections = copy.deepcopy(detected_locs[:i + 1])
            curr_detections = curr_detections[curr_detections[:, 0] != -1]
            curr_detections[:, 2] = detected_locs[i, 2] - curr_detections[:, 2]
            if self.always_include_start:
                curr_detections = np.concatenate((curr_detections[0:1], curr_detections[-self.sequence_length:]))
            else:
                curr_detections = curr_detections[-self.sequence_length:]  # only take last 512 samples
            # Pad the processed_detections to be of seq_len
            if curr_detections.shape[0] < self.sequence_length:
                empty_sequence = np.zeros((self.sequence_length - curr_detections.shape[0], 3))
                curr_detections = np.concatenate((empty_sequence, curr_detections), 0)
            processed_detections.append(curr_detections)

        return np.stack(processed_detections)

    def _produce_input(self, idx):
        return torch.from_numpy(self.blue_obs_np[idx]).float()


class VectorPrisonerDataset(BasePrisonerDataset):
    def __init__(self, folder_path, sequence_length, num_heads, step_length, include_current, multi_head, view='blue',
                 get_start_location=False):
        self.view = view
        self.get_start_location = get_start_location
        print(self.view)
        super().__init__(folder_path, sequence_length, num_heads, step_length, include_current, multi_head)

    def _load_data(self, folder_path):
        # Load files from a folder containing each episode
        for idx, file_name in enumerate(sorted(os.listdir(folder_path))):
            if file_name == "_graphs.npz":
                continue
            np_file = np.load(os.path.join(folder_path, file_name), allow_pickle=True)
            detected_location = np_file["detected_locations"]
            if idx == 0:
                if self.view == "blue":
                    self.detected_location = detected_location
                    if self.get_start_location:
                        starting_location = np.repeat(np_file["red_locations"][0] / 2428,
                                                      len(np_file["blue_observations"])).reshape(-1,
                                                                                                 2)  # Add the start location to the end of each vector in the episode
                        self.blue_obs_np = np.concatenate((np_file["blue_observations"], starting_location), 1)
                    else:
                        # Add starting location as a detection
                        starting_location = np_file["red_locations"][0] / 2428
                        self.blue_obs_np = np_file["blue_observations"][:, :-2]
                        # self.blue_obs_np[0, -2:] = starting_location

                elif self.view == "red":
                    self.red_obs_np = np_file["red_observations"]
                elif self.view == "both":
                    self.blue_obs_np = np_file["blue_observations"]
                    self.red_obs_np = np_file["red_observations"]
                else:
                    raise ValueError("View must be either red or blue")
                self.red_locs = np_file["red_locations"] / 2428
                self.dones = np_file["dones"]

            else:
                if self.view == "blue":
                    self.detected_location = np.append(self.detected_location, detected_location)

                    if self.get_start_location:
                        starting_location = np.repeat(np_file["red_locations"][0] / 2428,
                                                      len(np_file["blue_observations"])).reshape(-1,
                                                                                                 2)  # Add the start location to the end of each vector in the episode
                        blue_obs_np = np.concatenate((np_file["blue_observations"], starting_location), 1)
                        self.blue_obs_np = np.concatenate((self.blue_obs_np, blue_obs_np), 0)

                    else:
                        # Add starting location as a detection
                        starting_location = np_file["red_locations"][0] / 2428
                        curr_blue_obs = np_file["blue_observations"][:, :-2]
                        # curr_blue_obs[0, -2:] = starting_location
                        self.blue_obs_np = np.concatenate((self.blue_obs_np, curr_blue_obs), 0)
                elif self.view == "red":
                    self.red_obs_np = np.concatenate((self.red_obs_np, np_file["red_observations"]), 0)
                elif self.view == "both":
                    self.blue_obs_np = np.concatenate((self.blue_obs_np, np_file["blue_observations"]), 0)
                    self.red_obs_np = np.concatenate((self.red_obs_np, np_file["red_observations"]), 0)

                self.red_locs = np.append(self.red_locs, np_file["red_locations"] / 2428, 0)
                self.dones = np.append(self.dones, np_file["dones"])

    def _produce_input(self, idx):
        # First episode does not have reset marker
        if idx < self.done_locations[0] + 1:
            episode_start_idx = 0
        else:
            # Get index of the episode's start
            episode_start_idx = self.done_locations[np.where(self.done_locations <= idx - 1)[0][-1]] + 1
        assert idx >= episode_start_idx

        if idx - episode_start_idx >= self.sequence_length:
            if self.view == "red":
                red_sequence = self.red_obs_np[idx - self.sequence_length:idx]
            blue_sequence = self.blue_obs_np[idx - self.sequence_length:idx]
        else:
            if self.view == "red":
                red_sequence = self._process_start_observations(self.red_obs_np, idx, episode_start_idx)
            blue_sequence = self._process_start_observations(self.blue_obs_np, idx, episode_start_idx)

        if self.view == "red":
            return red_sequence
        elif self.view == "blue":
            return blue_sequence
        elif self.view == "both":
            return blue_sequence, red_sequence
        else:
            raise ValueError("View must be either red or blue")


class ContrastiveVectorPrisonerDataset(BasePrisonerDataset):
    def __init__(self, folder_path, sequence_length, num_heads, step_length, include_current, multi_head, view='blue'):
        self.view = view
        print(self.view)
        super().__init__(folder_path, sequence_length, num_heads, step_length, include_current, multi_head)

    def _load_data(self, folder_path):
        starting_seed = int(sorted(os.listdir(folder_path))[0].split('_')[1])
        num_files = len(os.listdir(folder_path))
        for seed in range(starting_seed, starting_seed + num_files):
            file_name = f"seed_{seed}_known_44_unknown_33.npz"
            np_file = np.load(os.path.join(folder_path, file_name), allow_pickle=True)
            # self.max_agent_size = max(self.max_agent_size, np.squeeze(np_file["agent_observations"]).shape[1])
            if seed == starting_seed:
                if self.view == "blue":
                    self.blue_obs_np = np_file["blue_observations"]
                elif self.view == "red":
                    self.red_obs_np = np_file["red_observations"] / 2428
                elif self.view == "both":
                    self.blue_obs_np = np_file["blue_observations"]
                    self.red_obs_np = np_file["red_observations"]
                self.red_locs = np_file["red_locations"] / 2428
                self.dones = np_file["dones"]
                # Store the locations of timestep where the fugitive was (+ve) and wasn't (-ve) observed
                self.positive_observation_timesteps = np_file["positive_observation_timesteps"]
                self.negative_observation_timesteps = np_file["negative_observation_timesteps"]

            else:
                if self.view == "blue":
                    self.blue_obs_np = np.concatenate((self.blue_obs_np, np_file["blue_observations"]), 0)
                elif self.view == "red":
                    self.red_obs_np = np.concatenate((self.red_obs_np, np_file["red_observations"] / 2428), 0)
                elif self.view == "both":
                    self.blue_obs_np = np.concatenate((self.blue_obs_np, np_file["blue_observations"]), 0)
                    self.red_obs_np = np.concatenate((self.red_obs_np, np_file["red_observations"]), 0)

                self.red_locs = np.append(self.red_locs, np_file["red_locations"] / 2428, 0)
                self.dones = np.append(self.dones, np_file["dones"])

                self.positive_observation_timesteps = np.concatenate(
                    (self.positive_observation_timesteps, np_file["positive_observation_timesteps"]), 0)
                self.negative_observation_timesteps = np.concatenate(
                    (self.negative_observation_timesteps, np_file["negative_observation_timesteps"]), 0)

    def _produce_input(self, idx):
        # First episode does not have reset marker
        if idx < self.done_locations[0] + 1:
            episode_start_idx = 0
        else:
            # Get index of the episode's start
            episode_start_idx = self.done_locations[np.where(self.done_locations <= idx - 1)[0][-1]] + 1
        assert idx >= episode_start_idx

        if self.view == "red":
            red_sequence = self._process_start_observations(self.red_obs_np, idx, episode_start_idx)
        positive_blue_sequence, negative_blue_sequence = self._process_contrastive_observations(self.blue_obs_np, idx,
                                                                                                episode_start_idx)

        # Needed to calculate delta t for contrastive loss
        last_positive_observation = self.positive_observation_timesteps[idx][-1]
        last_negative_observation = self.negative_observation_timesteps[idx][-1]

        sample = (positive_blue_sequence, negative_blue_sequence,
                  last_positive_observation, last_negative_observation)

        if self.view == "red":
            return red_sequence
        elif self.view == "blue":
            return sample
        elif self.view == "both":
            return sample, red_sequence
        else:
            raise ValueError("View must be either red or blue")

    def _process_contrastive_observations(self, np_array, idx, episode_start_idx):
        """ If we're indexing at the start of an episode, need to pad the start with zeros"""

        positive_blue_sequence, negative_blue_sequence = [], []

        last_obs = np_array[idx]
        # shape = (self.sequence_length - (idx - episode_start_idx + 1),) + last_obs.shape
        empty_sequence = np.zeros_like(last_obs)

        # TODO: @Manisha Optimize this by using numpy instead of for loop
        for t in self.positive_observation_timesteps[idx]:
            if t == -1 or t < episode_start_idx:
                positive_blue_sequence.append(empty_sequence)
            else:
                positive_blue_sequence.append(np_array[t])

        for t in self.negative_observation_timesteps[idx]:
            if t == -1 or t < episode_start_idx:
                negative_blue_sequence.append(empty_sequence)
            else:
                negative_blue_sequence.append(np_array[t])

        positive_blue_sequence = np.array(positive_blue_sequence)
        negative_blue_sequence = np.array(negative_blue_sequence)

        return positive_blue_sequence, negative_blue_sequence


if __name__ == "__main__":
    # path = "/nethome/sye40/PrisonerEscape/datasets/gnn_map_0_run_300_eps_0.1_norm.npz"
    # np_file = np.load(path, allow_pickle=True)

    seq_len = 16
    step_length = 5
    num_heads = 12

    # path = "/nethome/sye40/PrisonerEscape/datasets/train/gnn_map_0_run_300_eps_0.1_norm_random_cameras"
    # path = "/nethome/sye40/PrisonerEscape/datasets/small_train"
    path = "/nethome/sye40/PrisonerEscape/datasets/test_same/gnn_map_0_run_100_eps_0.1_norm"
    dataset = GNNPrisonerDataset(path, seq_len, num_heads, step_length,
                                 include_current=True, multi_head=True, one_hot=True,
                                 timestep=True, detected_location=True)
    print(len(dataset))
    train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    x, y = dataset[120]
    print(x[0].shape)
    # print(x[0][15, 0]*4320)

    # np_file_path = "/nethome/sye40/PrisonerEscape/datasets/seed_corrected/map_0_run_100_eps_0_norm.npz"
    # seq_len = 4
    # future_step = 10
    # red_blue_dataset = VectorPrisonerDataset(np_file_path, seq_len, num_heads, step_length, include_current=True, multi_head = True, view='blue')

    # print(red_blue_dataset[0])

    # import time
    # now = time.time()
    for x, y in train_dataloader:
        # print(x[3].shape)
        print(x[0].shape)
        break
        # for i in x:
        #     print(i.shape)
        # break
