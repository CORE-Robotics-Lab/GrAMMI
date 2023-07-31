import numpy as np
import os
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
import copy


class PaddedDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, num_heads, step_length, include_current, multi_head, env_name, max_seq_len=512,
                 always_include_start=False):
        """ This dataset is uses the pad_packed_sequence function to ensure the entire detection history is located in the LSTM """

        print(f"Training on Padded Dataset w/ {env_name}")

        self.max_seq_len = max_seq_len  # force all sequences to be this max length
        self.always_include_start = always_include_start

        self.env_name = env_name
        if env_name == "prisoner":
            self.env_dim_x = 2428
            self.env_dim_y = 2428
        elif env_name == "smuggler":
            self.env_dim_x = 7884.466935319577
            self.env_dim_y = 3538.368

        # Currently just try with Filtering?
        # Need self.red_locs and self.dones
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
        self.process_first_graph = True
        self._load_data(folder_path)
        self.dones_shape = self.dones[0].shape

        # These mark the end of each episode
        self.done_locations = np.where(self.dones == True)[0]

    def _load_data(self, folder_path):
        np_files = []
        for file_name in sorted(os.listdir(folder_path)):
            np_file = np.load(os.path.join(folder_path, file_name), allow_pickle=True)
            # print(np_file)
            # self.max_agent_size = max(self.max_agent_size, np.squeeze(np_file["agent_observations"]).shape[1])
            np_files.append(np_file)

        for np_file in np_files:
            self._load_graph(np_file)

    def _load_graph(self, file):
        agent_obs = file["agent_observations"]

        if self.env_name == "prisoner":
            agent_obs = agent_obs[:, -7:]  # just getting moving agents, 5 search parties and 2 helicopters

        num_agents = agent_obs.shape[1]

        detected_location = file["detected_locations"]
        timesteps = file["timestep_observations"]

        # Modify detected locations such that there is always a detection at the first timestep
        # This is so that we can use the first timestep as the input to the LSTM
        detected_location[0] = copy.deepcopy(file["red_locations"][0]) / np.array([[self.env_dim_x, self.env_dim_y]])

        if self.process_first_graph:
            self.process_first_graph = False
            self.agent_obs = agent_obs
            self.num_agents = [num_agents] * agent_obs.shape[0]
            self.hideouts = file["hideout_observations"]
            self.timesteps = timesteps
            self.dones = file["dones"]
            self.red_locs = file["red_locations"] / (np.array([[self.env_dim_x, self.env_dim_y]]))
            self.detected_locations = self._preprocess_detections(detected_location, timesteps)
            self.detected_location = detected_location  # for visualization
            self.last_k_fugitive_detections = file["last_k_fugitive_detections"]
        else:
            self.num_agents.extend([num_agents] * agent_obs.shape[0])
            self.agent_obs = np.concatenate((self.agent_obs, agent_obs), 0)
            self.hideouts = np.append(self.hideouts, file["hideout_observations"], 0)
            self.red_locs = np.append(self.red_locs,
                                      file["red_locations"] / (np.array([[self.env_dim_x, self.env_dim_y]])), 0)
            self.timesteps = np.append(self.timesteps, timesteps)
            self.dones = np.append(self.dones, file["dones"])
            self.last_k_fugitive_detections = np.append(self.last_k_fugitive_detections,
                                                        file["last_k_fugitive_detections"], 0)
            self.detected_locations = self.detected_locations + self._preprocess_detections(detected_location,
                                                                                            timesteps)
            self.detected_location = np.append(self.detected_location, detected_location, 0)  # for visualization

            # print(max([len(x) for x in self._preprocess_detections(detected_location, timesteps)]))

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
                curr_detections = np.concatenate((curr_detections[0:1], curr_detections[-self.max_seq_len:]))
            else:
                curr_detections = curr_detections[-self.max_seq_len:]  # only take last 512 samples

            processed_detections.append(curr_detections)
        return processed_detections

    def _produce_input(self, idx):
        return torch.from_numpy(self.detected_locations[idx]).float()

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
        return torch.from_numpy(red_loc).float()

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
        return torch.from_numpy(red_loc).float()


class PaddedDatasetVAE(torch.utils.data.Dataset):
    def __init__(self, folder_path, num_heads, step_length, include_current, multi_head, env_name, max_seq_len=512,
                 always_include_start=False):
        """ This dataset is uses the pad_packed_sequence function to ensure the entire detection history is located in the LSTM """

        print(f"Training on Padded Dataset w/ {env_name}")

        self.max_seq_len = max_seq_len  # force all sequences to be this max length
        self.always_include_start = always_include_start

        self.env_name = env_name
        if env_name == "prisoner":
            self.env_dim_x = 2428
            self.env_dim_y = 2428
        elif env_name == "smuggler":
            self.env_dim_x = 7884.466935319577
            self.env_dim_y = 3538.368

        # Currently just try with Filtering?
        # Need self.red_locs and self.dones
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
        self.process_first_graph = True
        self._load_data(folder_path)
        self.dones_shape = self.dones[0].shape

        # These mark the end of each episode
        self.done_locations = np.where(self.dones == True)[0]

    def _load_data(self, folder_path):
        np_files = []
        for file_name in sorted(os.listdir(folder_path)):
            np_file = np.load(os.path.join(folder_path, file_name), allow_pickle=True)
            # print(np_file)
            # self.max_agent_size = max(self.max_agent_size, np.squeeze(np_file["agent_observations"]).shape[1])
            np_files.append(np_file)

        for np_file in np_files:
            self._load_graph(np_file)

    def _load_graph(self, file):
        agent_obs = file["agent_observations"]

        if self.env_name == "prisoner":
            agent_obs = agent_obs[:, -7:]  # just getting moving agents, 5 search parties and 2 helicopters

        num_agents = agent_obs.shape[1]

        detected_location = file["detected_locations"]
        timesteps = file["timestep_observations"]

        # Modify detected locations such that there is always a detection at the first timestep
        # This is so that we can use the first timestep as the input to the LSTM
        detected_location[0] = copy.deepcopy(file["red_locations"][0]) / np.array([[self.env_dim_x, self.env_dim_y]])

        if self.process_first_graph:
            self.process_first_graph = False
            self.agent_obs = agent_obs
            self.num_agents = [num_agents] * agent_obs.shape[0]
            self.hideouts = file["hideout_observations"]
            self.timesteps = timesteps
            self.dones = file["dones"]
            self.red_locs = file["red_locations"] / (np.array([[self.env_dim_x, self.env_dim_y]]))
            # Prev red states refers to the location of the red agent at the previous timestep, repeat the first red loc
            # and pop the last red loc to create the prev red states array
            self.prev_red_locs = copy.deepcopy(self.red_locs)
            self.prev_red_locs = np.delete(np.concatenate((self.prev_red_locs[0:1], self.prev_red_locs)), -1, 0)
            self.detected_locations = self._preprocess_detections(detected_location, timesteps)
            self.detected_location = detected_location  # for visualization
            self.last_k_fugitive_detections = file["last_k_fugitive_detections"]
        else:
            self.num_agents.extend([num_agents] * agent_obs.shape[0])
            self.agent_obs = np.concatenate((self.agent_obs, agent_obs), 0)
            self.hideouts = np.append(self.hideouts, file["hideout_observations"], 0)
            self.red_locs = np.append(self.red_locs,
                                      file["red_locations"] / (np.array([[self.env_dim_x, self.env_dim_y]])), 0)

            # Prev red states refers to the location of the red agent at the previous timestep, repeat the first red loc
            # and pop the last red loc to create the prev red states array
            prev_red_locs = file["red_locations"] / (np.array([[self.env_dim_x, self.env_dim_y]]))
            prev_red_locs = np.delete(np.concatenate((prev_red_locs[0:1], prev_red_locs)), -1, 0)
            self.prev_red_locs = np.append(self.prev_red_locs, prev_red_locs, 0)
            self.timesteps = np.append(self.timesteps, timesteps)
            self.dones = np.append(self.dones, file["dones"])
            self.last_k_fugitive_detections = np.append(self.last_k_fugitive_detections,
                                                        file["last_k_fugitive_detections"], 0)
            self.detected_locations = self.detected_locations + self._preprocess_detections(detected_location,
                                                                                            timesteps)
            self.detected_location = np.append(self.detected_location, detected_location, 0)  # for visualization

            # print(max([len(x) for x in self._preprocess_detections(detected_location, timesteps)]))

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
                curr_detections = np.concatenate((curr_detections[0:1], curr_detections[-self.max_seq_len:]))
            else:
                curr_detections = curr_detections[-self.max_seq_len:]  # only take last 512 samples

            processed_detections.append(curr_detections)
        return processed_detections

    def _produce_input(self, idx):
        return (torch.from_numpy(self.detected_locations[idx]).float(),
                torch.from_numpy(self.prev_red_locs[idx]).float())

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
        return torch.from_numpy(red_loc).float()

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
        return torch.from_numpy(red_loc).float()


class PaddedDatasetAgent(PaddedDataset):
    def __init__(self, folder_path, num_heads, step_length, include_current, multi_head, env_name, agent_length):
        super().__init__(folder_path, num_heads, step_length, include_current, multi_head, env_name)
        self.agent_length = agent_length

    def get_agent_obs(self, idx):
        if idx < self.done_locations[0] + 1:
            episode_start_idx = 0
        else:
            # Get index of the episode's start
            episode_start_idx = self.done_locations[np.where(self.done_locations <= idx - 1)[0][-1]] + 1
        assert idx >= episode_start_idx

        if idx - episode_start_idx <= self.agent_length:
            agent_obs = self.agent_obs[episode_start_idx:idx + 1]
        else:
            agent_obs = self.agent_obs[idx - self.agent_length + 1:idx + 1]

        a = torch.from_numpy(agent_obs).float()

        return a

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        agent_obs = self.get_agent_obs(idx)
        n_agents = np.expand_dims(self.num_agents[idx], 0)
        n_agents = torch.from_numpy(n_agents).int()
        return x, agent_obs, n_agents, y


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)

    yy = torch.stack(yy, axis=0)

    return xx_pad, x_lens, yy


def pad_collate_vae(batch):
    (xx_total, yy) = zip(*batch)
    x_lens = [len(x[0]) for x in xx_total]
    prev_red_state = tuple([x[1] for x in xx_total])
    xx = [x[0] for x in xx_total]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)

    yy = torch.stack(yy, axis=0)

    return xx_pad, prev_red_state, x_lens, yy


def pad_collate_agent_gnn(batch):
    (xx, agent_obs, n_agents, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]

    outer = []
    # Flatten the agent obs because we want to pass to LSTM to GNN separately
    for a in agent_obs:
        num_agents = a.shape[1]
        for i in range(num_agents):
            outer.append(a[:, i, :])
    agent_lens = [len(x) for x in outer]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    agent_obs_pad = pad_sequence(outer, batch_first=True, padding_value=0)

    yy = torch.stack(yy, axis=0)
    n_agents = torch.stack(n_agents, axis=0)

    return xx_pad, x_lens, agent_obs_pad, agent_lens, n_agents, yy


def pad_collate_agent(batch):
    (xx, agent_obs, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    agent_lens = [len(x) for x in agent_obs]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    agent_obs_pad = pad_sequence(agent_obs, batch_first=True, padding_value=0)

    yy = torch.stack(yy, axis=0)
    return xx_pad, x_lens, agent_obs_pad, agent_lens, yy


if __name__ == "__main__":
    # data_path = "datasets/october_datasets/4_detect/train"
    # data_path = "/data/prisoner_datasets/4_detect/train"
    data_path = "/data/prisoner_datasets/balance_game/AStar/test"
    # data_path = "/data/smuggler_datasets/smuggler_paper_2_helo_40/test"
    # data_path = "/data/smuggler_datasets/smuggler_paper_3_helo_40/test"
    num_heads = 1
    step_length = 0
    include_current = False
    multi_head = False
    dataset = PaddedDataset(data_path, num_heads, step_length, include_current, multi_head, env_name="prisoner")

    print(dataset[0])

    # dataset = PaddedDatasetAgent(data_path, num_heads, step_length, include_current, multi_head, env_name="prisoner", agent_length = 16)
    # data_loader = DataLoader(dataset = dataset, batch_size = 32, shuffle=True, collate_fn=pad_collate_agent_gnn)
    # import torch.nn as nn
    # rnn = nn.GRU(3, 5, 1, batch_first=True)
    # agent_rnn = nn.GRU(3, 8, 1, batch_first=True)
    # for i, (x, x_lens, agent_obs, agent_lens, n_agents, y) in enumerate(data_loader):

    #     print(n_agents)

    #     x_packed = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
    #     agent_packed = pack_padded_sequence(agent_obs, agent_lens, batch_first=True, enforce_sorted=False)

    #     # print(x_packed.data.shape)
    #     hidden = torch.zeros(1, x.shape[0], 5)
    #     out_packed, hidden = rnn(x_packed, hidden)
    #     print(out_packed.data.shape, hidden.shape)

    #     hidden = torch.zeros(1, agent_obs.shape[0], 8)
    #     out_packed_agent, hidden = agent_rnn(agent_packed, hidden)
    #     print(out_packed.data.shape, hidden.shape)

    #     break
