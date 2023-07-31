import numpy as np
import os
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
import copy

class PaddedDatasetParticle(torch.utils.data.Dataset):
    def __init__(self, folder_path, num_heads, step_length, include_current, multi_head):
        """ This dataset is uses the pad_packed_sequence function to ensure the entire detection history is located in the LSTM """
        
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

        self.timestep_max = 60

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
            self._load_file(np_file)

    def _load_file(self, file):
        detected_location = file["detected_locations"]
        timesteps = file["timestep_observations"]

        # Modify detected locations such that there is always a detection at the first timestep
        # This is so that we can use the first timestep as the input to the LSTM
        detected_location[0] = copy.deepcopy(file["red_locations"][0])

        if self.process_first_graph:
            self.process_first_graph = False
            self.hideouts = file["hideout_observations"]
            self.timesteps = timesteps
            self.dones = file["dones"]
            self.red_locs = file["red_locations"]
            self.detected_locations = self._preprocess_detections(detected_location, timesteps)
            self.detected_location = detected_location # for visualization
        else:
            self.hideouts = np.append(self.hideouts, file["hideout_observations"], 0)
            self.red_locs = np.append(self.red_locs, file["red_locations"], 0)
            self.timesteps = np.append(self.timesteps, timesteps)
            self.dones = np.append(self.dones, file["dones"])
            self.detected_locations = self.detected_locations + self._preprocess_detections(detected_location, timesteps)
            self.detected_location = np.append(self.detected_location, detected_location, 0) # for visualization

    def _preprocess_detections(self, detected_locs, timestamps):
        """ Given a numpy array of [T x 2] where if there is a detection, the value is (x, y) and if there is not, the value is (-1, -1)
        
        For each row in the array, return all previous detections before that row

        Also need to add the time difference between each step so we return a [dt, x, y] for each detection
        """
        processed_detections = []

        timestamps = np.expand_dims(timestamps, 1)
        detected_locs = np.concatenate((detected_locs, timestamps), axis=1)
        for i in range(detected_locs.shape[0]):
            curr_detections = copy.deepcopy(detected_locs[:i+1])
            curr_detections = curr_detections[curr_detections[:, 0] != -1]
            curr_detections[:, 2] = detected_locs[i, 2] - curr_detections[:, 2]
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
                begin_locs = self.red_locs[idx+self.step_length:next_done:self.step_length]
                # unsqueeze a dimension in numpy
                end_loc = np.expand_dims(self.red_locs[next_done], axis=0)
                end_loc = np.repeat(end_loc, self.num_heads - len(begin_locs), axis=0)
                red_loc = np.concatenate((begin_locs, end_loc))
        else:
            red_loc = self.red_locs[idx+self.step_length:target_red_loc_idx+self.step_length:self.step_length]
        
        if self.include_current:
            # add the current location to the prediction
            red_loc = np.concatenate((np.expand_dims(self.red_locs[idx], 0), red_loc), axis=0)
        return torch.from_numpy(red_loc).float()


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)

    yy = torch.stack(yy, axis=0)

    return xx_pad, x_lens, yy

if __name__ == "__main__":
    # data_path = "/data/manisha/datasets/fixed_policy/corner_start_fixed_hideouts/test"
    data_path = "/home/sean/PrisonerEscape/datasets/particle_dataset"
    num_heads = 1
    step_length = 0
    include_current = False
    multi_head = False
    
    dataset = PaddedDatasetParticle(data_path, num_heads, step_length, include_current, multi_head)
    # print(dataset.done_locations)
    # print(dataset[264])
    
    # data_loader = DataLoader(dataset = dataset, batch_size = 32, shuffle=True, collate_fn=pad_collate)
    # import torch.nn as nn
    # rnn = nn.GRU(3, 5, 1, batch_first=True)
    # for i, (x, x_lens, y) in enumerate(data_loader):
    #     # print(x.shape)
    #     # print(x_lens)
    #     # print(y)
    #     x_packed = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
    #     print(x_packed.data.shape)
    #     hidden = torch.zeros(1, x.shape[0], 5)
    #     out_packed, hidden = rnn(x_packed, hidden)
    #     print(out_packed.data.shape, hidden.shape)
    #     output_padded, output_lengths = pad_packed_sequence(out_packed, batch_first=True)
    #     print(output_padded.shape)
    #     break
