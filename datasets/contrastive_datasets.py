"""
Datasets for contrastive models
"""
import copy

import numpy as np
import os
from torch.utils.data import DataLoader
import torch
from datasets.dataset import BasePrisonerDataset

# Contrastive Datasets with different combinations of anchor, positive and negative observations
class ContrastiveGNNDataset(BasePrisonerDataset):
    """
    Generic Class to load dataset for different Contrastive Learning methods
    """
    def __init__(self, folder_path, sequence_length, num_heads, step_length, include_current, multi_head,
                 one_hot=False, timestep=False, detected_location=False, get_last_k_fugitive_detections=False,
                 num_contrastive_samples=10, contrastive_sample_type="regular", multiple_negative_samples=False):
        """

        :param folder_path:
        :param sequence_length:
        :param num_heads:
        :param step_length:
        :param include_current:
        :param multi_head:
        :param one_hot:
        :param timestep:
        :param detected_location:
        :param get_last_k_fugitive_detections:
        :param num_contrastive_samples:
        :param contrastive_sample_type: "regular", "future_red_locs","none"
        :param multiple_negative_samples: If true, for each anchor get multiple negative samples
        """

        self.graphs = []
        self.max_agent_size = 0
        self.process_first_graph = True
        self.agent_dict = {}
        self.one_hot_bool = one_hot
        self.timestep_bool = timestep
        self.detected_location_bool = detected_location
        self.last_k_fugtitive_detection_bool = False  # TODO
        if multiple_negative_samples:
            self.num_contrastive_samples = num_contrastive_samples
        else:
            self.num_contrastive_samples = 0
        self.contrastive_sample_type = contrastive_sample_type
        self.positive_observation_timesteps = None
        self.negative_observation_timesteps = None

        print("Number of negative samples: ", self.num_contrastive_samples)
        super().__init__(folder_path, sequence_length, num_heads, step_length, include_current, multi_head)

    def _load_data(self, folder_path):
        starting_seed = int(sorted(os.listdir(folder_path))[0].split('_')[1])
        num_files = len(os.listdir(folder_path))
        np_files = []
        for file_name in sorted(os.listdir(folder_path)):
            np_file = np.load(os.path.join(folder_path, file_name), allow_pickle=True)
            self.max_agent_size = max(self.max_agent_size, np.squeeze(np_file["agent_observations"]).shape[1])
            np_files.append(np_file)

        for np_file in np_files:
            self._load_graph(np_file)

        print("Completed loading data...")

    def _load_graph(self, file):
        agent_obs = file["agent_observations"]
        agent_obs = np.squeeze(agent_obs)
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

        self.agent_dict = file["agent_dict"].item()

        num_timesteps = agent_obs.shape[0]
        one_hots = self._create_one_hot_agents(num_timesteps)
        one_hots = np.pad(one_hots, ((0, 0), (0, self.max_agent_size - num_agents), (0, 0)), 'constant')

        if self.process_first_graph:
            self.num_agents = [num_agents] * agent_obs.shape[0]
            self.agent_obs = agent_obs
            self.hideouts = file["hideout_observations"]
            self.timesteps = timesteps
            self.dones = file["dones"]
            self.red_locs = file["red_locations"]/2428
            self.detected_location = detected_location
            self.one_hots = one_hots
            self.process_first_graph = False
            # Store the locations of timestep where the fugitive was (+ve) and wasn't (-ve) observed
            if "positive_observation_timesteps" in file and "negative_observation_timesteps" in file:
                self.positive_observation_timesteps = file["positive_observation_timesteps"]
                self.negative_observation_timesteps = file["negative_observation_timesteps"]

        else:
            self.num_agents.extend([num_agents] * agent_obs.shape[0])
            self.agent_obs = np.concatenate((self.agent_obs, agent_obs), 0)
            self.hideouts = np.append(self.hideouts, file["hideout_observations"], 0)
            self.red_locs = np.append(self.red_locs, file["red_locations"]/2428, 0)
            self.timesteps = np.append(self.timesteps, timesteps)
            self.detected_location = np.append(self.detected_location, detected_location)
            self.one_hots = np.concatenate((self.one_hots, one_hots), 0)
            self.dones = np.append(self.dones, file["dones"])

            if "positive_observation_timesteps" in file and "negative_observation_timesteps" in file:
                self.positive_observation_timesteps = np.concatenate(
                    (self.positive_observation_timesteps, file["positive_observation_timesteps"]), 0)
                self.negative_observation_timesteps = np.concatenate(
                    (self.negative_observation_timesteps, file["negative_observation_timesteps"]), 0)

    def _create_one_hot_agents(self, timesteps):
        one_hot_base = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        agents = [self.agent_dict["num_known_cameras"] + self.agent_dict["num_unknown_cameras"],
                  self.agent_dict["num_helicopters"], self.agent_dict["num_search_parties"]]
        a = np.repeat(one_hot_base, agents, axis=0)  # produce [num_agents, 3] (3 for each in one-hot)
        # one_hot = np.repeat(np.expand_dims(a, 0), self.sequence_length, axis=0) # produce [seq_len, num_agents, 3]
        one_hot = np.repeat(np.expand_dims(a, 0), timesteps, axis=0)  # produce [timesteps, num_agents, 3]
        return one_hot

    def _produce_input(self, idx):
        # First episode does not have reset marker
        if idx < self.done_locations[0] + 1:
            episode_start_idx = 0
        else:
            # Get index of the episode's start
            episode_start_idx = self.done_locations[np.where(self.done_locations <= idx - 1)[0][-1]] + 1

        assert idx >= episode_start_idx

        # Get anchor sample
        if idx - episode_start_idx >= self.sequence_length:
            anchor_sequence = self.agent_obs[idx - self.sequence_length:idx]
            if self.one_hot_bool:
                anchor_sequence = np.concatenate((anchor_sequence, self.one_hots[idx - self.sequence_length:idx]), axis=2)
        else:
            anchor_sequence = self._process_start_observations(self.agent_obs, idx, episode_start_idx)
            if self.one_hot_bool:
                anchor_sequence = np.concatenate((anchor_sequence, self._process_start_observations(self.one_hots, idx, episode_start_idx)), axis=2)

        if self.contrastive_sample_type == "None":
            sample = [anchor_sequence,
                      self.hideouts[idx], np.expand_dims(self.timesteps[idx], 0), np.expand_dims(self.num_agents[idx], 0)]

            if not self.multi_head and self.num_heads > 1:
                # Consider it as auto-regressive output
                # Include detected location and target locs for autoregressive decoder
                targets = self._produce_multi_step_output(idx)
                if self.detected_location[idx] == -1:
                    decoder_input = np.array([-1.0, -1.0])
                else:
                    decoder_input = self.red_locs[idx]

                sample.append(decoder_input)
                sample.append(targets)

            return sample

        elif self.num_contrastive_samples > 1:
            # Get positive, negative and anchor. Use detections at random timesteps as the negative samples
            if self.positive_observation_timesteps:
                raise Exception("Cannot find detected timesteps for contrastive learning with random negatives")
            else:
                positive_gnn_sequence = self._get_sequence_of_detections(self.agent_obs, idx, episode_start_idx)
                negative_gnn_sequence = self._get_random_detected_timesteps(self.agent_obs, idx)
                sample = [anchor_sequence, positive_gnn_sequence, negative_gnn_sequence,
                          self.hideouts[idx], np.expand_dims(self.timesteps[idx], 0),
                          np.expand_dims(self.num_agents[idx], 0)]

                if not self.multi_head and self.num_heads > 1:
                    # Consider it as auto-regressive output
                    # Include detected location and target locs for autoregressive decoder
                    targets = self._produce_multi_step_output(idx)
                    if self.detected_location[idx] == -1:
                        decoder_input = np.array([-1.0, -1.0])
                    else:
                        decoder_input = self.red_locs[idx]

                    sample.append(decoder_input)
                    sample.append(targets)

                return sample

        elif self.contrastive_sample_type == "future_red_locs":
            # Use current blue as anchor and future red loc as augmented samples
            positive_gnn_sequence = self._get_next_red_states(idx)
            sample = [anchor_sequence, positive_gnn_sequence, self.hideouts[idx], np.expand_dims(self.timesteps[idx], 0),
                      np.expand_dims(self.num_agents[idx], 0)]

            if not self.multi_head and self.num_heads > 1:
                # Consider it as auto-regressive output
                # Include detected location and target locs for autoregressive decoder
                targets = self._produce_multi_step_output(idx)
                if self.detected_location[idx] == -1:
                    decoder_input = np.array([-1.0, -1.0])
                else:
                    decoder_input = self.red_locs[idx]

                sample.append(decoder_input)
                sample.append(targets)

            return sample

        elif self.contrastive_sample_type == "last_k_detects":
            pass

        elif self.contrastive_sample_type == "regular":
            # Use detected samples as anchor and non-detected samples as augmented
            if self.positive_observation_timesteps is None or self.negative_observation_timesteps is None:
                raise Exception("Cannot find detected and undetected timesteps for contrastive learning")
            else:
                positive_gnn_sequence = self._get_sequence_of_detections(self.agent_obs, idx, episode_start_idx)
                negative_gnn_sequence = self._get_sequence_of_non_detections(self.agent_obs, idx, episode_start_idx)

                if self.one_hot_bool:
                    if idx - episode_start_idx >= self.sequence_length:
                        positive_gnn_sequence = np.concatenate(
                            (positive_gnn_sequence, self.one_hots[idx - self.sequence_length:idx]),
                            axis=2)

                        negative_gnn_sequence = np.concatenate(
                            (negative_gnn_sequence, self.one_hots[idx - self.sequence_length:idx]),
                            axis=2)

                    else:
                        positive_gnn_sequence = np.concatenate(
                            (positive_gnn_sequence,
                             self._process_start_observations(self.one_hots, idx, episode_start_idx)), axis=2)

                        negative_gnn_sequence = np.concatenate(
                            (negative_gnn_sequence,
                             self._process_start_observations(self.one_hots, idx, episode_start_idx)),
                            axis=2)

                # Needed to calculate delta t for contrastive loss
                last_positive_observation = self.positive_observation_timesteps[idx][-1]
                last_negative_observation = self.negative_observation_timesteps[idx][-1]

                time_diff = np.abs(last_positive_observation - last_negative_observation)

                sample = [positive_gnn_sequence, negative_gnn_sequence,
                          self.hideouts[idx], np.expand_dims(self.timesteps[idx], 0),
                          np.expand_dims(self.num_agents[idx], 0),
                          time_diff]

                if not self.multi_head and self.num_heads > 1:
                    # Consider it as auto-regressive output
                    # Include detected location and target locs for autoregressive decoder
                    targets = self._produce_multi_step_output(idx)
                    if self.detected_location[idx] == -1:
                        decoder_input = np.array([-1.0, -1.0])
                    else:
                        decoder_input = self.red_locs[idx]

                    sample.append(decoder_input)
                    sample.append(targets)

                return sample

        else:
            raise Exception("Incorrect arguments for getting contrastive samples")


    def _get_random_detected_timesteps(self, np_array, idx):
        """ If we're indexing at the start of an episode, need to pad the start with zeros
            Return only negative examples for the current anchor!
        """

        negative_gnn_sequence = []

        last_obs = np_array[idx]
        # shape = (self.sequence_length - (idx - episode_start_idx + 1),) + last_obs.shape
        empty_sequence = np.zeros_like(last_obs)

        # Get contrastive samples (detected fugitive locs) from other timesteps
        total_steps = self.agent_obs.shape[0]

        contrastive_sequence_timesteps = []

        current_fugitive_location = self.red_locs[idx]
        delta = 250 / 2428

        k = 0
        detected_locs = self.detected_location.reshape(-1, 2)
        detected_timesteps = np.where(detected_locs[:, 0] != -1)

        while k < self.num_contrastive_samples:
            contrastive_t = np.random.choice(detected_timesteps[0], size=1)[0]
            while np.linalg.norm(self.red_locs[contrastive_t] - current_fugitive_location) < delta:
                contrastive_t = np.random.randint(total_steps)
            contrastive_sequence_timesteps.append(contrastive_t)
            k += 1

        # Get sequence of contrastive samples
        contrastive_sequence = []

        for t in contrastive_sequence_timesteps:
            # t = t[0]
            if t < self.done_locations[0] + 1:
                contrastive_episode_start_idx = 0

            else:
                # Get index of the episode's start
                contrastive_episode_start_idx = self.done_locations[np.where(self.done_locations <= t - 1)[0][-1]] + 1
            assert t >= contrastive_episode_start_idx

            if t - contrastive_episode_start_idx >= self.sequence_length:
                if len(contrastive_sequence) == 0:
                    contrastive_sequence = np_array[t - self.sequence_length:t]
                else:
                    contrastive_sequence = np.concatenate((contrastive_sequence, np_array[t - self.sequence_length:t]))
            else:
                for _ in range(self.sequence_length - (t - contrastive_episode_start_idx)):
                    if len(contrastive_sequence) == 0:
                        contrastive_sequence = empty_sequence[np.newaxis, :]
                    else:
                        contrastive_sequence = np.vstack((contrastive_sequence, empty_sequence[np.newaxis, :]))
                diff = t - contrastive_episode_start_idx
                contrastive_sequence = np.vstack((contrastive_sequence, np_array[contrastive_episode_start_idx:t]))

        contrastive_sequence = np.array(contrastive_sequence)
        contrastive_sequence = contrastive_sequence.reshape(self.num_contrastive_samples, self.sequence_length, -1,
                                                            last_obs.shape[-1])
        negative_gnn_sequence = np.array(contrastive_sequence)
        return negative_gnn_sequence

    def _get_next_red_states(self, idx):
        """
        Get the sequence of next red states (currently used in CPC)
        :param idx: sample index
        :param episode_start_idx:
        :return:
        """
        num_heads = 11
        step_length = 5
        future_step = 11 * 5  # step length * num_heads
        target_red_loc_idx = idx + future_step

        # Episode terminating index
        episode_end_idx = self.done_locations[np.where(self.done_locations >= idx)[0][0]]

        if target_red_loc_idx > episode_end_idx:
            # if we are at the end of the episode, just use the last location
            if idx + self.step_length > episode_end_idx:
                end_loc = np.expand_dims(self.red_locs[episode_end_idx], axis=0)
                red_sequence = np.repeat(end_loc, num_heads+1, axis=0)
            else:
                # if there are steps between the current and end of the episode
                begin_locs = self.red_locs[idx:episode_end_idx:step_length]
                # unsqueeze a dimension in numpy
                end_loc = np.expand_dims(self.red_locs[episode_end_idx], axis=0)
                end_loc = np.repeat(end_loc, num_heads + 1 - len(begin_locs), axis=0)
                red_sequence = np.concatenate((begin_locs, end_loc))
        else:
            red_sequence = self.red_locs[idx:target_red_loc_idx+step_length:step_length]


        # ------------------------------------------ For short horizon ---------------------------------------------- #
        # # Episode terminating index
        # episode_end_idx = self.done_locations[np.where(self.done_locations >= idx)[0][0]]
        #
        # red_sequence = []
        # # Get current and future red locations
        # if idx + self.red_sequence_length <= episode_end_idx:
        #     red_sequence = self.red_locs[idx:idx + self.red_sequence_length]
        #
        # else:
        #     last_obs = self.red_locs[idx]
        #     shape = (idx + self.red_sequence_length - (episode_end_idx + 1),) + last_obs.shape
        #     empty_sequences = np.zeros(shape)
        #     red_sequence = self.red_locs[idx:episode_end_idx + 1]
        #     red_sequence = np.concatenate((empty_sequences, red_sequence), axis=0)
        # ----------------------------------------------------------------------------------------------------------- #
        return red_sequence

    def _get_sequence_of_detections(self, np_array, idx, episode_start_idx=0):
        """ If we're indexing at the start of an episode, need to pad the start with zeros
            Return only positive examples (t+1... t+seq_len+1) for the current anchor!
        """
        positive_gnn_sequence = []

        last_obs = np_array[idx]
        # shape = (self.sequence_length - (idx - episode_start_idx + 1),) + last_obs.shape
        empty_sequence = np.zeros_like(last_obs)

        # TODO: @Manisha Optimize this by using numpy instead of for loop
        for t in self.positive_observation_timesteps[idx]:
            if t == -1 or t < episode_start_idx:
                positive_gnn_sequence.append(empty_sequence)
            else:
                positive_gnn_sequence.append(np_array[t])

        positive_gnn_sequence = np.array(positive_gnn_sequence)

        return positive_gnn_sequence

    def _get_sequence_of_non_detections(self, np_array, idx, episode_start_idx=0):
        """ If we're indexing at the start of an episode, need to pad the start with zeros
            Return only negative examples i.e., samples where the fugitive was
            not detected for the current anchor!
        """
        negative_gnn_sequence = []

        last_obs = np_array[idx]
        # shape = (self.sequence_length - (idx - episode_start_idx + 1),) + last_obs.shape
        empty_sequence = np.zeros_like(last_obs)

        # TODO: @Manisha Optimize this by using numpy instead of for loop
        for t in self.negative_observation_timesteps[idx]:
            if t == -1 or t < episode_start_idx:
                negative_gnn_sequence.append(empty_sequence)
            else:
                negative_gnn_sequence.append(np_array[t])

        negative_gnn_sequence = np.array(negative_gnn_sequence)

        return negative_gnn_sequence


class SimCLRGNNDataset(ContrastiveGNNDataset):
    def __init__(self, folder_path, sequence_length, num_heads, step_length, include_current, multi_head,
                 one_hot=False, timestep=False, detected_location=False, get_last_k_fugitive_detections=False,
                 modify_timestep_flag=False, modify_camera_flag=False, modify_dynamic_agent_flag=False):
        """
        Create augmented data samples by forcing different timesteps / changing the locations of the blue agents
        to ensure that the fugitive is detected. The input function returns the original sample and the augmented data.
        To be used with the SimCLR architecture.
        :param folder_path: To load dataset from
        :param sequence_length:
        :param num_heads:
        :param step_length:
        :param include_current:
        :param multi_head:
        :param one_hot:
        :param timestep:
        :param detected_location:
        :param get_last_k_fugitive_detections:
        :param modify_timestep_flag: Flag for augmenting data
        :param modify_camera_flag: Flag for augmenting data
        :param modify_dynamic_agent_flag: Flag for augmenting data
        """

        super().__init__(folder_path, sequence_length, num_heads, step_length, include_current, multi_head,
                         one_hot, timestep, detected_location, get_last_k_fugitive_detections)

        # Data augmentation flags
        # modify timestep iff we're including the timestep info in the agent obs
        self.modify_timestep_flag = modify_timestep_flag and self.timestep_bool
        # modify blue agent locations to track the fugitive iff we're including the detected location in the agent obs
        self.modify_camera_flag = modify_camera_flag and self.detected_location_bool
        self.modify_dynamic_agent_flag = modify_dynamic_agent_flag and self.detected_location_bool

    # -------------------------- Methods for data augmentation -------------------------- #
    def modify_timesteps(self, data):
        """
        Take a sequence of blue observations and modify the timesteps to be earlier or later in the rollout
        :param data: sequence of blue observations [seq_len, num_agents, feat_dim]
        :return: data --> modified
        Only useful if timestep info is passed in to the GNN + LSTM encoder
        """
        feat_dim = data.shape[-1]
        num_agents = data.shape[1]
        max_timesteps = 4320
        if self.detected_location_bool:
            # detected location is appended
            timestep_idx = 5
        else:
            timestep_idx = 3

        original_timesteps = data[:, :, timestep_idx]

        timestep_start = original_timesteps[0, 0] * max_timesteps

        # Sample a timestep at random (within max_timesteps)
        new_timestep_start = np.random.choice(np.arange(0, max_timesteps - self.sequence_length))
        modified_timesteps = np.arange(new_timestep_start, new_timestep_start + self.sequence_length)[:, np.newaxis]

        # Repeat and Normalize the timesteps
        modified_timesteps = np.repeat(modified_timesteps, num_agents, axis=1) / max_timesteps

        data[:, :, timestep_idx] = modified_timesteps

        return data

    def modify_camera_locs(self, data, red_locs, camera_idx=-1):
        """
        Take a sequence of blue observations and modify a single camera (idx) to be at the location
        of the fugitive. Modify the detected location [seq_len, num_agents, feat_dim]
        :param data: sequence of blue observations
        :param red_locs: ground truth fugitive location sequence
        :return: data --> modified
        """
        feat_dim = data.shape[-1]
        num_agents = data.shape[1]
        camera_loc_idx = 1
        detected_loc_idx = 3
        # Assume that we are appending detected loc (otherwise, there's hardly any point in
        # simply changing the camera location)

        num_cameras = self.agent_dict["num_known_cameras"] + self.agent_dict["num_unknown_cameras"]

        if camera_idx == -1:
            # Sample a random camera index
            camera_idx = np.random.choice(np.arange(0, num_cameras))

        # Change camera_loc to the fugitive loc at the end of sequence
        fugitive_loc = red_locs[-1]  # Can take other timesteps in the seq as well

        # Each agent has [d, x, y]. Modify the location
        data[:, camera_idx, camera_loc_idx:camera_loc_idx+2] = fugitive_loc

        # Modify the detected location of the agent for the timestep, where the fugitive is
        # in the vicinity of camera detection
        data[-1, camera_idx, 0] = 1.0  # TODO: Should be one whenever the ground truth fugitive (red_loc) is in the vicinity of the camera
        data[-1, camera_idx, detected_loc_idx:detected_loc_idx+2] = fugitive_loc
        return data

    def modify_dynamic_agent_locs(self, data, red_locs, agent_idx=-1):
        """
        Take a sequence of blue observations and modify a single dynamic agent (search party or helo) (idx) to be at the location
        of the fugitive. Modify the detected location [seq_len, num_agents, feat_dim]
        :param data: sequence of blue observations
        :param red_locs: ground truth fugitive location sequence
        :return: data --> modified
        """
        feat_dim = data.shape[-1]
        num_agents = data.shape[1]
        agent_loc_idx = 1
        detected_loc_idx = 3
        # Assume that we are appending detected loc (otherwise, there's hardly any point in
        # simply changing the camera location)

        num_cameras = self.agent_dict["num_known_cameras"] + self.agent_dict["num_unknown_cameras"]
        num_dynamic_agents = self.agent_dict["num_helicopters"] + self.agent_dict["num_search_parties"]

        if agent_idx == -1:
            # Sample a random moving index
            agent_idx = np.random.choice(np.arange(0, num_dynamic_agents))

        # Make the agent continuously track the fugitive
        # Each agent has [d, x, y]. Modify the location
        data[:, num_agents - agent_idx - 1, agent_loc_idx:agent_loc_idx+2] = red_locs

        # Modify the detected location of the agent for the timestep, where the fugitive is
        # in the vicinity of agent detection
        # The dynamic agents are appended at the end...
        data[:, num_agents - agent_idx - 1, 0] = 1.0
        data[:, num_agents - agent_idx - 1, detected_loc_idx:detected_loc_idx + 2] = red_locs
        return data

    def modify_fugitive_locs(self, data, red_locs, agent_idx=-1):
        """
        Take a sequence of blue observations and modify the fugitive's location to be at the location
        of any dynamic agent. Modify the detected location.
        :param data: sequence of blue observations [seq_len, num_agents, feat_dim]
        :param red_locs: ground truth fugitive location sequence
        :return: data --> modified
        """
        return data

    def modify_num_agents(self, data):
        """
        TODO: Modify the number of agents in the env
        :param data: sequence of blue observations [batch_size, seq_len, num_agents, feat_dim]
        :return: data -> modified
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------------------- #
    def _produce_input(self, idx):
        """
        Overwrite the base class contrastive learning's loader
        Get augmented observation based on the data augmentation flags
        :param idx:
        :return:
        """

        gt_red_locs = []

        # First episode does not have reset marker
        if idx < self.done_locations[0] + 1:
            episode_start_idx = 0
        else:
            # Get index of the episode's start
            episode_start_idx = self.done_locations[np.where(self.done_locations <= idx - 1)[0][-1]] + 1
        assert idx >= episode_start_idx

        if idx - episode_start_idx >= self.sequence_length:
            agent_obs = self.agent_obs[idx - self.sequence_length:idx]
            gt_red_locs = self.red_locs[idx - self.sequence_length:idx]

            if self.one_hot_bool:
                # Produce for each agent in agent_obs, include one-hot of the agent type
                # Agents obs is [Seq_len x Num_agents x 3]
                # One hot should be [Seq_len x Num_agents x 3]
                # Agent obs is in order known cameras, unknown cameras, helicopters, search parties
                agent_obs = np.concatenate((agent_obs, self.one_hots[idx - self.sequence_length:idx]), axis=2)

        else:
            agent_obs = self._process_start_observations(self.agent_obs, idx, episode_start_idx)
            gt_red_locs = self._process_start_observations(self.red_locs, idx, episode_start_idx)
            if self.one_hot_bool:
                agent_obs = np.concatenate((agent_obs, self._process_start_observations(self.one_hots, idx, episode_start_idx)), axis=2)

        # Get augmented obs based on different data augmentations
        augmented_obs = copy.deepcopy(agent_obs)
        if self.modify_timestep_flag:
            augmented_obs = self.modify_timesteps(augmented_obs)

        if self.modify_camera_flag:
            augmented_obs = self.modify_camera_locs(augmented_obs, gt_red_locs)

        if self.modify_dynamic_agent_flag:
            augmented_obs = self.modify_dynamic_agent_locs(augmented_obs, gt_red_locs)

        sample = [agent_obs,
                  augmented_obs,
                  self.hideouts[idx],
                  np.expand_dims(self.timesteps[idx], 0),
                  np.expand_dims(self.num_agents[idx], 0)]

        if not self.multi_head and self.num_heads > 1:
            # Consider it as auto-regressive output
            # Include detected location and target locs for autoregressive decoder
            targets = self._produce_multi_step_output(idx)
            if self.detected_location[idx] == -1:
                decoder_input = np.array([-1.0, -1.0])
            else:
                decoder_input = self.red_locs[idx]

            sample.append(decoder_input)
            sample.append(targets)

        return sample