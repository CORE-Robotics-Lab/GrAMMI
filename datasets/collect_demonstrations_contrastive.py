""" This script collects a triple of agent observations, hideout location, and timestep
and the list of last k timesteps when the fugitive was, and was not observed for computing the contrastive loss.

Currently looking at vector implementation (not graph dataset collection.
"""
import sys
sys.path.append('/nethome/mnatarajan30/codes/PrisonerEscape/')

import gym
import os
import numpy as np
import torch
import pickle
import random
from tqdm import tqdm
from simulator.prisoner_env import PrisonerBothEnv
from simulator.terrain import Terrain, TerrainType
from fugitive_policies.heuristic import HeuristicPolicy
from simulator.gnn_wrapper import PrisonerGNNEnv
from blue_policies.heuristic import BlueHeuristic

from simulator import initialize_prisoner_environment
import argparse


def collect_demonstrations(epsilon, num_runs,
                           starting_seed,
                           random_cameras,
                           folder_name,
                           heuristic_type,
                           show=False):
    """ Collect demonstrations for the homogeneous gnn where we assume all agents are the same.
    :param env: Environment to collect demonstrations from
    :param policy: Policy to use for demonstration collection
    :param repeat_stops: Number of times to repeat the last demonstration

    """

    if heuristic_type == "Normal":
        path = f"datasets/{folder_name}/contrastive_vector_map_{map_num}_run_{num_runs}_eps_{epsilon}_{heuristic_type}"
    else:
        path = f"datasets/{folder_name}/contrastive_vector_map_{map_num}_run_{num_runs}_{heuristic_type}"

    if random_cameras:
        path += "_random_cameras"

    if not os.path.exists(path):
        os.makedirs(path)

    global_timestep = 0
    for seed in tqdm(range(starting_seed, starting_seed + num_runs)):
        # print(len(buffer))
        red_observations = []
        blue_observations = []
        detected_locations = []
        agent_observations = []
        hideout_observations = []
        timestep_observations = []
        positive_observation_timesteps_list = []
        negative_observation_timesteps_list = []
        red_locations = []
        dones = []

        num_random_known_cameras = random.randint(25, 30)
        num_random_unknown_cameras = random.randint(25, 30)

        env = initialize_prisoner_environment(map_num,
                                              observation_step_type=observation_step_type,
                                              epsilon=epsilon,
                                              random_cameras=random_cameras,
                                              num_random_known_cameras=num_random_known_cameras,
                                              num_random_unknown_cameras=num_random_unknown_cameras,
                                              seed=seed,
                                              heuristic_type=heuristic_type,
                                              store_last_k_fugitive_detections=False)

        env = PrisonerGNNEnv(env)
        policy = BlueHeuristic(env, debug=False)

        gnn_obs, blue_obs = env.reset()
        policy.reset()
        blue_obs_dict = env.blue_obs_names._idx_dict
        prediction_obs_dict = env.prediction_obs_names._idx_dict
        policy.init_behavior()
        red_obs_names = env.prediction_obs_names
        positive_observation_timesteps = [-1, -1, -1, -1, -1, -1, -1, -1]
        negative_observation_timesteps = [-1, -1, -1, -1, -1, -1, -1, -1]

        done = False
        while not done:
            blue_actions = policy.predict(blue_obs)
            gnn_obs, blue_obs, reward, done, _ = env.step(blue_actions)

            prisoner_location = env.get_prisoner_location()
            detected_location = blue_obs[-2:]
            # blue_observation = env.get_blue_observation()
            red_observation = env.get_prediction_observation()
            wrapped_red = red_obs_names(red_observation)

            if env.is_detected:
                positive_observation_timesteps.pop(0)
                positive_observation_timesteps.append(global_timestep)  # Append current timestep

            else:
                negative_observation_timesteps.pop(0)
                negative_observation_timesteps.append(global_timestep)

            positive_observation_timesteps_list.append(np.array(positive_observation_timesteps))
            negative_observation_timesteps_list.append(np.array(negative_observation_timesteps))

            blue_observations.append(blue_obs)
            agent_observations.append(gnn_obs[0])
            hideout_observations.append(gnn_obs[1])
            timestep_observations.append(gnn_obs[2])
            red_observations.append(red_observation)
            red_locations.append(prisoner_location)
            dones.append(done)
            detected_locations.append(detected_location)

            global_timestep += 1

            if show:
                env.render('heuristic', show=True, fast=True)

            if done:
                if env.unwrapped.timesteps >= 2000:
                    print(f"Got stuck, {env.unwrapped.timesteps}")
                break

        # red_observations = np.stack(red_observations)
        # blue_observations = np.stack(blue_observations)
        # positive_observation_timesteps_list = np.stack(positive_observation_timesteps_list)
        # negative_observation_timesteps_list = np.stack(negative_observation_timesteps_list)
        # detected_locations = np.stack(detected_locations)

        red_locations = np.stack(red_locations) / 2428

        agent_dict = {"num_known_cameras": env.num_known_cameras,
                      "num_unknown_cameras": env.num_unknown_cameras,
                      "num_helicopters": env.num_helicopters,
                      "num_search_parties": env.num_search_parties}

        np.savez(path + f"/seed_{seed}_known_{env.num_known_cameras}_unknown_{env.num_unknown_cameras}.npz",
                 blue_observations=blue_observations,
                 red_observations=red_observations,
                 positive_observation_timesteps=positive_observation_timesteps_list,
                 negative_observation_timesteps=negative_observation_timesteps_list,
                 detected_locations=detected_locations,
                 red_locations=red_locations,
                 dones=dones,
                 prediction_dict=prediction_obs_dict,
                 blue_dict=blue_obs_dict,
                 agent_observations=agent_observations,
                 hideout_observations=hideout_observations,
                 timestep_observations=timestep_observations,
                 agent_dict=agent_dict
                 )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_num', type=int, default=0, help='Environment to use')

    args = parser.parse_args()
    map_num = args.map_num

    starting_seed = 0; num_runs = 300; epsilon = 0.1; folder_name = "train_same_new"

    random_cameras = False
    observation_step_type = "Blue"
    # camera_configuration = "/nethome/sye40/PrisonerEscape/simulator/camera_locations/reduced.txt"

    heuristic_type = "RRT"

    # blue_obs_dict = env.blue_obs_names._idx_dict
    # prediction_obs_dict = env.prediction_obs_names._idx_dict

    print("Collecting Training dataset....")
    collect_demonstrations(epsilon=epsilon,
                           heuristic_type=heuristic_type,
                           random_cameras=random_cameras,
                           folder_name=folder_name,
                           starting_seed=starting_seed,
                           num_runs=num_runs,
                           show=False)

    print("------------------------------------------")
    print("------------------------------------------")
    starting_seed = 500; num_runs = 100; epsilon = 0.1; folder_name = "test_same_new"
    print("Collecting Test dataset....")
    collect_demonstrations(epsilon=epsilon,
                           heuristic_type=heuristic_type,
                           random_cameras=random_cameras,
                           folder_name=folder_name,
                           starting_seed=starting_seed,
                           num_runs=num_runs,
                           show=False)