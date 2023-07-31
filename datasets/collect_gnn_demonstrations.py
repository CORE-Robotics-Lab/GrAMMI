""" This script collects a triple of agent observations, hideout location, and timestep. """

import gym
import numpy as np
import torch
import pickle

from tqdm import tqdm
from simulator.prisoner_env import PrisonerBothEnv
from simulator.terrain import Terrain, TerrainType
from fugitive_policies.heuristic import HeuristicPolicy
from simulator.gnn_wrapper import PrisonerGNNEnv
from blue_policies.heuristic import BlueHeuristic

from simulator import initialize_prisoner_environment
import argparse

def collect_demonstrations(env, policy, num_runs, show=False):
    """ Collect demonstrations for the homogeneous gnn where we assume all agents are the same. 
    :param env: Environment to collect demonstrations from
    :param policy: Policy to use for demonstration collection
    :param repeat_stops: Number of times to repeat the last demonstration
    
    """        
    # red_observations = []
    agent_observations = []
    hideout_observations = []
    timestep_observations = []
    detected_locations = []

    red_locations = []
    dones = []
    

    for _ in tqdm(range(num_runs)):
        # print(len(buffer))
        gnn_obs, blue_obs = env.reset()
        policy.reset()
        policy.init_behavior()

        done = False
        while not done:
            blue_actions = policy.predict(blue_obs)
            gnn_obs, blue_obs, reward, done, _ = env.step(blue_actions)

            prisoner_location = env.get_prisoner_location()
            detected_location = blue_obs[-2:]

            agent_observations.append(gnn_obs[0])
            timestep_observations.append(gnn_obs[1])
            hideout_observations.append(gnn_obs[2])
            red_locations.append(prisoner_location)
            dones.append(done)
            detected_locations.append(detected_location)

            if show:
                env.render('heuristic', show=True, fast=True)

            if done:
                if env.unwrapped.timesteps >= 2000:
                    print(f"Got stuck, {env.unwrapped.timesteps}")
                break

    agent_observations = np.stack(agent_observations)
    hideout_observations = np.stack(hideout_observations)
    timestep_observations = np.stack(timestep_observations)
    detected_locations = np.stack(detected_locations)

    red_locations = np.stack(red_locations)/2428

    return agent_observations, hideout_observations, timestep_observations, detected_locations, red_locations, dones

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_num', type=int, default=0, help='Environment to use')
    
    args = parser.parse_args()
    map_num = args.map_num

    seed = 0; num_runs = 300
    # seed = 1; num_runs = 100
    epsilon = 0.1
    random_cameras=False
    observation_step_type = "Blue"
    # camera_configuration = "/nethome/sye40/PrisonerEscape/simulator/camera_locations/reduced.txt"
    env = initialize_prisoner_environment(map_num,
                                        observation_step_type = observation_step_type, 
                                        epsilon=epsilon, 
                                        random_cameras=random_cameras,
                                        seed=seed)
    env = PrisonerGNNEnv(env)

    gnn_obs, blue_obs = env.reset()
    blue_heuristic = BlueHeuristic(env, debug=False)

    print(f"Collecting data for {map_num} with training shape {env.prediction_observation_space.shape}")
    blue_obs_dict = env.blue_obs_names._idx_dict
    prediction_obs_dict = env.prediction_obs_names._idx_dict

    agent_observations, hideout_observations, timestep_observations, detected_locations, red_locations, dones \
        = collect_demonstrations(env, blue_heuristic, num_runs=num_runs, show=False)
    
    print(agent_observations.shape)
    path = f"datasets/gnn_map_{map_num}_run_{num_runs}_eps_{epsilon}_norm"
    if random_cameras:
        path += "_random_cameras"

    np.savez(path + ".npz", 
                agent_observations=agent_observations,
                hideout_observations=hideout_observations,
                timestep_observations=timestep_observations, 
                detected_locations = detected_locations,
                red_locations=red_locations, 
                dones=dones,
                prediction_dict = prediction_obs_dict,
                blue_dict = blue_obs_dict
                )