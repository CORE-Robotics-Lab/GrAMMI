# Collect a single csv for a given run

""" This script collects a triple of blue observations, red observations, and red locations. """
import os
import sys
sys.path.append(os.getcwd())

import gym
import numpy as np
import torch
import pickle

from tqdm import tqdm
from simulator.prisoner_env import PrisonerBothEnv
from simulator.terrain import Terrain, TerrainType
from fugitive_policies.heuristic import HeuristicPolicy
from fugitive_policies.rrt_star_adversarial_avoid import RRTStarAdversarialAvoid

from simulator import initialize_prisoner_environment
import argparse
import pandas as pd

def collect_demonstrations(env, policy, show=False):
    """ Collect demonstrations for multi-step prediction. 
    :param env: Environment to collect demonstrations from
    :param policy: Policy to use for demonstration collection
    :param repeat_stops: Number of times to repeat the last demonstration
    
    """        
    red_observations = []
    blue_observations = []
    red_locations = []
    dones = []
    

    # for _ in tqdm(range(num_runs)):
        # print(len(buffer))
    observation = env.reset()
    red_obs_names = env.prediction_obs_names
    # print(red_obs_names._idx_dict['prisoner_loc'])
    done = False
    while not done:
        action = policy.predict(observation)[0]

        observation, reward, done, infos = env.step(action)
        prisoner_location = env.get_prisoner_location()
        blue_observation = env.get_blue_observation()
        
        red_observation = env.get_prediction_observation()
        wrapped_red = red_obs_names(red_observation)
        # print(wrapped_red['prisoner_loc'], np.array(prisoner_location)/2428)

        red_observations.append(red_observation)
        blue_observations.append(blue_observation)
        red_locations.append(prisoner_location)
        dones.append(done)
        if show:
            env.render('heuristic', show=True, fast=True)

        if done:
            break
    
    red_observations = np.stack(red_observations)
    blue_observations = np.stack(blue_observations)
    red_locations = np.stack(red_locations)/2428

    return red_observations, blue_observations, red_locations, dones

if __name__ == "__main__":

    seed = 4
    map_num = 0
    epsilon = 0.1
    env = initialize_prisoner_environment(map_num, seed=seed, heuristic_type="RRT")

    # heuristic = HeuristicPolicy(env, epsilon=epsilon)
    heuristic = RRTStarAdversarialAvoid(env, max_speed=7.5, n_iter=2000)

    print(f"Collecting data for {map_num} with training shape {env.prediction_observation_space.shape}")
    blue_obs_dict = env.blue_obs_names._idx_dict
    prediction_obs_dict = env.prediction_obs_names._idx_dict

    red_observations, blue_observations, red_locations, dones = collect_demonstrations(env, heuristic, show=True)
    np.savez(f"datasets/map_{map_num}_seed_{seed}_single_run.npz", 
                red_observations=red_observations, 
                blue_observations=blue_observations, 
                red_locations=red_locations, 
                dones=dones,
                prediction_dict = prediction_obs_dict,
                blue_dict = blue_obs_dict
                )
    print(blue_obs_dict)

    names = [""] * blue_observations.shape[1]

    for key in blue_obs_dict:
        start_key, end_key = blue_obs_dict[key]
        print(start_key, end_key)
        if end_key - start_key == 2:
            names[start_key:end_key] = [key + "_x", key + "_y"]
        else:
            names[start_key:end_key] = [key] * (end_key - start_key) 

    combined_observations = np.concatenate([blue_observations, red_locations], axis=1)
    names.append("prisoner_loc_x")
    names.append("prisoner_loc_y")

    print(names)
    df = pd.DataFrame(combined_observations, columns=names)

    
    df.to_csv(f"datasets/map_{map_num}_seed_{seed}.csv")

