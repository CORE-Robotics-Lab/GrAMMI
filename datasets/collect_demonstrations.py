""" This script collects a triple of blue observations, red observations, and red locations. """

import gym
import numpy as np
import torch
import pickle

import os
import sys
sys.path.append(os.getcwd())

from tqdm import tqdm
from simulator.prisoner_env import PrisonerBothEnv
from simulator.terrain import Terrain, TerrainType
from fugitive_policies.heuristic import HeuristicPolicy
from fugitive_policies.rrt_star_adversarial_heuristic import RRTStarAdversarial
from fugitive_policies.rrt_star_adversarial_avoid import RRTStarAdversarialAvoid

from blue_policies.heuristic import BlueHeuristic
from simulator.prisoner_perspective_envs import PrisonerEnv
from simulator.load_environment import load_environment

# from simulator import initialize_prisoner_environment
import argparse

def collect_demonstrations(env, policy, num_runs, show=False):
    """ Collect demonstrations for multi-step prediction. 
    :param env: Environment to collect demonstrations from
    :param policy: Policy to use for demonstration collection
    :param repeat_stops: Number of times to repeat the last demonstration
    
    """        
    red_observations = []
    blue_observations = []
    red_locations = []
    dones = []
    

    for _ in tqdm(range(num_runs)):
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
                if env.unwrapped.timesteps >= 2000:
                    print("Got stuck")
                observation = env.reset()
                break
    
    red_observations = np.stack(red_observations)
    blue_observations = np.stack(blue_observations)
    red_locations = np.stack(red_locations)/2428

    return red_observations, blue_observations, red_locations, dones

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_num', type=int, default=0, help='Environment to use')
    
    # seed = 0; num_runs = 300
    seed = 1; num_runs = 100
    args = parser.parse_args()
    map_num = args.map_num

    # epsilon = 0.1
    # env = initialize_prisoner_environment(map_num, seed=seed, heuristic_type="Normal")
    env = load_environment('simulator/configs/fixed_policy.yaml')
    blue_policy = BlueHeuristic(env, debug=False)
    env = PrisonerEnv(env, blue_policy)

    heuristic = HeuristicPolicy(env, epsilon=0)
    # heuristic = RRTStarAdversarialAvoid(env, max_speed=7.5, n_iter=2000)

    print(f"Collecting data for {map_num} with training shape {env.prediction_observation_space.shape}")
    blue_obs_dict = env.blue_obs_names._idx_dict
    prediction_obs_dict = env.prediction_obs_names._idx_dict

    red_observations, blue_observations, red_locations, dones = \
        collect_demonstrations(env, heuristic, num_runs=num_runs, show=True)
    assert np.round(blue_observations.max()) <= 1
    assert np.round(blue_observations.min()) >= -1

    assert np.round(red_observations.max()) <= 1
    assert np.round(red_observations.min()) >= -1

    np.savez(f"datasets/random_start_locations/map_{map_num}_run_{num_runs}_rrt.npz",
                red_observations=red_observations, 
                blue_observations=blue_observations, 
                red_locations=red_locations, 
                dones=dones,
                prediction_dict = prediction_obs_dict,
                blue_dict = blue_obs_dict
                )
