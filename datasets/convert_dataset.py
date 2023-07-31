""" Convert a folder of gnn models to a single numpy file with blue observations """

import os
import numpy as np
# path = "/nethome/sye40/PrisonerEscape/datasets/test/gnn_map_0_run_100_eps_0.1_Normal"
# save_path = "/nethome/sye40/PrisonerEscape/datasets/post_red_obs_fix/map_0_100_gnn_save.npz"

# path="/nethome/sye40/PrisonerEscape/datasets/train/gnn_map_0_run_300_eps_0.1_Normal"
# save_path="/nethome/sye40/PrisonerEscape/datasets/post_red_obs_fix/map_0_300_gnn_save.npz" 

path = "/data/manisha/datasets/fixed_policy/corner_start_fixed_hideouts/train/"
save_path = "/data/manisha/datasets/fixed_policy/corner_start_fixed_hideouts/train_gnn_save.npz"

# get all files in the folder
files = os.listdir(path)
# print(files)

blue_obs = []
red_locs = []
dones = []

for file in files:
    arr = np.load(os.path.join(path, file), allow_pickle=True)
    blue_ob = arr["blue_observations"]
    red_loc = arr["red_locations"]/2428
    done = arr["dones"]

    print(done)

    dones.append(done)
    blue_obs.append(blue_ob)
    red_locs.append(red_loc)

blue_obs = np.concatenate(blue_obs)
red_locs = np.concatenate(red_locs)
dones = np.concatenate(dones)

print(blue_obs.shape, red_locs.shape, dones.shape)

np.savez(save_path,
            blue_observations = blue_obs,
            red_locations=red_locs, 
            dones=dones,
            agent_dict = arr["agent_dict"])