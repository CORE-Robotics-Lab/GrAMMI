""" Convert a folder of gnn models to a single numpy file with blue observations """

import os
import numpy as np

path = "/nethome/sye40/PrisonerEscape/datasets/post_red_obs_fix/map_0_run_300_rrt.npz"

datas = np.load(path, allow_pickle=True)

res = np.concatenate((datas["blue_observations"], datas["red_locations"]), axis=1)

print(datas["blue_observations"].shape)
print(res.shape)
np.savetxt("datasets/start.csv", res, delimiter=",")