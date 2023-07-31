# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import random

path = "/nethome/sye40/PrisonerEscape/datasets/post_red_obs_fix/map_0_run_300_heuristic_eps_0.1.npz"
title = "Heuristic State Occupancy"
save_location = "evaluate/figs/heuristic_occupancy.png"

path = "/nethome/sye40/PrisonerEscape/datasets/post_red_obs_fix/map_0_400_RRT_gnn_save.npz"
title = "RRT* State Occupancy"
save_location = "evaluate/figs/rrt_occupancy.png"

a = np.load(path)
red_locs =  a["red_locations"]
x = red_locs[:, 0]
y = red_locs[:, 1]
fig = plt.subplots(figsize =(10, 7))
# Creating plot
plt.hexbin(x, y, bins=30)
plt.title(title)
  
plt.savefig(save_location)

# show plot
# plt.show()