import pandas as pd
import numpy as np

blue_obs_dict_sample = "/data/prisoner_datasets/balance_game/AStar/train/seed_400_known_44_unknown_33.npz"
data = np.load(blue_obs_dict_sample, allow_pickle=True)
blue_obs_dict = data['blue_obs_dict'].item()

# seed = 91

data_path = "/data/contrastive_learning_datasets/balance_game_astar/fixed_cams_astar_include_camera_at_start/gnn_map_0_run_100_AStar/seed_594_known_44_unknown_33.npz"

data = np.load(data_path, allow_pickle=True)


def get_dataframe(data):
    blue_observations = data['blue_observations']
    red_locations = data['red_locations'] / 2428
    # blue_obs_dict = data['blue_obs_dict'].item()

    names = [""] * blue_observations.shape[1]

    for key in blue_obs_dict:
        start_key, end_key = blue_obs_dict[key]
        if end_key - start_key == 2:
            names[start_key:end_key] = [key + "_x", key + "_y"]
        else:
            names[start_key:end_key] = [key] * (end_key - start_key)

    combined_observations = np.concatenate([blue_observations, red_locations], axis=1)
    names.append("prisoner_loc_x")
    names.append("prisoner_loc_y")

    df = pd.DataFrame(combined_observations, columns=names)
    return df

df = get_dataframe(data)
df.to_csv(f"datasets/test.csv")