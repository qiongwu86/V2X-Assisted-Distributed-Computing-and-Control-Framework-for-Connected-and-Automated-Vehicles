from metadrive.scenario import utils as sd_utils
import os
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy

dataset_path = "../exp_waymo"
print(f"Reading the summary file from Waymo data at: {dataset_path}")
waymo_dataset_summary = sd_utils.read_dataset_summary(dataset_path)
print(f"The dataset summary is a {type(waymo_dataset_summary)}, with lengths {len(waymo_dataset_summary)}.")

waymo_scenario_summary, waymo_scenario_ids, waymo_scenario_files = waymo_dataset_summary

# Just pick the first scenario
scenario_pkl_file = waymo_scenario_ids[0]

# Get the relative path to the .pkl file
print("The pkl file relative path: ", waymo_scenario_files[scenario_pkl_file])  # An empty path

# Get the absolute path to the .pkl file
abs_path_to_pkl_file = os.path.join(dataset_path, waymo_scenario_files[scenario_pkl_file], scenario_pkl_file)
print("The pkl file absolute path: ", abs_path_to_pkl_file)

# Call utility function in MD and get the Scenario Description object
scenario = sd_utils.read_scenario_data(abs_path_to_pkl_file)

print(f"\nThe raw data type after reading the .pkl file is {type(scenario)}")
