from metadrive.scenario import utils as sd_utils
import os
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from my_scenario_env import MyScenarioEnv

dataset_path = "../exp_waymo"
# print(f"Reading the summary file from Waymo data at: {dataset_path}")
# waymo_dataset_summary = sd_utils.read_dataset_summary(dataset_path)
# print(f"The dataset summary is a {type(waymo_dataset_summary)}, with lengths {len(waymo_dataset_summary)}.")
#
# waymo_scenario_summary, waymo_scenario_ids, waymo_scenario_files = waymo_dataset_summary
#
# # Just pick the first scenario
# scenario_pkl_file = waymo_scenario_ids[0]
#
# # Get the relative path to the .pkl file
# print("The pkl file relative path: ", waymo_scenario_files[scenario_pkl_file])  # An empty path
#
# # Get the absolute path to the .pkl file
# abs_path_to_pkl_file = os.path.join(dataset_path, waymo_scenario_files[scenario_pkl_file], scenario_pkl_file)
# print("The pkl file absolute path: ", abs_path_to_pkl_file)
#
# # Call utility function in MD and get the Scenario Description object
# scenario = sd_utils.read_scenario_data(abs_path_to_pkl_file)
#
# print(f"\nThe raw data type after reading the .pkl file is {type(scenario)}")

env = MyScenarioEnv(
    dict(
        physics_world_step_size=1e-1,
        decision_repeat=1,
    data_directory=os.path.abspath("."),
    agent_policy=ReplayEgoCarPolicy,
    use_render=True,
    log_level=50,
    # ===== Main Camera =====
    # A True value makes the camera follow the reference line instead of the vehicle, making its movement smooth
    use_chase_camera_follow_lane=True,
    # Height of the main camera
    camera_height=2.2,
    # Distance between the camera and the vehicle. It is the distance projecting to the x-y plane.
    camera_dist=7.5,
    # Pitch of main camera. If None, this will be automatically calculated
    camera_pitch=10,  # degree
    # Smooth the camera movement
    camera_smooth=True,
    # How many frames used to smooth the camera
    camera_smooth_buffer_size=20,
    # FOV of main camera
    camera_fov=65,
    # Setting the camera position for the Top-down Camera for 3D viewer (pressing key "B" to activate it)
    top_down_camera_initial_x=0,
    top_down_camera_initial_y=0,
    top_down_camera_initial_z=50,
    )
)
try:
    env.reset(seed=0)
    for t in range(10000):
        o, r, tm, tc, info = env.step([0, 0])
        env.render(
                   window=True,
                   screen_record=False,
                   screen_size=(700, 700))
        if info["replay_done"]:
            break
    # env.top_down_renderer.generate_gif()
finally:
    env.close()

