from dynamic_models import KinematicModel
from all_solvers import DistributedMPCLIPOPT
from utilits import *
import numpy as np

mpc_config = DistributedMPCLIPOPT.default_config
mpc_config['run_iter'] = 5
mpc_config['safe_factor'] = 3.5
mpc_config['safe_th'] = 1.8
mpc_config['pred_len'] = 30
mpc_config['other_veh_num'] = 8
mpc_config['comfort'] = 2.5
mpc_config['warm_start'] = False
mpc_config['Qu'] = 0.3 * np.diag([1.0, 0.8])
mpc_config["sensing_distance"] = 50
mpc_config['kernel'] = 'ipopt'

KinematicModel.initialize(KinematicModel.default_config)
DistributedMPCLIPOPT.initialize(DistributedMPCLIPOPT.default_config)


# load traj
# log_file = np.load("output_dir/traj_log/cross_double_3_round")

# trajs, step_num, traj_info, map_info = multi_cross(_points=8)
trajs, _, info_round, map_info = cross_traj_double_lane_2(
    _run_time=15.0,
    _round_distance=12,
    _road_width=8.5
)
# np.save('output_dir/traj_log/cross_double_3_round', info_round)
for car_id_, traj in enumerate(trajs):
    DistributedMPCLIPOPT(traj[0], traj, car_id_)

all_info = DistributedMPCLIPOPT.simulate()

# gen_video_from_info(all_info, trajs, draw_nominal=False, _map_info=map_info,
#                     _custom_lim=((-45, 45), (-45, 45))
#                     )
nlp_solve_info = NLP_RESULT_INFO.extract_info_from_info_all(all_info)
PickleSave(nlp_solve_info, "output_dir/solve_info/lnlp_solve_info_12")
PickleSave(all_info, "output_dir/solve_info/lnlp_all_info_12")
pass
