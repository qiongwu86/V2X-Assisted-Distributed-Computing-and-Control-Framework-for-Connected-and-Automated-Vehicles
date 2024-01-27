from dynamic_models import KinematicModel
from all_solvers import DistributedMPCLIPOPT
from utilits import *
import numpy as np

mpc_config = DistributedMPCLIPOPT.default_config
mpc_config['run_iter'] = 3
mpc_config['safe_factor'] = 8
mpc_config['safe_th'] = 2.5
mpc_config['pred_len'] = 30
mpc_config['other_veh_num'] = 2
mpc_config['comfort'] = (4.5, 0.0)
mpc_config['warm_start'] = True
mpc_config['Qu'] = 1 * np.diag([0.1, 0.6])
mpc_config["sensing_distance"] = 100
mpc_config['kernel'] = 'qrsqp'

KinematicModel.initialize(KinematicModel.default_config)
DistributedMPCLIPOPT.initialize(DistributedMPCLIPOPT.default_config)


# load traj
# log_file = np.load("output_dir/traj_log/cross_double_3_round")

# trajs, step_num, traj_info, map_info = multi_cross(_points=8)
trajs, _, info_round, map_info = cross_traj_T(
    _road_width=8.5,
    _init_road_length=15,
    _over_road_length=40
)
# np.save('output_dir/traj_log/cross_double_3_round', info_round)
for car_id_, traj in enumerate(trajs):
    DistributedMPCLIPOPT(traj[0], traj, car_id_)

all_info = DistributedMPCLIPOPT.simulate()

# gen_video_from_info(all_info, trajs, draw_nominal=False, _map_info=map_info,
#                     _custom_lim=((-45, 45), (-45, 45))
#                     )
nlp_solve_info = NLP_RESULT_INFO.extract_info_from_info_all(all_info)
PickleSave(nlp_solve_info, "output_dir/solve_info/sqp_solve_info_T")
PickleSave(all_info, "output_dir/solve_info/sqp_all_info_T")
pass
