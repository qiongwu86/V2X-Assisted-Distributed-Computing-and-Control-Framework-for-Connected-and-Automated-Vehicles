from dynamic_models import KinematicModel
from all_solvers import DistributedMPCIPOPT
from utilits import *

mpc_config = DistributedMPCIPOPT.default_config
mpc_config['run_iter'] = 5
mpc_config['safe_factor'] = 2.5
mpc_config['safe_th'] = 1.7
mpc_config['pred_len'] = 30
mpc_config['other_veh_num'] = 3
mpc_config['comfort'] = 1.5

KinematicModel.initialize(KinematicModel.default_config)
DistributedMPCIPOPT.initialize(DistributedMPCIPOPT.default_config)


# trajs, step_num, traj_info, map_info = multi_cross(_points=8)
trajs, _, _, map_info = cross_traj_double_lane(_round=1,
                                               _over_road_length=30)
for car_id_, traj in enumerate(trajs):
    DistributedMPCIPOPT(traj[0], traj, car_id_)

all_info = DistributedMPCIPOPT.simulate()

gen_video_from_info(all_info, trajs, draw_nominal=False, _map_info=map_info)
# all_info_qp = OSQP_RESULT_INFO.extract_info_from_info_all(all_info)
pass
