import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from dynamic_models import *
from utilits import *
from all_solvers import *

plt.rcParams.update({'font.size': 15,
                     "text.usetex": True})

WHAT = '12'
LANG = 'EN'
plt.rcParams['font.family'] = 'Times New Roman'
# matplotlib.rcParams['mathtext.default'] = 'regular'
# plt.rcParams['font.sans-serif'] = 'SimHei'
# plt.rcParams['axes.unicode_minus']=False

mpc_config = DistributedMPC.default_config
mpc_config['run_iter'] = 2
mpc_config['safe_factor'] = 1
mpc_config['safe_th'] = 2.5
mpc_config['pred_len'] = 30
mpc_config['other_veh_num'] = 11
mpc_config['comfort'] = (4.5, 0.0)
mpc_config['priority'] = False
mpc_config['warm_start'] = False
mpc_config['Qu'] = 1 * np.diag([0.1, 0.6])
mpc_config["sensing_distance"] = 100
mpc_config["Qx"] = 0.5 * np.diag((.5, 1.0, 0.0, 0))

KinematicModel.initialize(KinematicModel.default_config)
DistributedMPC.initialize(DistributedMPC.default_config)


# generate 1-dim traj
rho_sigma_list = [(1e-1, 1e-1)] + [(1e-0, 1e-0)] * 2 + [(1e+0, 1e+0)] * 10 + [(1e+1, 1e+1)] * 10 + [(1e+2, 1e+2)] * 30
custom_rho_sigma_num = len(rho_sigma_list)

LongitudeModel.initialize(LongitudeModel.default_config)
generator = TrajDataGenerator(100, 50, 150, 110, 150, 3, 20, 15, (-5, 5), (-5, 5), sigma=5, rho=5, safe_dist=10)
x_func, y_func = gen_xs_ys(110, 150, np.deg2rad(10), 10, 4)
map_info = gen_ramp_mapinfo(110, 150, np.deg2rad(10), 10, 4)
map_info.plot_map_solo(x_lim=(100, 170), y_lim=(-10, 5))
_, trajs = generator.generate_all_vdata()
# PickleSave(trajs, './output_dir/temp01')
trajs = PickleRead('./output_dir/temp01')

iter_times = len(rho_sigma_list)
for vid, vdata in trajs.items():
    ADMM(vdata)

all_info = ADMM.STEP_ALL(iter_times, rho_sigma_list=rho_sigma_list)
ADMM.plot_y_var()


save_list = [_ for _ in range(iter_times)]
all_lines = list()
for i, one_step_info in enumerate(all_info):
    if i in save_list:
        # plt.plot([0, 100], [110, 110], color='black')
        # plt.plot([0, 100], [150, 150], color='black')
        fig, ax = plt.subplots(figsize=(8, 4.8))
        text = ax.text(0.9, 0.10, '', transform=ax.transAxes, ha='right')
        text.set_text("迭代次数: {0}".format(i) if LANG == 'CN' else "Iterations: {0}".format(i))
        for v_id, one_traj in one_step_info.items():
            road, u, s = one_traj
            line = plt.plot(
                one_traj[2][:, 0],
                # color='red' if road == 'main' else 'green',
                color='red' if road == 'main' else 'green',
                linestyle='-' if road == 'main' else '--',
                lw=0.9,
                label=('主干路' if LANG == 'CN' else 'main') if road == 'main' else ('匝道' if LANG == 'CN' else 'merge')
            )
            ax.set_xlim(-5, 90)
            ax.set_ylim(0, 300)
            all_lines.append(line)
        # for admm in all_ADMM:
        #     plt.plot([admm.data.tdvp[0], admm.data.tdvp[0]], [0, 250], color='black', lw=0.5)
        plt.legend(['主干路', '匝道'] if LANG == 'CN' else ['main', 'merge'])
        plt.xlabel('时间[0.1s]' if LANG == 'CN' else 'time[0.1s]')
        plt.ylabel('纵向位置[m]' if LANG == 'CN' else 'longitudinal position[m]')
        plt.savefig('output_dir/traj_plan/traj_plan_{}.svg'.format(i), dpi=300, bbox_inches='tight', pad_inches=.01)
        plt.close()

# exit()
# generate 2-dim traj
two_dim_trajs = {}
for v_id, v_data in all_info[-1].items():
    v_road, v_control, v_state = v_data
    final_velocity = v_state[-1, 1]
    final_s = v_state[-1, 0]
    v_s = np.hstack((v_state[:, 0], np.array([final_s + 0.1 * i * final_velocity for i in range(50)])))
    x = np.array([x_func(_) for _ in v_s])
    y = np.array([y_func(_) for _ in v_s]) if v_road == 'merge' else np.zeros_like(x)
    two_dim_traj = np.vstack((x, y, np.zeros_like(x), np.zeros_like(x))).transpose()
    init_velocity = v_state[0, 1]
    # plt.plot(x, y)
    # plt.show()
    # plt.close()
    # two_dim_traj[0, ]
    two_dim_traj[0, 3] = init_velocity
    two_dim_trajs[v_id] = two_dim_traj
    print(v_id)

for car_id_, traj in two_dim_trajs.items():
    DistributedMPC(traj[0], traj, car_id_)

all_info = DistributedMPC.simulate()

gen_video_from_info(all_info, list(two_dim_trajs.values()), draw_nominal=True, save_frame=False,
                    _map_info=map_info,
                    _draw_all_nominal=False,
                    _custom_lim=((0, 110), (-10, 5))
                    )

# osqp_solve_info = OSQP_RESULT_INFO.extract_info_from_info_all(all_info)
# PickleSave(osqp_solve_info, "output_dir/solve_info/on_ramp")
# PickleSave(all_info, "output_dir/solve_info/on_ramp")

