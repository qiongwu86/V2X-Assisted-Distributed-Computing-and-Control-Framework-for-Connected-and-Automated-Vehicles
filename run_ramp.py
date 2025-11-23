import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from dynamic_models import *
from utilits import *
from all_solvers import *
from typing import List, Text

np.random.seed(2)

plt.rcParams.update({'font.size': 15,
                     "text.usetex": True})

WHAT = '12'
LANG = 'EN'
plt.rcParams.update({
    "font.family": "Times New Roman",   # 字体
    "mathtext.fontset": "stix",         # 数学字体风格
    "font.size": 8,                     # 字号
    "axes.labelsize": 9,                # 坐标轴标签字号
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.0,             # 线宽 1pt
    "lines.markersize": 4,              # 标记大小
})

class ColorSet:
    color_set: List = ["#00008B", "#800000", "#006400", "#4B0082",
                       "#0000FF", "#FF0000", "#dbdc56", "#9400D3",
                       "#1E90FF", "#FF4500", "#7CFC00", "#FF00FF"]
    color_ptr: int = 0

    @staticmethod
    def get_next_color() -> Text:
        if ColorSet.color_ptr == len(ColorSet.color_set):
            ColorSet.color_ptr = 0
        color = ColorSet.color_set[ColorSet.color_ptr]
        ColorSet.color_ptr += 1
        return color

    @staticmethod
    def reset():
        ColorSet.color_ptr = 0

mpc_config = DistributedMPC.default_config
mpc_config['run_iter'] = 2
mpc_config['safe_factor'] = 1
mpc_config['safe_th'] = 3.5
mpc_config['pred_len'] = 30
mpc_config['other_veh_num'] = 11
mpc_config['comfort'] = (4.5, 0.0)
mpc_config['priority'] = False
mpc_config['warm_start'] = False
mpc_config['Qu'] = 1 * np.diag([0.1, 0.6])
mpc_config["sensing_distance"] = 100
mpc_config["Qx"] = 0.5 * np.diag((1.0, 1.0, 0.0, 0))

KinematicModel.initialize(KinematicModel.default_config)
DistributedMPC.initialize(DistributedMPC.default_config)


# generate 1-dim traj
# generator = TrajDataGenerator(100, 50, 150, 110, 150, 10, 15, 15, (-3, 3), (-3, 3), sigma=5, rho=5, safe_dist=15, over_length=15, over_velocity=15)
# [(2e-1, 2e-1)] + [(1e-0, 1e-0)] * 7 + [(3e+0, 3e+0)] * 10 + [(1e+1, 1e+1)] * 23 + [(5e+1, 5e+1)] * 10
# [(1e+0, 1e+0)] * 3 + [(5e+0, 5e+0)] * 3 + [(5e+0, 5e+0)] * 10 + [(2e+1, 2e+1)] * 20 + [(5e+1, 5e+1)] * 20
# [(1e+0, 1e+0)] * 1 + [(5e+0, 5e+0)] * 5 + [(2e+1, 2e+1)] * 10 + [(5e+1, 5e+1)] * 10 + [(1e+2, 1e+2)] * 10
# [(1e+1, 1e+1)] * 3 + [(2e+1, 2e+1)] * 10 + [(5e+1, 5e+1)] * 20 + [(5e+2, 5e+2)] * 0
# [(1e+1, 1e+1)] * 3 + [(2e+1, 2e+1)] * 15 + [(4+1, 4e+1)] * 15
# [(1e+1, 1e+1)] * 30 + [(2e+1, 2e+1)] * 10 + [(1e+2, 1e+2)] * 10
rho_sigma_list = [(1e-0, 1e-0)] * 30 + [(1e+0, 1e+0)] * 20 + [(1e+1, 1e+1)] * 50 + [(2e+1, 2e+1)] * 20
custom_rho_sigma_num = len(rho_sigma_list)

LongitudeModel.initialize(LongitudeModel.default_config)
generator = TrajDataGenerator(100, 50, 150, 110, 150, 10, 15, 15, (-3, 3), (-3, 3), sigma=5, rho=5, safe_dist=15, over_length=15, over_velocity=15)
x_func, y_func = gen_xs_ys(110, 150, np.deg2rad(10), 10, 4)
map_info = gen_ramp_mapinfo(110, 150, np.deg2rad(10), 10, 4)
map_info.plot_map_solo(x_lim=(100, 170), y_lim=(-10, 5))
_, trajs = generator.generate_all_vdata()
# exit()
PickleSave(trajs, './output_dir/temp01')
# trajs = PickleRead('./output_dir/temp01')

iter_times = len(rho_sigma_list)
for vid, vdata in trajs.items():
    ADMM(vdata)

all_info = ADMM.STEP_ALL(iter_times, rho_sigma_list=rho_sigma_list)
var_y = ADMM.plot_y_var()
#######################################
fig, ax = plt.subplots(figsize=(3.4, 1.3))
plt.yscale('log')
ax.plot(var_y[1:50])
plt.xlabel('迭代次数' if LANG == 'CN' else 'Iterations')
plt.ylabel('$\mathrm{y^{i,k}}$的方差' if LANG == 'CN' else 'Var$(y^{i,k})$')
plt.grid()
# plt.show()
# plt.subplots_adjust(left=0.125, right=0.986, bottom=0.235, top=0.979)
plt.savefig('output_dir/traj_plan/sigma.eps', bbox_inches='tight', pad_inches=.01)
# plt.savefig('output_dir/traj_plan/sigma.eps')
plt.close()
# exit(0)
#######################################


save_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] + [_ for _ in range(iter_times) if not (_ % 10)] + [iter_times-1]
save_list = [iter_times-1]
all_lines = list()
for i, one_step_info in enumerate(all_info):
    if i in save_list:
        # plt.plot([0, 100], [110, 110], color='black')
        # plt.plot([0, 100], [150, 150], color='black')
        fig, ax = plt.subplots(figsize=(2.1, 1.4))
        text = ax.text(0.9, 0.10, '', transform=ax.transAxes, ha='right')
        text.set_text("迭代次数: {0}".format(i) if LANG == 'CN' else "Iterations: {0}".format(1+i))
        ColorSet.reset()
        for v_id, one_traj in one_step_info.items():
            road, u, s = one_traj
            line = plt.plot(
                one_traj[2][:, 0],
                # color='red' if road == 'main' else 'green',
                color=ColorSet.get_next_color(),
                linestyle='-' if road == 'main' else '--',
                lw=0.5,
                label=('主干路' if LANG == 'CN' else 'main') if road == 'main' else ('匝道' if LANG == 'CN' else 'merge'),
                # linewidth=0.5
            )
            ax.set_xlim(0, 110)
            ax.set_ylim(0, 300)
            all_lines.append(line)
        # for admm in all_ADMM:
        #     plt.plot([admm.data.tdvp[0], admm.data.tdvp[0]], [0, 250], color='black', lw=0.5)
        # if i == 0:
        #     plt.legend(['主干路', '匝道'] if LANG == 'CN' else ['main', 'merge'])
        plt.xlabel('时间[0.1s]' if LANG == 'CN' else '$k$')
        if i == 0:
            plt.ylabel('纵向位置[m]' if LANG == 'CN' else '$s^i_k$[m]')
        else:
            ax.yaxis.set_visible(False)
            ax.set_yticklabels([])
        plt.savefig('output_dir/traj_plan/traj_plan_{}.svg'.format(i), dpi=100, bbox_inches='tight', pad_inches=.01)
        plt.close()
# exit(0)
#
sum=0
sum_list = []
fig, ax = plt.subplots(figsize=(2.5, 1.8)) # 3.3
ColorSet.reset()
for v_id, one_traj in all_info[-1].items():
    road, u, s = one_traj
    sum += np.linalg.norm(u)
    sum_list.append(np.linalg.norm(u))
    line = plt.plot(
        one_traj[2][:, 2],  # one_traj[2][:, 2],
        # color='red' if road == 'main' else 'green',
        color=ColorSet.get_next_color(),
        linestyle='-' if road == 'main' else '--',
        lw=0.5,
        label=('主干路' if LANG == 'CN' else 'main') if road == 'main' else ('匝道' if LANG == 'CN' else 'merge')
    )
    ax.set_xlim(0, 110)
    ax.set_ylim(-6, 6)
    all_lines.append(line)
plt.grid(linewidth=0.25)

# for admm in all_ADMM:
#     plt.plot([admm.data.tdvp[0], admm.data.tdvp[0]], [0, 250], color='black', lw=0.5)
# plt.legend(['主干路', '匝道'] if LANG == 'CN' else ['main', 'merge'])
plt.xlabel('时间[0.1s]' if LANG == 'CN' else 'Time Step $k$')
plt.ylabel('纵向位置[m]' if LANG == 'CN' else 'Longitudinal Acc $a^i_k[\mathrm{m/s^2}]$')
# plt.ylabel('纵向位置[m]' if LANG == 'CN' else 'Longitudinal Pos $s^i_k[\mathrm{m}]$')
plt.grid(linewidth=0.25)
####################################
# 设置主图要放大的区域（x: 2~4，y: 0.5~1.2）
zoom_x_start, zoom_x_end = 0, 20
zoom_y_start, zoom_y_end = -1.7, 1.7
ColorSet.reset()
# 添加内嵌子图（位置：主图右上角，大小：主图的50%×50%）
axins = inset_axes(ax, width="25%", height="40%", loc='upper right')
# 在子图中绘制放大区域的曲线
for v_id, one_traj in all_info[-1].items():
    road, u, s = one_traj
    line = axins.plot(
        one_traj[2][:, 2],
        # color='red' if road == 'main' else 'green',
        color=ColorSet.get_next_color(),
        linestyle='-' if road == 'main' else '--',
        lw=0.5,
        label=('主干路' if LANG == 'CN' else 'main') if road == 'main' else ('匝道' if LANG == 'CN' else 'merge')
    )
axins.plot([0 for _ in range(100)], color='black', linestyle='-', lw=0.25)
# 限定子图的显示范围（局部放大区域）
plt.grid(linewidth=0.25)
axins.xaxis.set_visible(False)
# axins.yaxis.set_visible(False)
axins.set_xlim(zoom_x_start, zoom_x_end)
axins.set_ylim(zoom_y_start, zoom_y_end)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="red", lw=1.0)
####################################
plt.subplots_adjust(left=0.13, right=0.99, bottom=0.2, top=0.95)
# plt.show()
plt.savefig('output_dir/traj_plan/our_a.eps', dpi=50, bbox_inches='tight', pad_inches=.01)
plt.close()
print(sum)
print(sum_list)

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

all_info_new = DistributedMPC.simulate()

gen_video_from_info(all_info_new, list(two_dim_trajs.values()), draw_nominal=True, save_frame=True,
                    _map_info=map_info,
                    _draw_all_nominal=False,
                    _custom_lim=((100, 160), (-10, 3))
                    )

osqp_solve_info = OSQP_RESULT_INFO.extract_info_from_info_all(all_info_new)
PickleSave(osqp_solve_info, "output_dir/solve_info/on_ramp")
PickleSave(all_info_new, "output_dir/solve_info/on_ramp")


fig, ax = plt.subplots(figsize=(3.4, 1.0))  # 高度适中即可
ColorSet.reset()
for vid, ref_traj in two_dim_trajs.items():
    x_delta = list()
    for _ in range(len(all_info_new)):
        x_delta.append(ref_traj[_, 1] - all_info_new[_][vid]['old_state'][1])  # last idx
    road = all_info[0][vid][0]
    plt.plot(
        x_delta,
        # color='red' if road == 'main' else 'green',
        color=ColorSet.get_next_color(),
        linestyle='-' if road == 'main' else '--',
        lw=1.0 if vid in [4, 5] else 0.5,
        label=('主干路' if LANG == 'CN' else 'main') if road == 'main' else ('匝道' if LANG == 'CN' else 'merge')
    )
ax.set_xlim(0, 100)
plt.xlabel('时间[0.1s]' if LANG == 'CN' else 'Time Step $k$')
plt.ylabel('纵向位置[m]' if LANG == 'CN' else '$\Delta y[\mathrm{m}]$')
x_box = [50, 67, 67, 50, 50]  # x坐标：左下→右下→右上→左上→左下（闭合）
y_box = [-0.11, -0.11, 0.11, 0.11, -0.11]  # y坐标：对应x的四个顶点
# ax.plot(x_box, y_box, 'r-', linewidth=1)  # 'r-' 表示红色实线
# plt.show()
plt.grid()
plt.savefig('output_dir/traj_plan/delta_y.eps', bbox_inches='tight', pad_inches=.01)
plt.close()


