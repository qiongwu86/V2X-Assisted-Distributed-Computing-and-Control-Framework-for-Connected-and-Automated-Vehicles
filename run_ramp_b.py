import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from dynamic_models import *
from utilits import *
from all_solvers import *
from typing import List, Text

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
mpc_config['run_iter'] = 3
mpc_config['safe_factor'] = 1
mpc_config['safe_th'] = 2.5
mpc_config['pred_len'] = 30
mpc_config['other_veh_num'] = 11
mpc_config['comfort'] = (4.5, 0.0)
mpc_config['priority'] = False
mpc_config['warm_start'] = False
mpc_config['Qu'] = 1 * np.diag([0.1, 0.6])
mpc_config["sensing_distance"] = 100
mpc_config["Qx"] = 0.5 * np.diag((.1, 1.0, 0.0, 0))

KinematicModel.initialize(KinematicModel.default_config)
DistributedMPC.initialize(DistributedMPC.default_config)


# generate 1-dim traj
# [(2e-1, 2e-1)] + [(1e-0, 1e-0)] * 7 + [(3e+0, 3e+0)] * 10 + [(1e+1, 1e+1)] * 23 + [(5e+1, 5e+1)] * 10
# [(1e+0, 1e+0)] * 3 + [(5e+0, 5e+0)] * 3 + [(5e+0, 5e+0)] * 10 + [(2e+1, 2e+1)] * 20 + [(5e+1, 5e+1)] * 20
# [(1e+0, 1e+0)] * 1 + [(5e+0, 5e+0)] * 5 + [(2e+1, 2e+1)] * 10 + [(5e+1, 5e+1)] * 10 + [(1e+2, 1e+2)] * 10
# [(1e+1, 1e+1)] * 3 + [(2e+1, 2e+1)] * 10 + [(5e+1, 5e+1)] * 20 + [(5e+2, 5e+2)] * 0
# [(1e+1, 1e+1)] * 3 + [(2e+1, 2e+1)] * 15 + [(4+1, 4e+1)] * 15
rho_sigma_list = [(1e+1, 1e+1)] * 3 + [(2e+1, 2e+1)] * 20 + [(4+1, 4e+1)] * 20
custom_rho_sigma_num = len(rho_sigma_list)

LongitudeModel.initialize(LongitudeModel.default_config)
generator = TrajDataGenerator(100, 50, 150, 110, 150, 10, 15, 15, (-5, 5), (-3, 3), sigma=5, rho=5, safe_dist=20, over_length=20, over_velocity=15)
x_func, y_func = gen_xs_ys(110, 150, np.deg2rad(10), 10, 4)
map_info = gen_ramp_mapinfo(110, 150, np.deg2rad(10), 10, 4)
map_info.plot_map_solo(x_lim=(100, 170), y_lim=(-10, 5))
_, init_info = generator.generate_all_vdata()
init_info = PickleRead('./output_dir/temp01')

TrajB_solver = TrajB(init_info)
results = TrajB_solver.solve()

ColorSet.reset()
sum = 0
sum_list = []
fig, ax = plt.subplots(figsize=(2.5, 1.8))
for v_id, one_traj in reversed(results.items()):
    road, u, s = one_traj
    sum += np.linalg.norm(u)
    sum_list.append(np.linalg.norm(u))
    line = plt.plot(
        s[:, 2],
        # color='red' if road == 'main' else 'green',
        color=ColorSet.get_next_color(),
        linestyle='-' if road == 'main' else '--',
        lw=0.5,
        label=('主干路' if LANG == 'CN' else 'main') if road == 'main' else ('匝道' if LANG == 'CN' else 'merge')
    )
    ax.set_xlim(0, 110)
    ax.set_ylim(-6, 6)
    # all_lines.append(line)
# for admm in all_ADMM:
#     plt.plot([admm.data.tdvp[0], admm.data.tdvp[0]], [0, 250], color='black', lw=0.5)
# plt.legend(['主干路', '匝道'] if LANG == 'CN' else ['main', 'merge'])
plt.xlabel('时间[0.1s]' if LANG == 'CN' else 'Time Step $k$')
# plt.ylabel('纵向位置[m]' if LANG == 'CN' else 'Longitudinal Acc $a^i_k[\mathrm{m/s^2}]$')
# plt.ylabel('纵向位置[m]' if LANG == 'CN' else 'Longitudinal Pos $s^i_k[\mathrm{m}]$')
# ax.set_yticks([])
ax.tick_params(axis='y', left=False, labelleft=False)
plt.grid(linewidth=0.5)

##################
x_box = [50, 80, 80, 50, 50]  # x坐标：左下→右下→右上→左上→左下（闭合）
y_box = [70, 70, 140, 140, 70]  # y坐标：对应x的四个顶点
# x_box = [30, 70, 70, 30, 30]  # x坐标：左下→右下→右上→左上→左下（闭合）
# y_box = [20, 20, 80, 80, 20]  # y坐标：对应x的四个顶点
# ax.plot(x_box, y_box, 'r-', linewidth=1)  # 'r-' 表示红色实线
##################

plt.subplots_adjust(left=0.13, right=0.99, bottom=0.2, top=0.95)
# plt.show()
plt.savefig('output_dir/traj_plan/blnl_a.eps', dpi=50, bbox_inches='tight', pad_inches=.01)
plt.close()
print(sum)
print(sum_list)

pass

