"""
最小距离
"""
import matplotlib.pyplot as plt
from utilits import *
import numpy as np
from typing import List, Dict, Text, Tuple
from matplotlib import patches
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

"""
1: 2.5558414419678037,
2: 3.0430011356100164,
3: 3.272546326507824,
4: 3.403050345982561,
5: 3.9632087142485877
"""

WHAT = '12'
LANG = 'EN'
plt.rcParams.update({'font.size': 10,
                     "text.usetex": True})
plt.rcParams['font.family'] = 'Times New Roman'


def calculate_min_dist_once(_pos: List[np.ndarray]) -> float:
    _dist = []
    _points_num = len(_pos)
    for _pos_1_idx in range(_points_num):
        for _pos_2_idx in range(_pos_1_idx+1, _points_num):
            _dist.append(np.linalg.norm(_pos[_pos_1_idx] - _pos[_pos_2_idx]))
    return min(_dist)


def calculate_min_dist(_data_for_calculate: Dict) -> Dict[Text, np.ndarray]:
    _ret_dict = {}
    for _alg_name, _alg_data in _data_for_calculate.items():
        _ret_dict[_alg_name] = list()
        _points_num = len(_alg_data)
        for _t_data in _alg_data:
            _pos_list = [_v_data['new_state'][:2] for _v_data in _t_data.values()]
            _ret_dict[_alg_name].append(calculate_min_dist_once(_pos_list))
        _ret_dict[_alg_name] = np.array(_ret_dict[_alg_name])
    return _ret_dict


info_dict = {
    'Our': PickleRead('output_dir/solve_info/osqp_all_info_{}'.format(WHAT)),
    'OSQP-CS': PickleRead('output_dir/solve_info/osqp_all_info_cs_{}'.format(WHAT)),
    'IPOPT': PickleRead('output_dir/solve_info/nlp_all_info_{}'.format(WHAT)),
    'LD-IPOPT': PickleRead('output_dir/solve_info/lnlp_all_info_{}'.format(WHAT)),
    # 'SQP': PickleRead('output_dir/solve_info/sqp_all_info_{}'.format(WHAT))
}

def color_generator():
    yield "red"
    yield "blue"
    yield "green"
    yield "black"

colors = color_generator()

a = calculate_min_dist(info_dict)
fig, ax = plt.subplots()
[plt.plot([_ for _ in range(_alg_dist.shape[0])], _alg_dist, label=_alg_name if LANG=='EN' else en_to_cn(_alg_name), lw=0.75, color=next(colors))
 for _alg_name, _alg_dist in a.items()]
plt.xlabel('time[0.1m]' if LANG == 'EN' else '时间[0.1$\mathrm{s}$]')
plt.ylabel('minimum distance[m]' if LANG == 'EN' else '距离[$\mathrm{m}$]')
plt.legend()

_min_dist = {alg_name: np.min(dists) for alg_name, dists in a.items()}
# _min_dist_t = min([np.argmin(dist) for dist in a.values()])

colors = color_generator()
dimpc_utilts.ColorSet.reset()
_ax_in = ax.inset_axes((0.15, 0.5, 0.4, 0.45))
_ax_in.set_xticks([])
_ax_in.set_yticks([])
_ax_in.set_xlim(57, 83)
_ax_in.set_ylim(3.7, 5)
_ax_in.spines[:].set_color(None)
mark_inset(ax, _ax_in, loc1=3, loc2=3, fc="none", ec="red", lw=2)
rect = patches.Rectangle(
    (13, 8.25),
    54,
    4.7,
    fill=False,
    edgecolor='red',
    lw=2,
    alpha=0.8
)
ax.add_patch(rect)
for _alg_name, _alg_dist in a.items():
    _ax_in.plot(_alg_dist, lw=0.75, color=next(colors))

plt.grid()
plt.show()
print(_min_dist)
# plt.plot([0, 120], [_min_dist, _min_dist])
# plt.scatter(float(_min_dist_t), _min_dist, color='r')


dist_alpha =\
    {1: 2.5558414419678037,
2: 3.0430011356100164,
3: 3.272546326507824,
4: 3.403050345982561,
5: 3.9632087142485877}
fig, axs = plt.subplots()
plt.scatter([_ for _ in dist_alpha.keys()], [_ for _ in dist_alpha.values()])
plt.plot([_ for _ in dist_alpha.keys()], [_ for _ in dist_alpha.values()], linestyle='--', color='gray')
plt.xlabel(r'安全系数$\mathrm{\alpha}$' if LANG == 'CN' else r'safe factor $\alpha$')
plt.ylabel(r'最小距离[$\mathrm{m}$]' if LANG == 'CN' else r'minimum distance')
plt.xticks([_ for _ in range(0, 6)])
plt.grid()
plt.show()
print()



