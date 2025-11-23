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
            _pos_list = [_v_data['new_state'][:2] for _v_data in _t_data.values() if _v_data['new_state'][0] > 110]
            if len(_pos_list) >= 2:
                _ret_dict[_alg_name].append(calculate_min_dist_once(_pos_list))
        _ret_dict[_alg_name] = np.array(_ret_dict[_alg_name])
    return _ret_dict


info_dict = {
    'Our': PickleRead('output_dir/solve_info/on_ramp'),
    # 'OSQP-CS': PickleRead('output_dir/solve_info/osqp_all_info_cs_{}'.format(WHAT)),
    # 'IPOPT': PickleRead('output_dir/solve_info/nlp_all_info_{}'.format(WHAT)),
    # 'LD-IPOPT': PickleRead('output_dir/solve_info/lnlp_all_info_{}'.format(WHAT)),
    # 'SQP': PickleRead('output_dir/solve_info/sqp_all_info_{}'.format(WHAT))
}

def color_generator():
    yield "red"
    yield "blue"
    yield "green"
    yield "black"

colors = color_generator()

a = calculate_min_dist(info_dict)
# fig, ax = plt.subplots()
# [plt.plot([_ for _ in range(_alg_dist.shape[0])], _alg_dist, label=_alg_name if LANG=='EN' else en_to_cn(_alg_name), lw=0.75, color=next(colors))
#  for _alg_name, _alg_dist in a.items()]
# plt.xlabel('time[0.1m]' if LANG == 'EN' else '时间[0.1$\mathrm{s}$]')
# plt.ylabel('minimum distance[m]' if LANG == 'EN' else '距离[$\mathrm{m}$]')
# plt.legend()

_min_dist = {alg_name: np.min(dists) for alg_name, dists in a.items()}
print(_min_dist)
# _min_dist_t = min([np.argmin(dist) for dist in a.values()])
# (0.5, 2.5) : 5.0346408897819295
# (  1, 2.5) : 5.119633695256756
# (1.5, 2.5) : 5.140903634319407
# (2.0, 2.5) : 5.149505367037195
# (2.5, 2.5) : 5.157407316198906

# (0.5, 3.0) : 5.479289425672136
# (  1, 3.0) : 5.585737618279801
# (1.5, 3.0) : 5.62584234875519
# (2.0, 3.0) : 5.647769236661723
# (2.5, 3.0) : 5.662815657026249

# (0.5, 3.5) : 5.927640552495048
# (  1, 3.5) : 6.0812013535475264
# (1.5, 3.5) : 6.144590218576108
# (2.0, 3.5) : 6.17849029722521
# (2.5, 3.5) : 6.198619284694945

