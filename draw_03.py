import matplotlib.pyplot as plt
from utilits import *
import numpy as np
from typing import List, Dict, Text, Tuple
from matplotlib import patches
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


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
    'proposed': PickleRead('output_dir/solve_info/osqp_all_info_3'),
    # 'OSQP-CS': PickleRead('output_dir/solve_info/osqp_all_info_cs_3'),
    # 'IPOPT': PickleRead('output_dir/solve_info/nlp_all_info_3'),
    # 'LD-IPOPT': PickleRead('output_dir/solve_info/lnlp_all_info_3'),
    # 'SQP': PickleRead('output_dir/solve_info/lnlp_all_info_3')
}

a = calculate_min_dist(info_dict)
[plt.plot(_alg_dist) for _alg_name, _alg_dist in a.items()]

_min_dist = min([np.min(dist) for dist in a.values()])
_min_dist_t = min([np.argmin(dist) for dist in a.values()])
print(_min_dist_t)
plt.plot([0, 120], [_min_dist, _min_dist])
plt.scatter(float(_min_dist_t), _min_dist, color='r')
plt.grid()
plt.show()


