"""
轨迹绘制
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Text, Tuple
from dynamic_models import *
from utilits import *
from all_solvers import *
from typing import List, Text

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


trajs = PickleRead('./output_dir/temp01')


def extract_data(_all_info: Dict, _data_name: Text = '', _one_nominal: bool = False) -> Dict:
    _data_num = len(_all_info)
    assert _data_num > 0
    _veh_ids = [_v_id for _v_id in _all_info[0].keys()][:]
    _ret_dict = {
        _veh_id: dict(
            state=np.zeros((_data_num, 4)),
            control=np.zeros((_data_num, 2)),
            x_nominal=np.zeros((_data_num, 30 + 1, 4)),
            u_nominal=np.zeros((_data_num, 30, 2))
        ) for _veh_id in _veh_ids
    }
    for _t, one_time_data in enumerate(_all_info):
        for _v_id in _veh_ids:
            _ret_dict[_v_id]['state'][_t] = one_time_data[_v_id]['new_state']
            _ret_dict[_v_id]['control'][_t] = one_time_data[_v_id]['control']
            if _one_nominal:
                _ret_dict[_v_id]['x_nominal'][_t] = one_time_data[_v_id]['nominal'][0]
                _ret_dict[_v_id]['u_nominal'][_t] = one_time_data[_v_id]['nominal'][1]
            else:
                _ret_dict[_v_id]['x_nominal'][_t] = one_time_data[_v_id]['nominal'][-1][0]
                _ret_dict[_v_id]['u_nominal'][_t] = one_time_data[_v_id]['nominal'][-1][1]
    for _v_id, _v_data in _ret_dict.items():
        _v_data['delta_phi'] = np.array([_v_data['state'][1:, 2] - _v_data['state'][:-1, 2]]).transpose()
    return _ret_dict


info_dict = {
    'Our': PickleRead('output_dir/solve_info/on_ramp'),
}

extracted_data_dict = {
    'Our': extract_data(info_dict['Our']),
    # 'SQP': extract_data(info_dict['SQP'], _one_nominal=True),
}
pass
