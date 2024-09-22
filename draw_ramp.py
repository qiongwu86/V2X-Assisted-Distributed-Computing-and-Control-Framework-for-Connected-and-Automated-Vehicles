"""
轨迹绘制
"""
import matplotlib.pyplot as plt
from utilits import *
import numpy as np
from typing import List, Dict, Text, Tuple
from matplotlib import patches
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from utilits import dimpc_utilts

PREDICT_LEN = 30

WHAT = '12'
LANG = 'CN'
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus']=False


def extract_data(_all_info: Dict, _data_name: Text = '', _one_nominal: bool = False) -> Dict:
    _data_num = len(_all_info)
    assert _data_num > 0
    _veh_ids = [_v_id for _v_id in _all_info[0].keys()][:]
    _ret_dict = {
        _veh_id: dict(
            state=np.zeros((_data_num, 4)),
            control=np.zeros((_data_num, 2)),
            x_nominal=np.zeros((_data_num, PREDICT_LEN + 1, 4)),
            u_nominal=np.zeros((_data_num, PREDICT_LEN, 2))
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


def draw_data(
        _extracted_data: Dict,
        name: Text,
        _idx: int,
        _sub_fig_pos: List,
        _y_title: Text,
        _x_title: Text,
        _y_lim: Tuple,
        _x_lim: Tuple,
        _insert_axes=None,
        _rect_dict=None,
        _save_name=None,
        _draw_title=False
):
    fig, _ax = plt.subplots(len(_extracted_data), 1, figsize=(1, 2 * len(_extracted_data)))
    # for _ax, _alg_name in zip(axs, _extracted_data):
    _alg_name='ramp'
    _alg_data = _extracted_data[_alg_name]
    if _draw_title:
        _ax.set_title(_alg_name if LANG == "EN" else en_to_cn(_alg_name), fontsize=10, fontweight=1, rotation='vertical', x=1.02, y=0.4, va='center')
    _ax.set_ylim(*_y_lim)
    _ax.set_xlim(*_x_lim)
    _ax.grid(True)
    dimpc_utilts.ColorSet.reset()
    for _v_id, _v_data in _alg_data.items():
        _ax.plot(_v_data[name][:, _idx], lw=0.75, color=dimpc_utilts.ColorSet.get_next_color())

    plt.subplots_adjust(left=0.09, right=0.97, top=0.963, bottom=0.055, hspace=0.2, wspace=0.2)
    if _save_name is not None:
        plt.savefig(_save_name)
    plt.show()


info_dict = {
    'ramp': PickleRead('output_dir/solve_info/on_ramp'),
}

extracted_data_dict = {
    'ramp': extract_data(info_dict['ramp']),
}

draw_data(extracted_data_dict,
          name='state',
          _idx=0,
          _sub_fig_pos=[
              [0.09, 0.80, 0.85, 0.14],
              [0.09, 0.60, 0.85, 0.20],
              [0.09, 0.40, 0.85, 0.20],
              [0.09, 0.20, 0.85, 0.20],
              [0.09, 0.00, 0.85, 0.20]
          ],
          _y_title='acc[$m/s^2$]' if LANG == 'EN' else '加速度[$\mathrm{m/s^2}$]',
          _x_title='time[ms]' if LANG == 'EN' else '时间[0.1$\mathrm{s}$]',
          _y_lim=(0, 300),
          _x_lim=(-0, 90),
          # _rect_dict=rect_dict,
          _save_name='output_dir/figs/acc.svg',
          # _insert_axes=delta_phi_insert_dict
          )

# 手动记录
run_time = [
    [0.0019079579247368705, 0.06296261151631673],  # 3
    [0.0020849632494377365, 0.06880378723144531],  # 4
    [0.00212470834905451, 0.07011537551879883],  # 5
    [0.002143118116590712, 0.0707228978474935],  # 6
    [0.0022584814013856835, 0.07452988624572755],  # 7
    [0.00254571076595422, 0.08400845527648926],  # 8
    [0.0028129129698782262, 0.09282612800598146],  # 9
    [0.0030634562174479164, 0.10109405517578124],  # 10
]
