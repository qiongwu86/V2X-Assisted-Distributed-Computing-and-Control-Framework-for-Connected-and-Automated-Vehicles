import matplotlib.pyplot as plt
from utilits import *
import numpy as np
from typing import List, Dict, Text, Tuple
from matplotlib import patches
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from utilits import dimpc_utilts

PREDICT_LEN = 30


def extract_data(_all_info: Dict, _data_name: Text = '', _one_nominal: bool = False) -> Dict:
    _data_num = len(_all_info)
    assert _data_num > 0
    _veh_ids = [_v_id for _v_id in _all_info[0].keys()]
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
        _save_name=None
):
    fig, axs = plt.subplots(len(_extracted_data), 1, figsize=(8, 1.5 * len(_extracted_data)))
    for _ax, _alg_name in zip(axs, _extracted_data):
        _alg_data = _extracted_data[_alg_name]
        _ax.set_title(_alg_name, fontsize=10, fontweight=1, rotation='vertical', x=1.02, y=0.4, va='center')
        _ax.set_ylim(*_y_lim)
        _ax.set_xlim(*_x_lim)
        _ax.set_ylabel(_y_title)
        _ax.grid(True)
        for _v_id, _v_data in _alg_data.items():
            _ax.plot(_v_data[name][:, _idx], lw=0.75, color=dimpc_utilts.ColorSet.get_next_color())
        if _rect_dict is not None:
            for _rect in _rect_dict.values():
                if _alg_name in _rect['alg_name']:
                    rect = patches.Rectangle(
                        (_rect['pos'][0], _rect['pos'][1]),
                        _rect['w'],
                        _rect['l'],
                        fill=False,
                        edgecolor=_rect['color'],
                        lw=1.5,
                        alpha=0.8
                    )
                    _ax.add_patch(rect)

        if _insert_axes is not None:
            for _ins_data in _insert_axes.values():
                if _alg_name in _ins_data['alg_name']:
                    _ax_in = _ax.inset_axes(_ins_data['ins_pos'])
                    _ax_in.set_xticks([])
                    _ax_in.set_yticks([])
                    _ax_in.set_xlim(_ins_data['x_lim'])
                    _ax_in.set_ylim(_ins_data['y_lim'])
                    _ax_in.spines[:].set_color('b')
                    # mark_inset(_ax, _ax_in, loc1=3, loc2=2, fc="none", ec='r', lw=1)
                    rect = patches.Rectangle(
                        (_ins_data['x_lim'][0], _ins_data['y_lim'][0]),
                        _ins_data['x_lim'][1]-_ins_data['x_lim'][0],
                        _ins_data['y_lim'][1]-_ins_data['y_lim'][0],
                        fill=False,
                        edgecolor='b',
                        lw=1,
                        alpha=0.8
                    )
                    _ax.add_patch(rect)
                    for _v_id, _v_data in _alg_data.items():
                        _ax_in.plot(_v_data[name][:, _idx], lw=0.75, color=dimpc_utilts.ColorSet.get_next_color())
    [_ax.set_position(_sub_fig_pos[i]) for i, _ax in enumerate(axs)]
    axs[-1].set_xlabel(_x_title)
    if _save_name is not None:
        plt.savefig(_save_name)
    plt.show()


info_dict = {
    'proposed': PickleRead('output_dir/solve_info/osqp_all_info_T'),
    'OSQP-CS': PickleRead('output_dir/solve_info/osqp_all_info_cs_T'),
    'IPOPT': PickleRead('output_dir/solve_info/nlp_all_info_T'),
    'LD-IPOPT': PickleRead('output_dir/solve_info/lnlp_all_info_T'),
    'SQP': PickleRead('output_dir/solve_info/lnlp_all_info_T')
}

extracted_data_dict = {
    'proposed': extract_data(info_dict['proposed']),
    'OSQP-CS': extract_data(info_dict['OSQP-CS']),
    'LD-IPOPT': extract_data(info_dict['LD-IPOPT'], _one_nominal=True),
    'IPOPT': extract_data(info_dict['IPOPT'], _one_nominal=True),
    'SQP': extract_data(info_dict['SQP'], _one_nominal=True),
}

###################################
rect_dict = {
    '1': {
        'pos': (10, -0.25),
        'w': 12,
        'l': 0.5,
        'color': 'r',
        'alg_name': ['SQP', 'LD-IPOPT', 'IPOPT', 'proposed', 'OSQP-CS']
    },
    '2': {
        'pos': (70, -0.25),
        'w': 20,
        'l': 0.75,
        'color': 'b',
        'alg_name': ['SQP', 'LD-IPOPT', 'IPOPT', 'proposed', 'OSQP-CS']
    },
}
draw_data(extracted_data_dict,
          name='control',
          _idx=0,
          _sub_fig_pos=[[0.09, 0.84, 0.85, 0.14],
                        [0.09, 0.65, 0.85, 0.14],
                        [0.09, 0.46, 0.85, 0.14],
                        [0.09, 0.27, 0.85, 0.14],
                        [0.09, 0.08, 0.85, 0.14]],
          _y_title='acc[$m/s^2$]',
          _x_title='time[ms]',
          _y_lim=(-0.5, 1.0),
          _x_lim=(-5, 125),
          _rect_dict=rect_dict,
          _save_name='output_dir/figs/acc.svg'
          )

###################################################
delta_phi_insert_dict = {
    '1': {
        'ins_pos': (0.01, 0.15, 0.15, 0.8),
        'x_lim': (42, 53),
        'y_lim': (-0.05, 0.05),
        'alg_name': ['proposed', 'OSQP-CS']
    },
}
rect_dict = {
    '1': {
        'pos': (10, -0.05),
        'w': 12,
        'l': 0.1,
        'color': 'r',
        'alg_name': ['SQP', 'LD-IPOPT', 'IPOPT']
    },
    '2': {
        'pos': (42, -0.05),
        'w': 11,
        'l': 0.1,
        'color': 'b',
        'alg_name': ['SQP', 'LD-IPOPT', 'IPOPT']
    },
    '3': {
        'pos': (75, -0.05),
        'w': 15,
        'l': 0.1,
        'color': 'cyan',
        'alg_name': ['SQP', 'LD-IPOPT', 'IPOPT']
    }
}
draw_data(extracted_data_dict,
          name='delta_phi',
          _idx=0,
          _sub_fig_pos=[[0.09, 0.84, 0.85, 0.14],
                        [0.09, 0.65, 0.85, 0.14],
                        [0.09, 0.46, 0.85, 0.14],
                        [0.09, 0.27, 0.85, 0.14],
                        [0.09, 0.08, 0.85, 0.14]],
          _x_lim=(-20, 125),
          _y_title=r'$\Delta\phi$[$rad/100ms$]',
          _x_title='time[ms]',
          _y_lim=(-0.15, 0.15),
          _insert_axes=delta_phi_insert_dict,
          _rect_dict=rect_dict,
          _save_name='output_dir/figs/delta_phi.svg'
          )


pass
