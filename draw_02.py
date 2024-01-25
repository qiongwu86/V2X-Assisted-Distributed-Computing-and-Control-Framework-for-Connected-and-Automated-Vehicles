import matplotlib.pyplot as plt
from utilits import *
import numpy as np
import matplotlib as mpl
from typing import Dict, Text, List


def draw_one_svg(_data_for_draw: Dict, _draw_what: Text, _label: Text, _save_name: Text, _rescale: float, _sub_fig_pos: List):
    _data_for_draw = {
        title: extract_data(origin_data) for title, origin_data in info_dict.items()
    }
    _value_max = max([np.max(data[_draw_what]) for data in _data_for_draw.values()])
    _value_min = min([np.min(data[_draw_what]) for data in _data_for_draw.values()])
    print(_value_min, _value_max)

    # # fig = plt.figure()
    fig, axs = plt.subplots(len(info_dict)+1, 1, figsize=(8, 1.5*len(info_dict)))
    [
        _ax.imshow(_data_for_draw[_alg_name][_draw_what], cmap='plasma', vmin=_value_min, vmax=_value_max)
        for _alg_name, _ax in zip(_data_for_draw, axs)
    ]
    [
        _ax.set_title(_alg_name)
        for _alg_name, _ax in zip(_data_for_draw, axs)
    ]
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(_rescale*_value_min, _rescale*_value_max), cmap='plasma'),
                 cax=axs[-1], orientation='vertical', label=_label)

    [_ax.set_position(_sub_fig_pos[i]) for i, _ax in enumerate(axs)]
    # axs[0].set_position([0.05, 0.08, 0.75, 0.3])
    # axs[1].set_position([0.05, 0.5, 0.75, 0.3])
    # axs[-1].set_position([0.85, 0.08, 0.05, 0.7])
    plt.savefig('output_dir/figs/'+_save_name)
    plt.show()
    plt.close()


def extract_data(_all_info) -> Dict:
    _new_data = {v_id: dict() for v_id in _all_info.keys()}
    for v_id, _data in _all_info.items():
        _new_data[v_id]['run_time'] = np.sum(_data['run_time'], axis=1)
        _new_data[v_id]['iter_times'] = np.sum(_data['iter_times'], axis=1)

    _run_time_mat = np.vstack([
        _new_data[v_id]['run_time'] for v_id in _new_data.keys()
    ])
    _iter_times_mat = np.vstack([
        _new_data[v_id]['iter_times'] for v_id in _new_data.keys()
    ])
    return dict(run_time=_run_time_mat, iter_times=_iter_times_mat)


info_dict = {
    'proposed': PickleRead('output_dir/solve_info/osqp_solve_info_12'),
    'OSQP-CS': PickleRead('output_dir/solve_info/osqp_solve_info_cs_12'),
    # 'SQP': PickleRead('output_dir/solve_info/sqp_solve_info_12'),
    'IPOPT': PickleRead('output_dir/solve_info/nlp_solve_info_12'),
    'LD-IPOPT': PickleRead('output_dir/solve_info/lnlp_solve_info_12'),
}


draw_one_svg(
    _data_for_draw=info_dict,
    _draw_what='run_time',
    _label='run time[ms]',
    _save_name='run_time_all.svg',
    _rescale=1e3,
    _sub_fig_pos=[[0.05, -0.05, 0.75, 0.3],
                  [0.05, 0.15, 0.75, 0.3],
                  # [0.05, 0.35, 0.75, 0.3],
                  [0.05, 0.55, 0.75, 0.3],
                  [0.05, 0.75, 0.75, 0.3],
                  [0.85, 0.05, 0.05, 0.9]]
)

##########################################
info_dict = {
    'proposed': PickleRead('output_dir/solve_info/osqp_solve_info_12'),
    'OSQP-CS': PickleRead('output_dir/solve_info/osqp_solve_info_cs_12'),
    # 'IPOPT': PickleRead('output_dir/solve_info/nlp_solve_info'),
    # 'LD-IPOPT': PickleRead('output_dir/solve_info/lnlp_solve_info')
}

draw_one_svg(
    _data_for_draw=info_dict,
    _draw_what='run_time',
    _label='run time[ms]',
    _save_name='run_time_01.svg',
    _rescale=1e3,
    _sub_fig_pos=[[0.05, 0.08, 0.75, 0.3],
                  [0.05, 0.5, 0.75, 0.3],
                  [0.85, 0.08, 0.05, 0.7]]
)

draw_one_svg(
    _data_for_draw=info_dict,
    _draw_what='iter_times',
    _label='iter times',
    _save_name='iter_times_01.svg',
    _rescale=1,
    _sub_fig_pos=[[0.05, 0.08, 0.75, 0.3],
                  [0.05, 0.5, 0.75, 0.3],
                  [0.85, 0.08, 0.05, 0.7]]
)


##########################################
info_dict = {
    # proposed=PickleRead('output_dir/solve_info/osqp_solve_info'),
    # osqp_cs=PickleRead('output_dir/solve_info/osqp_solve_info_cs'),
    'IPOPT': PickleRead('output_dir/solve_info/nlp_solve_info_3'),
    'LD-IPOPT': PickleRead('output_dir/solve_info/lnlp_solve_info_3'),
    'SQP': PickleRead('output_dir/solve_info/sqp_solve_info_3'),
}
draw_one_svg(
    _data_for_draw=info_dict,
    _draw_what='run_time',
    _label='run time[ms]',
    _save_name='run_time_02.svg',
    _rescale=1e3,
    _sub_fig_pos=[[0.05, 0.08, 0.75, 0.3],
                  [0.05, 0.38, 0.75, 0.3],
                  [0.05, 0.68, 0.75, 0.3],
                  [0.85, 0.10, 0.05, 0.8]]
)

draw_one_svg(
    _data_for_draw=info_dict,
    _draw_what='iter_times',
    _label='iter times',
    _save_name='iter_times_02.svg',
    _rescale=1,
    _sub_fig_pos=[[0.05, 0.08, 0.75, 0.3],
                  [0.05, 0.38, 0.75, 0.3],
                  [0.05, 0.68, 0.75, 0.3],
                  [0.85, 0.10, 0.05, 0.8]]
)