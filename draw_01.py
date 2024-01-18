import matplotlib.pyplot as plt
from utilits import *
import numpy as np
import matplotlib as mpl
from typing import Tuple, Dict


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


##########################################
info_dict = {
    'proposed': PickleRead('output_dir/solve_info/osqp_solve_info'),
    'OSQP-CS': PickleRead('output_dir/solve_info/osqp_solve_info_cs'),
    # 'IPOPT': PickleRead('output_dir/solve_info/nlp_solve_info'),
    # 'LD-IPOPT': PickleRead('output_dir/solve_info/lnlp_solve_info')
}

data_for_draw = {
    title: extract_data(origin_data) for title, origin_data in info_dict.items()
}
run_time_max = max([np.max(data['run_time']) for data in data_for_draw.values()])
run_time_min = min([np.min(data['run_time']) for data in data_for_draw.values()])
print(run_time_min, run_time_max)

# # fig = plt.figure()
fig, axs = plt.subplots(len(info_dict)+1, 1, figsize=(8, 3))
[
    _ax.imshow(data_for_draw[_alg_name]['run_time'], cmap='plasma', vmin=run_time_min, vmax=run_time_max)
    for _alg_name, _ax in zip(data_for_draw, axs)
]
[
    _ax.set_title(_alg_name)
    for _alg_name, _ax in zip(data_for_draw, axs)
]
fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(1000*run_time_min, 1000*run_time_max), cmap='plasma'),
             cax=axs[-1], orientation='vertical', label='run time[ms]')

axs[0].set_position([0.05, 0.08, 0.75, 0.3])
axs[1].set_position([0.05, 0.5, 0.75, 0.3])
axs[-1].set_position([0.85, 0.08, 0.05, 0.7])
plt.savefig('output_dir/figs/run_time_01.svg')
plt.show()
plt.close()

iter_times_max = max([np.max(data['iter_times']) for data in data_for_draw.values()])
iter_times_min = min([np.min(data['iter_times']) for data in data_for_draw.values()])
print(iter_times_max, iter_times_min)

# # fig = plt.figure()
fig, axs = plt.subplots(len(info_dict)+1, 1, figsize=(8, 3))
[
    _ax.imshow(data_for_draw[_alg_name]['iter_times'], cmap='plasma', vmin=iter_times_min, vmax=iter_times_max)
    for _alg_name, _ax in zip(data_for_draw, axs)
]
[
    _ax.set_title(_alg_name)
    for _alg_name, _ax in zip(data_for_draw, axs)
]
fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(iter_times_min, iter_times_max), cmap='plasma'),
             cax=axs[-1], orientation='vertical', label='iter times')

axs[0].set_position([0.05, 0.08, 0.75, 0.3])
axs[1].set_position([0.05, 0.5, 0.75, 0.3])
axs[-1].set_position([0.85, 0.08, 0.05, 0.7])
plt.savefig('output_dir/figs/iter_times_01.svg')
plt.show()
plt.close()


##########################################
info_dict = {
    # proposed=PickleRead('output_dir/solve_info/osqp_solve_info'),
    # osqp_cs=PickleRead('output_dir/solve_info/osqp_solve_info_cs'),
    'IPOPT': PickleRead('output_dir/solve_info/nlp_solve_info'),
    'LD-IPOPT': PickleRead('output_dir/solve_info/lnlp_solve_info')
}
data_for_draw = {
    title: extract_data(origin_data) for title, origin_data in info_dict.items()
}

run_time_max = max([np.max(data['run_time']) for data in data_for_draw.values()])
run_time_min = min([np.min(data['run_time']) for data in data_for_draw.values()])
print(run_time_min, run_time_max)

# # fig = plt.figure()
fig, axs = plt.subplots(len(info_dict)+1, 1, figsize=(8, 3))
[
    _ax.imshow(data_for_draw[_alg_name]['run_time'], cmap='plasma', vmin=run_time_min, vmax=run_time_max)
    for _alg_name, _ax in zip(data_for_draw, axs)
]
[
    _ax.set_title(_alg_name)
    for _alg_name, _ax in zip(data_for_draw, axs)
]
fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(1000*run_time_min, 1000*run_time_max), cmap='plasma'),
             cax=axs[-1], orientation='vertical', label='run time[ms]')

axs[0].set_position([0.05, 0.08, 0.75, 0.3])
axs[1].set_position([0.05, 0.5, 0.75, 0.3])
axs[-1].set_position([0.85, 0.08, 0.05, 0.7])
plt.savefig('output_dir/figs/run_time_02.svg')
plt.show()
plt.close()

iter_times_max = max([np.max(data['iter_times']) for data in data_for_draw.values()])
iter_times_min = min([np.min(data['iter_times']) for data in data_for_draw.values()])
print(iter_times_max, iter_times_min)

# # fig = plt.figure()
fig, axs = plt.subplots(len(info_dict)+1, 1, figsize=(8, 3))
[
    _ax.imshow(data_for_draw[_alg_name]['iter_times'], cmap='plasma', vmin=iter_times_min, vmax=iter_times_max)
    for _alg_name, _ax in zip(data_for_draw, axs)
]
[
    _ax.set_title(_alg_name)
    for _alg_name, _ax in zip(data_for_draw, axs)
]
fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(iter_times_min, iter_times_max), cmap='plasma'),
             cax=axs[-1], orientation='vertical', label='iter times')

axs[0].set_position([0.05, 0.08, 0.75, 0.3])
axs[1].set_position([0.05, 0.5, 0.75, 0.3])
axs[-1].set_position([0.85, 0.08, 0.05, 0.7])
plt.savefig('output_dir/figs/iter_times_02.svg')
plt.show()
plt.close()

