import matplotlib.pyplot as plt
from utilits import *
import numpy as np
import matplotlib as mpl
from typing import Tuple, Dict

info_dict = dict(
    # proposed=PickleRead('output_dir/solve_info/osqp_solve_info'),
    # osqp_cs=PickleRead('output_dir/solve_info/osqp_solve_info_cs'),
    ipopt=PickleRead('output_dir/solve_info/nlp_solve_info'),
    ld_ipopt=PickleRead('output_dir/solve_info/lnlp_solve_info')
)


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
    return dict(run_time=_run_time_mat, iter_times=_run_time_mat)


data_for_draw = {
    title: extract_data(origin_data) for title, origin_data in info_dict.items()
}

run_time_max = max([np.max(data['iter_times']) for data in data_for_draw.values()])
run_time_min = min([np.min(data['iter_times']) for data in data_for_draw.values()])
print(run_time_min, run_time_max)


pass
# # fig = plt.figure()
fig, axs = plt.subplots(len(info_dict)+1, 1, figsize=(8, 3))
[
    _ax.imshow(data_for_draw[_alg_name]['iter_times'], cmap='plasma', vmin=run_time_min, vmax=run_time_max)
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
plt.show()
# _draw_subplots(axs[0], all_info_osqp)
# im1 = axs[0].set_title('proposed')
# _draw_subplots(axs[1], all_info_osqp_cs)
# im2 = axs[1].set_title('OSQP-CS')
# _draw_subplots(axs[2], all_info_nlp)
# im3 = axs[2].set_title('IPOPT')
# _draw_subplots(axs[3], all_info_lnlp)
# im4 = axs[3].set_title('LD-IPOPT')
#
# fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap='plasma'),
#              ax=axs[0], orientation='horizontal', label='a colorbar label')
#
# # fig.colorbar(im1, ax=axs.ravel().tolist())
# plt.show()

'''

import matplotlib.pyplot as plt
import numpy as np

# 创建数据
data = np.random.rand(5, 5)

# 绘制热力图
plt.imshow(data, cmap='plasma')

# 显示颜色条
plt.colorbar()

# 显示图形
plt.show()
'''
