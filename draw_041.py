"""
平均计算时间、迭代次数
"""
import matplotlib.pyplot as plt
from utilits import *
import numpy as np
import matplotlib as mpl
from typing import Dict, Text, List


WHAT = '12'
LANG = 'EN'
plt.rcParams.update({'font.size': 10,
                     "text.usetex": True})
plt.rcParams['font.family'] = 'Times New Roman'


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


# 定义数据
# algorithms = ['proposed', 'OSQP-CS']
# indicators = {'T cross': 'T', '1 round': '1', '2 round': '2', '3 round': '3'}
algorithms = ['Our']
indicators = {'3': '3', '4': '4', '5': '5', '6': '6', '7': '7',
              '8': '8', '9': '9', '10': '10'}
all_data = np.zeros((len(algorithms), len(indicators)))
_tmp = []
for ind_idx, ind_name in enumerate(indicators.keys()):
    info_dict = {
        'proposed': PickleRead('output_dir/solve_info/osqp_solve_info_{}'.format(indicators[ind_name])),
    }
    # info_dict['SQP']
    indicator_data = []
    for alg_name, alg_data in info_dict.items():
        mean_run_time = []
        for v_data in alg_data.values():
            mean_run_time.append(np.mean(v_data['run_time']))
        _tmp.append(np.max([1000*np.max(_data['run_time']) for _, _data in alg_data.items()]))
        indicator_data.append(1000 * np.mean(mean_run_time))
    all_data[:, ind_idx] = np.array(indicator_data)

print(all_data)
print(_tmp)

# D:\onramp_admm\.venv\Scripts\python.exe D:\onramp_admm\draw_041.py
# [[ 4.07416734  5.05435684  6.21942273  7.10118552  7.93798858  8.65174882
#    9.44424441 10.21078244]]
# [5.584328729248047, 10.65381636352539, 11.527995999145508, 13.94775630493164, 13.974142343139649, 14.331927618408203, 15.940529953002928, 18.393506045532227]
