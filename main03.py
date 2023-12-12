from env import EnvParam
from trace_generate import AddTrace
import matplotlib.pyplot as plt
from dynamic_model import OneDimDynamic, init_system
from utilit import ProcessTrace, SceneDraw
import pickle
from solver_admm import WeightedADMM, FullADMM
import numpy as np


init_system()

result = EnvParam.generate_init_state()
AddTrace(result, reshape=False)
ProcessTrace(result)
# gen add y
phi = np.pi / 10
all_traj = {}
for id in result:
    all_traj[id] = {'lane': result[id][0]}
    one_trace = result[id][-1]
    temp = one_trace.copy()
    to = np.nonzero(one_trace[:, 0] > 0)[0][0]
    if all_traj[id]['lane'] == 'main':
        temp[:, 1] = 0
        temp[:, 2] = 0
    else:
        temp[:to, 0] = -np.abs(one_trace[:to, 0]) * np.cos(phi)
        temp[:to, 1] = -np.abs(one_trace[:to, 0]) * np.sin(phi)
        temp[to:, 1] = 0.0
        temp[:to, 2] = phi
    all_traj[id]['X'] = temp

drawer = SceneDraw(len(all_traj), 91, all_traj)
drawer.GenVideo()




