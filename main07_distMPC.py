from trace_generate import TraceTestForMPC4
from utilit import SceneDraw
from solver_admm import ILMPC
from tqdm import tqdm
import numpy as np



managed_trace = TraceTestForMPC4()



# mpc
run_len = 150
traj_len = run_len
all_trace_for_draw = {}
for id in managed_trace:
    trace = managed_trace[id]
    mpc_solver = ILMPC(id, trace[0], run_len, trace)

    all_trace_for_draw[id] = {
        'X' : np.zeros((run_len, 4)), 
        'U' : np.zeros((run_len, 2)),
        'lane' : 'main'
    }

for t in tqdm(range(run_len)):
    ILMPC.UpdateNominalOther()
    for id in ILMPC.all_solver:
        x_current, _ = ILMPC.all_solver[id].Step()
        all_trace_for_draw[id]['X'][t] = x_current
        all_trace_for_draw[id]['U'][t] = _

Drawer = SceneDraw(len(all_trace_for_draw), traj_len, all_trace_for_draw)
Drawer.GenVideo()

