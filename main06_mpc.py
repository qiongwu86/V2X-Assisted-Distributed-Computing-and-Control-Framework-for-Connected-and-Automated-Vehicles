from solver_admm import ILMPC
from utilit import WatchOneVeh, PickleSave
from trace_generate import TraceTestForMPC3
import numpy as np
from tqdm import tqdm

ref_len = 550
run_len = 500

trace = TraceTestForMPC3()
# np.save('trace_origin.npy', trace)
mpc_solver = ILMPC(trace[0], run_len, trace)

act_trace = np.zeros((run_len, 4))
act_control = np.zeros((run_len, 2))
for t in tqdm(range(run_len)):
    x_current, _ = mpc_solver.Step()
    act_trace[t] = x_current
    act_control[t] = _
    
# np.save('act_trace.npy', act_trace)
# np.save('act_control.npy', act_control)
    
video = WatchOneVeh(act_trace, trace)
video.DrawVideo()

