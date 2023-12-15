from env import EnvParam
from trace_generate import AddTrace, TraceTestForMPC
import matplotlib.pyplot as plt
from dynamic_model import OneDimDynamic, init_system
from utilit import ProcessTrace, PickleSave, PickleRead, SceneDraw
import pickle
from solver_admm import WeightedADMM, FullADMM, ILMPC
from tqdm import tqdm
import numpy as np


init_system()

# 
result = EnvParam.generate_init_state()
PickleSave(result, 'init_state/init_state.pkl')
# result = PickleRead('init_state/init_state.pkl')
for _ in result:
    print("id: {0}, state {1}".format(_, result[_]))

# generate_trace
AddTrace(result, reshape=True)
all_trace_main = []
all_trace_merge = []
for id in result:
    lane, t0, tf, to, x0, xf, iop, iof, oiop, oiof, one_trace = result[id]
    # include init point
    # total_trace_length = int((tf - t0) / OneDimDynamic.Td) + 1
    if result[id][0] == "main":
        one_trace = one_trace.reshape((-1, 4))
        all_trace_main.append((to, one_trace[:, 0]))
    elif result[id][0] == "merge":
        one_trace = one_trace.reshape((-1, 4))
        all_trace_merge.append((to, one_trace[:, 0]))
    else:
        raise ValueError

fig = plt.figure()
for one_trace in all_trace_merge:
    to, line = one_trace
    to = to * 10
    plt.plot(line, color="green", linestyle='-')
for one_trace in all_trace_main:
    to, line = one_trace
    to = to * 10
    plt.plot(line, color="red", linestyle='-')
fig.savefig('figs/ref_traj.jpg')
plt.close()

# process for trajectory planning by ADMM
ProcessTrace(result)
for _ in result:
    print("id: {0}, state {1}".format(_, result[_][:-1]))

# solve
round = 100

for id in result:
    FullADMM(id, len(result), result[id], X_constrain=False)
    
for i in tqdm(range(round)):
    for id in FullADMM.all_solver:
        FullADMM.all_solver[id].UpdateY()
    for id in FullADMM.all_solver:
        FullADMM.all_solver[id].Solve()
    
    if (i+1) % 10 == 0:
        fig = plt.figure()

        # for one_trace in all_trace_merge:
            # to, line = one_trace
            # to = to * 10
            # plt.plot(line, color="green", linestyle='--')
        # for one_trace in all_trace_main:
            # to, line = one_trace
            # to = to * 10
            # plt.plot(line, color="red", linestyle='--')

        for id in FullADMM.all_solver:
            trace = FullADMM.all_solver[id].x.reshape((-1, 4))[:, 0]
            color = 'red' if FullADMM.all_solver[id].lane == 'main' else 'green'
            plt.plot(trace, color=color)
        plt.savefig('figs/' + "full" +"_test_"+str(i+1)+".jpg")
        plt.close()
    
# 
all_trace_opt = FullADMM.CollectTraces()
for id in all_trace_opt:
    lane, trace = all_trace_opt[id]
    x, v, a = trace[-1]
    print('final state: ID:{0:3.0f}, x: {1:5.2f}, v: {2:5.2f}, a: {3:5.2f}'.format(id, x, v, a))

managed_trace = EnvParam.TraceArrange(all_trace_opt)
for id in managed_trace:
    trace = managed_trace[id]
    fig = plt.figure()
    plt.plot(trace[:, 0], trace[:, 1])
    plt.savefig('tmp/'+str(id) + '.jpg')
    plt.close()


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

