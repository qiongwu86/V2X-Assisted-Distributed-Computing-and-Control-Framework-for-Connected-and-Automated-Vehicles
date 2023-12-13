from env import EnvParam
from trace_generate import AddTrace, TraceTestForMPC
import matplotlib.pyplot as plt
from dynamic_model import OneDimDynamic, init_system
from utilit import ProcessTrace, PickleSave, PickleRead
import pickle
from solver_admm import WeightedADMM, FullADMM, ILMPC


init_system()

# 
result = EnvParam.generate_init_state()
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
for one_trace in all_trace_merge:
    to, line = one_trace
    to = to * 10
    plt.plot(line, color="green")
for one_trace in all_trace_main:
    to, line = one_trace
    to = to * 10
    plt.plot(line, color="red")
plt.savefig("figs/ref_trace.jpg")


# process for trajectory planning by ADMM
ProcessTrace(result)
for _ in result:
    print("id: {0}, state {1}".format(_, result[_][:-1]))

# solve
round = 20

for id in result:
    FullADMM(id, len(result), result[id], X_constrain=False)
    
for i in range(round):
    print('---------------------------------')
    for id in FullADMM.all_solver:
        FullADMM.all_solver[id].UpdateY()
    for id in FullADMM.all_solver:
        FullADMM.all_solver[id].Solve()
    
    if (i+1) % 10 == 0:
        fig = plt.figure()
        for id in FullADMM.all_solver:
            trace = FullADMM.all_solver[id].x.reshape((-1, 4))[:, 0]
            color = 'red' if FullADMM.all_solver[id].lane == 'main' else 'green'
            plt.plot(trace, color=color)
        plt.plot([0, len(trace)], [0, 0], color='black')
        fig.savefig('figs/' + "full" +"_test_"+str(i+1)+".jpg")
    
# 


