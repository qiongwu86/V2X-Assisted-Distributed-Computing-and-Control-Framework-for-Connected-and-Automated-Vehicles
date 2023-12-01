from env import EnvParam
from trace_generate import AddTrace
import matplotlib.pyplot as plt
from dynamic_model import OneDimDynamic, init_system
from utilit import ProcessTrace, PickleSave, PickleRead
import pickle
from solver_admm import WeightedADMM


init_system()

result = EnvParam.generate_init_state()
# PickleSave(result, 'init_state.pkl')
result = PickleRead('init_state/init_state.pkl')

for _ in result:
    print("id: {0}, state {1}".format(_, result[_]))

# generate_trace
all_trace_main = []
all_trace_merge = []
AddTrace(result, reshape=True)
# for id in result:
    # lane, t0, tf, to, x0, xf, iop, iof, oiop, oiof, one_trace = result[id]
    # # include init point
    # # total_trace_length = int((tf - t0) / OneDimDynamic.Td) + 1
    # if result[id][0] == "main":
        # all_trace_main.append((to, one_trace[:, 0]))
    # elif result[id][0] == "merge":
        # all_trace_merge.append((to, one_trace[:, 0]))
    # else:
        # raise ValueError

ProcessTrace(result)

for _ in result:
    print("id: {0}, state {1}".format(_, result[_][:-1]))

WeightedADMM.makeAD(result)
for id in result:
    WeightedADMM(id, len(result), result[id])
    
for i in range(50):
    for id in WeightedADMM.all_solver:
        WeightedADMM.all_solver[id].Solve()
    
fig = plt.figure()
for id in WeightedADMM.all_solver:
    trace = WeightedADMM.all_solver[id].x.reshape((-1, 4))[:, 0]
    color = 'red' if WeightedADMM.all_solver[id].lane == 'main' else 'green'
    plt.plot(trace, color=color)
    # to, line = one_trace
    # to = to * 10
    # plt.plot(line, color="green")
    # plt.plot([to, to], [-100, 80], color="black")
# for one_trace in all_trace_merge:
    # to, line = one_trace
    # to = to * 10
    # plt.plot(line, color="red")
    # plt.plot([to, to], [-100, 80], color="black")

# plt.plot([0, 175], [0, 0], color='black')


fig.savefig("test1.jpg")
# fig.show()



