from env import EnvParam
from trace_generate import AddTrace
import matplotlib.pyplot as plt
from dynamic_model import OneDimDynamic, init_system
from utilit import ProcessTrace, PickleSave, PickleRead
import pickle
from solver_admm import WeightedADMM, FullADMM


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


ProcessTrace(result)

for _ in result:
    print("id: {0}, state {1}".format(_, result[_][:-1]))

graph = 'full'
round = 50

#################
WeightedADMM.makeAD(result, graph)
for id in result:
    WeightedADMM(id, len(result), result[id])
    
for i in range(round):
    print('---------------------------------')
    for id in WeightedADMM.all_solver:
        WeightedADMM.all_solver[id].SolveP1()
    for id in WeightedADMM.all_solver:
        WeightedADMM.all_solver[id].SolveP2()
    
    if (i+1) % 10 == 0:
        fig = plt.figure()
        for id in WeightedADMM.all_solver:
            trace = WeightedADMM.all_solver[id].x.reshape((-1, 4))[:, 0]
            color = 'red' if WeightedADMM.all_solver[id].lane == 'main' else 'green'
            plt.plot(trace, color=color)
        plt.plot([0, len(trace)], [0, 0], color='black')
        fig.savefig('figs/' + graph+"_test_"+str(i+1)+".jpg")
#################
# for id in result:
    # FullADMM(id, len(result), result[id])
    
# for i in range(round):
    # print('---------------------------------')
    # for id in FullADMM.all_solver:
        # FullADMM.all_solver[id].SolveP1()
    # for id in FullADMM.all_solver:
        # FullADMM.all_solver[id].SolveP2()
    
# fig = plt.figure()
# for id in FullADMM.all_solver:
    # trace = FullADMM.all_solver[id].x.reshape((-1, 4))[:, 0]
    # color = 'red' if FullADMM.all_solver[id].lane == 'main' else 'green'
    # plt.plot(trace, color=color)




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


# fig.show()



