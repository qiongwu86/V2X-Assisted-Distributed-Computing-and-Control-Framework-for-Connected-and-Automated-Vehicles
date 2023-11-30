from env import EnvParam
from trace_generate import AddTrace
import matplotlib.pyplot as plt
from dynamic_model import OneDimDynamic, init_system
from utilit import ProcessTrace

init_system()

result = EnvParam.generate_init_state()
for _ in result:
    print("id: {0}, state {1}".format(_, result[_]))

# generate_trace
all_trace_main = []
all_trace_merge = []
AddTrace(result, reshape=False)
for id in result:
    lane, t0, tf, to, x0, xf, iop, iof, oiop, oiof, one_trace = result[id]
    # include init point
    # total_trace_length = int((tf - t0) / OneDimDynamic.Td) + 1
    if result[id][0] == "main":
        all_trace_main.append((to, one_trace[:, 0]))
    elif result[id][0] == "merge":
        all_trace_merge.append((to, one_trace[:, 0]))
    else:
        raise ValueError

ProcessTrace(result)

for _ in result:
    print("id: {0}, state {1}".format(_, result[_][:-1]))

    
fig = plt.figure()
for one_trace in all_trace_main:
    to, line = one_trace
    to = to * 10
    plt.plot(line, color="green")
    plt.plot([to, to], [-100, 80], color="black")
for one_trace in all_trace_merge:
    to, line = one_trace
    to = to * 10
    plt.plot(line, color="red")
    plt.plot([to, to], [-100, 80], color="black")

plt.plot([0, 175], [0, 0], color='black')


fig.savefig("test.jpg")
fig.show()



