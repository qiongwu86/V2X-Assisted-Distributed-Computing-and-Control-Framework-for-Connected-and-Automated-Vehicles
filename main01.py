import numpy as np
from trace_generate import generate_test_trace
from dynamic_models.dynamic_model import init_system
from solver_admm import InitSolver, SolverAdmm
import matplotlib.pyplot as plt

veh_num = 5
T_nums = 100

init_system()
InitSolver(T_nums, veh_num)

all_init_state = dict()
all_init_state[1] = np.array([25, 13, 0, 0])
all_init_state[2] = np.array([21, 12, 0, 0])
all_init_state[3] = np.array([13, 10, 0, 0])
all_init_state[4] = np.array([9, 11, 0, 0])
all_init_state[5] = np.array([5, 16, 0, 0])

all_final_state = dict()
for i in range(veh_num):
    all_final_state[i+1] = np.array([130+25-5*i, 13, 0, 0])

all_trace = dict()
for i in range(veh_num):
    id = i+1
    all_trace[id] = generate_test_trace(T_nums, all_init_state[id], all_final_state[id])

all_solver = [SolverAdmm(id, all_init_state[id], all_trace[id]) for id in range(1, veh_num+1)]
for i in range(10):
    y_dict = dict()
    for s in all_solver:
        y_dict[s.ID] = np.array([s.y])
    for s in all_solver:
        other_y = list()
        for os in all_solver:
            if os == s:
                continue
            other_y.append(y_dict[os.ID])
        other_y = np.concatenate(other_y)
        s.step(other_y)
        #print(s.x)

        
plt.figure()
x_data = np.arange(T_nums)
true_state = dict()
for s in all_solver:
    temp = s.x.reshape((-1, 4))
    true_state[s.ID] = temp[:, 0]
    plt.plot(x_data, true_state[s.ID])
plt.show()
    
        
