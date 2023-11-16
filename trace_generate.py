import numpy as np
from dynamic_model import OneDimDynamic

SDIM = OneDimDynamic.SDIM
CDIM = OneDimDynamic.CDIM

def generate_test_trace(T_nums: int, init_state: np.ndarray, final_state: np.ndarray) -> np.ndarray:
    ref_traj = np.zeros((T_nums, SDIM+CDIM))
    temp = (final_state[0]-init_state[0]) / T_nums
    for i in range(T_nums):
        ref_traj[i, 0] = init_state[0] + (i+1) * temp
    #assert init_state[1] == final_state[1]
    ref_traj[:, 1] = final_state[1]
    ref_traj = ref_traj.reshape((-1))
    print(ref_traj)
    return ref_traj