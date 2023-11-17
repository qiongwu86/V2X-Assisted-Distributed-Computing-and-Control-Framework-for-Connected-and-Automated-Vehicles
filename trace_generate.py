import numpy as np
import math
from numpy.linalg import inv
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

def generate_trace(T_nums: int, k: float, T_a: float, t0: float, tf: float, x0: np.ndarray, xf: np.ndarray) -> np.ndarray:
    c_i = edge_condition(k, T_a, t0, tf, x0, xf)
    time_step_length = (tf - t0) / T_nums
    trace = np.zeros((T_nums, SDIM+CDIM))
    c1, c2, c3, c4, c5, c6 = c_i
    s_t = lambda t: c1 * t**3 / (12*k) + c2 * t**2 / (4*k) + c5 * t + c6 - c3 * T_a ** 3 * math.exp(t/T_a) / (4*k) + c4 * T_a ** 2 * math.exp(-t/T_a)
    v_t = lambda t: c1 * t**2 / (4*k) + c2 * t / (2*k) + c5 - c3 * T_a ** 2 * math.exp(t/T_a) / (4*k) - c4 * T_a * math.exp(-t/T_a)
    for i in range(T_nums):
        t = t0 + (i+1) * time_step_length
        trace[i][0] = s_t(t)
        trace[i][1] = v_t(t)
    trace = trace.reshape((-1))
    return trace
        

def edge_condition(k: float, T_a: float, t0: float, tf: float, x0: np.ndarray, xf: np.ndarray) -> np.ndarray:    
    result = np.zeros((SDIM * 2, SDIM * 2))
    ### init state
    # line 0, s0
    result[0][0] = t0 ** 3 / (12*k)
    result[0][1] = t0 ** 2 / (4*k)
    result[0][2] = - (T_a ** 3 * math.exp(t0/T_a)) / (4*k)
    result[0][3] = T_a ** 2 * math.exp(-t0/T_a)
    result[0][4] = t0
    result[0][5] = 1
    # line 1, v0
    result[1][0] = t0 ** 2 / (4*k)
    result[1][1] = t0 ** 1 / (2*k)
    result[1][2] = - (T_a ** 2 * math.exp(t0/T_a)) / (4*k)
    result[1][3] = - T_a ** 1 * math.exp(-t0/T_a)
    result[1][4] = 1
    result[1][5] = 0
    # line 2, a0
    result[2][0] = t0 / (2*k)
    result[2][1] = 1 / (2*k)
    result[2][2] = - (T_a * math.exp(t0/T_a)) / (4*k)
    result[2][3] = math.exp(-t0/T_a)
    result[2][4] = 0
    result[2][5] = 0
    ## final state
    # line 3, sf
    result[3][0] = tf ** 3 / (12*k)
    result[3][1] = tf ** 2 / (4*k)
    result[3][2] = - (T_a ** 3 * math.exp(tf/T_a)) / (4*k)
    result[3][3] = T_a ** 2 * math.exp(-tf/T_a)
    result[3][4] = tf
    result[3][5] = 1
    # line 4, vf
    result[4][0] = tf ** 2 / (4*k)
    result[4][1] = tf ** 1 / (2*k)
    result[4][2] = - (T_a ** 2 * math.exp(tf/T_a)) / (4*k)
    result[4][3] = - T_a ** 1 * math.exp(-tf/T_a)
    result[4][4] = 1
    result[4][5] = 0
    # line 5, af
    result[5][0] = tf / (2*k)
    result[5][1] = 1 / (2*k)
    result[5][2] = - (T_a * math.exp(tf/T_a)) / (4*k)
    result[5][3] = math.exp(-tf/T_a)
    result[5][4] = 0
    result[5][5] = 0
    ## inv
    inv_A = inv(result)
    b = np.concatenate((x0[:3], xf[:3]))
    ## c1-c6
    c = inv_A @ b
    return c

