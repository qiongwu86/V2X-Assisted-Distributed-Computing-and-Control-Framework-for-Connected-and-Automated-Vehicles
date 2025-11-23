from casadi import *
from utilits import suppress_stdout_stderr

delta_T = 0.1
_state = casadi.SX.sym('state', 4)
_control = casadi.SX.sym('control', 2)
_state_dot = casadi.vertcat(
    _state[3, 0] * casadi.cos(_state[2, 0] + casadi.arctan(0.5 * casadi.tan(_control[1, 0]))),
    _state[3, 0] * casadi.sin(_state[2, 0] + casadi.arctan(0.5 * casadi.tan(_control[1, 0]))),
    _state[3, 0] * casadi.sin(casadi.arctan(0.5 * casadi.tan(_control[1, 0]))) / (0.5 * 3.5),
    _control[0, 0]
)
_state_next = casadi.Function('state_next_func', [_state, _control], [delta_T * _state_dot + _state])

# suppress = suppress_stdout_stderr()
OTHER_MAX = 4
PRED_LEN = 30


# param
x_0 = MX.sym('x_0', 4)
x_ref_list = vertcat(*[MX.sym('x_ref_'+str(i+1), 4) for i in range(PRED_LEN)])
x_nominal_other_list = vertcat(*[
    vertcat(*[MX.sym('x_no_'+str(o)+str(i+1), 4) for i in range(PRED_LEN)])
    for o in range(OTHER_MAX)
])

# dec var
x_1_T_list = [MX.sym('x_'+str(i+1), 4) for i in range(PRED_LEN)]
u_0_T_1_list = [MX.sym('u_'+str(i), 2) for i in range(PRED_LEN)]

# constrain
lbg_list = [0, 0, 0, 0]
ubg_list = [0, 0, 0, 0]
g_list = [x_1_T_list[0] - _state_next(x_0, u_0_T_1_list[0])]
for i in range(1, PRED_LEN):
    g_list.append(x_1_T_list[i] - _state_next(x_1_T_list[i-1], u_0_T_1_list[i]))
    lbg_list += [0, 0, 0, 0]
    ubg_list += [0, 0, 0, 0]

x_1_T = vertcat(*x_1_T_list)
u_0_T_1 = vertcat(*u_0_T_1_list)
g = vertcat(*g_list)
lbg = vertcat(*lbg_list)
ubg = vertcat(*ubg_list)
x_ref = vertcat(*x_ref_list)


_Qx = diag([1.0, 1.0, 0.0, 0.0])
Qx = kron(MX.eye(PRED_LEN), _Qx)

_Qu = diag([0.1, 0.1])
Qu = kron(MX.eye(PRED_LEN), _Qu)


J = (x_1_T - x_ref).T @ Qx @ (x_1_T - x_ref) + u_0_T_1.T @ Qu @ u_0_T_1

nlp = {
    'x': vertcat(u_0_T_1, x_1_T),
    'f': J,
    'g': g,
    'p': vertcat(x_0, x_ref)
}

prob_option = dict(
    # verbose=False,
    # verbose_init=False,
    # print_in=False,
    # print_out=False,
    # print_time=False,
    record_time=True,
)

x_0_value = np.zeros((4,))

px_ref = np.array([i*0.3 for i in range(PRED_LEN)])
py_ref = np.zeros_like(px_ref)
phi_ref = np.zeros_like(px_ref)
v_ref = np.zeros_like(px_ref)
x_ref_value = np.vstack((px_ref, py_ref, phi_ref, v_ref)).transpose().reshape(-1)

p = np.concatenate((x_0_value, x_ref_value))

# print(S_ipopt)
with suppress_stdout_stderr():
    S_ipopt = nlpsol('S_ipopt', 'ipopt', nlp, prob_option)
    r_ipopt = S_ipopt(lbg=lbg, ubg=ubg, p=p)
x_opt_ipopt = r_ipopt['x']
print('x_opt: ', x_opt_ipopt, S_ipopt.stats())

# with suppress_stdout_stderr():
#     r_ipopt = S_ipopt(lbg=lbg, ubg=ubg, p=[0.0, 0.0, 0.0, 3.0])
# x_opt_ipopt = r_ipopt['x']
# print('x_opt: ', x_opt_ipopt)

#
# with suppress_stdout_stderr():
#     S_sqp = nlpsol('S_sqpmethod', 'feasiblesqpmethod', qp, prob_option)
#     r_sqp = S_sqp(x0=[30, 30], lbg=1, p=0.6)
# x_opt_sqp = r_sqp['x']
# print('x_opt: ', x_opt_sqp)
