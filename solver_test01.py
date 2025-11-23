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
x_0 = MX.sym('x_0', 4)
_x_current = x_0
x_others_nominal = MX.sym('xo_nomi', PRED_LEN * 4, 1, OTHER_MAX)

u_0_T_1_list = list()
x_1_T_list = list()
for i in range(PRED_LEN):
    _u_current = MX.sym('u_'+str(i), 2)
    u_0_T_1_list.append(_u_current)

    _x_next = _state_next(_x_current, _u_current)
    x_1_T_list.append(_x_next)

    _x_current = _x_next

x_1_T = vertcat(*x_1_T_list)
u_0_T_1 = vertcat(*u_0_T_1_list)

_Qx = diag([1.0, 1.0, 0.0, 0.0])
Qx = kron(MX.eye(PRED_LEN), _Qx)

_Qu = diag([0.1, 0.1])
Qu = kron(MX.eye(PRED_LEN), _Qu)

J = x_1_T.T @ Qx @ x_1_T + u_0_T_1.T @ Qu @ u_0_T_1

nlp = {
    'x': vertcat(u_0_T_1),
    'f': J,
    # 'g': x**2+y**2,
    'p': x_0
}
prob_option = dict(
    verbose=False,
    verbose_init=False,
    print_in=False,
    print_out=False,
    print_time=False,
    record_time=True,
)

# print(S_ipopt)
# with suppress_stdout_stderr():
S_ipopt = nlpsol('S_ipopt', 'ipopt', nlp, prob_option)
r_ipopt = S_ipopt(p=[0.5, 0.5, 0.0, 3.0])
x_opt_ipopt = r_ipopt['x']
print('x_opt: ', x_opt_ipopt)
#
# with suppress_stdout_stderr():
#     S_sqp = nlpsol('S_sqpmethod', 'feasiblesqpmethod', qp, prob_option)
#     r_sqp = S_sqp(x0=[30, 30], lbg=1, p=0.6)
# x_opt_sqp = r_sqp['x']
# print('x_opt: ', x_opt_sqp)


# x = SX.sym('x')
# y = SX.sym('y')
# p = SX.sym('p')
# qp = {
#     'x': vertcat(x, y),
#     'f': x**2+p*y**2,
#     'g': x**2+y**2,
#     'p': p
# }
# prob_option = dict(
#     verbose=False,
#     verbose_init=False,
#     print_in=False,
#     print_out=False,
#     print_time=False,
#     record_time=True,
# )
#
# # print(S_ipopt)
# with suppress_stdout_stderr():
#     S_ipopt = nlpsol('S_ipopt', 'ipopt', qp, prob_option)
#     r_ipopt = S_ipopt(x0=[-20, 2], lbg=1, p=0.5)
# x_opt_ipopt = r_ipopt['x']
# print('x_opt: ', x_opt_ipopt)
#
# with suppress_stdout_stderr():
#     S_sqp = nlpsol('S_sqpmethod', 'feasiblesqpmethod', qp, prob_option)
#     r_sqp = S_sqp(x0=[30, 30], lbg=1, p=0.6)
# x_opt_sqp = r_sqp['x']
# print('x_opt: ', x_opt_sqp)
