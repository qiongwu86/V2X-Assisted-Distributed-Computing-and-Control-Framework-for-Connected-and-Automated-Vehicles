import casadi as ca
import numpy as np
from dynamic import KinematicModel


LENGTH = 3.5
DELTA_T = 0.1
PRED_LEN = 30

x_t = ca.SX.sym('x_t', 4)
u_t = ca.SX.sym('u_t', 2)
x_dot = ca.vertcat(
    x_t[3, 0] * ca.cos(x_t[2, 0] + ca.arctan(0.5*ca.tan(u_t[1, 0]))),
    x_t[3, 0] * ca.sin(x_t[2, 0] + ca.arctan(0.5*ca.tan(u_t[1, 0]))),
    x_t[3, 0] * ca.sin(ca.arctan(0.5*ca.tan(u_t[1, 0]))) / (0.5 * LENGTH),
    u_t[0, 0]
)
F1 = ca.Function('F1', [x_t, u_t], [DELTA_T * x_dot+x_t])

x0 = ca.MX.sym('x_0', 4)
x_current = x0
x_list = []
u_list = []
for i in range(PRED_LEN):
    u_current = ca.MX.sym('u_'+str(i), 2)
    # x_current_ = ca.MX.sym('x_'+str(i+1), 4)
    x_current_ = F1(x_current, u_current)
    x_list.append(x_current_)
    u_list.append(u_current)

    x_current = x_current_

x_1_T = ca.vertcat(*x_list)
u_1_T = ca.vertcat(*u_list)
ABC_all = ca.Function(
    "ABC",
    [x0, u_1_T],
    [
        ca.jacobian(x_1_T, x0),
        ca.jacobian(x_1_T, u_1_T),
        x_1_T-ca.jacobian(x_1_T, x0)@x0-ca.jacobian(x_1_T, u_1_T)@u_1_T
    ],
    ['x0', 'u_1_T'],
    ['A', 'B', 'C']
)
x_1_T_f = ca.Function('x_1_T_f', [x0, u_1_T], [x_1_T])
# Xi[1: T] = F(x0, u[0: T-1])
x0 = np.array([0.5, 1.0, 0.0, 3])
u_1_T = np.vstack((np.random.random(30)+1, 0.1*np.random.random(30))).transpose()
A, B, C = ABC_all(x0=x0, u_1_T=u_1_T.reshape(-1)).values()
x_1_T_test = np.array(x_1_T_f(x0, u_1_T.reshape(-1)))

d1 = KinematicModel(KinematicModel.default_config)
x_1_T_right = d1.roll_out(init_state=x0, control=u_1_T, include_init=False)
A_, B_, C_ = d1.dynamic_constrain(u_bar=u_1_T, x_0=x0)

x_1_T_test = x_1_T_test.reshape((-1, 4))

print(A, B, C)
