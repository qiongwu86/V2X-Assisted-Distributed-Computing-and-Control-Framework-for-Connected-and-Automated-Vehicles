import casadi
import numpy as np
import scipy


class LongitudeModel:

    default_config = dict(
        delta_Td=0.1,
        delta_Ta=0.1
    )

    def __init__(self, config):
        # param
        self.delta_Td = config["delta_Td"]
        self.delta_Ta = config["delta_Ta"]
        # var
        self._state = casadi.SX.sym('state', 3)
        self._control = casadi.SX.sym('control', 1)
        #
        Ad, Bd = self._init_system()
        self.Ad = Ad
        self.Bd = Bd

    def _init_system(self):
        Ac = np.array([[0, 1, 0], [0, 0, 1], [0, 0, -1 / self.delta_Ta]])
        Bc = np.array([[0], [0], [1 / self.delta_Ta]])
        Cc = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        Dc = np.array([[0], [0], [0]])
        sys_d = scipy.signal.cont2discrete([Ac, Bc, Cc, Dc], self.delta_Td, "zoh")
        Ad, Bd, _, _, _ = sys_d
        return Ad, Bd

    def step_once(self, init_state: np.ndarray, control: float):
        assert init_state.shape == (3,)
        next_state = self.Ad @ init_state + self.Bd * control
        return next_state

    def roll_out(self, init_state: np.ndarray, control: np.ndarray, include_init: bool = True):
        assert control.shape[1] == 1 and init_state.shape == (3,)
        steps = control.shape[0]
        traj = np.zeros((steps+1, 3))
        traj[0] = init_state
        for t in range(steps):
            traj[t+1] = self.step_once(traj[t], control[t])
        if not include_init:
            traj = traj[1:]
        return traj


class KinematicModel:

    default_config = dict(
        length=4.0,
        width=1.7,
        acc_min=-5.0,
        acc_max=5.0,
        steer_min=-np.deg2rad(34),
        steer_max=np.deg2rad(34),
        # acc_min=-np.inf,
        # acc_max=np.inf,
        # steer_min=-np.inf,
        # steer_max=np.inf,
        delta_T=0.1,
        pred_len=30,
        safe_th=2.0
    )

    def __init__(self, config: dict):
        # param
        self.length = config["length"]
        self.width = config["width"]
        self.acc_min = config["acc_min"]
        self.acc_max = config["acc_max"],
        self.steer_min = config["steer_min"],
        self.steer_max = config["steer_max"],
        self.delta_T = config["delta_T"]
        self.pred_len = config["pred_len"]
        self.safe_th = config["safe_th"]

        # dynamic step & rollout
        self._delta_fun = self._delta_state()
        # dynamic constrain, X = AA x0 + BB U_all + GG
        self._AA_fun, self._BB_fun, self._GG_fun = self._dynamic_constrain()
        # safe constrain || min(dist(x_ego|x_other), 0) ||^2 = || k * x_ego + b || ^ 2
        self._k_fun, self._b_fun, self._dist_fun = self._safe_constrain_constructor()

    def _delta_state(self):
        # var
        self._state = casadi.SX.sym('state', 4)
        self._control = casadi.SX.sym('control', 2)
        delta = casadi.SX.sym('delta', 4)
        beta = casadi.arctan(0.5 * casadi.tan(self._control[1, 0]))
        delta[0, 0] = self._state[3, 0] * casadi.cos(self._state[2, 0] + beta)
        delta[1, 0] = self._state[3, 0] * casadi.sin(self._state[2, 0] + beta)
        delta[2, 0] = self._state[3, 0] * casadi.sin(beta) / (0.5*self.length)
        delta[3, 0] = self._control[0, 0]

        delta_fun = casadi.Function('delta_state', [self._state, self._control], [delta])
        return delta_fun

    def step_once(self, init_state: np.ndarray, control: np.ndarray):
        assert init_state.shape == (4,) and control.shape == (2,)
        next_state = self.delta_T * self._delta_fun(init_state, control) + init_state
        return np.array(next_state).reshape(-1)

    def roll_out(self, init_state: np.ndarray, control: np.ndarray, include_init: bool = True, include_final: bool = True):
        assert init_state.shape == (4,) and control.shape[1] == 2
        steps = control.shape[0]
        traj = np.zeros((steps + 1, 4))
        traj[0] = init_state
        for t in range(steps):
            traj[t + 1] = self.step_once(traj[t], control[t])
        if not include_init:
            traj = traj[1:]
        if not include_final:
            traj = traj[:-1]
        return traj

    def _dynamic_constrain(self):
        u_all = casadi.SX.sym("u_all", 2, self.pred_len)
        x_all = casadi.SX.sym("x_all", 4, self.pred_len)  # it comes from roll out

        def _jkl(x_t, u_t, _t: str):
            assert x_t.shape == (4, 1) and u_t.shape == (2, 1)
            beta = casadi.arctan(0.5 * casadi.tan(u_t[1, 0]))
            delta = casadi.SX(4, 1)
            delta[0, 0] = x_t[3, 0] * casadi.cos(x_t[2, 0] + beta)
            delta[1, 0] = x_t[3, 0] * casadi.sin(x_t[2, 0] + beta)
            delta[2, 0] = x_t[3, 0] * casadi.sin(beta) / (0.5 * self.length)
            delta[3, 0] = u_t[0, 0]
            J = casadi.jacobian(delta, x_t) * self.delta_T + np.eye(4)
            K = casadi.jacobian(delta, u_t) * self.delta_T
            L = self.delta_T * (delta - casadi.jacobian(delta, x_t) @ x_t - casadi.jacobian(delta, u_t) @ u_t)
            return J, K, L

        Js = casadi.SX.sym("Js", 4, 4, self.pred_len)
        Ks = casadi.SX.sym("Ks", 4, 2, self.pred_len)
        Ls = casadi.SX.sym("Ls", 4, 1, self.pred_len)
        for i in range(self.pred_len):
            Js[i], Ks[i], Ls[i] = _jkl(x_all[:, i], u_all[:, i], str(i))

        AA = casadi.SX(self.pred_len * 4, 4)
        AA[: 4, :] = Js[0]
        for i in range(1, self.pred_len):
            AA[i*4: (i+1)*4, :] = Js[i] @ AA[(i-1)*4: i*4, :]

        BB = casadi.SX(self.pred_len * 4, 2 * self.pred_len)
        for c in range(self.pred_len):
            BB[c * 4:(c + 1) * 4, c * 2:(c + 1) * 2] = Ks[c]
            for r in range(c + 1, self.pred_len):
                BB[r * 4: (r + 1) * 4, c * 2:(c + 1) * 2] = Js[r] @ BB[(r - 1) * 4: r * 4, c * 2:(c + 1) * 2]

        GG = casadi.SX(self.pred_len * 4, 1)
        GG[: 4] = Ls[0]
        for i in range(1, self.pred_len):
            GG[i*4: (i+1)*4] = Ls[i] + Js[i] @ GG[(i-1)*4: i*4]

        AA_fun = casadi.Function("AA_fun", [x_all, u_all], [AA])
        BB_fun = casadi.Function("BB_fun", [x_all, u_all], [BB])
        GG_fun = casadi.Function("GG_fun", [x_all, u_all], [GG])

        return AA_fun, BB_fun, GG_fun

    def dynamic_constrain(self, u_bar, x_0=None, x_bar=None):
        assert u_bar is not None and u_bar.shape == (self.pred_len, 2)
        if x_0 is not None:
            # need calculate x_bar first
            assert x_bar is None and x_0.shape == (4,)
            x_bar = self.roll_out(x_0, u_bar, include_final=False)
        elif x_bar is not None:
            assert x_bar.shape == (self.pred_len, 4) and x_0 is None
        else:
            raise ValueError

        AA = np.array(self._AA_fun(x_bar.transpose(), u_bar.transpose()))
        BB = np.array(self._BB_fun(x_bar.transpose(), u_bar.transpose()))
        GG = np.array(self._GG_fun(x_bar.transpose(), u_bar.transpose())).reshape(-1)

        return AA, BB, GG

    def _safe_constrain_constructor(self):
        self.pred_len = self.pred_len
        x_ego = casadi.SX.sym('x_ego', 4 * self.pred_len)
        x_other = casadi.SX.sym('x_other', 4 * self.pred_len)

        V = 0.5 * (self.length - self.width)
        D = self.width
        th = self.safe_th

        dist = casadi.SX.sym('dist', 4 * self.pred_len)
        for i in range(self.pred_len):
            dist[0 + i * 4] = (x_ego[0 + i * 4] - x_other[0 + i * 4] + V * (casadi.cos(x_ego[2 + i * 4]) - casadi.cos(x_other[2 + i * 4]))) ** 2 \
                            + (x_ego[1 + i * 4] - x_other[1 + i * 4] + V * (casadi.sin(x_ego[2 + i * 4]) - casadi.sin(x_other[2 + i * 4]))) ** 2 \
                            - D**2 - th

            dist[1 + i * 4] = (x_ego[0 + i * 4] - x_other[0 + i * 4] + V * (casadi.cos(x_ego[2 + i * 4]) + casadi.cos(x_other[2 + i * 4]))) ** 2 \
                            + (x_ego[1 + i * 4] - x_other[1 + i * 4] + V * (casadi.sin(x_ego[2 + i * 4]) + casadi.sin(x_other[2 + i * 4]))) ** 2 \
                            - D**2 - th

            dist[2 + i * 4] = (x_ego[0 + i * 4] - x_other[0 + i * 4] + V * (-casadi.cos(x_ego[2 + i * 4]) - casadi.cos(x_other[2 + i * 4]))) ** 2 \
                            + (x_ego[1 + i * 4] - x_other[1 + i * 4] + V * (-casadi.sin(x_ego[2 + i * 4]) - casadi.sin(x_other[2 + i * 4]))) ** 2 \
                            - D**2 - th

            dist[3 + i * 4] = (x_ego[0 + i * 4] - x_other[0 + i * 4] + V * (-casadi.cos(x_ego[2 + i * 4]) + casadi.cos(x_other[2 + i * 4]))) ** 2 \
                            + (x_ego[1 + i * 4] - x_other[1 + i * 4] + V * (-casadi.sin(x_ego[2 + i * 4]) + casadi.sin(x_other[2 + i * 4]))) ** 2 \
                            - D**2 - th

        dist_over = casadi.fmin(dist, casadi.SX.zeros(4 * self.pred_len))
        dist_fun = casadi.Function('F_dist', [x_ego, x_other], [dist_over])

        J = casadi.jacobian(dist_over, x_ego)
        k_fun = casadi.Function('F_k', [x_ego, x_other], [J])
        b_fun = casadi.Function('F_b', [x_ego, x_other], [dist_over - J @ x_ego])

        return k_fun, b_fun, dist_fun

    def safe_constrain(self, x_ego, x_other):
        assert x_ego.shape == (self.pred_len, 4) and x_other.shape == (self.pred_len, 4)
        k = self._k_fun(x_ego.reshape(-1), x_other.reshape(-1))
        b = self._b_fun(x_ego.reshape(-1), x_other.reshape(-1))
        return np.array(k), np.array(b).reshape(-1)

    def dist_calculator(self, x_ego, x_other):
        assert x_ego.shape == (self.pred_len, 4) and x_other.shape == (self.pred_len, 4)
        dist = self._dist_fun(x_ego.reshape(-1), x_other.reshape(-1))
        return np.array(dist)


if __name__ == '__main__':
    pass
    # d1 = KinematicModel(KinematicModel.default_config)
    # _x_0 = np.array([1.5, 0.0, 0.0, 2.0])
    # _u_bar = np.random.randn(30, 2)
    # a, b, c = d1.dynamic_constrain(_u_bar, x_0=_x_0)
    # print(a.shape, b.shape, c.shape)
    # _x_all = d1.roll_out(_x_0, _u_bar, include_final=False)
    # print(_x_all)
    # dist = d1.dist_calculator(_x_all, _x_all+np.random.randn(30, 4))
    # print(dist)
    # k, b = d1.safe_constrain(_x_all, _x_all+np.random.randn(30, 4))
    # print(k.shape, b.shape)
    #
    # _u_ = np.array(
    #     [
    #         2.0*np.sin(0.1*np.arange(0, 100, 1)),
    #         0.1*np.sin(0.1*np.arange(0, 100, 1)),
    #     ]
    # )
    # _u_ = _u_.transpose()
    # x_all = np.array([_x_0])
    # for i in range(len(_u_)):
    #     x_next = d1.step_once(x_all[i], _u_[i])
    #     x_all = np.vstack((x_all, x_next))
    #
    # import matplotlib.pyplot as plt
    # plt.plot(x_all[:, 0], x_all[:, 1])
    # plt.show()
