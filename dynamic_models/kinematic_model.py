import casadi
import numpy as np
from typing import Dict, Callable


class KinematicModel:

    default_config = dict(
        length=3.5,
        width=1.7,
        acc_min=-7.0,
        acc_max=7.0,
        steer_min=-np.deg2rad(34),
        steer_max=np.deg2rad(34),
        delta_T=0.1,
        pred_len=30,
        safe_th=5.0
    )

    length: float = 3.5
    width: float = 1.7
    acc_min: float = -7.0
    acc_max: float = 7.0
    steer_min: float = -np.deg2rad(34)
    steer_max: float = np.deg2rad(34)
    delta_T: float = 0.1
    pred_len: int = 30
    safe_th: float = 5.0

    _delta_fun: Callable = None
    _AA_fun: Callable = None
    _BB_fun: Callable = None
    _GG_fun: Callable = None
    _k_fun: Callable = None
    _b_fun: Callable = None
    _dist_fun: Callable = None

    _initialized: bool = False

    @classmethod
    def is_initialized(cls):
        return cls._initialized

    @classmethod
    def initialize(cls, config: Dict):
        if cls._initialized:
            raise RuntimeError
        cls._initialized = True
        # param
        cls.length = config["length"]
        cls.width = config["width"]
        cls.acc_min = config["acc_min"]
        cls.acc_max = config["acc_max"]
        cls.steer_min = config["steer_min"]
        cls.steer_max = config["steer_max"]
        cls.delta_T = config["delta_T"]
        cls.pred_len = config["pred_len"]
        cls.safe_th = config["safe_th"]

        cls._delta_fun = cls._state_dot()
        cls._AA_fun, cls._BB_fun, cls._GG_fun = cls._dynamic_constrain()
        cls._k_fun, cls._b_fun, cls._dist_fun = cls._safe_constrain_constructor()

    @classmethod
    def _state_dot(cls):
        # var
        _state = casadi.SX.sym('state', 4)
        _control = casadi.SX.sym('control', 2)
        _state_dot = casadi.vertcat(
            _state[3, 0] * casadi.cos(_state[2, 0] + casadi.arctan(0.5 * casadi.tan(_control[1, 0]))),
            _state[3, 0] * casadi.sin(_state[2, 0] + casadi.arctan(0.5 * casadi.tan(_control[1, 0]))),
            _state[3, 0] * casadi.sin(casadi.arctan(0.5 * casadi.tan(_control[1, 0]))) / (0.5*cls.length),
            _control[0, 0]
        )
        delta_fun = casadi.Function('delta_state', [_state, _control], [_state_dot])
        return delta_fun

    @classmethod
    def step_once(cls, init_state: np.ndarray, control: np.ndarray):
        assert cls._initialized
        assert init_state.shape == (4,) and control.shape == (2,)
        next_state = cls.delta_T * cls._delta_fun(init_state, control) + init_state
        return np.array(next_state).reshape(-1)

    @classmethod
    def roll_out(cls, init_state: np.ndarray, control: np.ndarray, include_init: bool = True, include_final: bool = True):
        assert cls._initialized
        assert init_state.shape == (4,) and control.shape[1] == 2
        steps = control.shape[0]
        traj = np.zeros((steps + 1, 4))
        traj[0] = init_state
        for t in range(steps):
            traj[t + 1] = cls.step_once(traj[t], control[t])
        if not include_init:
            traj = traj[1:]
        if not include_final:
            traj = traj[:-1]
        return traj

    @classmethod
    def _dynamic_constrain(cls):
        x_t = casadi.SX.sym('x_t', 4)
        u_t = casadi.SX.sym('u_t', 2)
        x_dot = casadi.vertcat(
            x_t[3, 0] * casadi.cos(x_t[2, 0] + casadi.arctan(0.5 * casadi.tan(u_t[1, 0]))),
            x_t[3, 0] * casadi.sin(x_t[2, 0] + casadi.arctan(0.5 * casadi.tan(u_t[1, 0]))),
            x_t[3, 0] * casadi.sin(casadi.arctan(0.5 * casadi.tan(u_t[1, 0]))) / (0.5 * cls.length),
            u_t[0, 0]
        )
        F1 = casadi.Function('F1', [x_t, u_t], [cls.delta_T * x_dot + x_t])

        x0 = casadi.MX.sym('x_0', 4)
        x_current = x0
        x_list = []
        u_list = []
        for i in range(cls.pred_len):
            u_current = casadi.MX.sym('u_' + str(i), 2)
            x_current_ = F1(x_current, u_current)
            x_list.append(x_current_)
            u_list.append(u_current)

            x_current = x_current_

        x_1_T = casadi.vertcat(*x_list)
        u_1_T = casadi.vertcat(*u_list)
        AA_fun = casadi.Function(
            "A_fun",
            [x0, u_1_T],
            [casadi.jacobian(x_1_T, x0)],
        )
        BB_fun = casadi.Function(
            "B_fun",
            [x0, u_1_T],
            [casadi.jacobian(x_1_T, u_1_T)],
        )
        GG_fun = casadi.Function(
            "G_fun",
            [x0, u_1_T],
            [x_1_T - casadi.jacobian(x_1_T, x0) @ x0 - casadi.jacobian(x_1_T, u_1_T) @ u_1_T],
        )
        return AA_fun, BB_fun, GG_fun

    @classmethod
    def dynamic_constrain(cls, u_bar: np.ndarray, x_0: np.ndarray):
        assert cls._initialized
        assert u_bar is not None and u_bar.shape == (cls.pred_len, 2)
        assert x_0.shape == (4,)

        AA = np.array(cls._AA_fun(x_0, u_bar.reshape(-1)))
        BB = np.array(cls._BB_fun(x_0, u_bar.reshape(-1)))
        GG = np.array(cls._GG_fun(x_0, u_bar.reshape(-1))).reshape(-1)

        return AA, BB, GG

    @classmethod
    def _safe_constrain_constructor(cls):
        x_ego = casadi.SX.sym('x_ego', 4)
        x_other = casadi.SX.sym('x_other', 4)

        V = 0.5 * (cls.length - cls.width)
        D = cls.width
        th = cls.safe_th

        _dist_inner = casadi.fmin(
            casadi.vertcat(
                (x_ego[0] - x_other[0] + V * (casadi.cos(x_ego[2]) - casadi.cos(x_other[2]))) ** 2 +
                (x_ego[1] - x_other[1] + V * (casadi.sin(x_ego[2]) - casadi.sin(x_other[2]))) ** 2 -
                D**2 - th,
                (x_ego[0] - x_other[0] + V * (casadi.cos(x_ego[2]) + casadi.cos(x_other[2]))) ** 2 +
                (x_ego[1] - x_other[1] + V * (casadi.sin(x_ego[2]) + casadi.sin(x_other[2]))) ** 2 -
                D**2 - th,
                (x_ego[0] - x_other[0] + V * (-casadi.cos(x_ego[2]) - casadi.cos(x_other[2]))) ** 2 +
                (x_ego[1] - x_other[1] + V * (-casadi.sin(x_ego[2]) - casadi.sin(x_other[2]))) ** 2 -
                D**2 - th,
                (x_ego[0] - x_other[0] + V * (-casadi.cos(x_ego[2]) + casadi.cos(x_other[2]))) ** 2 +
                (x_ego[1] - x_other[1] + V * (-casadi.sin(x_ego[2]) + casadi.sin(x_other[2]))) ** 2 -
                D**2 - th,
            ),
            casadi.SX.zeros(4)
        )
        _dist_inner_func = casadi.Function('dist_inner_func', [x_ego, x_other], [_dist_inner])

        _dist_all_list = []
        _x_ego_all = casadi.MX.sym('_x_ego_all', 4*cls.pred_len)
        _x_other_all = casadi.MX.sym('_x_other_all', 4*cls.pred_len)
        for i in range(cls.pred_len):
            _dist_all_list.append(
                _dist_inner_func(_x_ego_all[i*4: (i+1)*4], _x_other_all[i*4: (i+1)*4])
            )
        _dist_all = casadi.vertcat(*_dist_all_list)

        _dist_all_func = casadi.Function('dist_all_func', [_x_ego_all, _x_other_all], [_dist_all])
        _k_func = casadi.Function('F_k', [_x_ego_all, _x_other_all], [casadi.jacobian(_dist_all, _x_ego_all)])
        _b_func = casadi.Function('F_b', [_x_ego_all, _x_other_all], [_dist_all-casadi.jacobian(_dist_all, _x_ego_all)@_x_ego_all])

        return _k_func, _b_func, _dist_all_func

    @classmethod
    def safe_constrain(cls, x_ego: np.ndarray, x_other: np.ndarray):
        assert cls._initialized
        assert x_ego.shape == (cls.pred_len, 4) and x_other.shape == (cls.pred_len, 4)
        k = cls._k_fun(x_ego.reshape(-1), x_other.reshape(-1))
        b = cls._b_fun(x_ego.reshape(-1), x_other.reshape(-1))
        return np.array(k), np.array(b).reshape(-1)

    @classmethod
    def dist_calculator(cls, x_ego: np.ndarray, x_other: np.ndarray):
        assert cls._initialized
        assert x_ego.shape == (cls.pred_len, 4) and x_other.shape == (cls.pred_len, 4)
        dist = cls._dist_fun(x_ego.reshape(-1), x_other.reshape(-1))
        return np.array(dist)
