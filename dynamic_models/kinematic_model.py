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
    )

    length: float = 3.5
    width: float = 1.7
    acc_min: float = -7.0
    acc_max: float = 7.0
    steer_min: float = -np.deg2rad(34)
    steer_max: float = np.deg2rad(34)
    delta_T: float = 0.1

    _delta_fun: Callable = None

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
        cls._delta_fun = cls._state_dot()

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
