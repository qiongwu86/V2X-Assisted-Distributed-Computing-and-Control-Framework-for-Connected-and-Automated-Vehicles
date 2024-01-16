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
