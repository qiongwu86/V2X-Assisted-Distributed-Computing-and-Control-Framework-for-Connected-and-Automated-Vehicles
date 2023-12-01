import numpy as np
from scipy import signal
import scipy


def init_system(Ta: float = 0.1, Td: float = 0.1):
    OneDimDynamic.Ta = Ta
    OneDimDynamic.Td = Td
    OneDimDynamic.Ac = np.array([[0, 1, 0], [0, 0, 1], [0, 0, -1/OneDimDynamic.Ta]])
    OneDimDynamic.Bc = np.array([[0], [0], [1/OneDimDynamic.Ta]])
    OneDimDynamic.Cc = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    OneDimDynamic.Dc = np.array([[0], [0], [0]])
    sys_d = scipy.signal.cont2discrete([OneDimDynamic.Ac, OneDimDynamic.Bc, OneDimDynamic.Cc, OneDimDynamic.Dc], OneDimDynamic.Td, "zoh")
    OneDimDynamic.Ad, OneDimDynamic.Bd, OneDimDynamic.Cd, OneDimDynamic.Dd, _ = sys_d
    


class OneDimDynamic:
    sys_initd = False

    SDIM = 3
    CDIM = 1

    U_min = -5.0
    U_max = +5.0

    def __init__(self, init_state: np.ndarray, Td: float = 0.1, Ta: float = 0.1, save_trace: bool = False) -> None:
        if not OneDimDynamic.sys_initd:
            print("Init system with default param: Ta = 0.1, Td = 0.1")
            init_system()
        # state := [s, v, a], control := [a_r]
        self.state: np.ndarray = init_state
        self.save_trace = save_trace
        if save_trace:
            self.state_num = 1
            self.trace = np.zeros((1, OneDimDynamic.CDIM+OneDimDynamic.SDIM))
            self.trace[self.state_num - 1][:OneDimDynamic.SDIM] = self.state

    def step(self, c_var: np.array) -> np.array:
        assert len(c_var) == 1
        self.state = OneDimDynamic.Ad @ self.state + OneDimDynamic.Bd @ c_var
        if self.save_trace:
            self.trace = np.append(self.trace, np.zeros((1, OneDimDynamic.CDIM+OneDimDynamic.SDIM)), axis=0)
            self.trace[self.state_num-1][OneDimDynamic.SDIM:] = c_var
            self.state_num += 1
            self.trace[self.state_num - 1][:OneDimDynamic.SDIM] = self.state
        return self.state

    def get_trace(self) -> np.array:
        return self.trace

    def get_discrete_system() -> tuple:
        return (OneDimDynamic.Ad, OneDimDynamic.Bd, OneDimDynamic.Cd, OneDimDynamic.Dd)
