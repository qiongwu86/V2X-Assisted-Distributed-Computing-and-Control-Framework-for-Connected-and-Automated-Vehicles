import numpy as np
from scipy import signal
import scipy


L = 4.5

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


class BicycleModel:
    Td = 0.1
    L = 4.5
    SDIM = 4
    CDIM = 2
    a_max = 2.0
    a_min = -2.0
    psi_max = 15 * np.pi / 180
    psi_min = -15 * np.pi / 180
    u_max = np.array([a_max, psi_max])
    u_min = np.array([a_min, psi_min])
    def __init__(self) -> None:
        pass

    @staticmethod
    def _GenA(x_bar: np.ndarray, u_bar: np.ndarray) -> np.ndarray:
        x, y, phi, v = x_bar[0], x_bar[1], x_bar[2], x_bar[3]
        a, psi = u_bar[0], u_bar[1]
        result = np.eye(4)
        # line[0]
        result[0][2] += -BicycleModel.Td * v * np.sin(phi)
        result[0][3] += BicycleModel.Td * np.cos(phi)
        # line[1]
        result[1][2] += BicycleModel.Td * v * np.cos(phi)
        result[1][3] += BicycleModel.Td * np.sin(phi)
        # line[2]
        result[2][3] += BicycleModel.Td * np.tan(psi) / BicycleModel.L
        return result

    @staticmethod
    def _GenB(x_bar: np.ndarray, u_bar: np.ndarray) -> np.ndarray:
        x, y, phi, v = x_bar[0], x_bar[1], x_bar[2], x_bar[3]
        a, psi = u_bar[0], u_bar[1]
        result = np.zeros((4, 2))
        result[2][1] = BicycleModel.Td * v / (BicycleModel.L * np.cos(psi) ** 2)
        result[3][0] = BicycleModel.Td 
        return result

    @staticmethod
    def _Geng(x_bar: np.ndarray, u_bar: np.ndarray) -> np.ndarray:
        x, y, phi, v = x_bar[0], x_bar[1], x_bar[2], x_bar[3]
        a, psi = u_bar[0], u_bar[1]
        result = np.zeros((4,))
        result[0] = BicycleModel.Td * v * phi * np.sin(phi)
        result[1] = -BicycleModel.Td * v * phi * np.cos(phi)
        result[2] = -BicycleModel.Td * v * psi / (BicycleModel.L * np.cos(psi) ** 2)
        return result

    @staticmethod
    def step(x, u) -> np.ndarray:
        px, py, phi, v = x[0], x[1], x[2], x[3]
        a, psi = u[0], u[1]
        x_ = BicycleModel.Td * np.array([v*np.cos(phi), v*np.sin(phi), v*np.tan(psi) / BicycleModel.L, a]) + x
        return x_

    @staticmethod
    def roll(x: np.ndarray, u_all: np.ndarray, T_nums: int) -> np.ndarray:
        assert u_all.shape[0] == T_nums 
        all_x = np.zeros((T_nums+1, BicycleModel.SDIM))
        all_x[0] = x
        for i in range(1, T_nums+1):
            all_x[i] = BicycleModel.step(all_x[i-1], u_all[i-1])
        return all_x

    @staticmethod
    def _GenABg(x_bar: np.ndarray, u_bar: np.ndarray) -> tuple:
        A = BicycleModel._GenA(x_bar, u_bar)
        B = BicycleModel._GenB(x_bar, u_bar)
        g = BicycleModel._Geng(x_bar, u_bar)
        return (A, B, g)

    @staticmethod
    def MakeDynamicConstrain(x0_bar: np.ndarray, u_bars: np.ndarray, T_nums: int) -> tuple:
        A_saves = np.zeros((T_nums, BicycleModel.SDIM, BicycleModel.SDIM))
        B_saves = np.zeros((T_nums, BicycleModel.SDIM, BicycleModel.CDIM))
        g_save = np.zeros((T_nums, BicycleModel.SDIM))

        x_bars = BicycleModel.roll(x0_bar, u_bars, T_nums)[:-1]
        for i in range(T_nums):
            A_saves[i], B_saves[i], g_save[i] = BicycleModel._GenABg(x_bars[i], u_bars[i])

        '''
        X = AA * x0 + BB * U + GG
        '''
        AA = np.zeros((BicycleModel.SDIM * T_nums, BicycleModel.SDIM))
        AA[0*BicycleModel.SDIM: BicycleModel.SDIM] = A_saves[0]
        for i in range(1, T_nums):
            AA[i*BicycleModel.SDIM: (i+1)*BicycleModel.SDIM] = A_saves[i] @ AA[(i-1)*BicycleModel.SDIM: i*BicycleModel.SDIM]

        BB = np.zeros((BicycleModel.SDIM * T_nums, BicycleModel.CDIM * T_nums))
        for c in range(T_nums):
            BB[c*BicycleModel.SDIM:(c+1)*BicycleModel.SDIM, c*BicycleModel.CDIM:(c+1)*BicycleModel.CDIM] = B_saves[c]
            for r in range(c+1, T_nums):
                BB[r*BicycleModel.SDIM: (r+1)*BicycleModel.SDIM, c*BicycleModel.CDIM:(c+1)*BicycleModel.CDIM] = A_saves[r] @ BB[(r-1)*BicycleModel.SDIM: r*BicycleModel.SDIM, c*BicycleModel.CDIM:(c+1)*BicycleModel.CDIM]

        GG = np.zeros((BicycleModel.SDIM * T_nums,))
        GG[: BicycleModel.SDIM] = g_save[0]
        for i in range(1, T_nums):
            GG[i*BicycleModel.SDIM: (i+1)*BicycleModel.SDIM] = g_save[i] + A_saves[i] @ GG[(i-1)*BicycleModel.SDIM: i*BicycleModel.SDIM]

        np.save('AA.npy', AA)
        np.save('BB.npy', BB)
        np.save('GG.npy', GG)
        return (AA, BB, GG)
