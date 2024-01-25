import numpy as np
from typing import Tuple, Dict, Callable, List
import osqp
from dynamic_models import LongitudeModel
from utilits import VData, TrajDataGenerator, suppress_stdout_stderr, PickleRead, PickleSave
from scipy import sparse


class ADMM:

    _all_y: List[np.ndarray] = list()

    @classmethod
    def communicate(cls):
        cls._all_y.clear()
        for admm in _all_ADMM:
            cls._all_y.append(admm._y.copy())

    def __init__(self, config: VData):
        self._data = config
        _all_ADMM.append(self)

        self._osqp = osqp.OSQP()
        self._K, self._B = LongitudeModel.gen_M_N(self._data.T)
        self._G_i, self._H_i = self._gen_G_H()
        self._A_i, self._b_i = self._gen_A_b()
        self._M_i, self._l_pos, self._u_pos = self._gen_M_lu()

        self._p: np.ndarray = np.zeros((self._data.T * self._data.veh_num,))
        self._s: np.ndarray = np.zeros((self._data.T * self._data.veh_num,))
        self._r: np.ndarray = np.zeros((self._data.T * self._data.veh_num,))
        self._x: np.ndarray = np.zeros((1 * self._data.T,))
        self._y: np.ndarray = np.zeros((self._data.T * self._data.veh_num,))
        self._z: np.ndarray = np.zeros((self._data.T * self._data.veh_num,))

        self._init_osqp()

    def _gen_M_lu(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        M = np.zeros((3, self._data.T * 3))
        M[0, 1: self._data.tdvp[0] * 3: 3] = 1.0 / (self._data.tdvp[0])
        M[1, 1: self._data.tdvq[0] * 3: 3] = 1.0 / (self._data.tdvp[0])
        M[2, 1: self._data.tdf[0] * 3: 3] = 1.0 / (self._data.tdf[0])
        l_pos = np.array([
            10 * (self._data.tdvp[1] - self.X_0[0]) / (self._data.tdvp[0] - self._data.tx0[0]),
            -np.inf,
            # 10 * (self._data.tdf[1] - self.X_0[0]) / (self._data.tdf[0] - self._data.tx0[0])
            -np.inf
        ])
        u_pos = np.array([
            np.inf,
            10 * (self._data.tdvq[1] - self.X_0[0]) / (self._data.tdvq[0] - self._data.tx0[0]),
            np.inf
        ])
        return M, l_pos, u_pos

    def _gen_G_H(self) -> Tuple[np.ndarray, np.ndarray]:
        G = np.zeros((self._data.T * self._data.veh_num, 3 * self._data.T))
        H = np.zeros((self._data.T * self._data.veh_num,))
        MAT = np.kron(np.eye(self._data.T), np.array([1.0, 0.0, 0.0]))
        # before p
        if self._data.tdvp[2]:
            G[self._data.vid * self._data.T: self._data.vid * self._data.T + self._data.tdvp[0]] \
                = -MAT[: self._data.tdvp[0]]
            H[self._data.vid * self._data.T: self._data.vid * self._data.T + self._data.tdvp[0]] \
                = self._data.safe_dist
        else:
            print("veh {} has no prev before step {}".format(self._data.vid, self._data.tdvp[0]))
        # after q
        if self._data.tdvq[2]:
            G[self._data.vid * self._data.T + self._data.tdvq[0]: (self._data.vid + 1) * self._data.T] \
                = -MAT[self._data.tdvq[0]:]
            H[self._data.vid * self._data.T + self._data.tdvq[0]: (self._data.vid + 1) * self._data.T] \
                = self._data.safe_dist
        else:
            print("veh {} has no prev after step {}".format(self._data.vid, self._data.tdvq[0]))
        # other
        if self._data.otivp[2]:
            tm, o_vid, _ = self._data.otivp
            G[o_vid * self._data.T: o_vid * self._data.T + tm] = MAT[: tm]
        else:
            print("veh {} is not leader before".format(self._data.vid))
        if self._data.otivq[2]:
            tm, o_vid, _ = self._data.otivq
            G[o_vid * self._data.T + tm: (o_vid + 1) * self._data.T] = MAT[tm:]
        else:
            print("veh {} is not leader after".format(self._data.vid))
        return G, H

    def _gen_A_b(self) -> Tuple[np.ndarray, np.ndarray]:
        A = self._G_i @ self._B
        b = self._H_i - self._G_i @ self._K @ self._data.tx0[1]
        return A, b

    def _init_osqp(self):
        _P = sparse.csc_matrix(
            np.eye(self._data.T)
            + (self._A_i.transpose() @ self._A_i) / (2 * (self._data.sigma + 2 * self._data.rho * self._data.d_i))
        )
        _q = np.zeros((self._data.T,))
        #
        _A = sparse.csc_matrix(np.vstack((np.eye(self._data.T), self._M_i @ self._B)))
        _l = np.concatenate((
            np.ones((self._data.T,)) * -5,
            self._l_pos - self._M_i @ self._K @ self._data.tx0[1]
        ))
        _u = np.concatenate((
            np.ones((self._data.T,)) * 5,
            self._u_pos - self._M_i @ self._K @ self._data.tx0[1]
        ))
        with suppress_stdout_stderr():
            self._osqp.setup(_P, _q, _A, _l, _u)

    def _solve_x(self) -> np.ndarray:
        self._osqp.update(
            q=(self._A_i.transpose() @ self._r) / (2 * (self._data.sigma + 2 * self._data.rho * self._data.d_i)))
        with suppress_stdout_stderr():
            res = self._osqp.solve()
        self._x = res.x
        return self._x

    def step(self):
        # p
        self._p = self._p + self._data.rho * sum([self._y - oy for oy in self._all_y])
        # q
        self._s = self._s + self._data.sigma * (self._y - self._z)
        # r
        self._r = self._data.sigma * self._z \
                + self._data.rho * (sum([self._y + oy for oy in self._all_y]) - 2 * self._y) \
                - (self._b_i + self._p + self._s)
        # x
        self._solve_x()
        # y
        self._y = (self._A_i @ self._x + self._r) / (self._data.sigma + 2 * self._data.rho * self._data.d_i)
        # z
        self._z = np.clip(self._y + self._s / self._data.sigma, -np.inf, 0)

    @classmethod
    def STEP(cls):
        for admm in _all_ADMM:
            admm.step()

    @property
    def VID(self) -> int:
        return self._data.vid

    @property
    def results(self):
        return self._x

    @property
    def X_0(self):
        return self._data.tx0[1]



_all_ADMM: List[ADMM] = list()


if __name__ == '__main__':
    LongitudeModel.initialize(LongitudeModel.default_config)
    generator = TrajDataGenerator(100, 50, 150, 110, 150, 10, 20, 15, (-3, 3), (-5, 5), sigma=0.3, rho=0.03)
    trajs = generator.generate_all_vdata()
    PickleSave(trajs, '../output_dir/temp01')
    # trajs = PickleRead('../output_dir/temp01')
    for vid, vdata in trajs.items():
        ADMM(vdata)
    for i in range(100):
        ADMM.communicate()
        ADMM.STEP()
    import matplotlib.pyplot as plt
    x_main = [(LongitudeModel.M() @ admm.X_0 + LongitudeModel.N() @ admm.results) for admm in _all_ADMM]
    x_main_ = [x.reshape((-1, 3))[:, 0] for x in x_main]

    [plt.plot(_) for _ in x_main_]
    plt.show()
    pass
