import numpy as np
from typing import Tuple, Dict, Callable, List
import osqp
from dynamic_models import LongitudeModel
from utilits import VData, TrajDataGenerator, suppress_stdout_stderr, PickleRead, PickleSave, veh_constrain
from scipy import sparse
import tqdm
import matplotlib.pyplot as plt
import time


WHAT = 'T'
LANG = 'EN'
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
# plt.rcParams['axes.unicode_minus'] = False
just_neighbor = False

y_var_list = list()
run_time = list()

class ADMM:
    _all_y: List[np.ndarray] = list()

    @property
    def data(self):
        return self._data

    @classmethod
    def communicate(cls):
        cls._all_y.clear()
        for admm in all_ADMM:
            cls._all_y.append(admm._y.copy())

    def __init__(self, config: VData):
        self._data = config
        all_ADMM.append(self)

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

        # self._init_osqp()

    def _get_neighbors_y(self) -> List[np.ndarray]:
        result = []
        for admm in all_ADMM:
            if admm._data.vid in self._data.neighbors:
                result.append(admm._y)
        assert 1 <= len(result) <= 4
        return result

    def _gen_M_lu(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if veh_constrain:
            M = np.zeros((3, self._data.T * 3))
            M[0, 1: self._data.tdvp[0] * 3: 3] = 1.0 / (self._data.tdvp[0])
            M[1, 1: self._data.tdvq[0] * 3: 3] = 1.0 / (self._data.tdvq[0])
            M[2, 1: self._data.tdf[0] * 3: 3] = 1.0 / (self._data.tdf[0])
            l_pos = np.array([
                10 * (self._data.tdvp[1] - self.X_0[0]) / (self._data.tdvp[0] - self._data.tx0[0]),
                -np.inf,
                10 * (self._data.tdf[1][0] - self.X_0[0]) / (self._data.tdf[0] - self._data.tx0[0])
                # 10 * (self._data.tdf[1] - self.X_0[0]) / (self._data.tdf[0] - self._data.tx0[0]) if self._data.last
                # else -np.inf
            ])
            u_pos = np.array([
                np.inf,
                10 * (self._data.tdvq[1] - self.X_0[0]) / (self._data.tdvq[0] - self._data.tx0[0]),
                10 * (self._data.tdf[1][1] - self.X_0[0]) / (self._data.tdf[0] - self._data.tx0[0])
                # np.inf
            ])
        else:
            M = np.zeros((3, self._data.T * 3))
            M[0, self._data.tdvp[0] * 3] = 1.0
            M[1, self._data.tdvq[0] * 3] = 1.0
            M[2, self._data.tdf[0] * 3-3] = 1.0
            l_pos = np.array([
                self._data.tdvp[1],
                -np.inf,
                self._data.tdf[1][0]
                # 10 * (self._data.tdf[1] - self.X_0[0]) / (self._data.tdf[0] - self._data.tx0[0]) if self._data.last
                # else -np.inf
            ])
            u_pos = np.array([
                np.inf,
                self._data.tdvq[1],
                self._data.tdf[1][1]
                # np.inf
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
            self._osqp = osqp.OSQP()
            self._osqp.setup(_P, _q, _A, _l, _u)

    def _solve_x(self) -> np.ndarray:
        self._init_osqp()
        self._osqp.update(
            q=(self._A_i.transpose() @ self._r) / (2 * (self._data.sigma + 2 * self._data.rho * self._data.d_i)))
        with suppress_stdout_stderr():
            res = self._osqp.solve()
        self._x = res.x
        return self._x

    def step(self):
        start_time = time.time()
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
        end_time = time.time()
        return end_time - start_time

    def _update_rs(self, rho: float, sigma: float):
        self._data.rho = rho
        self._data.sigma = sigma

    @classmethod
    def STEP(cls, rho_sigma: Tuple[float, float] = None) -> Dict[int, Tuple[str, np.ndarray, np.ndarray]]:
        ADMM.communicate()
        ret_dict = dict()
        run_time_list = list()
        for admm in all_ADMM:
            if rho_sigma is not None:
                admm._update_rs(rho_sigma[0], rho_sigma[1])
            run_time_list.append(admm.step())
            u = admm._x.copy()
            s = (LongitudeModel.M() @ admm.X_0 + LongitudeModel.N() @ admm.results).reshape((-1, 3))
            road = admm._data.road
            ret_dict[admm._data.vid] = (road, u, s)
        print(np.sum(np.var(cls._all_y, axis=0)))
        y_var_list.append(np.sum(np.var(cls._all_y, axis=0)))
        run_time.append([np.sum(run_time_list), np.mean(run_time_list)])
        return ret_dict

    @classmethod
    def STEP_ALL(cls, step_nums: int, rho_sigma_list: List = None) -> List:
        if rho_sigma_list is not None:
            assert len(rho_sigma_list) == step_nums
        ALL_DATA = list()
        for step in tqdm.tqdm(range(step_nums)):
            if rho_sigma_list is not None:
                ALL_DATA.append(cls.STEP(rho_sigma_list[step]))
            else:
                ALL_DATA.append(cls.STEP())
        all = np.array(run_time)
        print(np.mean(all[:, 1]), np.sum(all[:, 1]))
        return ALL_DATA

    @property
    def VID(self) -> int:
        return self._data.vid

    @property
    def results(self):
        return self._x

    @property
    def X_0(self):
        return self._data.tx0[1]

    @property
    def x(self):
        return self._x

    @staticmethod
    def plot_y_var():
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_var_list[1:])
        plt.yscale('log')
        plt.xlabel('迭代次数' if LANG == 'CN' else 'Iterations')
        plt.ylabel('$\mathrm{y^{i,k}}$的方差' if LANG == 'CN' else 'Var$(y^{i,k})$')
        plt.grid()
        plt.show()
        plt.savefig('output_dir/traj_plan/方差.svg', dpi=300, bbox_inches='tight', pad_inches=.1)
        plt.close()


all_ADMM: List[ADMM] = list()

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    rho_sigma_list = [(1e-1, 1e-1)] + [(1e-1, 1e-1)]*2 + [(1e+0, 1e+0)]*10 + [(1e+1, 1e+1)]* 10 + [(1e+2, 1e+2)]* 30
                     # [(1e+1, 1e+1) for _ in range(10)]
    custom_rho_sigma_num = len(rho_sigma_list)

    LongitudeModel.initialize(LongitudeModel.default_config)
    generator = TrajDataGenerator(100, 50, 150, 110, 150, 10, 20, 15, (-5, 5), (-5, 5), sigma=5, rho=5, safe_dist=10)
    _, trajs = generator.generate_all_vdata()

    iter_times = len(rho_sigma_list)
    # rho_sigma_list = rho_sigma_list + [(1e-3, 1e-3) for _ in range(iter_times - custom_rho_sigma_num)]

    # PickleSave(trajs, '../output_dir/temp01')
    # trajs = PickleRead('../output_dir/temp01')
    for vid, vdata in trajs.items():
        ADMM(vdata)

    all_info = ADMM.STEP_ALL(iter_times, rho_sigma_list=rho_sigma_list)

    save_list = [_ for _ in range(iter_times)][-1:]
    for i, one_step_info in enumerate(all_info):
        if i in save_list:
            plt.plot([0, 100], [110, 110], color='black')
            plt.plot([0, 100], [150, 150], color='black')
            for v_id, one_traj in one_step_info.items():
                road, u, s = one_traj
                plt.plot(
                    one_traj[2][:, 1],
                    # color='red' if road == 'main' else 'green',
                    color='black',
                    linestyle='-' if road == 'main' else '--',
                    lw=0.5
                )
            for admm in all_ADMM:
                plt.plot([admm.data.tdvp[0], admm.data.tdvp[0]], [0, 250], color='black', lw=0.5)
            plt.savefig('../output_dir/traj_plan/traj_plan_{}.jpg'.format(i))
            plt.close()
    for admm in all_ADMM:
        plt.plot(admm.x, color='black', lw=0.5)
    plt.show()
    pass
