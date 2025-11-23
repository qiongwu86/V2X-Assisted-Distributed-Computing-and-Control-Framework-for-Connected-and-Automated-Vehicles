from dynamic_models import LongitudeModel
import numpy as np
from typing import Tuple, Dict, List, AnyStr
from utilits import VData
from scipy import sparse
import osqp


class VehNeighbor:
    def __init__(self, beforeP: int = None, beforeQ: int = None):
        self.beforeP: int = beforeP
        self.beforeQ: int = beforeQ


class TrajB:
    def __init__(self, init_info: Dict[int, VData]):
        self.veh_num: int = len(init_info)
        self.init_info: Dict[int, VData] = init_info
        self.vehNeighbors: Dict[int, VehNeighbor] = dict()
        self.results: Dict[int, np.ndarray] = {}
        self.time_steps = self.init_info[0].T
        self.results[100] = np.zeros(3 * self.time_steps)
        self._prepare()
        self._tau, self._delta = LongitudeModel.gen_M_N(self.init_info[0].T)
        self.d_safe = self.init_info[0].safe_dist

    def _prepare(self):
        for vid in range(self.veh_num):
            v_info: VData = self.init_info[vid]
            v_road: AnyStr = v_info.road

            assert vid == v_info.vid
            # self.results[vid] = np.zeros(3*self.time_steps)

            self.vehNeighbors[vid] = VehNeighbor()
            if vid == 0:
                continue

            found_beforeP = found_beforeQ = False
            for _before in reversed(range(vid)):

                # # add
                # self.vehNeighbors[vid].beforeP = _before
                # self.vehNeighbors[vid].beforeQ = _before
                # found_beforeP = found_beforeQ = True
                # # add

                if found_beforeP and found_beforeQ:
                    break

                if v_road is self.init_info[_before].road:
                    if not found_beforeP:
                        self.vehNeighbors[vid].beforeP = _before
                        found_beforeP = True
                    if not found_beforeQ:
                        self.vehNeighbors[vid].beforeQ = _before
                        found_beforeQ = True
                else:
                    if not found_beforeQ:
                        found_beforeQ = True
                        self.vehNeighbors[vid].beforeQ = _before

    def _get_a_ALU(self, vid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        time_steps = self.init_info[vid].T
        A = np.eye(time_steps + 1)
        l = np.concatenate((-5 * np.ones(time_steps), np.zeros(1)))
        u = np.concatenate((5 * np.ones(time_steps), np.array([np.inf])))
        return A, l, u

    def _get_b_ALU(self, vid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        init_state = self.init_info[vid].tx0[1]
        time_steps = self.init_info[vid].T
        merge_time = self.init_info[vid].tdvp[0]
        before_P = self.vehNeighbors[vid].beforeP
        before_Q = self.vehNeighbors[vid].beforeQ

        E1P1 = np.kron(
            np.diag(
                np.concatenate(
                    (
                        np.ones(merge_time),
                        np.zeros(time_steps - merge_time)
                    )
                )
            ),
            np.array([1, 0, 0])
        )
        # E1P2 = np.zeros((time_steps, 1))
        # E1 = np.hstack((E1P1, E1P2))
        E1 = E1P1

        E2P1 = np.kron(
            np.diag(
                np.concatenate(
                    (
                        np.zeros(merge_time),
                        np.ones(time_steps - merge_time)
                    )
                )
            ),
            np.array([1, 0, 0])
        )
        # E2P2 = np.zeros((time_steps, 1))
        # E2 = np.hstack((E2P1, E2P2))
        E2 = E2P1

        E3 = np.kron(np.eye(time_steps), np.array([-1, 0, 0]))

        # _tmp = 0
        # K = np.vstack((np.ones((time_steps - _tmp, 1)), np.zeros(_tmp)))
        A = np.hstack((E3 @ self._delta, np.ones((time_steps, 1))))

        d_tmp = np.concatenate(
            (
                np.ones(merge_time) if before_P is not None else (-np.ones(merge_time) * np.inf),
                np.ones(time_steps - merge_time) if before_Q is not None else (
                            -np.ones(time_steps - merge_time) * np.inf)
            )
        )

        # _tmp = E1 @ (self.results[before_P] if before_P is not None else np.zeros(3 * self.time_steps))

        l = self.d_safe * d_tmp \
            - E1 @ (self.results[before_P] if before_P is not None else np.zeros(3 * self.time_steps)) \
            - E2 @ (self.results[before_Q] if before_Q is not None else np.zeros(3 * self.time_steps)) \
            - E3 @ self._tau @ init_state
        u = np.inf * np.ones(time_steps)
        return A, l, u

    def _get_c_ALU(self, vid: int):
        merge_time = self.init_info[vid].tdvp[0]
        time_steps = self.init_info[vid].T
        J = np.zeros((2, 3 * time_steps))
        J[0][-3] = 1.0
        for i in range(merge_time):
            J[1][i * 3 + 1] = 1
        # _tmp = J @ self._delta
        A = np.hstack((J @ self._delta, np.zeros((2, 1))))

        _tmp = np.array([
            self.init_info[vid].tdf[1][0],
            (self.init_info[vid].tdvp[1]-self.init_info[vid].tx0[1][0]) / 0.1
            # -np.inf
        ])
        l = _tmp - J @ self._tau @ self.init_info[vid].tx0[1]

        _tmp = np.array([
            self.init_info[vid].tdf[1][1],
            (self.init_info[vid].tdvq[1]-self.init_info[vid].tx0[1][0]) / 0.1
            # np.inf
        ])
        u = _tmp - J @ self._tau @ self.init_info[vid].tx0[1]

        # l1 = self.init_info[vid].tdf[1][0] - _tmp @ self._tau @ self.init_info[vid].tx0[1]
        # l2 = np.array([(self.init_info[vid].tdvp[0] / 0.1)]) - _tmp @ self._tau @ self.init_info[vid].tx0[1]
        # l = np.concatenate((l1, l2))
        #
        # u1 = self.init_info[vid].tdf[1][1] - _tmp @ self._tau @ self.init_info[vid].tx0[1]
        # u2 = np.array([(self.init_info[vid].tdvp[1] + self.init_info[vid].tdvq[1]) / 0.1]) - _tmp @ self._tau @ self.init_info[vid].tx0[1]
        # u = np.concatenate((u1, u2))
        return A, l, u

    def solve(self):
        all_results = dict()
        for vid in range(self.veh_num):
            As = []
            ls = []
            us = []
            alu = self._get_a_ALU(vid)
            As.append(alu[0])
            ls.append(alu[1])
            us.append(alu[2])
            alu = self._get_c_ALU(vid)
            As.append(alu[0])
            ls.append(alu[1])
            us.append(alu[2])
            if vid != 0:
                alu = self._get_b_ALU(vid)
                As.append(alu[0])
                ls.append(alu[1])
                us.append(alu[2])

            P = sparse.csc_matrix(
                np.diag(np.concatenate((
                    np.ones(self.time_steps),
                    np.array([1e2])
                )))
            )
            Q = np.zeros(self.time_steps + 1)
            A = sparse.csc_matrix(np.vstack(As))
            l = np.concatenate(ls)
            u = np.concatenate(us)

            _osqp = osqp.OSQP()
            _osqp.setup(P, Q, A, l, u)
            res = _osqp.solve()
            u = res.x[:-1].copy()
            init_state = self.init_info[vid].tx0[1]
            s = (self._tau @ init_state + self._delta @ u)
            self.results[vid] = s

            all_results[vid] = (self.init_info[vid].road, u, s.reshape((-1, 3)))

        return all_results
