import numpy as np
import osqp

from dynamic import KinematicModel
from scipy import sparse
from utilit import suppress_stdout_stderr
from typing import Dict, List


class OSQP_RESULT_INFO:
    RUN_TIME = 0
    SOLVE_TIME = 1
    STATUS = 2
    ITER_TIMES = 3
    @staticmethod
    def get_info_from_result(res) -> List:
        return [res.info.run_time, res.info.solve_time, res.info.status, res.info.iter]


class DistributedMPC:
    default_config = dict(
        Qx=0.1 * np.diag((1.0, 1.0, 0, 0)),
        Qu=0.07 * np.eye(2),
        safe_factor=10.0,
        init_iter=10,
        run_iter=10,
        sensing_distance=25.0,
        priority=False
    )

    _init_iter = default_config["init_iter"]
    _run_iter = default_config["run_iter"]

    def __init__(self, config: dict, init_state: np.ndarray, ref_traj: np.ndarray, mpc_id: int):
        _all_mpc[mpc_id] = self
        self._kinematic_model = KinematicModel(KinematicModel.default_config)
        self._mpc_id = mpc_id
        self._pred_len = self._kinematic_model.pred_len
        self._safe_factor = config["safe_factor"]
        self._sending_dist = config["sensing_distance"]
        self._priority = config["priority"]

        assert ref_traj.shape[1] == 4
        self._ref_traj = ref_traj

        assert init_state.shape == (4,)
        self._x_t: np.ndarray = init_state
        self._x_nominal: np.ndarray = np.vstack((self._x_t, np.zeros((self._pred_len, 4))))
        self._u_nominal: np.ndarray = np.zeros((self._pred_len + 0, 2))
        self._t: int = 0
        self._x_nominal_others: list[np.ndarray] = list()
        # hyper param
        self._Qx_big = np.kron(np.eye(self._pred_len), config["Qx"])
        self._Qx_big[-4:, -4:] = 10 * config["Qx"]
        self._Qu_big = np.kron(np.eye(self._pred_len), config["Qu"])
        # initialize
        self._init_nominal()

    @property
    def mpc_ID(self):
        return self._mpc_id

    @property
    def position(self):
        return self._x_t[:2]

    def _get_others_in_distance(self, distance: float) -> list:
        ret_id = list()
        for mpc_id, mpc in _all_mpc.items():
            if mpc_id == self._mpc_id:
                continue
            if np.linalg.norm(self.position - mpc.position) <= distance:
                ret_id.append(mpc_id)
        return ret_id

    def _update_x_nominal_others(self):
        self._x_nominal_others.clear()
        ids_in_range = self._get_others_in_distance(self._sending_dist)
        for mpc_id in ids_in_range:
            if self._priority:
                if mpc_id > self._mpc_id:
                    self._x_nominal_others.append(_all_mpc[mpc_id]._x_nominal.copy())
            else:
                self._x_nominal_others.append(_all_mpc[mpc_id]._x_nominal.copy())

    def _init_nominal(self) -> List:
        _ret_info = list()
        for i in range(self._init_iter):
            A, B, G, ks, bs = self._get_all_necessary_for_qp(self._u_nominal, x_nominal=self._x_nominal)
            P, Q, A, l, u = self._get_pqalu(self._x_t, self._ref_traj[self._t: self._t + self._pred_len], A, B, G, ks,
                                            bs)
            with suppress_stdout_stderr():
                prob = osqp.OSQP()
                prob.setup(P, Q, A, l, u)
                result = prob.solve()
                u_opt = np.array(result.x).reshape((self._pred_len, 2))
            self._u_nominal = u_opt
            self._x_nominal = self._kinematic_model.roll_out(self._x_t, u_opt)
        return _ret_info

    def _get_all_necessary_for_qp(self,
                                  u_nominal: np.ndarray,
                                  x_0: np.ndarray = None,
                                  x_nominal: np.ndarray = None,
                                  x_nominal_others: list[np.ndarray] = None):
        # param check and get A B G
        assert u_nominal is not None and u_nominal.shape == (self._pred_len + 0, 2)
        if x_0 is not None:
            assert x_0.shape == (4,) and x_nominal is None
            A, B, G = self._kinematic_model.dynamic_constrain(u_nominal, x_0=x_0)
        elif x_nominal is not None:
            assert x_nominal.shape == (self._pred_len + 1, 4) and x_0 is None
            A, B, G = self._kinematic_model.dynamic_constrain(u_nominal, x_bar=x_nominal[:-1])
        else:
            raise ValueError

        # k and b
        ks = list()
        bs = list()
        if x_nominal_others is not None:
            # need calculate k, b
            for i in range(len(x_nominal_others)):
                x_nominal_other = x_nominal_others[i]
                assert x_nominal_other.shape == (self._pred_len + 1, 4)
                k, b = self._kinematic_model.safe_constrain(x_nominal[1:], x_nominal_other[1:])
                ks.append(k)
                bs.append(b)

        return A, B, G, ks, bs

    def _get_pqalu(self, x_t: np.ndarray, x_ref: np.ndarray,
                   A: np.ndarray, B: np.ndarray, G: np.ndarray, ks: list[np.ndarray], bs: list[np.ndarray]):
        # param check
        assert x_t.shape == (4,)
        assert x_ref.shape == (self._pred_len, 4)
        assert A.shape == (self._pred_len * 4, 4)
        assert B.shape == (self._pred_len * 4, self._pred_len * 2)
        assert G.shape == (self._pred_len * 4,)
        assert isinstance(ks, list) and all([k.shape == (self._pred_len * 4, self._pred_len * 4) for k in ks])
        assert isinstance(bs, list) and all([b.shape == (self._pred_len * 4,) for b in bs])

        P = sparse.csc_matrix(B.transpose() @ self._Qx_big @ B + self._Qu_big +
                              self._safe_factor * sum([(k @ B).transpose() @ (k @ B) for k in ks]))
        Q = B.transpose() @ self._Qx_big @ (A @ x_t + G - x_ref.reshape(-1)) + \
            self._safe_factor * sum([(k @ B).transpose() @ (k @ A @ x_t + k @ G + b) for k, b in zip(ks, bs)])
        # not include x constrain now
        A = sparse.csc_matrix(np.eye(self._pred_len * 2))
        l = np.kron(np.ones((self._pred_len,)), np.array([KinematicModel.default_config["acc_min"],
                                                          KinematicModel.default_config["steer_min"]]))
        u = np.kron(np.ones((self._pred_len,)), np.array([KinematicModel.default_config["acc_max"],
                                                          KinematicModel.default_config["steer_max"]]))

        assert P.shape == (self._pred_len * 2, self._pred_len * 2)
        assert Q.shape == (self._pred_len * 2,)
        assert A.shape == (self._pred_len * 2, self._pred_len * 2)
        assert l.shape == (self._pred_len * 2,)
        assert u.shape == (self._pred_len * 2,)

        return P, Q, A, l, u

    def get_nominal(self):
        return self._x_nominal, self._u_nominal

    def step(self):
        for i in range(self._run_iter):
            self._inner_optimize()
        self._step_forward_from_nominal()
        return self._x_t

    def _inner_optimize(self) -> List:
        A, B, G, ks, bs = self._get_all_necessary_for_qp(self._u_nominal, x_nominal=self._x_nominal,
                                                         x_nominal_others=self._x_nominal_others)
        P, Q, A, l, u = self._get_pqalu(self._x_t, self._ref_traj[self._t: self._t + self._pred_len], A, B, G, ks,
                                        bs)
        # with suppress_stdout_stderr():
        prob = osqp.OSQP()
        prob.setup(P, Q, A, l, u)
        result = prob.solve()
        u_opt = np.array(result.x).reshape((self._pred_len, 2))
        self._u_nominal = u_opt
        self._x_nominal = self._kinematic_model.roll_out(self._x_t, u_opt)
        return OSQP_RESULT_INFO.get_info_from_result(result)

    def _step_forward_from_nominal(self):
        u = self._u_nominal[0]
        self._x_t = self._kinematic_model.step_once(self._x_t, u)
        self._u_nominal = np.vstack((self._u_nominal[1:], self._u_nominal[-1]))
        self._x_nominal = self._kinematic_model.roll_out(self._x_t, self._u_nominal)
        self._t += 1
        return u

    @staticmethod
    def step_all() -> Dict:
        step_info = dict()
        for mpc_id, mpc in _all_mpc.items():
            step_info[mpc_id] = dict()
        with suppress_stdout_stderr():
            # collect old state
            for mpc_id, mpc in _all_mpc.items():
                step_info[mpc_id]["old_state"] = mpc._x_t

            # iter and optimize
            for mpc_id, mpc in _all_mpc.items():
                step_info[mpc_id]["nominal"] = list()
                step_info[mpc_id]["osqp_res"] = list()
            for i in range(DistributedMPC._run_iter):
                # update nominal
                for mpc_id, mpc in _all_mpc.items():
                    mpc._update_x_nominal_others()
                # optimize and get osqp info
                for mpc_id, mpc in _all_mpc.items():
                    step_info[mpc_id]["osqp_res"].append(mpc._inner_optimize())
                # collect nominal
                for mpc_id, mpc in _all_mpc.items():
                    step_info[mpc_id]["nominal"].append(mpc.get_nominal())

            # step forward
            for mpc_id, mpc in _all_mpc.items():
                step_info[mpc_id]["control"] = mpc._step_forward_from_nominal()

            # collect new state
            for mpc_id, mpc in _all_mpc.items():
                step_info[mpc_id]["new_state"] = mpc._x_t
        return step_info

    def test(self, x0, u_bar):
        x_roll = self._kinematic_model.roll_out(x0, u_bar)
        A, B, G, ks, bs = self._get_all_necessary_for_qp(u_bar, x_nominal=x_roll)
        x_mat = A @ x0 + B @ u_bar.reshape(-1) + G
        print(sum(x_mat - x_roll[1:].reshape(-1)), len(x_mat))
        for i in range(len(x_mat)):
            print("{0:5.2f} | {1:5.2f}".format(x_mat[i], x_roll.reshape(-1)[i]))

        from dynamic_model import BicycleModel
        AA, BB, GG = BicycleModel.MakeDynamicConstrain(x0, u_bar, 30)
        print(np.sum(np.abs(A - AA)), np.sum(np.abs(B - BB)), np.sum(np.abs(G - GG)))


_all_mpc: Dict[int, DistributedMPC] = {}


if __name__ == "__main__":
    pass
    # init_state = np.array((0,0, np.pi * 0.5, 0))
    # points_num = 1000
    # radius = 20
    # speed = 2
    # beta = (speed/radius) / 10
    # rads = [beta*i for i in range(points_num)]
    # ref_traj_x = radius - np.cos(rads) * radius
    # ref_traj_y = np.sin(rads) * radius
    # ref_traj_phi = np.zeros((points_num, ))
    # ref_traj_v = speed * np.ones((points_num,))
    # ref_traj = np.vstack((ref_traj_x, ref_traj_y, ref_traj_phi, ref_traj_v)).transpose()
    # print(ref_traj.shape)
    # mpc = DistributedMPC(DistributedMPC.default_config, init_state, ref_traj, 10)
    # plt.plot(ref_traj[:, 0], ref_traj[:, 1])
    # # x_n, u_n = mpc.get_nominal()
    # # plt.plot(x_n[:,0], x_n[:,1])
    # # plt.show()
    # log = np.zeros((50, 4))
    # for i in range(50):
    #     state = mpc.step()
    #     log[i] = state
    # plt.plot(log[:, 0], log[:, 1])
    # plt.show()
    #
    # x_n, u_n = mpc.get_nominal()
    # mpc.test(x0=x_n[0], u_bar=u_n)

    # init_state_01 = np.array((0,0.5, 0, 0))
    # init_state_02 = np.array((100,-0.5, np.pi, 0))
    #
    # points_num = 1000
    # radius = 20
    # speed = 1
    # beta = (speed/radius) / 10
    # rad_01 = [beta*i for i in range(points_num)]
    # rad_02 = [-beta*i for i in range(points_num)]
    #
    # # ref_traj_x = radius - np.cos(rad_01) * radius
    # ref_traj_x = np.arange(0, 100, 0.1)
    # ref_traj_y = 0.5 * np.ones((len(ref_traj_x),))
    # ref_traj_phi = np.zeros((len(ref_traj_x),))
    # ref_traj_v = speed * np.ones((points_num, ))
    # ref_traj_01 = np.vstack((ref_traj_x, ref_traj_y, ref_traj_phi, ref_traj_v)).transpose()
    #
    # ref_traj_x = np.arange(100, 0, -0.1)
    # ref_traj_y = -0.5 * np.ones((len(ref_traj_x),))
    # ref_traj_phi = np.zeros((len(ref_traj_x),))
    # ref_traj_v = speed * np.ones((points_num, ))
    # ref_traj_02 = np.vstack((ref_traj_x, ref_traj_y, ref_traj_phi, ref_traj_v)).transpose()
    #
    # plt.plot(ref_traj_01[:, 0], ref_traj_01[:, 1])
    # plt.plot(ref_traj_02[:, 0], ref_traj_02[:, 1])
    # # plt.show()
    #
    # mpc_01 = DistributedMPC(DistributedMPC.default_config, init_state_01, ref_traj_01, 1)
    # mpc_02 = DistributedMPC(DistributedMPC.default_config, init_state_02, ref_traj_02, 2)
    #
    # run_steps = 600
    # mpc_01_log = np.zeros((run_steps, 4))
    # mpc_02_log = np.zeros((run_steps, 4))
    # for i in range(run_steps):
    #     step_info = DistributedMPC.step_all()
    #     mpc_01_log[i] = step_info["new_state"][1]
    #     mpc_02_log[i] = step_info["new_state"][2]
    #
    # plt.plot(mpc_01_log[:, 0], mpc_01_log[:, 1])
    # plt.plot(mpc_02_log[:, 0], mpc_02_log[:, 1])
    # plt.show()

    # # T
    # traj1, traj2, traj3 = _gen_trace_01()
    # plt.plot(traj1[:, 0], traj1[:, 1])
    # plt.plot(traj2[:, 0], traj2[:, 1])
    # mpc_01 = DistributedMPC(DistributedMPC.default_config, traj1[0], traj1, 1)
    # mpc_02 = DistributedMPC(DistributedMPC.default_config, traj2[0], traj2, 2)
    # mpc_03 = DistributedMPC(DistributedMPC.default_config, traj3[0], traj3, 3)
    # run_steps = min((traj1.shape[0], traj2.shape[0], traj3.shape[0])) - KinematicModel.default_config["pred_len"]
    # # mpc_01_log = np.zeros((run_steps, 4))
    # # mpc_02_log = np.zeros((run_steps, 4))
    # step_infos = list()
    # for i in range(run_steps):
    #     step_info = DistributedMPC.step_all()
    #     step_infos.append(step_info)
    # #     mpc_01_log[i] = step_info["new_state"][1]
    # #     mpc_02_log[i] = step_info["new_state"][2]
    # # plt.plot(mpc_01_log[:, 0], mpc_01_log[:, 1])
    # # plt.plot(mpc_02_log[:, 0], mpc_02_log[:, 1])
    # # plt.axis('equal')
    # # plt.show()
    # _draw_from_info(step_infos)
