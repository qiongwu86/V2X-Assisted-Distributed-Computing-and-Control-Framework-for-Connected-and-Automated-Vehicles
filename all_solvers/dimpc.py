import numpy as np
import osqp
from dynamic_models import KinematicModel
from scipy import sparse
from utilits import suppress_stdout_stderr, OSQP_RESULT_INFO
from typing import Dict, List, Tuple


class DistributedMPC:

    _initialized: bool = False

    default_config = dict(
        Qx=0.1 * np.diag((1.0, 1.0, 0, 0)),
        Qu=0.05 * np.eye(2),
        safe_factor=10.0,
        init_iter=5,
        run_iter=3,
        sensing_distance=25.0,
        priority=False
    )

    _init_iter: int = 5
    _run_iter: int = 10
    _Qx_big: np.ndarray = None
    _Qu_big: np.ndarray = None
    _safe_factor: float = 10.0,
    _sensing_distance: float = 25.0,
    _priority: bool = False
    _pred_len: int = KinematicModel.pred_len

    @classmethod
    def initialize(cls, config: Dict):
        if cls._initialized:
            raise RuntimeError
        cls._initialized = True
        cls._init_iter = config["init_iter"]
        cls._run_iter = config["run_iter"]
        cls._Qx_big = np.kron(np.eye(cls._pred_len), config["Qx"])
        cls._Qx_big[-4:, -4:] = 10 * config["Qx"]
        cls._Qu_big = np.kron(np.eye(cls._pred_len), config["Qu"])
        cls._safe_factor = config["safe_factor"]
        cls._sensing_distance = config["sensing_distance"]
        cls._priority = config["priority"]

    def __init__(self, init_state: np.ndarray, ref_traj: np.ndarray, mpc_id: int):
        assert DistributedMPC._initialized
        assert KinematicModel.is_initialized()

        self._mpc_id = mpc_id
        _all_mpc[mpc_id] = self

        self._kinematic_model = KinematicModel()

        assert ref_traj.shape[1] == 4
        self._ref_traj = ref_traj

        assert init_state.shape == (4,)
        self._x_t: np.ndarray = init_state
        self._u_nominal: np.ndarray = np.zeros((self._pred_len + 0, 2))
        self._x_nominal = self._kinematic_model.roll_out(self._x_t, self._u_nominal)
        self._x_nominal_others: List[np.ndarray] = list()
        self._t: int = 0
        self._init_nominal()

    @property
    def mpc_ID(self):
        return self._mpc_id

    @property
    def position(self):
        return self._x_t[:2]

    def _update_x_nominal_others(self):
        # get veh in sensing range
        ids_in_range = list()
        for mpc_id, mpc in _all_mpc.items():
            if mpc_id == self._mpc_id:
                continue
            if np.linalg.norm(self.position - mpc.position) <= self._sensing_distance:
                ids_in_range.append(mpc_id)
        # get nominal
        self._x_nominal_others.clear()
        for mpc_id in ids_in_range:
            if self._priority:
                if mpc_id > self._mpc_id:
                    self._x_nominal_others.append(_all_mpc[mpc_id]._x_nominal.copy())
            else:
                self._x_nominal_others.append(_all_mpc[mpc_id]._x_nominal.copy())

    def _init_nominal(self):
        for i in range(self._init_iter):
            A, B, G, ks, bs = self._get_all_necessary_for_qp()
            P, Q, A, l, u = self._get_pqalu(A, B, G, ks, bs)
            with suppress_stdout_stderr():
                prob = osqp.OSQP()
                prob.setup(P, Q, A, l, u)
                result = prob.solve()
                u_opt = np.array(result.x).reshape((self._pred_len, 2))
            self._u_nominal = u_opt
            self._x_nominal = self._kinematic_model.roll_out(self._x_t, u_opt)

    def _get_all_necessary_for_qp(self):
        # param check and get A B G
        assert self._u_nominal is not None and self._u_nominal.shape == (self._pred_len + 0, 2)
        assert self._x_t.shape == (4,)
        A, B, G = self._kinematic_model.dynamic_constrain(u_bar=self._u_nominal, x_0=self._x_t)

        # k and b
        ks = list()
        bs = list()
        if self._x_nominal_others is not None:
            # need calculate k, b
            for i in range(len(self._x_nominal_others)):
                x_nominal_other = self._x_nominal_others[i]
                assert x_nominal_other.shape == (self._pred_len + 1, 4)
                k, b = self._kinematic_model.safe_constrain(self._x_nominal[1:], x_nominal_other[1:])
                ks.append(k)
                bs.append(b)

        return A, B, G, ks, bs

    def _get_pqalu(self, A: np.ndarray, B: np.ndarray, G: np.ndarray, ks: list[np.ndarray], bs: list[np.ndarray]):
        # param check
        assert A.shape == (self._pred_len * 4, 4)
        assert B.shape == (self._pred_len * 4, self._pred_len * 2)
        assert G.shape == (self._pred_len * 4,)
        assert isinstance(ks, list) and all([k.shape == (self._pred_len * 4, self._pred_len * 4) for k in ks])
        assert isinstance(bs, list) and all([b.shape == (self._pred_len * 4,) for b in bs])
        x_ref = self._ref_traj[self._t+1: self._t+1+self._pred_len].reshape(-1)

        P = sparse.csc_matrix(B.transpose() @ self._Qx_big @ B + self._Qu_big +
                              self._safe_factor * sum([(k @ B).transpose() @ (k @ B) for k in ks]))
        Q = B.transpose() @ self._Qx_big @ (A @ self._x_t + G - x_ref.reshape(-1)) + \
            self._safe_factor * sum([(k @ B).transpose() @ (k @ A @ self._x_t + k @ G + b) for k, b in zip(ks, bs)])
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

    def get_nominal(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._x_nominal, self._u_nominal

    def _inner_optimize(self) -> List:
        A, B, G, ks, bs = self._get_all_necessary_for_qp()
        P, Q, A, l, u = self._get_pqalu(A, B, G, ks, bs)
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


_all_mpc: Dict[int, DistributedMPC] = {}

if __name__ == "__main__":
    pass
