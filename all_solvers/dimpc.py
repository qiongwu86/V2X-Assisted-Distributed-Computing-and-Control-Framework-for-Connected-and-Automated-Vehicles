import numpy as np
import osqp
from dynamic_models import KinematicModel
from scipy import sparse
from utilits import suppress_stdout_stderr, OSQP_RESULT_INFO
from typing import Dict, List, Tuple, Callable
import casadi
import tqdm
import time


class DistributedMPC:

    _initialized: bool = False

    default_config = dict(
        Qx=0.5 * np.diag((1.0, 1.0, 0.0, 0)),
        Qu=0.1 * np.diag([1.0, 0.1]),
        comfort=(1.0, 0.3),
        safe_factor=10.0,
        safe_th=5.0,
        init_iter=5,
        run_iter=3,
        sensing_distance=25.0,
        priority=False,
        pred_len=30,
        other_veh_num=3,
        warm_start=True,
        osqp_check_termination=1
    )

    _init_iter: int = 5
    _run_iter: int = 10
    _comfort: Tuple[float] = (1.0, 0.3)
    _safe_factor: float = 10.0,
    _safe_th: float = 4.0
    _sensing_distance: float = 25.0,
    _priority: bool = False
    _pred_len: int = 30
    _other_veh_num: int = 3
    _warm_start: bool = True
    _osqp_check_termination: int = 0

    _Q_comfort: np.ndarray = None
    _Qx_big: np.ndarray = None
    _Qu_big: np.ndarray = None

    _AA_fun: Callable = None
    _BB_fun: Callable = None
    _GG_fun: Callable = None
    _k_fun: Callable = None
    _b_fun: Callable = None
    _dist_fun: Callable = None

    @classmethod
    def initialize(cls, config: Dict):
        if cls._initialized:
            raise RuntimeError
        cls._initialized = True
        cls._init_iter = config["init_iter"]
        cls._run_iter = config["run_iter"]
        cls._comfort = config['comfort']
        cls._safe_factor = config["safe_factor"]
        cls._safe_th = config["safe_th"]
        cls._sensing_distance = config["sensing_distance"]
        cls._priority = config["priority"]
        cls._pred_len = config["pred_len"]
        cls._other_veh_num = config["other_veh_num"]
        cls._warm_start = config["warm_start"]
        cls._osqp_check_termination = config["osqp_check_termination"]

        cls._Qx_big = np.kron(np.eye(cls._pred_len), config["Qx"])
        cls._Qx_big[-4:, -4:] = 5 * config["Qx"]
        cls._Qu_big = np.kron(np.eye(cls._pred_len), config["Qu"])
        cls._Q_comfort = cls._gen_Q_comfort()

        cls._AA_fun, cls._BB_fun, cls._GG_fun = cls._dynamic_constrain()
        cls._k_fun, cls._b_fun, cls._dist_fun = cls._safe_constrain_constructor()

    @classmethod
    def _gen_Q_comfort(cls) -> np.ndarray:
        Q_comfort = np.zeros((2*(cls._pred_len-1), cls._pred_len*4))
        for i in range(cls._pred_len-1):
            Q_comfort[i*2, i * 4 + 2] = -cls._comfort[0]
            Q_comfort[i*2+1, i * 4 + 3] = -cls._comfort[1]
            Q_comfort[i*2, (i + 1) * 4 + 2] = cls._comfort[0]
            Q_comfort[i*2+1, (i + 1) * 4 + 3] = cls._comfort[1]
        return Q_comfort

    def __init__(self, init_state: np.ndarray, ref_traj: np.ndarray, mpc_id: int):
        assert DistributedMPC._initialized
        assert KinematicModel.is_initialized()

        self._mpc_id = mpc_id
        _all_mpc[mpc_id] = self

        assert ref_traj.shape[1] == 4
        self._ref_traj = ref_traj

        assert init_state.shape == (4,)
        self._x_t: np.ndarray = init_state
        self._u_nominal: np.ndarray = np.zeros((self._pred_len + 0, 2))
        self._x_nominal: np.ndarray = KinematicModel.roll_out(self._x_t, self._u_nominal)
        self._x_nominal_others: List[np.ndarray] = list()
        self._t: int = 0
        self._init_nominal()
        self._y_warm: np.ndarray = np.zeros((self._pred_len * 2))

    @property
    def max_step(self):
        return self._ref_traj.shape[0] - self._pred_len - 1

    @property
    def mpc_ID(self):
        return self._mpc_id

    @property
    def position(self):
        return self._x_t[:2]

    def _update_x_nominal_others(self):
        # get veh in sensing range
        ids_in_range_dict = dict()
        for mpc_id, mpc in _all_mpc.items():
            if mpc_id == self._mpc_id:
                continue
            if np.linalg.norm(self.position - mpc.position) <= self._sensing_distance:
                ids_in_range_dict[mpc_id] = np.linalg.norm(self.position - mpc.position)
        ids_in_range = sorted(ids_in_range_dict.keys(), key=lambda _id: ids_in_range_dict[_id])
        # get nominal
        self._x_nominal_others.clear()
        for mpc_id in ids_in_range:
            if len(self._x_nominal_others) >= self._other_veh_num:
                break
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
            self._x_nominal = KinematicModel.roll_out(self._x_t, u_opt)

    def _get_all_necessary_for_qp(self):
        # param check and get A B G
        assert self._u_nominal is not None and self._u_nominal.shape == (self._pred_len + 0, 2)
        assert self._x_t.shape == (4,)
        A, B, G = self.dynamic_constrain(u_bar=self._u_nominal, x_0=self._x_t)

        # k and b
        ks = list()
        bs = list()
        if self._x_nominal_others is not None:
            # need calculate k, b
            for i in range(len(self._x_nominal_others)):
                x_nominal_other = self._x_nominal_others[i]
                assert x_nominal_other.shape == (self._pred_len + 1, 4)
                k, b = self.safe_constrain(self._x_nominal[1:], x_nominal_other[1:])
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

        MAT_1 = self._Qx_big + self._safe_factor * sum([k.transpose() @ k for k in ks]) + self._Q_comfort.transpose() @ self._Q_comfort
        P = sparse.csc_matrix(B.transpose() @ MAT_1 @ B + self._Qu_big)
        Q = B.transpose() @ (
                MAT_1 @ (A @ self._x_t + G) + self._safe_factor * sum([k.transpose()@b for k, b in zip(ks, bs)]) - self._Qx_big @ x_ref
        )

        '''
        P = sparse.csc_matrix(
            B.transpose() @ self._Qx_big @ B + self._Qu_big +
            self._safe_factor * sum([(k @ B).transpose() @ (k @ B) for k in ks]) +
            (self._Q_comfort @ B).transpose() @ (self._Q_comfort @ B)
        )
        Q = B.transpose() @ self._Qx_big @ (A @ self._x_t + G - x_ref) + \
            self._safe_factor * sum([(k @ B).transpose() @ (k @ A @ self._x_t + k @ G + b) for k, b in zip(ks, bs)]) + \
            (self._Q_comfort @ B).transpose() @ (self._Q_comfort @ A @ self._x_t + self._Q_comfort @ G)
        '''

        # not include x constrain now
        A = sparse.csc_matrix(np.eye(self._pred_len * 2))
        l = np.kron(np.ones((self._pred_len,)), np.array([KinematicModel.acc_min,
                                                          KinematicModel.steer_min]))
        u = np.kron(np.ones((self._pred_len,)), np.array([KinematicModel.acc_max,
                                                          KinematicModel.steer_max]))

        assert P.shape == (self._pred_len * 2, self._pred_len * 2)
        assert Q.shape == (self._pred_len * 2,)
        assert A.shape == (self._pred_len * 2, self._pred_len * 2)
        assert l.shape == (self._pred_len * 2,)
        assert u.shape == (self._pred_len * 2,)

        return P, Q, A, l, u

    def get_nominal(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._x_nominal, self._u_nominal

    def _inner_optimize(self) -> Tuple:
        start_time = time.time()
        A, B, G, ks, bs = self._get_all_necessary_for_qp()
        P, Q, A, l, u = self._get_pqalu(A, B, G, ks, bs)
        end_time = time.time()
        # with suppress_stdout_stderr():
        prob = osqp.OSQP()
        prob.setup(P, Q, A, l, u, check_termination=self._osqp_check_termination)
        if self._warm_start:
            prob.warm_start(x=self._u_nominal.reshape(-1), y=self._y_warm)
        result = prob.solve()
        u_opt = np.array(result.x).reshape((self._pred_len, 2))
        self._u_nominal = u_opt
        self._x_nominal = KinematicModel.roll_out(self._x_t, u_opt)
        self._y_warm = result.y
        osqp_result = list(OSQP_RESULT_INFO.get_info_from_result(result))
        osqp_result[OSQP_RESULT_INFO.RUN_TIME] += (end_time-start_time)
        return tuple(osqp_result)

    def _step_forward_from_nominal(self):
        u = self._u_nominal[0]
        self._x_t = KinematicModel.step_once(self._x_t, u)
        self._u_nominal = np.vstack((self._u_nominal[1:], self._u_nominal[-1]))
        self._x_nominal = KinematicModel.roll_out(self._x_t, self._u_nominal)
        self._t += 1
        return u

    @classmethod
    def _dynamic_constrain(cls):
        x_t = casadi.SX.sym('x_t', 4)
        u_t = casadi.SX.sym('u_t', 2)
        x_dot = casadi.vertcat(
            x_t[3, 0] * casadi.cos(x_t[2, 0] + casadi.arctan(0.5 * casadi.tan(u_t[1, 0]))),
            x_t[3, 0] * casadi.sin(x_t[2, 0] + casadi.arctan(0.5 * casadi.tan(u_t[1, 0]))),
            x_t[3, 0] * casadi.sin(casadi.arctan(0.5 * casadi.tan(u_t[1, 0]))) / (0.5 * KinematicModel.length),
            u_t[0, 0]
        )
        F1 = casadi.Function('F1', [x_t, u_t], [KinematicModel.delta_T * x_dot + x_t])

        x0 = casadi.MX.sym('x_0', 4)
        x_current = x0
        x_list = []
        u_list = []
        for i in range(cls._pred_len):
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
        assert u_bar is not None and u_bar.shape == (cls._pred_len, 2)
        assert x_0.shape == (4,)

        AA = np.array(cls._AA_fun(x_0, u_bar.reshape(-1)))
        BB = np.array(cls._BB_fun(x_0, u_bar.reshape(-1)))
        GG = np.array(cls._GG_fun(x_0, u_bar.reshape(-1))).reshape(-1)

        return AA, BB, GG

    @classmethod
    def _safe_constrain_constructor(cls):
        x_ego = casadi.SX.sym('x_ego', 4)
        x_other = casadi.SX.sym('x_other', 4)

        V = 0.5 * (KinematicModel.length - KinematicModel.width)
        D = KinematicModel.width
        th = cls._safe_th

        _dist_inner = casadi.fmin(
            casadi.vertcat(
                casadi.sqrt(
                    (x_ego[0] - x_other[0] + V * (casadi.cos(x_ego[2]) - casadi.cos(x_other[2]))) ** 2 +
                    (x_ego[1] - x_other[1] + V * (casadi.sin(x_ego[2]) - casadi.sin(x_other[2]))) ** 2
                ) - D - th,
                casadi.sqrt(
                    (x_ego[0] - x_other[0] + V * (casadi.cos(x_ego[2]) + casadi.cos(x_other[2]))) ** 2 +
                    (x_ego[1] - x_other[1] + V * (casadi.sin(x_ego[2]) + casadi.sin(x_other[2]))) ** 2
                ) - D - th,
                casadi.sqrt(
                    (x_ego[0] - x_other[0] + V * (-casadi.cos(x_ego[2]) - casadi.cos(x_other[2]))) ** 2 +
                    (x_ego[1] - x_other[1] + V * (-casadi.sin(x_ego[2]) - casadi.sin(x_other[2]))) ** 2
                ) - D - th,
                casadi.sqrt(
                    (x_ego[0] - x_other[0] + V * (-casadi.cos(x_ego[2]) + casadi.cos(x_other[2]))) ** 2 +
                    (x_ego[1] - x_other[1] + V * (-casadi.sin(x_ego[2]) + casadi.sin(x_other[2]))) ** 2
                ) - D - th,
            ),
            casadi.SX.zeros(4)
        )
        _dist_inner_func = casadi.Function('dist_inner_func', [x_ego, x_other], [_dist_inner])

        _dist_all_list = []
        _x_ego_all = casadi.MX.sym('_x_ego_all', 4*cls._pred_len)
        _x_other_all = casadi.MX.sym('_x_other_all', 4*cls._pred_len)
        for i in range(cls._pred_len):
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
        assert x_ego.shape == (cls._pred_len, 4) and x_other.shape == (cls._pred_len, 4)
        k = cls._k_fun(x_ego.reshape(-1), x_other.reshape(-1))
        b = cls._b_fun(x_ego.reshape(-1), x_other.reshape(-1))
        return np.array(k), np.array(b).reshape(-1)

    @classmethod
    def dist_calculator(cls, x_ego: np.ndarray, x_other: np.ndarray):
        assert cls._initialized
        assert x_ego.shape == (cls._pred_len, 4) and x_other.shape == (cls._pred_len, 4)
        dist = cls._dist_fun(x_ego.reshape(-1), x_other.reshape(-1))
        return np.array(dist)

    @staticmethod
    def step_all() -> Dict:
        step_info = dict()
        for mpc_id, mpc in _all_mpc.items():
            step_info[mpc_id] = dict()
        # collect old state
        for mpc_id, mpc in _all_mpc.items():
            step_info[mpc_id]["old_state"] = mpc._x_t
        # iter and optimize
        for mpc_id, mpc in _all_mpc.items():
            step_info[mpc_id]["nominal"] = list()
            step_info[mpc_id]["osqp_res"] = list()
        with suppress_stdout_stderr():
            for i in range(DistributedMPC._run_iter):
                # update nominal
                for mpc_id, mpc in _all_mpc.items():
                    mpc._update_x_nominal_others()
                # optimize and get osqp info and collect nominal
                for mpc_id, mpc in _all_mpc.items():
                    step_info[mpc_id]["osqp_res"].append(mpc._inner_optimize())
                    step_info[mpc_id]["nominal"].append(mpc.get_nominal())
        # step forward
        for mpc_id, mpc in _all_mpc.items():
            step_info[mpc_id]["control"] = mpc._step_forward_from_nominal()
        # collect new state
        for mpc_id, mpc in _all_mpc.items():
            step_info[mpc_id]["new_state"] = mpc._x_t
        return step_info

    @staticmethod
    def simulate() -> List:
        all_info = list()
        max_step = min([mpc_obj.max_step for mpc_obj in _all_mpc.values()])
        print("max step: {}".format(max_step))
        for _ in tqdm.tqdm(range(max_step)):
            all_info.append(DistributedMPC.step_all())
        return all_info


_all_mpc: Dict[int, DistributedMPC] = {}

if __name__ == "__main__":
    pass
