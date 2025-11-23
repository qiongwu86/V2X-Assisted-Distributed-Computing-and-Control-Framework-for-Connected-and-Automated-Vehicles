import numpy as np
from dynamic_models import KinematicModel
from utilits import suppress_stdout_stderr, NLP_RESULT_INFO
from typing import Dict, List, Tuple, Callable
import casadi as ca
import tqdm


class DistributedMPCIPOPT:

    _initialized: bool = False

    default_config = dict(
        Qx=0.1 * np.diag((1.0, 1.0, 0.0, 0)),
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
        kernel='ipopt'
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
    _kernel: str = 'ipopt'

    _Q_comfort: np.ndarray = None
    _Qx_big: np.ndarray = None
    _Qu_big: np.ndarray = None
    _nlp_solver = None
    _lbx = None
    _ubx = None
    _lbg = None
    _ubg = None

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
        cls._kernel = config["kernel"]

        cls._Qx_big = np.kron(np.eye(cls._pred_len), config["Qx"])
        cls._Qx_big[-4:, -4:] = 5 * config["Qx"]
        cls._Qu_big = np.kron(np.eye(cls._pred_len), config["Qu"])
        cls._Q_comfort = cls._gen_Q_comfort()
        cls._nlp_solver, cls._lbx, cls._ubx, cls._lbg, cls._ubg = cls._gen_nlp_solver()

    @classmethod
    def _gen_nlp_solver(cls):
        delta_T = KinematicModel.delta_T
        _state = ca.SX.sym('state', 4)
        _control = ca.SX.sym('control', 2)
        _state_dot = ca.vertcat(
            _state[3, 0] * ca.cos(_state[2, 0] + ca.arctan(0.5 * ca.tan(_control[1, 0]))),
            _state[3, 0] * ca.sin(_state[2, 0] + ca.arctan(0.5 * ca.tan(_control[1, 0]))),
            _state[3, 0] * ca.sin(ca.arctan(0.5 * ca.tan(_control[1, 0]))) / (0.5 * 3.5),
            _control[0, 0]
        )
        _state_next = ca.Function('state_next_func', [_state, _control], [delta_T * _state_dot + _state])

        # param
        x_0 = ca.MX.sym('x_0', 4)
        x_ref = ca.vertcat(*[ca.MX.sym('x_ref_' + str(i + 1), 4) for i in range(cls._pred_len)])
        x_nominal_other_list = [ca.MX.sym('x_no', cls._pred_len*4) for _ in range(cls._other_veh_num)]
        x_nominal_other = ca.vertcat(*x_nominal_other_list)

        # dec var
        x_1_T = ca.vertcat(*[ca.MX.sym('x_' + str(i + 1), 4) for i in range(cls._pred_len)])
        u_0_T_1 = ca.vertcat(*[ca.MX.sym('u_' + str(i), 2) for i in range(cls._pred_len)])

        # constrain - g
        lbg = [0, 0, 0, 0]
        ubg = [0, 0, 0, 0]
        g_list = [x_1_T[:4] - _state_next(x_0, u_0_T_1[:2])]
        for i in range(1, cls._pred_len):
            g_list.append(x_1_T[i*4: (i+1)*4] - _state_next(x_1_T[(i - 1)*4: i*4], u_0_T_1[i*2: (i+1)*2]))
            lbg += [0, 0, 0, 0]
            ubg += [0, 0, 0, 0]
        g = ca.vertcat(*g_list)

        # constrain - x
        lbx = [
                i for pair in zip(KinematicModel.acc_min*np.ones((cls._pred_len,)),
                                  KinematicModel.steer_min*np.ones((cls._pred_len,))) for i in pair
        ] + [-np.inf for _ in range(cls._pred_len*4)]

        ubx = [
                i for pair in zip(KinematicModel.acc_max*np.ones((cls._pred_len,)),
                                  KinematicModel.steer_max*np.ones((cls._pred_len,))) for i in pair
        ] + [+np.inf for _ in range(cls._pred_len*4)]

        Qx = ca.DM(cls._Qx_big)
        Qu = ca.DM(cls._Qu_big)
        Qc = ca.DM(cls._Q_comfort)

        # obj
        _fake_dist_func = cls._safe_constrain_constructor()
        safe_vec = ca.vertcat(*[_fake_dist_func(x_1_T, a_nomi) for a_nomi in x_nominal_other_list])
        J = (x_1_T - x_ref).T @ Qx @ (x_1_T - x_ref) + \
            u_0_T_1.T @ Qu @ u_0_T_1 + \
            (Qc @ x_1_T).T @ (Qc @ x_1_T) + \
            safe_vec.T @ safe_vec

        nlp = {
            'x': ca.vertcat(u_0_T_1, x_1_T),
            'f': J,
            'g': g,
            'p': ca.vertcat(x_0, x_ref, x_nominal_other)
        }

        prob_option = dict(
            # verbose=False,
            # verbose_init=False,
            # print_in=False,
            # print_out=False,
            # print_time=False,
            record_time=True,
        )

        return ca.nlpsol('S_ipopt', cls._kernel, nlp, prob_option), lbx, ubx, lbg, ubg

    @classmethod
    def _safe_constrain_constructor(cls):
        x_ego = ca.SX.sym('x_ego', 4)
        x_other = ca.SX.sym('x_other', 4)

        V = 0.5 * (KinematicModel.length - KinematicModel.width)
        D = KinematicModel.width
        th = cls._safe_th

        _dist_inner = ca.fmin(
            ca.vertcat(
                ca.sqrt(
                    (x_ego[0] - x_other[0] + V * (ca.cos(x_ego[2]) - ca.cos(x_other[2]))) ** 2 +
                    (x_ego[1] - x_other[1] + V * (ca.sin(x_ego[2]) - ca.sin(x_other[2]))) ** 2
                ) - D - th,
                ca.sqrt(
                    (x_ego[0] - x_other[0] + V * (ca.cos(x_ego[2]) + ca.cos(x_other[2]))) ** 2 +
                    (x_ego[1] - x_other[1] + V * (ca.sin(x_ego[2]) + ca.sin(x_other[2]))) ** 2
                ) - D - th,
                ca.sqrt(
                    (x_ego[0] - x_other[0] + V * (-ca.cos(x_ego[2]) - ca.cos(x_other[2]))) ** 2 +
                    (x_ego[1] - x_other[1] + V * (-ca.sin(x_ego[2]) - ca.sin(x_other[2]))) ** 2
                ) - D - th,
                ca.sqrt(
                    (x_ego[0] - x_other[0] + V * (-ca.cos(x_ego[2]) + ca.cos(x_other[2]))) ** 2 +
                    (x_ego[1] - x_other[1] + V * (-ca.sin(x_ego[2]) + ca.sin(x_other[2]))) ** 2
                ) - D - th,
            ),
            ca.SX.zeros(4)
        )
        _dist_inner_func = ca.Function('dist_inner_func', [x_ego, x_other], [_dist_inner])

        _dist_all_list = []
        _x_ego_all = ca.MX.sym('_x_ego_all', 4*cls._pred_len)
        _x_other_all = ca.MX.sym('_x_other_all', 4*cls._pred_len)
        for i in range(cls._pred_len):
            _dist_all_list.append(
                _dist_inner_func(_x_ego_all[i*4: (i+1)*4], _x_other_all[i*4: (i+1)*4])
            )

        _dist_all = ca.vertcat(*_dist_all_list)
        _dist_all_func = ca.Function('dist_all_func', [_x_ego_all, _x_other_all], [_dist_all])

        return _dist_all_func

    @classmethod
    def _gen_Q_comfort(cls) -> np.ndarray:
        Q_comfort = np.zeros((cls._pred_len-1, cls._pred_len*4))
        for i in range(cls._pred_len-1):
            Q_comfort[i, i * 4 + 2] = -cls._comfort[0]
            Q_comfort[i, i * 4 + 3] = -cls._comfort[1]
            Q_comfort[i, (i + 1) * 4 + 2] = cls._comfort[0]
            Q_comfort[i, (i + 1) * 4 + 3] = cls._comfort[1]
        return Q_comfort

    def __init__(self, init_state: np.ndarray, ref_traj: np.ndarray, mpc_id: int):
        assert DistributedMPCIPOPT._initialized
        assert KinematicModel.is_initialized()

        self._mpc_id = mpc_id
        _all_mpc[mpc_id] = self

        assert ref_traj.shape[1] == 4
        self._ref_traj = ref_traj

        assert init_state.shape == (4,)
        self._x_t: np.ndarray = init_state
        self._u_nominal: np.ndarray = np.zeros((self._pred_len + 0, 2))
        self._x_nominal = KinematicModel.roll_out(self._x_t, self._u_nominal)
        self._x_nominal_others: List[np.ndarray] = list()
        self._t: int = 0
        self._init_nominal()

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
            # if np.linalg.norm(self.position - mpc.position) <= self._sensing_distance:
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

    def _solve_ego_prob(self):
        # param: x0, x_ref
        x0 = np.concatenate((self._u_nominal.reshape(-1), self._x_nominal[1:].reshape(-1)))
        x_nominal_other_vec = np.concatenate([a_nominal[1:].reshape(-1) for a_nominal in self._x_nominal_others])
        p = np.concatenate((
            self._x_t,
            self._ref_traj[self._t+1: self._t+1+self._pred_len].reshape(-1),
            x_nominal_other_vec
        ))
        r_ipopt = self._nlp_solver(x0=x0, lbg=self._lbg, ubg=self._ubg, lbx=self._lbx, ubx=self._ubx, p=p)
        res = r_ipopt['x']
        self._u_nominal = np.array(res[: 2*self._pred_len]).reshape((self._pred_len, 2))
        self._x_nominal = KinematicModel.roll_out(self._x_t, self._u_nominal)

    def _init_nominal(self):
        # with suppress_stdout_stderr():
        #     self._solve_ego_prob()
        print("vehicle {} init complete.".format(self._mpc_id))

    def get_nominal(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._x_nominal, self._u_nominal

    def _step_forward_from_nominal(self):
        u = self._u_nominal[0]
        self._x_t = KinematicModel.step_once(self._x_t, u)
        self._u_nominal = np.vstack((self._u_nominal[1:], self._u_nominal[-1]))
        self._x_nominal = KinematicModel.roll_out(self._x_t, self._u_nominal)
        self._t += 1
        return u

    @classmethod
    def step_all(cls) -> Dict:
        step_info = dict()
        for mpc_id, mpc in _all_mpc.items():
            step_info[mpc_id] = dict()
        # collect old state
        for mpc_id, mpc in _all_mpc.items():
            step_info[mpc_id]["old_state"] = mpc._x_t
        # # # iter and optimize
        # for mpc_id, mpc in _all_mpc.items():
        #     step_info[mpc_id]["nominal"] = list()
        #     step_info[mpc_id]["nlp_res"] = list()
        # collect nominal
        for mpc_id, mpc in _all_mpc.items():
            mpc._update_x_nominal_others()
        with suppress_stdout_stderr():
            # optimize and get osqp info and collect nominal
            for mpc_id, mpc in _all_mpc.items():
                mpc._solve_ego_prob()
                step_info[mpc_id]["nlp_res"] = NLP_RESULT_INFO.get_info_from_result(cls._nlp_solver)
                step_info[mpc_id]["nominal"] = mpc.get_nominal()
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
            all_info.append(DistributedMPCIPOPT.step_all())
        return all_info


_all_mpc: Dict[int, DistributedMPCIPOPT] = {}

if __name__ == "__main__":
    pass
