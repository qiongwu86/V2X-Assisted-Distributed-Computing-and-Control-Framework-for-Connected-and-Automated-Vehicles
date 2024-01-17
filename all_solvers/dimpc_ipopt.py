import numpy as np
from dynamic_models import KinematicModel
from utilits import suppress_stdout_stderr
from typing import Dict, List, Tuple, Callable
import casadi as ca
import tqdm


class DistributedMPCIPOPT:

    _initialized: bool = False

    default_config = dict(
        Qx=0.1 * np.diag((1.0, 1.0, 1.5, 0)),
        Qu=0.1 * np.diag([1.0, 0.1]),
        comfort=1.0,
        safe_factor=10.0,
        safe_th=5.0,
        init_iter=5,
        run_iter=3,
        sensing_distance=25.0,
        priority=False,
        pred_len=30,
        other_veh_num=3
    )

    _init_iter: int = 5
    _run_iter: int = 10
    _comfort: float = 1.0
    _safe_factor: float = 10.0,
    _safe_th: float = 4.0
    _sensing_distance: float = 25.0,
    _priority: bool = False
    _pred_len: int = 30
    _other_veh_num: int = 3

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

        cls._Qx_big = np.kron(np.eye(cls._pred_len), config["Qx"])
        cls._Qx_big[-4:, -4:] = 10 * config["Qx"]
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
        x_nominal_other = ca.vertcat(*[
            ca.vertcat(*[ca.MX.sym('x_no_' + str(o) + str(i + 1), 4) for i in range(cls._pred_len)])
            for o in range(cls._pred_len)
        ])

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

        J = (x_1_T - x_ref).T @ Qx @ (x_1_T - x_ref) + u_0_T_1.T @ Qu @ u_0_T_1 + (Qc @ x_1_T).T @ (Qc @ x_1_T)

        nlp = {
            'x': ca.vertcat(u_0_T_1, x_1_T),
            'f': J,
            'g': g,
            'p': ca.vertcat(x_0, x_ref)
        }

        prob_option = dict(
            # verbose=False,
            # verbose_init=False,
            # print_in=False,
            # print_out=False,
            # print_time=False,
            record_time=True,
        )

        return ca.nlpsol('S_ipopt', 'ipopt', nlp, prob_option), lbx, ubx, lbg, ubg

    @classmethod
    def _gen_Q_comfort(cls) -> np.ndarray:
        Q_comfort = np.zeros((cls._pred_len-1, cls._pred_len*4))
        for i in range(cls._pred_len-1):
            Q_comfort[i, i*4+2] = -cls._comfort
            Q_comfort[i, (i+1)*4+2] = cls._comfort
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

    def _solve_ego_prob(self):
        # param: x0, x_ref
        p = np.concatenate((self._x_t, self._ref_traj[self._t+1: self._t+1+self._pred_len].reshape(-1)))
        r_ipopt = self._nlp_solver(lbg=self._lbg, ubg=self._ubg, lbx=self._lbx, ubx=self._ubx, p=p)
        res = r_ipopt['x']
        self._u_nominal = np.array(res[: 2*self._pred_len]).reshape((self._pred_len, 2))
        self._x_nominal = KinematicModel.roll_out(self._x_t, self._u_nominal)

    def _init_nominal(self):
        with suppress_stdout_stderr():
            self._solve_ego_prob()
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

    @staticmethod
    def step_all() -> Dict:
        step_info = dict()
        for mpc_id, mpc in _all_mpc.items():
            step_info[mpc_id] = dict()
        # collect old state
        for mpc_id, mpc in _all_mpc.items():
            step_info[mpc_id]["old_state"] = mpc._x_t
        # # iter and optimize
        for mpc_id, mpc in _all_mpc.items():
            step_info[mpc_id]["nominal"] = list()
        #     step_info[mpc_id]["osqp_res"] = list()
        with suppress_stdout_stderr():
            for mpc_id, mpc in _all_mpc.items():
                mpc._update_x_nominal_others()
            # optimize and get osqp info and collect nominal
            for mpc_id, mpc in _all_mpc.items():
                mpc._solve_ego_prob()
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
            all_info.append(DistributedMPCIPOPT.step_all())
        return all_info


_all_mpc: Dict[int, DistributedMPCIPOPT] = {}

if __name__ == "__main__":
    pass
