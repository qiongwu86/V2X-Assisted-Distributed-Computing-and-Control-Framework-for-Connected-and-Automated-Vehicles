import numpy as np
import scipy
from typing import Dict, Tuple

from numpy import ndarray


class LongitudeModel:

    default_config = dict(
        T_d=0.1,
        T_a=0.1
    )

    _Td: float = None
    _Ta: float = None

    _A: np.ndarray = None
    _B: np.ndarray = None

    _T: int = None
    _M: np.ndarray = None
    _N: np.ndarray = None

    @classmethod
    def M(cls):
        return cls._M

    @classmethod
    def N(cls):
        return cls._N

    @classmethod
    def initialize(cls, config: Dict):
        cls._Td = config['T_d']
        cls._Ta = config['T_a']
        Ac = np.array([[0, 1, 0], [0, 0, 1], [0, 0, -1 / cls._Td]])
        Bc = np.array([[0], [0], [1 / cls._Ta]])
        Cc = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        Dc = np.array([[0], [0], [0]])
        cls._A, cls._B, _, _, _ = scipy.signal.cont2discrete([Ac, Bc, Cc, Dc], cls._Td, "zoh")

    @classmethod
    def gen_M_N(cls, T: int) -> tuple[ndarray, ndarray]:
        assert cls._A is not None and cls._B is not None
        if cls._T is not None:
            if cls._T == T:
                return tuple((cls._M, cls._N))
        # ####################
        M = np.zeros((T * 3, 3))
        M[: 3] = cls._A
        for t in range(1, T):
            M[t * 3: (t+1) * 3] = M[(t-1) * 3: t * 3] @ cls._A
        ######################
        N = np.kron(np.eye(T), cls._B)
        temp = cls._B
        for t in range(1, T):
            temp = cls._A @ temp
            for _t in range(t, T):
                N[_t*3: (_t+1)*3, _t - t: _t - t + 1] = temp
        if cls._T is None:
            cls._T = T
            cls._M = M
            cls._N = N
        return tuple((M, N))
