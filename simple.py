import numpy as np
from numpy import ndarray
import scipy
from typing import Dict, Tuple


class Simple:
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
    _F: np.ndarray = None

    @classmethod
    def initialize(cls, config: Dict):
        # A and B
        cls._Td = config['T_d']
        cls._Ta = config['T_a']
        cls._T = config['T']
        Ac = np.array([[0, 1, 0], [0, 0, 1], [0, 0, -1 / cls._Td]])
        Bc = np.array([[0], [0], [1 / cls._Ta]])
        Cc = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        Dc = np.array([[0], [0], [0]])
        cls._A, cls._B, _, _, _ = scipy.signal.cont2discrete([Ac, Bc, Cc, Dc], cls._Td, "zoh")

        # M
        M = np.zeros((cls._T * 3, 3))
        M[: 3] = cls._A
        for t in range(1, cls._T):
            M[t * 3: (t + 1) * 3] = M[(t - 1) * 3: t * 3] @ cls._A
        # N
        N = np.kron(np.eye(cls._T), cls._B)
        temp = cls._B
        for t in range(1, cls._T):
            temp = cls._A @ temp
            for _t in range(t, cls._T):
                N[_t * 3: (_t + 1) * 3, _t - t: _t - t + 1] = temp
        cls._M = M
        cls._N = N

        # F
        cls._F = np.kron(np.eye(cls._T), np.array([[1, 0, 0]]))
