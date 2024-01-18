import os
from typing import List, Dict, Tuple
import numpy as np
import pickle


class OSQP_RESULT_INFO:
    RUN_TIME = 0
    SOLVE_TIME = 1
    STATUS = 2
    ITER_TIMES = 3

    _RUN_TIME = "run_time"
    _SOLVE_TIME = "solve_time"
    _STATUS = "status"
    _ITER_TIMES = "iter_times"

    @staticmethod
    def get_info_from_result(res) -> Tuple:
        return tuple((res.info.run_time, res.info.solve_time, res.info.status, res.info.iter))

    @staticmethod
    def extract_info_from_info_all(info_all: List) -> Dict[int, Dict[str, np.ndarray]]:
        veh_ids = info_all[0].keys()
        run_times = len(info_all[0][0]["osqp_res"])
        ret = {veh_id: {
            OSQP_RESULT_INFO._RUN_TIME: np.zeros((len(info_all), run_times)),
            OSQP_RESULT_INFO._ITER_TIMES: np.zeros((len(info_all), run_times)),
            OSQP_RESULT_INFO._SOLVE_TIME: np.zeros((len(info_all), run_times))
        } for veh_id in veh_ids}
        for i, one_time_info in enumerate(info_all):
            for veh_id in veh_ids:
                ret[veh_id][OSQP_RESULT_INFO._RUN_TIME][i, :] = \
                    np.array([res[OSQP_RESULT_INFO.RUN_TIME] for res in one_time_info[veh_id]["osqp_res"]])
                ret[veh_id][OSQP_RESULT_INFO._ITER_TIMES][i, :] = \
                    np.array([res[OSQP_RESULT_INFO.ITER_TIMES] for res in one_time_info[veh_id]["osqp_res"]])
                ret[veh_id][OSQP_RESULT_INFO._SOLVE_TIME][i, :] = \
                    np.array([res[OSQP_RESULT_INFO.SOLVE_TIME] for res in one_time_info[veh_id]["osqp_res"]])
        return ret


class NLP_RESULT_INFO:
    RUN_TIME = 0
    ITER_TIMES = 1

    _RUN_TIME = "run_time"
    _ITER_TIMES = "iter_times"

    @staticmethod
    def get_info_from_result(nlp_obj) -> Tuple:
        return tuple((nlp_obj.stats()["t_wall_total"], nlp_obj.stats()["iter_count"]))

    @staticmethod
    def extract_info_from_info_all(info_all: List) -> Dict[int, Dict[str, np.ndarray]]:
        veh_ids = info_all[0].keys()
        ret = {veh_id: {
            NLP_RESULT_INFO._RUN_TIME: np.zeros((len(info_all), 1)),
            NLP_RESULT_INFO._ITER_TIMES: np.zeros((len(info_all), 1)),
        } for veh_id in veh_ids}
        for i, one_time_info in enumerate(info_all):
            for veh_id in veh_ids:
                ret[veh_id][NLP_RESULT_INFO._RUN_TIME][i, :] = \
                    one_time_info[veh_id]["nlp_res"][NLP_RESULT_INFO.RUN_TIME]
                ret[veh_id][NLP_RESULT_INFO._ITER_TIMES][i, :] = \
                    one_time_info[veh_id]["nlp_res"][NLP_RESULT_INFO.ITER_TIMES]
        return ret


class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


def PickleSave(thing, name: str) -> None:
    with open(name, 'wb') as f:
        pickle.dump(thing, f)


def PickleRead(name: str):
    with open(name, 'rb') as f:
        obj = pickle.load(f)
    return obj


def moving_average_filter(data, window_size):
    """
    移动平均滤波算法
    :param data: 待滤波的数据
    :param window_size: 窗口大小
    :return: 滤波后的数据
    """
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='same')
