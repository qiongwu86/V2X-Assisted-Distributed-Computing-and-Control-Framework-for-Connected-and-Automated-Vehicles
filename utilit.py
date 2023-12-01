import os
import numpy as np
from dynamic_model import OneDimDynamic
import pickle
from env import EnvParam


def ProcessTrace(state_dict: dict):
    """construct data for optimization

    Args:
        state_dict (dict): {id: [lane, t0, tf, to, x0, xf, iop, iof, oiop, oiof, trace]}
                                [   0,  1,  2,  3,  4,  5,   6,   7,    8,    9,    10]
    Returns:
        dict: 
    """
    fun_int_to = lambda tf, t0, to :int(int((tf - t0)/OneDimDynamic.Td) * to / (tf-t0))
    ret_dict = {}
    for id in state_dict:
        lane, t0, tf, to, x0, xf, iop, iof, oiop, oiof, trace = state_dict[id]
        int_to = fun_int_to(tf, t0, to)
        state_dict[id][3] = int_to

    for id in state_dict:
        if state_dict[id][8] is not None:
            oiop = state_dict[id][8][0]
            state_dict[id][8] = (oiop, state_dict[oiop][3])
        if state_dict[id][9] is not None:
            oiof = state_dict[id][9][0]
            state_dict[id][9] = (oiof, state_dict[oiof][3])


def PickleSave(obj, name: str) -> None:
    with open(name, 'wb') as f:
        pickle.dump(obj, f)


def PickleRead(nama: str):
    with open(nama, 'rb') as f:
        obj = pickle.load(f)
    return obj

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
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

def GenerateSpecificTrace(main_num: int, merge_num: int, count: int = 10):
    while count:
        count -= 1
        result = EnvParam.generate_init_state()
        main_ = 0
        merge_ = 0
        for id in result:
            if result[id][0] == 'main':
                main_ += 1
            elif result[id][0] == 'merge':
                merge_ += 1
            else:
                raise ValueError
        if main_num == main_ and merge_ == merge_num:
            PickleSave(result, 'init_state.pkl')
            print("complete")
            return
    print("Faile")

        
    
