import numpy as np
from dynamic_model import OneDimDynamic


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


        
