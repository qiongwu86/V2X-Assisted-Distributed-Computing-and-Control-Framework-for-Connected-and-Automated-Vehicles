import numpy as np
from dynamic_model import OneDimDynamic


class EnvParam:

    L1: float = 50
    L2: float = 60
    L3: float = 60

    Dsafe = 5

    InitPosRandRange: np.ndarray = np.array([-5, 5])

    InitVelocity: float = 10.0
    InitVelocityRandRange: np.ndarray = np.array([-3, 3])
    InitDensity: float = 0.08
    InitDensityRandRange: np.ndarray = np.array([-0.05, 0.05])

    FinalVelocity: float = 10.0
    FinalVehDensity: float = 0.16
    
    GenR = lambda a, b: np.random.random() * (b-a) + a

    def __init__(self) -> None:
        pass

    @staticmethod
    def generate_init_state(t0: float = 0.0) -> dict:
        """generate initial state, final state and others

        Args:
            t0 (float, optional): initial time. Defaults to 0.0.

        Returns:
            tuple: dict := {
                id(int) : state(list),
                id(int) : state(list),
                ...
            }
            state := ('main/merge', t0, tf, to, init-state, final-state, iop, iof)
        """
        tf = t0 + EnvParam.L2 / EnvParam.InitVelocity + EnvParam.L3 / EnvParam.FinalVelocity

        state_init = list()
        # main lane
        start_point = -(EnvParam.L2 + EnvParam.GenR(*EnvParam.InitPosRandRange))
        while (start_point >= -(EnvParam.L1+EnvParam.L2)):
            state_init.append(
                ("main", (start_point, EnvParam.GenR(*EnvParam.InitVelocityRandRange) + EnvParam.InitVelocity, 0, 0))
            )
            start_point += -1/(EnvParam.InitDensity + EnvParam.GenR(*EnvParam.InitDensityRandRange))
        # merge lane
        start_point = -(EnvParam.L2 + EnvParam.GenR(*EnvParam.InitPosRandRange))
        while (start_point >= -(EnvParam.L1+EnvParam.L2)):
            state_init.append(
                ("merge", (start_point, EnvParam.GenR(*EnvParam.InitVelocityRandRange) + EnvParam.InitVelocity, 0, 0))
            )
            start_point += -1/(EnvParam.InitDensity + EnvParam.GenR(*EnvParam.InitDensityRandRange))

        # calculate order
        state_init = sorted(state_init, key=lambda state: -state[1][0])
        ret_dict = {}
        for i in range(len(state_init)):
            id = i + 1
            lane = state_init[i][0]
            state = state_init[i][1]
            over_s = EnvParam.L3 - i * (1/EnvParam.FinalVehDensity)
            over_v = EnvParam.FinalVelocity
            to = tf - over_s / EnvParam.FinalVelocity
            iop = None
            iof = (id-1) if (id > 1) else None
            ret_dict[id] = [lane, t0, tf, to, state, (over_s, over_v, 0.0, 0.0), iop, iof]
        
        # calculate iop
        for id in ret_dict:
            ego_lane = ret_dict[id][0]
            if id == 1:
                continue
            for id_p in range(id-1, 0, -1):
                lane_p = ret_dict[id_p][0]
                if ego_lane == lane_p:
                    ret_dict[id][6] = id_p
                    break

        # calculate oiop, oiof
        for id in ret_dict:
            ret_dict[id].append(None)
            ret_dict[id].append(None)
            for o_id in ret_dict:
                oiop, oiof = ret_dict[o_id][6: 8]
                oto = ret_dict[o_id][3]
                if oiop == id:
                    assert ret_dict[id][8] == None
                    ret_dict[id][8] = (o_id, oto)
                if oiof == id:
                    assert ret_dict[id][9] == None
                    ret_dict[id][9] = (o_id, oto)

        return ret_dict

            
            
                


        
        

