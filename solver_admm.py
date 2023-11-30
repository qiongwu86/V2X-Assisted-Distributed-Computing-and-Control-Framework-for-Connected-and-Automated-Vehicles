import numpy as np
import dynamic_model
#from dynamic_model import OneDimDynamic
import osqp
from scipy import sparse
import scipy.io
from env import EnvParam

SDIM = dynamic_model.OneDimDynamic.SDIM
CDIM = dynamic_model.OneDimDynamic.CDIM

def InitSolver(T_num: int, veh_num: int, u_min: float = -3.0, u_max: float = 3.0):
    # init matrixs
    SolverAdmm.T_num = T_num
    SolverAdmm.veh_num = veh_num
    ###########################################
    ## dynamic constrain: B @ (x0) = L @ (S) ##
    ###########################################
    # B
    B = np.zeros((SolverAdmm.T_num*SDIM, SDIM))
    for i in range(SolverAdmm.T_num):
        B[i*SDIM: (i+1)*SDIM, :] = np.linalg.matrix_power(dynamic_model.OneDimDynamic.Ad, i+1)
    SolverAdmm.B = B
    # L
    T1 = np.eye(T_num)
    T2 = np.concatenate((np.eye(SDIM), -dynamic_model.OneDimDynamic.Bd), axis=1)
    T3 = np.kron(T1, T2)
    G = np.zeros((T_num*SDIM, T_num*(SDIM+CDIM)))
    for i in range(1, T_num):
        T4 = -dynamic_model.OneDimDynamic.Bd
        for j in range(i, 0, -1):
            T4 = dynamic_model.OneDimDynamic.Ad @ T4
            G[i*SDIM: (i+1)*SDIM, (j-1)*(SDIM+CDIM)+SDIM: j*(SDIM+CDIM)] = T4
    SolverAdmm.L = G + T3
    ###########################################
    # state constrain: U_min <= K(S) <= U_max #
    ###########################################
    SolverAdmm.U_min = u_min * np.ones((T_num,))
    SolverAdmm.U_max = u_max * np.ones((T_num,))
    T5 = np.zeros((CDIM, CDIM+SDIM))
    T5[:, SDIM:] = np.eye(CDIM)
    # T5 = np.array([[0, 0, 0, 1]])
    SolverAdmm.K = np.kron(np.eye(T_num), T5)
    ###########################################
    ##### safe constrain: \sim M (S) >= 0 #####
    ############ M needs R ####################
    ###########################################
    SolverAdmm.R = np.kron(np.eye(T_num), np.array([[1, 0, 0, 0]]))
    ##########################################
    ########## (S-S_r)^T Q (S-S_r) ###########
    ##########################################
    Q_ = np.diag([1.0, 1.0, 0.1, 0.1])
    SolverAdmm.Q = np.kron(np.eye(T_num), Q_)

    

class SolverAdmm:
    # dynamic constrain: b1 

    def __init__(self, id: int, x_0: np.ndarray, ref_trace: np.ndarray, rho: float=0.1, sigma: float=0.1) -> None:
        self.ID: int = id
        self.rho = rho
        self.sigma = sigma
        self.ref_trace  = ref_trace
        # vars
        self.p: np.ndarray = np.zeros((SolverAdmm.T_num * (SolverAdmm.veh_num-1),))
        self.y: np.ndarray = np.zeros((SolverAdmm.T_num * (SolverAdmm.veh_num-1),))
        self.z: np.ndarray = np.zeros((SolverAdmm.T_num * (SolverAdmm.veh_num-1),))
        self.s: np.ndarray = np.zeros((SolverAdmm.T_num * (SolverAdmm.veh_num-1),))
        self.x: np.ndarray = None
        self.r: np.ndarray = None
        # step-7
        self.M = self._calculateM()
        self.k = self.sigma + 2 * self.rho * (SolverAdmm.veh_num-1)
        self.P7 = SolverAdmm.Q +  (0.5 / self.k) * np.transpose(self.M) @ self.M 
        self.q7 = None
        self.A7 = np.concatenate((SolverAdmm.L, SolverAdmm.K), axis=0)
        self.Bx0 = SolverAdmm.B @ x_0[:3]
        self.l7 = np.append(self.Bx0, SolverAdmm.U_min)
        self.u7 = np.append(self.Bx0, SolverAdmm.U_max)

        self.prob = osqp.OSQP()
        self.prob_setup = False
        
    def step(self, others_y: np.ndarray) -> np.ndarray:
        N_1yi = (SolverAdmm.veh_num-1) * self.y
        Ayj = others_y.sum(0)
        # for p_k+1
        self.p = self.p + self.rho * (N_1yi - Ayj)
        # for s_k+1
        self.s = self.s + self.sigma * (self.y - self.z)
        # for r_k+1
        self.r = self.rho * (N_1yi + Ayj) + self.sigma * self.z - self.p - self.s
        # for x_k+1
        self.step_for_x()
        # for y_k+1
        self.y = (1 / self.k) * (self.M @ self.x + self.r)
        # for z_k+1
        self.step_for_z()
        return self.y

    def _calculateM(self):
        M = np.zeros(((SolverAdmm.veh_num-1 + 2)*SolverAdmm.T_num, SolverAdmm.T_num*(SDIM+CDIM)))
        M[(self.ID-1)*SolverAdmm.T_num: self.ID*SolverAdmm.T_num, :] = -SolverAdmm.R
        M[self.ID*SolverAdmm.T_num: (self.ID+1)*SolverAdmm.T_num, :] = SolverAdmm.R
        M = M[SolverAdmm.T_num: -SolverAdmm.T_num, :]
        return M
        
    def step_for_x(self):
        if not self.prob_setup:
            P = sparse.csc_matrix(self.P7)
            q = (0.5 / self.k) * (self.r @ self.M) - self.ref_trace @ SolverAdmm.Q
            A = sparse.csc_matrix(self.A7)
            l = self.l7
            u = self.u7
            self.prob.setup(P, q, A, l, u)
            self.prob_setup = True
        else:
            q = (0.5 / self.k) * (np.transpose(self.r) @ self.M) - self.ref_trace @ SolverAdmm.Q
            self.prob.update(q=q)
        res = self.prob.solve()
        self.x = np.array(res.x)

    def step_for_z(self):
        temp = SolverAdmm.veh_num * (self.s + self.sigma * self.y)
        temp = np.clip(temp, 0, np.inf)
        self.z = (1/ self.sigma) * self.s + self.y - (1/(SolverAdmm.veh_num*self.sigma)) * temp

    def get_result(self):
        return self.x


class WeightedADMM:
    
    make_A_D: bool = False
    MatrixA: np.ndarray = None
    MatrixD: np.ndarray = None
    MatrixAD_dict: dict = None

    all_solver: dict = {}
    
    def makeAD(all_data: dict) -> None:
        if len(all_data) != 10:
            raise NotImplementedError

        if len(all_data) == 10:
            ADmat = scipy.io.loadmat('ADsave/veh_5_5.mat')
            WeightedADMM.MatrixA = ADmat['A']
            WeightedADMM.MatrixD = ADmat['D'].diagonal()
        main_set = [0, 1, 2, 3, 4]
        merge_set = [5, 6, 7, 8, 9]
        for id in all_data:
            lane = all_data[id][0]
            if lane == 'main':
                WeightedADMM.MatrixAD_dict[id] = main_set.pop(0)
            if lane == 'merge':
                WeightedADMM.MatrixAD_dict[id] = merge_set.pop(0)
        return
    
    def __init__(self,id:int, veh_num: int, T_nums: int, one_data: list) -> None:
        if not WeightedADMM.make_A_D:
            print("Uninitialize A and D")
            raise ValueError

        WeightedADMM.all_solver[id] = self
        lane, t0, tf, to, x0, xf, iop, iof, oiop, oiof, trace = one_data
        self.id = id
        assert T_nums == trace.shape[0] / (SDIM+CDIM)
        # variable
        self.x: np.ndarray = np.zeros((T_nums * (SDIM+CDIM),))
        self.y_self: np.ndarray = np.zeros((veh_num * T_nums,))
        self.y_all: np.ndarray = np.zeros(veh_num, (veh_num*T_nums))
        self.p: np.ndarray = np.zeros((veh_num * T_nums,))
        self.v: np.ndarray = None
        # hyper param
        self.a_j: np.ndarray = WeightedADMM.MatrixA[WeightedADMM.MatrixA_dict[self.id]]
        self.d_ii: float = WeightedADMM.MatrixD[WeightedADMM.MatrixD_dict[self.id]]
        # matrix
        self.ref_trace: np.ndarray = trace[CDIM+SDIM:]
        self.b_i: np.ndarray = EnvParam.Dsafe * np.ones((veh_num * T_nums,)) / veh_num
        self.A_i: np.ndarray = self.GenerateA_i(veh_num, T_nums, one_data)

    def GenerateA_i(self, veh_num: int, T_nums: int, one_data: list) -> np.ndarray:
        lane, t0, tf, to, x0, xf, iop, iof, oiop, oiof, trace = one_data
        A_i = np.ndarray((T_nums * veh_num, T_nums * (CDIM + SDIM)))
        temp = np.kron(np.eye(T_nums), np.diag((-1, 0, 0, 0)))
        A_i[(self.id-1)*T_nums: self.id * T_nums] = -temp
        A_i[(oiop[0]-1) * T_nums          : (oiop[0]-1) * T_nums + oiop[1]] = temp[       : oiop[1]]
        A_i[(oiof[0]-1) * T_nums + oiof[1]:               oiof[0] * T_nums] = temp[oiof[1]:        ]
        return A_i
    # @staticmethod
    # def generate_y_all(veh_num: int, y_shape: tuple) -> np.ndarray:
        # result = np.zeros(veh_num)
        # for i in range(veh_num):
            # id = i+1
            # result[id] = np.zeros(y_shape)
        # return result

    def _UpdateY(self, all_solver: list) -> None:
        for id in all_solver:
            self.y_all[id-1] = all_solver[id].y_self
        
    def Solve(self) -> None:
        # for v
        self.v = self.d_ii * self.y_self + self.a_j @ self.y_all - self.b_i - self.p
        # qp for x, t
        self.x, _ = self.SolveX()
        # for y
        self.y = np.clip(self.A_i @ self.x + self.v, 0, np.inf) / (2*self.d_ii)
        # communicate
        self._UpdateY(WeightedADMM.all_solver)
        # for p
        self.p = self.p + self.d_ii * self.y_self - self.a_j @ self.y_all
        
    def SolveX(self) -> tuple:
        
        pass

