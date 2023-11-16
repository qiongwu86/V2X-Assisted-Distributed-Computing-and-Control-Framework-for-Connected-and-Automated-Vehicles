import numpy as np
import dynamic_model
#from dynamic_model import OneDimDynamic
import osqp
from scipy import sparse

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

