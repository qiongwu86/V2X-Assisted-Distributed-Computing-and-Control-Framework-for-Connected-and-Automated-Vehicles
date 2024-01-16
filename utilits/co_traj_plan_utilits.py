import numpy as np
from dynamic_models.dynamic_model import OneDimDynamic
import pickle
from env import EnvParam
from matplotlib import pyplot as plt
from matplotlib import animation, patches

import numpy as np
import math
from numpy.linalg import inv
from dynamic_models.dynamic_model import OneDimDynamic, BicycleModel

SDIM = OneDimDynamic.SDIM
CDIM = OneDimDynamic.CDIM

def ProcessTrace(state_dict: dict):
    """construct data for optimization

    Args:
        state_dict (dict): {id: [lane, t0, tf, to, x0, xf, iop, iof, oiop, oiof, trace]}
                                [   0,  1,  2,  3,  4,  5,   6,   7,    8,    9,    10]
    Returns:
        dict:
    """
    fun_int_to = lambda tf, t0, to: int(int((tf - t0) / OneDimDynamic.Td) * to / (tf - t0))
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
            PickleSave(result, 'init_state/init_state.pkl')
            print("complete")
            return
    print("Faile")


class SceneDraw:
    L = 4.5
    W = 2.0

    L_ = L / 2
    W_ = W / 2

    pos_tran = lambda x, y, phi: (x + SceneDraw.W_ * np.sin(phi), y - SceneDraw.W_ * np.cos(phi))

    def __init__(self, car_nums: int, traj_len: int,
                 car_trajs: dict[int, dict['X': np.ndarray, 'U': np.ndarray, 'lane': str]]) -> None:
        assert car_nums == len(car_trajs)
        self.car_nums = car_nums
        self.car_trajs = car_trajs
        self.traj_len = traj_len
        for id in self.car_trajs:
            traj = car_trajs[id]
            assert self.traj_len == traj['X'].shape[0]

        self.car_objs = {id: patches.Rectangle((0, 0), SceneDraw.L, SceneDraw.W, fc='None') for id in car_trajs}

        def init_cars():
            for id in self.car_trajs:
                x, y, phi = self.car_trajs[id]['X'][0][0: 3]
                self.car_objs[id].set_xy(SceneDraw.pos_tran(x, y, phi))
                self.car_objs[id].set_angle(np.rad2deg(phi))
                if self.car_trajs[id]['lane'] == 'main':
                    self.car_objs[id].set_edgecolor('red')
                else:
                    self.car_objs[id].set_edgecolor('green')
            return

        init_cars()

    def GenVideo(self):
        fig, ax = plt.subplots()
        bg = patches.Rectangle((-50, -20), 220, 100, fc='black')
        plt.plot([-200, 200], [2, 2], color='black')
        plt.plot([-200, -60], [-2, -2], color='black')
        plt.plot([-200, 400], [-6, -6], color='black')
        plt.scatter([40], [-4], color='yellow')
        # ax.add_patch(bg)
        for id in self.car_objs:
            ax.add_patch(self.car_objs[id])

        def update(frame):
            for id in self.car_objs:
                x, y, phi, _ = self.car_trajs[id]['X'][frame]
                self.car_objs[id].set_xy(SceneDraw.pos_tran(x, y, phi))
                self.car_objs[id].set_angle(np.rad2deg(phi))
            plt.xlim(-150, 50)
            plt.ylim(-10, 10)
            ax.set_aspect('equal')
            ax.margins(0)

        anim = animation.FuncAnimation(fig, update, frames=self.traj_len, interval=100)
        writer = animation.FFMpegWriter(fps=10)
        anim.save('video/animation.mp4', writer=writer)
        plt.close()

        fig = plt.figure()
        for id in self.car_trajs:
            trace = self.car_trajs[id]['X']
            phis = trace[:, 2]
            # v = trace[:, 3]
            plt.plot(phis)
        fig.savefig('figs/phis.jpg')

        fig = plt.figure()
        for id in self.car_trajs:
            trace = self.car_trajs[id]['X']
            v = trace[:, 3]
            plt.plot(v)
        fig.savefig('figs/v.jpg')


class WatchOneVeh:
    L = 4
    W = 1.8

    L_ = L / 2
    W_ = W / 2

    pos_tran = lambda x, y, phi: (x + SceneDraw.W_ * np.sin(phi), y - SceneDraw.W_ * np.cos(phi))

    def __init__(self, one_trace: np.ndarray, ref_traj: np.ndarray) -> None:
        self.ref_traj = ref_traj
        self.one_trace = one_trace
        self.traj_len = one_trace.shape[0]
        self.car = patches.Rectangle((0, 0), WatchOneVeh.L, WatchOneVeh.W, fc='None', ec='red')
        self.x_lim_len = 10
        self.y_lim_len = 10
        self.x_lim_fun = lambda Xt: (Xt[0] - 0.5 * self.x_lim_len, Xt[0] + 0.5 * self.x_lim_len)
        self.y_lim_fun = lambda Xt: (Xt[1] - 0.5 * self.y_lim_len, Xt[1] + 0.5 * self.y_lim_len)

    def DrawVideo(self):
        fig, ax = plt.subplots()
        # circle = patches.Circle((0, 0), 20, ec = 'blue', fc='none')
        ref_car = patches.Rectangle((0, 0), WatchOneVeh.L, WatchOneVeh.W, fc='None', ec='blue')
        # ax.add_patch(circle)
        ax.add_patch(self.car)
        # ax.add_patch(ref_car)
        plt.plot(self.ref_traj[:self.traj_len, 0], self.ref_traj[: self.traj_len, 1], color='red')

        def update(frame):
            x, y, phi, v = self.one_trace[frame]
            self.car.set_xy(WatchOneVeh.pos_tran(x, y, phi))
            self.car.set_angle(np.rad2deg(phi))

            # x, y, phi, v = self.ref_traj[frame]
            # ref_car.set_xy(WatchOneVeh.pos_tran(x, y, phi))
            # ref_car.set_angle(np.rad2deg(phi))

            # plt.xlim(self.x_lim_fun((x, y)))
            # plt.ylim(self.y_lim_fun((x, y)))
            # plt.xlim(0, 150)
            # plt.ylim(0, 30)
            plt.xlim(self.x_lim_fun((x, y)))
            plt.ylim(-2, 5)

            ax.set_aspect('equal')
            ax.margins(0)

        anim = animation.FuncAnimation(fig, update, frames=self.traj_len, interval=100)
        writer = animation.FFMpegWriter(fps=10)
        anim.save('video/one_veh.mp4', writer=writer)
        plt.close()




def generate_test_trace(T_nums: int, init_state: np.ndarray, final_state: np.ndarray) -> np.ndarray:
    ref_traj = np.zeros((T_nums, SDIM + CDIM))
    temp = (final_state[0] - init_state[0]) / T_nums
    for i in range(T_nums):
        ref_traj[i, 0] = init_state[0] + (i + 1) * temp
    # assert init_state[1] == final_state[1]
    ref_traj[:, 1] = final_state[1]
    ref_traj = ref_traj.reshape((-1))
    print(ref_traj)
    return ref_traj


def AddTrace(all_state_dict: dict, reshape: bool = True, k: float = 1) -> None:
    for id in all_state_dict:
        lane, t0, tf, to, x0, xf, iop, iof, oiop, oiof = all_state_dict[id]
        T_nums = int((tf - t0) / OneDimDynamic.Td) + 1
        c_i = edge_condition(k, OneDimDynamic.Ta, t0, tf, x0, xf)
        # T_nums include init point
        trace = np.zeros((T_nums, SDIM + CDIM))
        c1, c2, c3, c4, c5, c6 = c_i
        s_t = lambda t: c1 * t ** 3 / (12 * k) + c2 * t ** 2 / (
                    4 * k) + c5 * t + c6 - c3 * OneDimDynamic.Ta ** 3 * math.exp(t / OneDimDynamic.Ta) / (
                                    4 * k) + c4 * OneDimDynamic.Ta ** 2 * math.exp(-t / OneDimDynamic.Ta)
        v_t = lambda t: c1 * t ** 2 / (4 * k) + c2 * t / (2 * k) + c5 - c3 * OneDimDynamic.Ta ** 2 * math.exp(
            t / OneDimDynamic.Ta) / (4 * k) - c4 * OneDimDynamic.Ta * math.exp(-t / OneDimDynamic.Ta)
        for i in range(T_nums):
            t = t0 + i * OneDimDynamic.Td
            trace[i][0] = s_t(t)
            trace[i][1] = v_t(t)
        if reshape:
            all_state_dict[id].append(trace.reshape((-1)))
        else:
            all_state_dict[id].append(trace)


def edge_condition(k: float, T_a: float, t0: float, tf: float, x0: np.ndarray, xf: np.ndarray) -> np.ndarray:
    result = np.zeros((SDIM * 2, SDIM * 2))
    ### init state
    # line 0, s0
    result[0][0] = t0 ** 3 / (12 * k)
    result[0][1] = t0 ** 2 / (4 * k)
    result[0][2] = - (T_a ** 3 * math.exp(t0 / T_a)) / (4 * k)
    result[0][3] = T_a ** 2 * math.exp(-t0 / T_a)
    result[0][4] = t0
    result[0][5] = 1
    # line 1, v0
    result[1][0] = t0 ** 2 / (4 * k)
    result[1][1] = t0 ** 1 / (2 * k)
    result[1][2] = - (T_a ** 2 * math.exp(t0 / T_a)) / (4 * k)
    result[1][3] = - T_a ** 1 * math.exp(-t0 / T_a)
    result[1][4] = 1
    result[1][5] = 0
    # line 2, a0
    result[2][0] = t0 / (2 * k)
    result[2][1] = 1 / (2 * k)
    result[2][2] = - (T_a * math.exp(t0 / T_a)) / (4 * k)
    result[2][3] = math.exp(-t0 / T_a)
    result[2][4] = 0
    result[2][5] = 0
    ## final state
    # line 3, sf
    result[3][0] = tf ** 3 / (12 * k)
    result[3][1] = tf ** 2 / (4 * k)
    result[3][2] = - (T_a ** 3 * math.exp(tf / T_a)) / (4 * k)
    result[3][3] = T_a ** 2 * math.exp(-tf / T_a)
    result[3][4] = tf
    result[3][5] = 1
    # line 4, vf
    result[4][0] = tf ** 2 / (4 * k)
    result[4][1] = tf ** 1 / (2 * k)
    result[4][2] = - (T_a ** 2 * math.exp(tf / T_a)) / (4 * k)
    result[4][3] = - T_a ** 1 * math.exp(-tf / T_a)
    result[4][4] = 1
    result[4][5] = 0
    # line 5, af
    result[5][0] = tf / (2 * k)
    result[5][1] = 1 / (2 * k)
    result[5][2] = - (T_a * math.exp(tf / T_a)) / (4 * k)
    result[5][3] = math.exp(-tf / T_a)
    result[5][4] = 0
    result[5][5] = 0
    ## inv
    inv_A = inv(result)
    b = np.concatenate((x0[:3], xf[:3]))
    ## c1-c6
    c = inv_A @ b
    return c


def TraceTestForMPC(T_nums: int, map: str = 'rectangle'):
    #
    ref_trace = np.zeros((T_nums, 4))
    L = 30
    W = 10
    total_lengeh = 2 * (L + W)
    u_bar = np.array([3 * np.sin(2 * np.pi * np.arange(0, T_nums) / 100), np.zeros((T_nums,))])
    u_bar = u_bar.transpose()
    x_bar = BicycleModel.roll(np.array([0, 0, 0, 8]), u_bar, T_nums)[1:]
    ref_trace[:, 0] = x_bar[:, 0]
    for i in range(int(T_nums / 50)):
        ref_trace[i * 50: (i + 1) * 50, 1] = 2 * (i % 2)
    ref_trace[:, 2] = 0
    ref_trace[:, 3] = 8

    return ref_trace


def TraceTestForMPC2(T_nums: int):
    radius = 20
    velocity = 10
    step_dist = 10 * 0.1
    step_angle = step_dist / radius
    generate_one_point = lambda angle: (radius * np.cos(angle), radius * np.sin(angle), angle + 0.5 * np.pi, velocity)
    all_state = [generate_one_point(i * step_angle + 0.5 * np.pi) for i in range(T_nums)]
    return np.array(all_state)


def TraceTestForMPC3():
    radius = 20
    L = 10
    H = 2
    angle = np.arctanh(H / L)
    velocity = 10
    l = 0
    trace = np.zeros((200, 4))
    trace[:, 0] = np.linspace(0, 200, 200)
    trace[50: 60, 1] = np.tan(angle) * (trace[50: 60, 0] - trace[50, 0])
    trace[50: 60, 2] = angle
    trace[:, 3] = velocity
    return trace


def TraceTestForMPC3():
    radius = 20
    L = 10
    H = 2
    angle = np.arctanh(H / L)
    velocity = 10
    l = 0
    trace = np.zeros((200, 4))
    trace[:, 0] = np.linspace(0, 200, 200)
    trace[50: 80, 1] = 2
    trace[80: 100, 1] = 1
    trace[:, 3] = velocity
    return trace


def TraceTestForMPC4() -> dict[int, np.ndarray]:
    result = {}
    # veh 1
    result[1] = np.zeros((200, 4))
    result[1][:, 0] = np.arange(0, 200)
    result[1][:, 1] = 0.1
    result[1][:, 2] = 0
    result[1][:, 3] = 10
    # veh 2
    result[2] = np.zeros((200, 4))
    result[2][:, 0] = np.arange(200, 0, -1)
    result[2][:, 1] = -0.1
    result[2][:, 2] = np.pi
    result[2][:, 3] = 10
    return result


