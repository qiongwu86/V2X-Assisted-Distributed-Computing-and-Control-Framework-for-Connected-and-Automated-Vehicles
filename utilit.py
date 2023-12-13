import os
import numpy as np
from dynamic_model import OneDimDynamic
import pickle
from env import EnvParam
from matplotlib import pyplot as plt
from matplotlib import animation, rc, patches


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
            PickleSave(result, 'init_state/init_state.pkl')
            print("complete")
            return
    print("Faile")

        
class SceneDraw:

    L = 4.5
    W = 2.0

    L_ = L/2
    W_ = W/2

    pos_tran = lambda x, y, phi : (x+SceneDraw.W_ * np.sin(phi), y-SceneDraw.W_*np.cos(phi))


    def __init__(self, car_nums: int, traj_len:int, car_trajs: dict[int, dict['X': np.ndarray, 'U': np.ndarray, 'lane': str]]) -> None:
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
        ax.add_patch(bg)
        for id in self.car_objs:
            ax.add_patch(self.car_objs[id])

        def update(frame):
            for id in self.car_objs:
                x, y, phi, _ = self.car_trajs[id]['X'][frame]
                self.car_objs[id].set_xy(SceneDraw.pos_tran(x, y, phi))
                self.car_objs[id].set_angle(np.rad2deg(phi))
            plt.xlim(-50, 60)
            plt.ylim(-20, 10)
            ax.set_aspect('equal')
            ax.margins(0)

        anim = animation.FuncAnimation(fig, update, frames=self.traj_len, interval=100)
        writer = animation.FFMpegWriter(fps=10)
        anim.save('video/animation.mp4', writer=writer)
        plt.close()
        
    
class WatchOneVeh:

    L = 4.5
    W = 2.0

    L_ = L/2
    W_ = W/2

    pos_tran = lambda x, y, phi : (x+SceneDraw.W_ * np.sin(phi), y-SceneDraw.W_*np.cos(phi))
    def __init__(self, one_trace: np.ndarray, ref_traj: np.ndarray) -> None:
        self.ref_traj = ref_traj
        self.one_trace = one_trace
        self.traj_len = one_trace.shape[0]
        self.car = patches.Rectangle((0, 0), WatchOneVeh.L, WatchOneVeh.W, fc='None', ec='red')
        self.x_lim_len = 10
        self.y_lim_len = 10
        self.x_lim_fun = lambda Xt: (Xt[0] - 0.5*self.x_lim_len, Xt[0] + 0.5*self.x_lim_len)
        self.y_lim_fun = lambda Xt: (Xt[1] - 0.5*self.y_lim_len, Xt[1] + 0.5*self.y_lim_len)

    def DrawVideo(self):
        fig, ax = plt.subplots()
        circle = patches.Circle((0, 0), 20, ec = 'blue', fc='none')
        ref_car = patches.Rectangle((0, 0), WatchOneVeh.L, WatchOneVeh.W, fc='None', ec='blue')
        ax.add_patch(circle)
        ax.add_patch(self.car)
        ax.add_patch(ref_car)
        plt.plot(self.ref_traj[:self.traj_len, 0], self.ref_traj[: self.traj_len, 1], color='red')

        def update(frame):
            x, y, phi, v = self.one_trace[frame]
            self.car.set_xy(WatchOneVeh.pos_tran(x, y, phi))
            self.car.set_angle(np.rad2deg(phi))

            x, y, phi, v = self.ref_traj[frame]
            ref_car.set_xy(WatchOneVeh.pos_tran(x, y, phi))
            ref_car.set_angle(np.rad2deg(phi))

            # plt.xlim(self.x_lim_fun((x, y)))
            # plt.ylim(self.y_lim_fun((x, y)))
            plt.xlim(-30, 30)
            plt.ylim(-30, 30)

            ax.set_aspect('equal')
            ax.margins(0)
        
        anim = animation.FuncAnimation(fig, update, frames=self.traj_len, interval=100)
        writer = animation.FFMpegWriter(fps=10)
        anim.save('video/one_veh.mp4', writer=writer)
        plt.close()

        
    
    
