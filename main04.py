from dynamic_model import BicycleModel
import numpy as np
from utilit import WatchOneVeh

x0 = np.zeros((BicycleModel.SDIM,))
x0[3] = 10
T_nums = 100
u_bar = np.array([5 * np.sin(2*np.pi*np.arange(0, T_nums)/100),
                  np.deg2rad(15) * np.cos(2*np.pi*np.arange(0, T_nums)/100)])
u_bar = u_bar.transpose()

x_bar = BicycleModel.roll(x0, u_bar, T_nums)

drawer = WatchOneVeh(x_bar)
drawer.DrawVideo()


