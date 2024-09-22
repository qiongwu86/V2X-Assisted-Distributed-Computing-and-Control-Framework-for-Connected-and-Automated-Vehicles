import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

WHAT = '12'
LANG = 'CN'
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus']=False


# # 定义参数
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# alpha_values = [0.5, 1, 2, 2, 5]
# beta_values = [0.5, 1, 2, 3, 2]
# x = np.linspace(0, 1, 100)
#
# # 画出不同参数下的贝塔分布的概率密度函数
# # fig, ax = plt.subplots()
# for a, b in zip(alpha_values, beta_values):
#     y = beta.pdf(x, a, b)
#     axs[0].plot(x, y, label=r'a={},b={}'.format(a, b))
# axs[0].set_title(r'$\mathrm{\beta}$分布')
# axs[0].set_xlabel('x')
# axs[0].set_ylabel(r'$\mathrm{P_{\beta}(x;a,b)}$')
# axs[0].legend()
#
# #
# def softplus(x):
#     return np.log(1 + np.exp(x))+1
#
# x = np.linspace(-10, 10, 100)
# y = softplus(x)
#
# axs[1].plot(x, y)
# # plt.title('')
# axs[1].set_title('函数$\mathrm{F(\cdot)}$')
# axs[1].set_xlabel('x')
# axs[1].set_ylabel('$\mathrm{F(x)}$')
# axs[1].grid(True)
# axs[1].set_ylim(-0.1, 5.0)
# axs[1].set_xlim(-10, 10.0)
# # axs[1].show()
#
# plt.show()]

def clip_function(x, a):
    return np.clip(x, 1-a, 1+a)

x = np.linspace(-10, 10, 1000)
a = 0.2
y = clip_function(x, a)

plt.plot(x, y)
plt.axhline(y=1-a, color='r', linestyle='--', label='1-a')
plt.axhline(y=1+a, color='g', linestyle='--', label='1+a')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.xlim(0.7, 1.3)
plt.ylim(0.7, 1.3)
plt.title('Clip Function: f(x) = clip(x, 1-a, 1+a)')
plt.grid(True)
plt.show()

