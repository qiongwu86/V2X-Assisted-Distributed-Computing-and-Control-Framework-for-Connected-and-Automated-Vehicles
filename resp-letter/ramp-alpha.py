import matplotlib.pyplot as plt
import numpy as np


# plt.rcParams.update({'font.size': 10,
#                      "text.usetex": True})
# plt.rcParams['font.family'] = 'Times New Roman'
#

plt.rcParams.update({
    "font.family": "Times New Roman",   # 字体
    "mathtext.fontset": "stix",         # 数学字体风格
    "font.size": 8,                     # 字号
    "axes.labelsize": 9,                # 坐标轴标签字号
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.0,             # 线宽 1pt
    "lines.markersize": 4,              # 标记大小
})

# 数据
ds2 = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
dmin_a25 = np.array([5.0346408897819295, 5.119633695256756, 5.140903634319407, 5.149505367037195, 5.157407316198906])
dmin_a30 = np.array([5.479289425672136, 5.585737618279801, 5.62584234875519, 5.647769236661723, 5.662815657026249])
dmin_a35 = np.array([5.927640552495048, 6.0812013535475264, 6.144590218576108, 6.17849029722521, 6.198619284694945])

# 设置图像大小（IEEE Transactions 单栏宽度约 3.5 英寸）
fig, ax = plt.subplots(figsize=(3.4, 1.5))  # 高度适中即可

# 绘制曲线
ax.plot(ds2, dmin_a25, 'o-', label=r'$d_s^2=2.5$')
ax.plot(ds2, dmin_a30, 's-', label=r'$d_s^2=3.0$')
ax.plot(ds2, dmin_a35, '^-', label=r'$d_s^2=3.5$')

# 坐标轴与标签
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$d_{\min}$ (m)')

ax.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5], ['0.5', '', '1.0', '', '1.5', '', '2.0', '', '2.5'])
ax.set_ylim((4.5, 6.5))

ax.legend(
    loc='lower center',           # 底部中央
    bbox_to_anchor=(0.5, 0.02),   # 控制位置（越小越往下）
    ncol=3,                       # 三个图例横排
    frameon=True                 # 无边框
)
ax.grid(True, linestyle='--', linewidth=0.5)

# 紧凑布局 + 保存去白边
plt.tight_layout(pad=0.0)
# plt.savefig('dmin_vs_ds2.eps', bbox_inches='tight', pad_inches=0.05)
plt.subplots_adjust(left=0.13, right=0.99, bottom=0.25, top=0.95)
plt.savefig('dmin_vs_ds2.eps')
# plt.show()


###################################################

# 数据
ds2 = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
dmin_a25 = np.array([1.9481696041145002, 2.556603625422026, 2.8589463344814106, 3.04482607831869, 3.343486403614612])
dmin_a30 = np.array([2.202158056413714, 2.8801360752721865, 3.223069734753703, 3.4207497679350585, 3.5485974174770094])
dmin_a35 = np.array([2.471033126378977, 3.1906418278950293, 3.562942572514665, 3.780747943174387, 3.90937894297565])

# 设置图像大小（IEEE Transactions 单栏宽度约 3.5 英寸）
fig, ax = plt.subplots(figsize=(3.4, 1.5))  # 高度适中即可

# 绘制曲线
ax.plot(ds2, dmin_a25, 'o-', label=r'$d_s^2=2.5$')
ax.plot(ds2, dmin_a30, 's-', label=r'$d_s^2=3.0$')
ax.plot(ds2, dmin_a35, '^-', label=r'$d_s^2=3.5$')

# 坐标轴与标签
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$d_{\min}$ (m)')

ax.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5], ['0.5', '', '1.0', '', '1.5', '', '2.0', '', '2.5'])
ax.set_ylim((1.5, 4.0))

ax.legend(
    loc='lower center',           # 底部中央
    bbox_to_anchor=(0.5, 0.02),   # 控制位置（越小越往下）
    ncol=3,                       # 三个图例横排
    frameon=True                 # 无边框
)
ax.grid(True, linestyle='--', linewidth=0.5)

# 紧凑布局 + 保存去白边
plt.tight_layout(pad=0.0)
# plt.savefig('dmin_vs_ds2.eps', bbox_inches='tight', pad_inches=0.05)
plt.subplots_adjust(left=0.13, right=0.99, bottom=0.25, top=0.95)
plt.savefig('dmin_vs_ds2_inter.eps')
# plt.show()
