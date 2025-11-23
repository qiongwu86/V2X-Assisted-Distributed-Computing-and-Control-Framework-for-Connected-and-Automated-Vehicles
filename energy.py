
# our
our_all = [21.585649429783423, 21.104027050581436, 26.026911433979606, 26.362941285323185, 34.96130363327162, 30.10813634085336, 33.452247522129575, 36.689340155231136]
our_10 = [1.6039334311634827, 2.013197642483083, 3.1278479374897854, 4.891842222807182, 1.2610695218353971, 4.766973941256116, 3.016100492620864, 0.8511747430809051, 6.511206024473663, 8.645994198020663]

# seq
seq_all = [25.740902076429876, 45.73356559640243, 59.38581356255912, 57.65369720355534, 68.4676963561604, 74.00645424357647, 92.46717177314122, 136.82329976398685]
seq_10 = [13.478671775427998, 7.008939898877902, 12.525826564295683, 12.979997279826382, 15.92621587273953, 15.382849467938792, 23.339525242626, 17.423708300454496, 12.468937265983904, 6.288628095816157]

# seq-nolim
seqnolim_all = [3.711050128740581, 40.57305066159922, 50.980456243809975, 63.30819952909786, 69.95581097164845, 96.44765023262104, 127.73274563129122, 221.5893579152213]
seqnolim_10 = [29.827116110684194, 28.335631366185442, 25.57541221736517, 29.67588345717733, 25.800806133259833, 17.619362863184094, 27.670404512286193, 18.326988763155974, 12.469124675211773, 6.288627816711293]

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

fig, ax = plt.subplots(figsize=(3.4, 2.2))


veh_num = np.array([3, 4, 5, 6, 7, 8, 9, 10])
# 绘制曲线
ax.plot(veh_num, our_all, 'o-', label=r'Alg. 2')
ax.plot(veh_num, seq_all, 's-', label=r'SeqA')
ax.plot(veh_num, seqnolim_all, '^-', label=r'SeqB')

# 坐标轴与标签
ax.set_xlabel(r'Number of CAVs $N$')
ax.set_ylabel(r'Object Value')

ax.set_xticks([3, 4, 5, 6, 7, 8, 9, 10])
# ax.set_ylim((4.5, 6.5))

ax.legend(
    # loc='lower center',           # 底部中央
    # bbox_to_anchor=(0.5, 0.02),   # 控制位置（越小越往下）
    # ncol=3,                       # 三个图例横排
    # frameon=True                 # 无边框
)
ax.grid(True, linestyle='--', linewidth=0.5)

# 紧凑布局 + 保存去白边
plt.tight_layout(pad=0.0)
# plt.savefig('dmin_vs_ds2.eps', bbox_inches='tight', pad_inches=0.05)
plt.subplots_adjust(left=0.13, right=0.99, bottom=0.2, top=0.95)
plt.savefig('output_dir/traj_plan/obj_all.eps')
plt.close()

###################################################

fig, ax = plt.subplots(figsize=(3.4, 2.2))


veh_num = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# 绘制曲线
ax.plot(veh_num, np.array(our_10) / our_all[-1], 'o-', label=r'Our')
ax.plot(veh_num, np.array(seq_10) / seq_all[-1], 's-', label=r'Seq')
ax.plot(veh_num, np.array(seqnolim_10) / seqnolim_all[-1], '^-', label=r'SeqNL')

# 坐标轴与标签
ax.set_xlabel(r'id of CAV $i$')
ax.set_ylabel(r'Object Value')

ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# ax.set_ylim((4.5, 6.5))

ax.legend(
    # loc='lower center',           # 底部中央
    # bbox_to_anchor=(0.5, 0.02),   # 控制位置（越小越往下）
    # ncol=3,                       # 三个图例横排
    # frameon=True                 # 无边框
)
ax.grid(True, linestyle='--', linewidth=0.5)

# 紧凑布局 + 保存去白边
plt.tight_layout(pad=0.0)
# plt.savefig('dmin_vs_ds2.eps', bbox_inches='tight', pad_inches=0.05)
plt.subplots_adjust(left=0.13, right=0.99, bottom=0.2, top=0.95)
plt.savefig('output_dir/traj_plan/obj_10.eps')

