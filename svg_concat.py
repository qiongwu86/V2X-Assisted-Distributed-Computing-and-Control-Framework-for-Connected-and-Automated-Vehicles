import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.imread('svg_temp/ref_traj_12.svg')
# 读取 SVG 图片
image_paths = ['svg_temp/ref_traj_12.svg', 'svg_temp/ref_traj_T.svg']
images = [plt.imread(path) for path in image_paths]

# 创建 2x2 的子图布局
fig, axs = plt.subplots(1, 2)

# 在每个子图中显示图片
for i in range(2):
    axs[0, i].imshow(images[i])
    axs[0, i].axis('off')  # 关闭坐标轴

plt.show()
