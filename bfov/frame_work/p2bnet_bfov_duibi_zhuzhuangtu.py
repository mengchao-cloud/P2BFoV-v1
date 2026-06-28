import matplotlib.pyplot as plt
import numpy as np
import os

save_dir = "./bfov/frame_work/"
os.makedirs(save_dir, exist_ok=True)

# 数据
labels = ['Low Latitude', 'Medium Latitude', 'High Latitude', 'All']
x = np.arange(len(labels))
width = 0.35

iou_p2bfov = [0.4818, 0.4788, 0.4048, 0.4861]
iou_p2bnet = [0.4027, 0.3148, 0.2419, 0.3814]

dev_p2bfov = [0.1281, 0.0939, 0.0726, 0.1210]
dev_p2bnet = [0.1663, 0.1421, 0.1194, 0.1611]

# 配色：原始配置
color_iou_bfov = '#1f77b4'   # 主蓝
color_iou_bnet = '#ff7f0e'   # 主橙

color_dev_bfov = '#212121'   # 深灰蓝
color_dev_bnet = '#795548'   # 深暖棕橙
# 绘图
fig, ax1 = plt.subplots(figsize=(8, 5))

# 左轴 IoU 柱状
ax1.set_ylabel('IoU', fontsize=11)
ax1.bar(x - width/2, iou_p2bfov, width, label='Ours IoU', color=color_iou_bfov)
ax1.bar(x + width/2, iou_p2bnet, width, label='P2BNet-bfov IoU', color=color_iou_bnet)
ax1.tick_params(axis='y')
ax1.set_ylim(0, 0.55)

# 右轴 偏离度 折线
ax2 = ax1.twinx()
ax2.set_ylabel('Deviation', fontsize=11)
ax2.plot(x, dev_p2bfov, marker='o', linewidth=2.5, markersize=7, color=color_dev_bfov, label='Ours Deviation')
ax2.plot(x, dev_p2bnet, marker='s', linewidth=2.5, markersize=7, color=color_dev_bnet, label='P2BNet-bfov Deviation')
ax2.tick_params(axis='y')
ax2.set_ylim(0, 0.2)

# X轴
ax1.set_xlabel('Latitude Level', fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)

# 图例缩小
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, handlelength=1.5, framealpha=0.9)

fig.tight_layout()

# 保存矢量PDF
out_path = os.path.join(save_dir, "IoU_Deviation_Latitude.pdf")
plt.savefig(out_path, format='pdf', bbox_inches='tight')
plt.close()

print("矢量图已保存至：", out_path)