import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, ax = plt.subplots(figsize=(10, 10))

# 参数设置
grid_size = 14
pool_size = 2
output_size = 7
cell_size = 1.0

# 颜色设置 - 区分上下两层
bottom_color = '#90EE90'   # 浅绿色 - 底层14x14（较浅）
pool_color = '#4682B4'     # 钢蓝色 - 上层7x7池化（较深，半透明）
border_color = '#1E3A8A'   # 深蓝色 - 池化块边框
bg_color = '#F8FAFC'       # 浅灰白 - 特写背景

# 1. 绘制底层 14x14 网格（细线，较浅，底层）
for i in range(grid_size + 1):
   ax.plot([i*cell_size, i*cell_size], [0, grid_size*cell_size], 
           color=bottom_color, linewidth=0.8, alpha=0.7, zorder=1)
   ax.plot([0, grid_size*cell_size], [i*cell_size, i*cell_size], 
           color=bottom_color, linewidth=0.8, alpha=0.7, zorder=1)

# 2. 绘制上层 7x7 池化块（半透明覆盖，粗边框，上层）
for i in range(output_size):
   for j in range(output_size):
       x = j * pool_size * cell_size
       y = i * pool_size * cell_size
       
       # 池化块 - 半透明蓝色覆盖，深色粗边框
       rect = patches.Rectangle(
           (x, y), pool_size*cell_size, pool_size*cell_size,
           linewidth=3.0,
           edgecolor=border_color,
           facecolor=pool_color,
           alpha=0.4,  # 半透明，能看到底层绿色网格
           zorder=2    # 上层
       )
       ax.add_patch(rect)

# 3. 左下角特写：展示 2x2→1 的池化过程（带白色背景框）
inset_x, inset_y = 0.6, 0.6  # 左下角位置
inset_size = 3.0
gap = 0.15  # 四个小格子之间的间隙，显示分离感

# 绘制白色圆角背景框
bg_rect = patches.FancyBboxPatch(
   (inset_x - 0.3, inset_y - 0.5), inset_size + 0.6, inset_size + 0.9,
   boxstyle="round,pad=0.05",
   facecolor='white',
   edgecolor='#64748B',
   linewidth=2,
   alpha=0.95,
   zorder=10
)
ax.add_patch(bg_rect)

# 绘制底层 4 个小格子（分离排列，显示"四个独立格子"）
small_cell = (inset_size - 3*gap) / 2
positions = [(0, 0), (1, 0), (0, 1), (1, 1)]  # 左下、右下、左上、右上

for px, py in positions:
   x_pos = inset_x + gap + px * (small_cell + gap)
   y_pos = inset_y + gap + py * (small_cell + gap)
   
   # 小格子 - 浅绿色底层（稍微阴影效果）
   # 先画阴影
   shadow = patches.Rectangle(
       (x_pos + 0.05, y_pos - 0.05), small_cell, small_cell,
       facecolor='gray',
       alpha=0.2,
       zorder=11
   )
   ax.add_patch(shadow)
   
   # 再画格子
   small_rect = patches.Rectangle(
       (x_pos, y_pos), small_cell, small_cell,
       facecolor=bottom_color,
       edgecolor='darkgreen',
       linewidth=1.5,
       alpha=0.9,
       zorder=12
   )
   ax.add_patch(small_rect)

# 绘制上层池化后的格子（半透明蓝色覆盖，显示"合并成一个"）
pool_rect = patches.Rectangle(
   (inset_x, inset_y + 0.1), inset_size, inset_size - 0.2,  # 稍微调整垂直位置
   facecolor=pool_color,
   edgecolor=border_color,
   linewidth=3.5,
   alpha=0.6,  # 比主图的池化块稍不透明，突出显示
   zorder=13   # 最上层
)
ax.add_patch(pool_rect)

# 添加文字说明
ax.text(inset_x + inset_size/2, inset_y - 0.3, '2×2 Pooling', 
       fontsize=12, ha='center', va='top', fontweight='bold', color='#1E3A8A')
ax.text(inset_x + inset_size/2, inset_y + inset_size + 0.2, '4 grids → 1 cell', 
       fontsize=10, ha='center', va='bottom', color='#059669', fontweight='bold')

# 图形设置
ax.set_xlim(-0.5, grid_size*cell_size + 1.0)
ax.set_ylim(-0.5, grid_size*cell_size + 1.0)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig('./bfov/frame_work/roi/grid_pooling_illustration.pdf', 
           format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('./bfov/frame_work/roi/grid_pooling_illustration.svg', 
           format='svg', bbox_inches='tight')
plt.show()