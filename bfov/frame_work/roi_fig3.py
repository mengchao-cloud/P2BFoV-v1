import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch, FancyArrowPatch

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

# Grid dimensions
H, W = 8, 16

# Color definitions
COLOR_INTERNAL = 'lightgreen'  # Internal region color
COLOR_BORDER = 'lightcoral'    # Border and mirror region color
COLOR_INTERP = 'gold'          # Bilinear interpolation 2x2 region highlight

# 1. Draw internal non-border region (green)
for y in range(1, H-1):
   for x in range(1, W-1):
       rect = Rectangle((x-0.5, y-0.5), 1, 1, facecolor=COLOR_INTERNAL, 
                       edgecolor='darkgreen', linewidth=1.5, alpha=0.8)
       ax.add_patch(rect)

# 2. Draw border cells (pale red) - outermost ring of valid region
# Top and bottom borders
for x in range(W):
   # Top border y=0
   rect = Rectangle((x-0.5, -0.5), 1, 1, facecolor=COLOR_BORDER, 
                   edgecolor='darkred', linewidth=1.5, alpha=0.6)
   ax.add_patch(rect)
   # Bottom border y=H-1
   rect = Rectangle((x-0.5, H-1.5), 1, 1, facecolor=COLOR_BORDER, 
                   edgecolor='darkred', linewidth=1.5, alpha=0.6)
   ax.add_patch(rect)

# Left and right borders (excluding corners to avoid overlap)
for y in range(1, H-1):
   # Left border x=0
   rect = Rectangle((-0.5, y-0.5), 1, 1, facecolor=COLOR_BORDER, 
                   edgecolor='darkred', linewidth=1.5, alpha=0.6)
   ax.add_patch(rect)
   # Right border x=W-1
   rect = Rectangle((W-1.5, y-0.5), 1, 1, facecolor=COLOR_BORDER, 
                   edgecolor='darkred', linewidth=1.5, alpha=0.6)
   ax.add_patch(rect)

# 3. Draw mirrored border region (same pale red color) with border coordinates
# Top mirror (y=-1), mirrors y=0
for x in range(W):
   rect = Rectangle((x-0.5, -1.5), 1, 1, facecolor=COLOR_BORDER, 
                   edgecolor='darkred', linewidth=1.5, linestyle='--', alpha=0.6)
   ax.add_patch(rect)
   # Display border coordinate (x, 0) instead of mirror coordinate (x, -1)
   ax.text(x, -1, f'({x},0)', ha='center', va='center', fontsize=8, 
           fontweight='bold', color='black')

# Bottom mirror (y=H), mirrors y=H-1
for x in range(W):
   rect = Rectangle((x-0.5, H-0.5), 1, 1, facecolor=COLOR_BORDER, 
                   edgecolor='darkred', linewidth=1.5, linestyle='--', alpha=0.6)
   ax.add_patch(rect)
   # Display border coordinate (x, H-1)
   ax.text(x, H, f'({x},{H-1})', ha='center', va='center', fontsize=8, 
           fontweight='bold', color='black')

# Left mirror (x=-1), mirrors x=0
for y in range(H):
   rect = Rectangle((-1.5, y-0.5), 1, 1, facecolor=COLOR_BORDER, 
                   edgecolor='darkred', linewidth=1.5, linestyle='--', alpha=0.6)
   ax.add_patch(rect)
   # Display border coordinate (0, y)
   ax.text(-1, y, f'(0,{y})', ha='center', va='center', fontsize=8, 
           fontweight='bold', color='black')

# Right mirror (x=W), mirrors x=W-1
for y in range(H):
   rect = Rectangle((W-0.5, y-0.5), 1, 1, facecolor=COLOR_BORDER, 
                   edgecolor='darkred', linewidth=1.5, linestyle='--', alpha=0.6)
   ax.add_patch(rect)
   # Display border coordinate (W-1, y)
   ax.text(W, y, f'({W-1},{y})', ha='center', va='center', fontsize=8, 
           fontweight='bold', color='black')

# 4. Add arrows from border cells to corresponding mirror centers
# Left border (x=0) to left mirror (x=-1)
for y in range(H):
   arrow = FancyArrowPatch((0, y), (-1, y), arrowstyle='->', 
                          mutation_scale=15, linewidth=1.5, color='navy', alpha=0.7)
   ax.add_patch(arrow)

# Right border (x=W-1) to right mirror (x=W)
for y in range(H):
   arrow = FancyArrowPatch((W-1, y), (W, y), arrowstyle='->', 
                          mutation_scale=15, linewidth=1.5, color='navy', alpha=0.7)
   ax.add_patch(arrow)

# Top border (y=0) to top mirror (y=-1)
for x in range(W):
   arrow = FancyArrowPatch((x, 0), (x, -1), arrowstyle='->', 
                          mutation_scale=15, linewidth=1.5, color='navy', alpha=0.7)
   ax.add_patch(arrow)

# Bottom border (y=H-1) to bottom mirror (y=H)
for x in range(W):
   arrow = FancyArrowPatch((x, H-1), (x, H), arrowstyle='->', 
                          mutation_scale=15, linewidth=1.5, color='navy', alpha=0.7)
   ax.add_patch(arrow)

# 5. Highlight 2x2 bilinear interpolation regions (uniform gold color)
queries = [
   (-0.3, 3.4, 'red'),    # Left of border, involves x=-1,0 and y=3,4
   (5.2, -0.4, 'blue'),   # Above border, involves x=5,6 and y=-1,0
   (15.1, 5.3, 'purple'), # Right of border, involves x=15,16 and y=5,6
   (7.5, 4.5, 'orange'),  # Internal, involves x=7,8 and y=4,5
]

for qx, qy, color in queries:
   # Calculate four corners for bilinear interpolation
   x0, x1 = int(np.floor(qx)), int(np.ceil(qx))
   y0, y1 = int(np.floor(qy)), int(np.ceil(qy))
   
   # Highlight 2x2 interpolation grid
   for cx in [x0, x1]:
       for cy in [y0, y1]:
           # Check if within display range (valid region + mirror border)
           if -1 <= cx <= W and -1 <= cy <= H:
               rect = Rectangle((cx-0.5, cy-0.5), 1, 1, facecolor=COLOR_INTERP, 
                               edgecolor='darkorange', linewidth=3, alpha=0.9, zorder=5)
               ax.add_patch(rect)
               
               # Display coordinates: if mirror region, show corresponding border coordinates
               display_cx, display_cy = cx, cy
               if cx == -1:
                   display_cx = 0
               elif cx == W:
                   display_cx = W-1
               if cy == -1:
                   display_cy = 0
               elif cy == H:
                   display_cy = H-1
               
               ax.text(cx, cy, f'({display_cx},{display_cy})', 
                      ha='center', va='center', fontsize=9, fontweight='bold', 
                      color='black', zorder=6)
   
   # Query point with circle marker (no coordinate label)
   ax.scatter(qx, qy, c=color, s=50, marker='o', edgecolors='black', 
             linewidths=2, zorder=10)

# Add legend in English
legend_elements = [
   Patch(facecolor=COLOR_INTERNAL, edgecolor='darkgreen', label='Internal Region'),
   Patch(facecolor=COLOR_BORDER, edgecolor='darkred', label='Border & Mirror Region'),
   Patch(facecolor=COLOR_INTERP, edgecolor='darkorange', label='Bilinear Interp 2×2 Region')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

# Set axis properties
ax.set_xlim(-2.5, W+1.5)
ax.set_ylim(-2.5, H+1.5)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig('./bfov/frame_work/roi/bilinear_interp_mirror.pdf', dpi=150, 
          bbox_inches='tight', facecolor='white')
plt.show()