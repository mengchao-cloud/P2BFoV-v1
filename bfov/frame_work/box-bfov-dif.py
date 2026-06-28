import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as mpatches

# Set font for international support
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# Create output directory if not exists
output_dir = './bfov/frame_work/box-bfov-dif/'
os.makedirs(output_dir, exist_ok=True)

# Data preparation
latitudes = ['Low Lat.', 'Mid Lat.', 'High Lat.', 'Overall']
x = np.arange(len(latitudes))
width = 0.35

# Experimental data
box_feat = {
   'miou': [0.4348, 0.4023, 0.31, 0.4361],
   'distance': [0.1379, 0.1032, 0.1235, 0.1139],
   'score': [0.3551, 0.3111, 0.3627, 0.3433]
}
tangent_feat = {
   'miou': [0.4418, 0.4788, 0.3748, 0.4461],
   'distance': [0.1281, 0.0939, 0.0726, 0.1210],
   'score': [0.5644, 0.5402, 0.4267, 0.5572]
}

# Color settings
color_box = '#3498db'
color_tangent = '#e74c3c'
color_bg = '#f8f9fa'

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('white')

# Unified settings
for ax in axes:
   ax.set_facecolor(color_bg)
   ax.grid(axis='y', alpha=0.3, linestyle='--')

# Subplot 1: mIoU
ax1 = axes[0]
bars1 = ax1.bar(x - width/2, box_feat['miou'], width, color=color_box, alpha=0.85, edgecolor='black', linewidth=0.5)
bars2 = ax1.bar(x + width/2, tangent_feat['miou'], width, color=color_tangent, alpha=0.85, edgecolor='black', linewidth=0.5)
ax1.set_ylabel('mIoU', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(latitudes, fontsize=11)
ax1.set_ylim(0, 0.55)

# Add value labels
for bars in [bars1, bars2]:
   for bar in bars:
       height = bar.get_height()
       ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}', 
               ha='center', va='bottom', fontsize=9, fontweight='bold')

# Subplot 2: Spherical Distance Offset
ax2 = axes[1]
bars3 = ax2.bar(x - width/2, box_feat['distance'], width, color=color_box, alpha=0.85, edgecolor='black', linewidth=0.5)
bars4 = ax2.bar(x + width/2, tangent_feat['distance'], width, color=color_tangent, alpha=0.85, edgecolor='black', linewidth=0.5)
ax2.set_ylabel('Spherical Distance Offset', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(latitudes, fontsize=11)
ax2.set_ylim(0, 0.15)

# Add value labels
for i, (b3, b4) in enumerate(zip(bars3, bars4)):
   h3, h4 = b3.get_height(), b4.get_height()
   ax2.text(b3.get_x() + b3.get_width()/2., h3 + 0.003, f'{h3:.3f}', 
           ha='center', va='bottom', fontsize=9, fontweight='bold')
   ax2.text(b4.get_x() + b4.get_width()/2., h4 + 0.003, f'{h4:.3f}', 
           ha='center', va='bottom', fontsize=9, fontweight='bold')
   # Annotate improvement at high latitude
   if i == 2:
       ax2.annotate('↓41%', xy=(b4.get_x() + b4.get_width()/2., h4), 
                   xytext=(0, 20), textcoords='offset points', ha='center', 
                   fontsize=9, color='green', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='green'))

# Subplot 3: Score
ax3 = axes[2]
bars5 = ax3.bar(x - width/2, box_feat['score'], width, color=color_box, alpha=0.85, edgecolor='black', linewidth=0.5)
bars6 = ax3.bar(x + width/2, tangent_feat['score'], width, color=color_tangent, alpha=0.85, edgecolor='black', linewidth=0.5)
ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(latitudes, fontsize=11)
ax3.set_ylim(0, 0.65)

# Add value labels and improvement percentages
for i, (b5, b6) in enumerate(zip(bars5, bars6)):
   h5, h6 = b5.get_height(), b6.get_height()
   ax3.text(b5.get_x() + b5.get_width()/2., h5 + 0.015, f'{h5:.3f}', 
           ha='center', va='bottom', fontsize=9, fontweight='bold')
   ax3.text(b6.get_x() + b6.get_width()/2., h6 + 0.015, f'{h6:.3f}', 
           ha='center', va='bottom', fontsize=9, fontweight='bold')
   # Annotate improvement for low, mid, and overall
   if i in [0, 1, 3]:
       improvement = ((h6 - h5) / h5) * 100
       ax3.annotate(f'+{improvement:.0f}%', 
                   xy=(b6.get_x() + b6.get_width()/2., h6), 
                   xytext=(0, 15), textcoords='offset points', 
                   ha='center', fontsize=9, color='darkred', fontweight='bold')

# Create single legend for the entire figure
legend_elements = [mpatches.Patch(facecolor=color_box, alpha=0.85, edgecolor='black', label='Box Feature'),
                  mpatches.Patch(facecolor=color_tangent, alpha=0.85, edgecolor='black', label='Tangent Plane')]
fig.legend(handles=legend_elements, loc='upper center', ncol=2, 
         bbox_to_anchor=(0.5, 1.02), fontsize=11, framealpha=0.9)

plt.tight_layout(rect=[0, 0, 1, 0.98])

# Save files
plt.savefig(f'{output_dir}/feature_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(f'{output_dir}/feature_comparison.pdf', bbox_inches='tight', facecolor='white')
plt.savefig(f'{output_dir}/feature_comparison.svg', bbox_inches='tight', facecolor='white')
print(f"Plots saved to:")
print(f" - {output_dir}/feature_comparison.png")
print(f" - {output_dir}/feature_comparison.pdf")
print(f" - {output_dir}/feature_comparison.svg")
plt.show()