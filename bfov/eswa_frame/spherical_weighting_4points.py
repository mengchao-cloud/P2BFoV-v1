import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

fig = plt.figure(figsize=(6, 5))

def draw_sphere(ax):
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    
    u = np.linspace(0, 2*np.pi, 200)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    
    ax.plot_surface(x, y, z, color="#cce5ff", alpha=1.0, edgecolor="none", 
                    rstride=1, cstride=1, antialiased=True)
    
    for lon in np.linspace(0, 2*np.pi, 12):
        v_vals = np.linspace(0, np.pi, 100)
        ax.plot(np.cos(lon)*np.sin(v_vals), np.sin(lon)*np.sin(v_vals), np.cos(v_vals), 
                linewidth=0.8, alpha=0.4, color='gray')
    
    for lat in np.linspace(-np.pi/2 + 0.1, np.pi/2 - 0.1, 7):
        r = np.cos(lat)
        u_vals = np.linspace(0, 2*np.pi, 200)
        ax.plot(r*np.cos(u_vals), r*np.sin(u_vals), np.sin(lat)*np.ones_like(u_vals), 
                linewidth=0.8, alpha=0.4, color='gray')

pts = np.array([
    [0.7, 0.2, 0.3],
    [-0.6, 0.4, 0.3],
    [0.2, -0.7, 0.3],
    [-0.3, -0.2, 0.8],
])
pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)

colors = ['red', 'green', 'blue', 'purple']

ax = fig.add_subplot(1, 1, 1, projection='3d')
draw_sphere(ax)
for p, c in zip(pts, colors):
    ax.scatter(*p, s=8, color=c, alpha=0.95, depthshade=True)
ax.view_init(elev=25, azim=-45)

os.makedirs('./bfov_shaken_output', exist_ok=True)
plt.savefig('./bfov_shaken_output/spherical_weighting_4points.png', dpi=300, bbox_inches='tight')
print("Saved to: ./bfov_shaken_output/spherical_weighting_4points.png")
