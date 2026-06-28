import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import os

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"

fig = plt.figure(figsize=(7, 10))

def draw_sphere(ax, title="", result=False):
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()

    u = np.linspace(0, 2*np.pi, 80)
    v = np.linspace(0, np.pi, 40)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(x, y, z, color="white", alpha=0.18, edgecolor="none")

    for t in np.linspace(0, 2*np.pi, 12):
        ax.plot(np.cos(t)*np.sin(v), np.sin(t)*np.sin(v), np.cos(v), 
                linewidth=0.5, alpha=0.35)
    for p in np.linspace(-0.75, 0.75, 7):
        r = np.sqrt(1-p**2)
        ax.plot(r*np.cos(u), r*np.sin(u), np.ones_like(u)*p, 
                linewidth=0.5, alpha=0.35)

    ax.set_title(title, fontsize=14, pad=8)

    if not result:
        pts = [
            (-0.55,  0.55,  0.35, "red"),
            ( 0.55,  0.45,  0.30, "orange"),
            (-0.55, -0.35, -0.10, "green"),
            ( 0.55, -0.25, -0.05, "royalblue"),
            ( 0.05, -0.65, -0.35, "purple"),
        ]

        for x0, y0, z0, c in pts:
            p = np.array([x0, y0, z0])
            p = p / np.linalg.norm(p)

            ax.scatter(*p, s=45, color=c, depthshade=False)
            ax.plot([0, p[0]], [0, p[1]], [0, p[2]], 
                    linestyle="--", linewidth=1.2, color=c, alpha=0.8)

            ax.scatter(p[0]*1.05, p[1]*1.05, p[2]*1.05, 
                       s=380, facecolors=c, edgecolors=c, alpha=0.12)

        ax.scatter(0, 0, 0, s=35, color="black", depthshade=False)
    else:
        q = np.array([0.18, -0.05, 0.45])
        q = q / np.linalg.norm(q)
        ax.scatter(*q, s=55, color="purple", depthshade=False)
        ax.scatter(q[0]*1.08, q[1]*1.08, q[2]*1.08, 
                   s=1500, facecolors="purple", edgecolors="purple", alpha=0.14)
        ax.text(q[0], q[1], q[2]+0.25, r"$(\bar{\theta},\bar{\phi})$", 
                fontsize=13, color="purple")

    ax.view_init(elev=18, azim=-65)

ax1 = fig.add_axes([0.18, 0.66, 0.64, 0.28], projection="3d")
draw_sphere(ax1, "Top-K BFoV Proposals on Sphere")

ax2 = fig.add_axes([0.12, 0.42, 0.25, 0.20], projection="3d")
draw_sphere(ax2, "Log Map")

ax3 = fig.add_axes([0.40, 0.43, 0.22, 0.17])
ax3.set_axis_off()
plane = plt.Polygon([[0.1,0.2],[0.9,0.2],[0.75,0.8],[0.25,0.8]], 
                    closed=True, facecolor="#eef2ff", edgecolor="#6070a0", linewidth=1.2)
ax3.add_patch(plane)

colors = ["red", "orange", "green", "royalblue", "purple"]
points = np.array([[0.28,0.68],[0.70,0.66],[0.25,0.35],[0.73,0.38],[0.48,0.28]])
center = np.array([0.50,0.50])

for p, c in zip(points, colors):
    ax3.plot([p[0], center[0]], [p[1], center[1]], linestyle="--", color=c, alpha=0.7)
    ax3.scatter(p[0], p[1], s=50, color=c)

ax3.scatter(center[0], center[1], s=60, color="black")
ax3.text(0.5, 0.05, "Weighted average\non tangent plane", ha="center", fontsize=12)

ax4 = fig.add_axes([0.66, 0.42, 0.25, 0.20], projection="3d")
draw_sphere(ax4, "Exp Map", result=True)

overlay = fig.add_axes([0, 0, 1, 1])
overlay.set_axis_off()

def arrow(x1, y1, x2, y2):
    overlay.add_patch(
        FancyArrowPatch((x1, y1), (x2, y2), 
                        arrowstyle="simple", 
                        mutation_scale=18, 
                        linewidth=0, 
                        color="#3b6fb6", 
                        alpha=0.9)
    )

arrow(0.48, 0.65, 0.48, 0.61)
arrow(0.35, 0.52, 0.40, 0.52)
arrow(0.62, 0.52, 0.67, 0.52)
arrow(0.50, 0.42, 0.50, 0.36)

ax5 = fig.add_axes([0.22, 0.12, 0.56, 0.26], projection="3d")
draw_sphere(ax5, "Weighted BFoV Result", result=True)

overlay.add_patch(
    FancyArrowPatch((0.30, 0.43), (0.70, 0.43), 
                    connectionstyle="arc3,rad=0.35", 
                    arrowstyle="->", 
                    mutation_scale=16, 
                    linewidth=1.5, 
                    linestyle="--", 
                    color="#3b6fb6")
)
overlay.text(0.50, 0.385, "iterate until convergence", ha="center", fontsize=12)

fig.text(0.5, 0.975, "Spherical Weighting for BFoV Proposals", 
         ha="center", va="top", fontsize=18, fontweight="bold")

fig.text(0.5, 0.055, 
         r"Center: Karcher Mean on $S^2$     |     FoV size: "
         r"$\bar{\alpha}=\sum_i w_i\alpha_i,\ \bar{\beta}=\sum_i w_i\beta_i$", 
         ha="center", fontsize=13)

os.makedirs('./bfov_shaken_output', exist_ok=True)
plt.savefig("./bfov_shaken_output/spherical_weighting_module.png", dpi=400, bbox_inches="tight")
plt.savefig("./bfov_shaken_output/spherical_weighting_module.pdf", bbox_inches="tight")
print("Saved to: ./bfov_shaken_output/spherical_weighting_module.png")
