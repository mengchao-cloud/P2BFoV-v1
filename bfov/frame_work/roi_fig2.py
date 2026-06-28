import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_tangent_plane_geometry():
   """创建切平面几何数据"""
   lat = np.radians(40)
   lon = np.radians(55)
   theta0 = np.pi/2 - lat
   phi0 = lon
   
   C = np.array([
       np.sin(theta0) * np.cos(phi0),
       np.sin(theta0) * np.sin(phi0),
       np.cos(theta0)
   ])
   
   e_u = np.array([-np.sin(phi0), np.cos(phi0), 0])
   e_u = e_u / np.linalg.norm(e_u)
   
   e_v = np.array([
       -np.cos(theta0)*np.cos(phi0),
       -np.cos(theta0)*np.sin(phi0),
       np.sin(theta0)
   ])
   e_v = e_v / np.linalg.norm(e_v)
   
   n = C.copy()
   W, H = 0.85, 0.65
   grid_size = 14
   
   u_grid = np.linspace(-W, W, grid_size)
   v_grid = np.linspace(-H, H, grid_size)
   
   plane_points = []
   for ui in u_grid:
       for vi in v_grid:
           p = C + ui * e_u + vi * e_v
           plane_points.append(p)
   
   plane_points = np.array(plane_points)
   sphere_points = plane_points / np.linalg.norm(plane_points, axis=1, keepdims=True)
   
   return C, e_u, e_v, n, W, H, grid_size, plane_points, sphere_points, u_grid, v_grid


def main():
   C, e_u, e_v, n, W, H, grid_size, plane_points, sphere_points, u_grid, v_grid = create_tangent_plane_geometry()
   
   # 单图布局，正方形画布
   fig, ax = plt.subplots(figsize=(8, 8))
   
   margin = 0.1
   ax.set_xlim(-W-margin, W+margin)
   ax.set_ylim(-H-margin, H+margin)
   ax.set_aspect('equal')
   ax.set_facecolor('#f8f8f8')
   
   u_vals = np.linspace(-W, W, grid_size)
   v_vals = np.linspace(-H, H, grid_size)
   
   # 绘制矩形边界
   rect = plt.Rectangle((-W, -H), 2*W, 2*H, fill=False, 
                        edgecolor='darkgreen', linewidth=2.5, linestyle='-')
   ax.add_patch(rect)
   
   # 绘制网格线
   for u in u_vals:
       ax.plot([u, u], [-H, H], 'g-', linewidth=1.5, alpha=0.7)
   for v in v_vals:
       ax.plot([-W, W], [v, v], 'g-', linewidth=1.5, alpha=0.7)
   
   # 绘制网格点
   uu, vv = np.meshgrid(u_vals, v_vals)
   ax.scatter(uu.flatten(), vv.flatten(), c='darkgreen', s=60, zorder=5, 
              edgecolors='white', linewidths=0.5)
   
   # 坐标轴箭头
   arrow_len_u = W + 0.05
   arrow_len_v = H + 0.05
   ax.annotate('', xy=(arrow_len_u, 0), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2.5))
   ax.annotate('', xy=(0, arrow_len_v), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2.5))
   
   # 坐标轴标签
   ax.text(arrow_len_u+0.02, 0, '$e_u$', fontsize=14, color='darkgreen', fontweight='bold')
   ax.text(0, arrow_len_v+0.02, '$e_v$', fontsize=14, color='darkgreen', fontweight='bold',
           rotation=90, va='bottom')
   
   # 中心点（红色圆点）
   ax.scatter([0], [0], c='red', s=100, marker='o', zorder=10, 
              edgecolors='darkred', linewidths=1.5)
   ax.text(0.05, 0.05, 'C', fontsize=13, color='red', fontweight='bold')
   
   ax.set_xlabel('u (tangential direction)', fontsize=11)
   ax.set_ylabel('v (tangential direction)', fontsize=11)
   
   # 保存
   script_dir = os.path.dirname(os.path.abspath(__file__))
   roi_dir = os.path.join(script_dir, 'roi')
   os.makedirs(roi_dir, exist_ok=True)
   
   pdf_path = os.path.join(roi_dir, 'fig2_tangent_plane.pdf')
   svg_path = os.path.join(roi_dir, 'fig2_tangent_plane.svg')
   
   plt.tight_layout(pad=0.2)
   plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white', pad_inches=0.1)
   plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white', pad_inches=0.1)
   
   print(f"图2已保存到: {pdf_path}")
   print(f"图2已保存到: {svg_path}")
   plt.close()


if __name__ == "__main__":
   main()