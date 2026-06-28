import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_geometry(ax, C, e_u, e_v, directions, plane_scale=1.5):
    """
    绘制几何元素（实体球面、纯净切平面、带箭头的偏移线、细投影线）
    """
    # 1. **实体球面**（更不透明，清晰可见）
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.3,  # 更不透明
                   linewidth=0, antialiased=True, shade=True)
    # 极淡的网格仅作定位参考
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.5, 
                     rstride=5, cstride=5, linewidth=0.3)

    # 计算带缩放的偏移量
    base_du = directions['E']['scale']
    base_dv = directions['N']['scale']
    base_size = max(base_du, base_dv)
    plane_size = base_size * plane_scale

    # 2. **纯净切平面**（仅半透明表面，无网格无边框）
    u_grid = np.linspace(-plane_size, plane_size, 15)
    v_grid = np.linspace(-plane_size, plane_size, 15)
    U, V = np.meshgrid(u_grid, v_grid)
    X_plane = C[0] + U*e_u[0] + V*e_v[0]
    Y_plane = C[1] + U*e_u[1] + V*e_v[1]
    Z_plane = C[2] + U*e_u[2] + V*e_v[2]
    
    ax.plot_surface(X_plane, Y_plane, Z_plane, color='gray', 
                   alpha=0.12, rstride=1, cstride=1, shade=True, linewidth=0)

    # 3. 中心点（较小）
    ax.scatter(*C, color='black', s=40, marker='o', 
              edgecolors='white', linewidth=1.5, zorder=10)
    ax.text(*(C + 0.05*C), r'$\mathbf{c}$', fontsize=10, fontweight='bold')

    # 4. **四个方向：切平面箭头 + 径向投影**
    for key, info in directions.items():
        # 带缩放的偏移向量（适中距离）
        offset_vec = info['scale'] * plane_scale * info['vec']
        P_tangent = C + offset_vec
        P_sphere = P_tangent / np.linalg.norm(P_tangent)
        color = info['color']

        # **切平面内的方向箭头**（细线带箭头，从中心指向抖动点）
        ax.quiver(*C, *offset_vec, color=color, arrow_length_ratio=0.12, 
                 linewidth=0.7, alpha=0.6, linestyle='-')
        
        # **径向投影线**（细线，从切平面跌落到球面）
        ax.plot([P_tangent[0], P_sphere[0]], 
               [P_tangent[1], P_sphere[1]], 
               [P_tangent[2], P_sphere[2]], 
               color=color, linewidth=1.0, alpha=0.7, linestyle='-')

        # 抖动点（极小）和投影点（极小）
        ax.scatter(*P_tangent, color=color, s=20, marker='x',  # 切平面上的x
                  linewidths=1.5, alpha=0.9, zorder=8)
        ax.scatter(*P_sphere, color=color, s=25, marker='o',  # 球面上的小圆点
                  edgecolors='black', linewidth=1, zorder=9)

        # 标签（靠近球面点）
        label_pos = P_sphere * 1.08
        ax.text(*label_pos, info['label'], color=color, fontsize=8, fontweight='bold')

def setup_axis(ax, elev, azim, title_suffix=""):
    """设置坐标轴——紧凑视野使球体显得更大"""
    ax.set_xlabel('X', fontsize=9)
    ax.set_ylabel('Y', fontsize=9)
    ax.set_zlabel('Z', fontsize=9)
    ax.set_title(f'Tangent Plane Jittering {title_suffix}', fontsize=11, fontweight='bold', pad=10)
    
    # **紧凑的坐标范围使球（半径1）占据主要画面**
    limit = 1.25
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    ax.set_box_aspect([1,1,1])
    
    # 隐藏刻度标签减少干扰
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    ax.tick_params(labelsize=7)
    
    ax.view_init(elev=elev, azim=azim)

def main():
    # ==================== 参数计算 ====================
    center_lon, center_lat = np.radians([30, 30])
    theta_u = np.radians(40)
    theta_v = np.radians(30)
    eta = 0.6
    
    C = np.array([
        np.cos(center_lat) * np.cos(center_lon),
        np.cos(center_lat) * np.sin(center_lon),
        np.sin(center_lat)
    ])
    
    e_u = np.array([-np.sin(center_lon), np.cos(center_lon), 0])
    theta0 = np.pi/2 - center_lat
    e_v = np.array([
        -np.cos(theta0) * np.cos(center_lon),
        -np.cos(theta0) * np.sin(center_lon),
        np.sin(theta0)
    ])
    
    du = np.tan(eta * theta_u / 2)
    dv = np.tan(eta * theta_v / 2)
    
    directions = {
        'E': {'label': r'$+\mathbf{e}_u$', 'vec': e_u, 'scale': du, 'color': '#228B22'},
        'W': {'label': r'$-\mathbf{e}_u$', 'vec': -e_u, 'scale': du, 'color': '#DC143C'},
        'N': {'label': r'$+\mathbf{e}_v$', 'vec': e_v, 'scale': dv, 'color': '#4169E1'},
        'S': {'label': r'$-\mathbf{e}_v$', 'vec': -e_v, 'scale': dv, 'color': '#DAA520'}
    }
    
    # ==================== 视角1：标准侧视 ====================
    fig1 = plt.figure(figsize=(9, 7), dpi=300)
    ax1 = fig1.add_subplot(111, projection='3d')
    draw_geometry(ax1, C, e_u, e_v, directions, plane_scale=1.5)  # 适中扩大
    setup_axis(ax1, elev=20, azim=45, title_suffix='(Standard View)')
    
    plt.tight_layout()
    path1 = './jitter_balanced_view1.png'
    plt.savefig(path1, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ View 1 saved: {path1}")
    plt.close()
    
    # ==================== 视角2：左下角低视角 ====================
    fig2 = plt.figure(figsize=(9, 7), dpi=300)
    ax2 = fig2.add_subplot(111, projection='3d')
    draw_geometry(ax2, C, e_u, e_v, directions, plane_scale=1.5)
    setup_axis(ax2, elev=8, azim=20, title_suffix='(Low-Left View)')
    
    plt.tight_layout()
    path2 = './jitter_balanced_view2.png'
    plt.savefig(path2, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ View 2 saved: {path2}")
    plt.close()

if __name__ == '__main__':
    main()