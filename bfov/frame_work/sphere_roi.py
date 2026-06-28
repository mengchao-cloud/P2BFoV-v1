import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon
import matplotlib.gridspec as gridspec
from scipy.spatial import ConvexHull
import warnings
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
    e_v = np.array([-np.cos(theta0)*np.cos(phi0), -np.cos(theta0)*np.sin(phi0), np.sin(theta0)])
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


def draw_solid_sphere(ax, alpha=0.25):
    """绘制实体球面（更实心）"""
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=alpha, color='skyblue',
                    edgecolor='none', rstride=2, cstride=2, shade=True, 
                    antialiased=True, linewidth=0)
    
    for theta in np.radians([60, 90, 120]):
        x_line = np.sin(theta) * np.cos(u)
        y_line = np.sin(theta) * np.sin(u)
        z_line = np.cos(theta) * np.ones_like(u)
        ax.plot(x_line, y_line, z_line, 'steelblue', alpha=0.3, linewidth=0.8)
    
    for phi in np.radians([0, 90, 180, 270]):
        x_line = np.sin(v) * np.cos(phi)
        y_line = np.sin(v) * np.sin(phi)
        z_line = np.cos(v)
        ax.plot(x_line, y_line, z_line, 'steelblue', alpha=0.3, linewidth=0.8)


def draw_wireframe_tangent_plane(ax, C, e_u, e_v, W, H, grid_size, plane_points, sphere_points, u_grid, v_grid, 
                                  show_projection_lines=True, line_color='darkgreen', point_color='darkgreen'):
    """绘制透视化的切平面（wireframe风格）"""
    
    for i in range(grid_size):
        line_u = np.array([C + u_grid[j]*e_u + v_grid[i]*e_v for j in range(grid_size)])
        line_v = np.array([C + u_grid[i]*e_u + v_grid[j]*e_v for j in range(grid_size)])
        ax.plot(line_u[:,0], line_u[:,1], line_u[:,2], color=line_color, alpha=0.8, linewidth=1.5)
        ax.plot(line_v[:,0], line_v[:,1], line_v[:,2], color=line_color, alpha=0.8, linewidth=1.5)
    
    ax.scatter(plane_points[:,0], plane_points[:,1], plane_points[:,2],
              c=point_color, s=80, alpha=0.9, edgecolors='white', linewidths=1,
              label='Tangent plane samples')
    
    if show_projection_lines:
        for i in range(0, len(plane_points), 3):
            ax.plot([plane_points[i,0], sphere_points[i,0]],
                    [plane_points[i,1], sphere_points[i,1]],
                    [plane_points[i,2], sphere_points[i,2]],
                    'm-', alpha=0.4, linewidth=1)
    
    ax.scatter(sphere_points[:,0], sphere_points[:,1], sphere_points[:,2],
              c='blue', s=25, alpha=0.8, label='Projected on sphere')
    
    sphere_grid = sphere_points.reshape(grid_size, grid_size, 3)
    for i in range(grid_size):
        ax.plot(sphere_grid[i,:,0], sphere_grid[i,:,1], sphere_grid[i,:,2], 
                'b-', alpha=0.3, linewidth=0.8)
        ax.plot(sphere_grid[:,i,0], sphere_grid[:,i,1], sphere_grid[:,i,2], 
                'b-', alpha=0.3, linewidth=0.8)


def draw_annotations(ax, C, e_u, e_v, n, W, H, scale=0.5):
    """绘制标注"""
    ax.scatter(*C, color='red', s=500, marker='*', edgecolors='darkred', 
               linewidths=2, zorder=10)
    ax.text(C[0]+0.1, C[1]+0.12, C[2]+0.15, 'C\ncenter', fontsize=12, 
            color='darkred', fontweight='bold')
    
    corner = C + W*e_u + H*e_v
    ax.text(corner[0]+0.08, corner[1]+0.08, corner[2]+0.08, 
            'Tangent\nPlane', fontsize=11, color='darkgreen', fontweight='bold')
    
    ax.quiver(*C, *(scale*e_u), color='darkgreen', arrow_length_ratio=0.25, 
              linewidth=3.5, alpha=0.9)
    ax.quiver(*C, *(scale*e_v), color='darkgreen', arrow_length_ratio=0.25, 
              linewidth=3.5, alpha=0.9)
    ax.quiver(*C, *(scale*n), color='crimson', arrow_length_ratio=0.25, 
              linewidth=3, linestyle='-', alpha=0.9)
    
    ax.text(C[0]+scale*e_u[0]+0.06, C[1]+scale*e_u[1]+0.06, C[2]+scale*e_u[2]+0.04, 
            '$e_u$', fontsize=14, color='darkgreen', fontweight='bold')
    ax.text(C[0]+scale*e_v[0]+0.06, C[1]+scale*e_v[1]+0.06, C[2]+scale*e_v[2]+0.04, 
            '$e_v$', fontsize=14, color='darkgreen', fontweight='bold')
    ax.text(C[0]+scale*n[0]+0.08, C[1]+scale*n[1]+0.08, C[2]+scale*n[2]+0.06, 
            '$n$', fontsize=12, color='crimson', fontweight='bold')
    
    ax.scatter([0], [0], [0], color='black', s=100, marker='o', zorder=10)
    ax.text(0.08, 0.08, 0.1, 'O', fontsize=12, fontweight='bold')
    ax.plot([0, C[0]], [0, C[1]], [0, C[2]], 'k--', alpha=0.5, linewidth=2)
    
    ax.quiver(0,0,0, 1.3,0,0, color='dimgray', arrow_length_ratio=0.08, linewidth=2, alpha=0.5)
    ax.quiver(0,0,0, 0,1.3,0, color='dimgray', arrow_length_ratio=0.08, linewidth=2, alpha=0.5)
    ax.quiver(0,0,0, 0,0,1.3, color='dimgray', arrow_length_ratio=0.08, linewidth=2, alpha=0.5)
    ax.text(1.4, 0, 0, 'X', fontsize=11, color='dimgray', fontweight='bold')
    ax.text(0, 1.4, 0, 'Y', fontsize=11, color='dimgray', fontweight='bold')
    ax.text(0, 0, 1.4, 'Z', fontsize=11, color='dimgray', fontweight='bold')


def remove_3d_background(ax):
    """移除3D坐标轴的背景平面"""
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax.grid(False)


def create_figure_1():
    """图1：两个视角 - 沿OC轴向外看和向里看（左下图视角）"""
    fig = plt.figure(figsize=(20, 10))
    
    C, e_u, e_v, n, W, H, grid_size, plane_points, sphere_points, u_grid, v_grid = create_tangent_plane_geometry()
    
    lat = np.arccos(C[2])
    lon = np.arctan2(C[1], C[0])
    lat_deg, lon_deg = np.degrees(lat), np.degrees(lon)
    
    # ========== 左图：沿OC轴向外看（标准视角） ==========
    ax1 = fig.add_subplot(121, projection='3d')
    
    draw_solid_sphere(ax1, alpha=0.2)
    draw_wireframe_tangent_plane(ax1, C, e_u, e_v, W, H, grid_size, plane_points, sphere_points, u_grid, v_grid)
    draw_annotations(ax1, C, e_u, e_v, n, W, H)
    remove_3d_background(ax1)
    
    ax1.set_xlim([-0.6, 1.4])
    ax1.set_ylim([-0.4, 1.6])
    ax1.set_zlim([-0.2, 1.3])
    ax1.set_box_aspect([1, 1, 1])
    ax1.view_init(elev=90-lat_deg+10, azim=lon_deg)
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.set_title('(a) View from sphere center outward along OC axis\n'
                  'Wireframe tangent plane on solid sphere',
                  fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    # ========== 右图：向左下方移动的视角 ==========
    ax2 = fig.add_subplot(122, projection='3d')
    
    draw_solid_sphere(ax2, alpha=0.25)
    draw_wireframe_tangent_plane(ax2, C, e_u, e_v, W, H, grid_size, plane_points, sphere_points, u_grid, v_grid)
    draw_annotations(ax2, C, e_u, e_v, n, W, H)
    remove_3d_background(ax2)
    
    ax2.set_xlim([-0.8, 1.2])
    ax2.set_ylim([-0.6, 1.4])
    ax2.set_zlim([-0.4, 1.2])
    ax2.set_box_aspect([1, 1, 1])
    
    # 向左下方移动：降低elev（向下），增加azim（向左）
    # 原视角: elev=90-lat_deg+10, azim=lon_deg
    # 新视角: 大幅降低elev，显著增加azim（向左转）
    ax2.view_init(elev=90-lat_deg-35, azim=lon_deg+290)
    
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    ax2.set_title('(b) View from lower-left of OC axis\n'
                  'Looking upward and rightward toward tangent plane',
                  fontsize=13, fontweight='bold', pad=10)
    ax2.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    plt.suptitle('Tangent Plane Sampling: Wireframe Grid on Solid Sphere (Two Perspectives)',
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('fig1_tangent_plane_core.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("图1已保存: fig1_tangent_plane_core.png")
    plt.close()
    return C, e_u, e_v, W, H, grid_size


def create_figure_2(C, e_u, e_v, W, H, grid_size):
    """图2：切平面 vs 球面（透视风格）"""
    fig = plt.figure(figsize=(18, 10))
    
    # 左图：2D切平面
    ax1 = fig.add_subplot(121)
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_aspect('equal')
    ax1.set_facecolor('#f8f8f8')
    
    u_vals = np.linspace(-W, W, grid_size)
    v_vals = np.linspace(-H, H, grid_size)
    
    rect = plt.Rectangle((-W, -H), 2*W, 2*H, fill=False, edgecolor='darkgreen', 
                         linewidth=2.5, linestyle='-')
    ax1.add_patch(rect)
    
    for u in u_vals:
        ax1.plot([u, u], [-H, H], 'g-', linewidth=1.5, alpha=0.7)
    for v in v_vals:
        ax1.plot([-W, W], [v, v], 'g-', linewidth=1.5, alpha=0.7)
    
    uu, vv = np.meshgrid(u_vals, v_vals)
    ax1.scatter(uu.flatten(), vv.flatten(), c='darkgreen', s=60, zorder=5, 
                edgecolors='white', linewidths=0.5)
    
    ax1.annotate('', xy=(0.9, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2.5))
    ax1.annotate('', xy=(0, 0.7), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2.5))
    ax1.text(0.95, 0, '$e_u$', fontsize=14, color='darkgreen', fontweight='bold')
    ax1.text(0, 0.75, '$e_v$', fontsize=14, color='darkgreen', 
             fontweight='bold', rotation=90, va='bottom')
    
    ax1.scatter([0], [0], c='red', s=250, marker='*', zorder=10, 
                edgecolors='darkred', linewidths=1.5)
    ax1.text(0.1, 0.1, 'C', fontsize=13, color='red', fontweight='bold')
    
    ax1.set_title('(a) Tangent Plane: Uniform 14×14 Grid\n'
                  'Local Cartesian coordinates (u, v)',
                  fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('u (tangential direction)', fontsize=11)
    ax1.set_ylabel('v (tangential direction)', fontsize=11)
    
    # ========== 右图：3D透视风格 ==========
    ax2 = fig.add_subplot(122, projection='3d')
    
    u = np.linspace(0, 2*np.pi, 60)
    v_s = np.linspace(0, np.pi, 60)
    x_s = np.outer(np.cos(u), np.sin(v_s))
    y_s = np.outer(np.sin(u), np.sin(v_s))
    z_s = np.outer(np.ones(np.size(u)), np.cos(v_s))
    ax2.plot_surface(x_s, y_s, z_s, alpha=0.15, color='lightblue', 
                     edgecolor='none', rstride=2, cstride=2)
    
    for theta in np.radians([60, 90, 120]):
        x_line = np.sin(theta) * np.cos(u)
        y_line = np.sin(theta) * np.sin(u)
        z_line = np.cos(theta) * np.ones_like(u)
        ax2.plot(x_line, y_line, z_line, 'steelblue', alpha=0.25, linewidth=1)
    
    u_vals = np.linspace(-W, W, grid_size)
    v_vals = np.linspace(-H, H, grid_size)
    
    plane_pts = []
    sphere_pts = []
    for ui in u_vals:
        for vi in v_vals:
            p_plane = C + ui*e_u + vi*e_v
            p_sphere = p_plane / np.linalg.norm(p_plane)
            plane_pts.append(p_plane)
            sphere_pts.append(p_sphere)
    
    plane_pts = np.array(plane_pts)
    sphere_pts = np.array(sphere_pts)
    
    plane_pts_grid = plane_pts.reshape(grid_size, grid_size, 3)
    for i in range(grid_size):
        ax2.plot(plane_pts_grid[i,:,0], plane_pts_grid[i,:,1], plane_pts_grid[i,:,2], 
                'g-', alpha=0.8, linewidth=1.5)
        ax2.plot(plane_pts_grid[:,i,0], plane_pts_grid[:,i,1], plane_pts_grid[:,i,2], 
                'g-', alpha=0.8, linewidth=1.5)
    
    ax2.scatter(plane_pts[:,0], plane_pts[:,1], plane_pts[:,2],
               c='darkgreen', s=50, alpha=0.9, edgecolors='white', linewidths=0.5,
               label='Tangent samples')
    
    for i in range(0, len(plane_pts), 3):
        ax2.plot([plane_pts[i,0], sphere_pts[i,0]],
                [plane_pts[i,1], sphere_pts[i,1]],
                [plane_pts[i,2], sphere_pts[i,2]],
                'm-', alpha=0.35, linewidth=0.8)
    
    ax2.scatter(sphere_pts[:,0], sphere_pts[:,1], sphere_pts[:,2],
               c='blue', s=25, alpha=0.8, label='Sphere samples')
    
    sphere_grid = sphere_pts.reshape(grid_size, grid_size, 3)
    for i in range(grid_size):
        ax2.plot(sphere_grid[i,:,0], sphere_grid[i,:,1], sphere_grid[i,:,2], 
                'b-', alpha=0.3, linewidth=0.8)
        ax2.plot(sphere_grid[:,i,0], sphere_grid[:,i,1], sphere_grid[:,i,2], 
                'b-', alpha=0.3, linewidth=0.8)
    
    ax2.scatter(*C, c='red', s=300, marker='*', edgecolors='darkred', linewidths=1.5)
    ax2.text(C[0]+0.08, C[1]+0.08, C[2]+0.08, 'C', fontsize=12, color='red', fontweight='bold')
    
    remove_3d_background(ax2)
    
    ax2.set_xlim([0.2, 1.0])
    ax2.set_ylim([0.4, 1.2])
    ax2.set_zlim([0.5, 1.1])
    ax2.set_box_aspect([1,1,1])
    ax2.view_init(elev=20, azim=45)
    
    ax2.set_xlabel('X', fontsize=10)
    ax2.set_ylabel('Y', fontsize=10)
    ax2.set_zlabel('Z', fontsize=10)
    
    ax2.set_title('(b) 3D View: Wireframe Tangent Plane → Sphere\n'
                  'Central projection preserves angular relationships',
                  fontsize=13, fontweight='bold', pad=10)
    
    fig.text(0.5, 0.02, 'Projection curves the grid while preserving local angles',
             ha='center', fontsize=12, style='italic', color='darkblue')
    
    plt.tight_layout()
    plt.savefig('fig2_tangent_vs_sphere.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("图2已保存: fig2_tangent_vs_sphere.png")
    plt.close()


def create_figure_3(C, e_u, e_v, W, H, grid_size):
    """图3：完整投影流程（BFOV只绘制彩色点，无填充）"""
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
    
    lat = np.arccos(C[2])
    lon = np.arctan2(C[1], C[0])
    lon_deg, lat_deg = np.degrees(lon), 90 - np.degrees(lat)
    
    u_vals = np.linspace(-W, W, grid_size)
    v_vals = np.linspace(-H, H, grid_size)
    
    # Step 1: Tangent Plane
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_aspect('equal')
    ax1.set_facecolor('#e8f5e9')
    
    uu, vv = np.meshgrid(u_vals, v_vals)
    for u in u_vals:
        ax1.axvline(u, color='green', alpha=0.5, linewidth=1)
    for v in v_vals:
        ax1.axhline(v, color='green', alpha=0.5, linewidth=1)
    ax1.scatter(uu.flatten(), vv.flatten(), c='darkgreen', s=50)
    
    ax1.set_title('Step 1: Tangent Plane\n(u, v) coordinates\nUniform grid', 
                  fontsize=11, fontweight='bold')
    ax1.set_xlabel('u')
    ax1.set_ylabel('v')
    
    # Step 2: Project to Sphere
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    
    u = np.linspace(0, 2*np.pi, 40)
    v_s = np.linspace(0, np.pi, 40)
    x_s = np.outer(np.cos(u), np.sin(v_s))
    y_s = np.outer(np.sin(u), np.sin(v_s))
    z_s = np.outer(np.ones(np.size(u)), np.cos(v_s))
    ax2.plot_surface(x_s, y_s, z_s, alpha=0.12, color='lightblue', edgecolor='none')
    
    sphere_pts = []
    for ui in u_vals:
        for vi in v_vals:
            p = C + ui*e_u + vi*e_v
            p_s = p / np.linalg.norm(p)
            sphere_pts.append(p_s)
    sphere_pts = np.array(sphere_pts)
    
    sphere_grid = sphere_pts.reshape(grid_size, grid_size, 3)
    for i in range(grid_size):
        ax2.plot(sphere_grid[i,:,0], sphere_grid[i,:,1], sphere_grid[i,:,2], 'b-', alpha=0.4, linewidth=1)
        ax2.plot(sphere_grid[:,i,0], sphere_grid[:,i,1], sphere_grid[:,i,2], 'b-', alpha=0.4, linewidth=1)
    
    ax2.scatter(sphere_pts[:,0], sphere_pts[:,1], sphere_pts[:,2], c='blue', s=25, alpha=0.8)
    ax2.scatter(*C, c='red', s=150, marker='*', edgecolors='darkred')
    
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0.3, 1])
    ax2.set_zlim([0.5, 1])
    ax2.set_box_aspect([1,1,1])
    ax2.view_init(elev=25, azim=50)
    remove_3d_background(ax2)
    ax2.set_xlabel('X', fontsize=9)
    ax2.set_ylabel('Y', fontsize=9)
    ax2.set_zlabel('Z', fontsize=9)
    ax2.set_title('Step 2: Project to Sphere\n(x, y, z) = p/||p||\nCurved grid', 
                  fontsize=11, fontweight='bold')
    
    # Step 3: Geographic coords
    ax3 = fig.add_subplot(gs[0, 2])
    
    sphere_pts_T = sphere_pts.T
    phi = np.arctan2(sphere_pts_T[1], sphere_pts_T[0])
    theta = np.arccos(np.clip(sphere_pts_T[2], -1, 1))
    lat_pts = 90 - np.degrees(theta)
    lon_pts = np.degrees(phi)
    
    ax3.set_xlim(lon_deg-15, lon_deg+15)
    ax3.set_ylim(lat_deg-12, lat_deg+12)
    ax3.set_aspect('equal')
    ax3.set_facecolor('#fff3e0')
    
    lon_grid = lon_pts.reshape(grid_size, grid_size).T
    lat_grid = lat_pts.reshape(grid_size, grid_size).T
    
    for i in range(grid_size):
        ax3.plot(lon_grid[i, :], lat_grid[i, :], 'b-', alpha=0.5, linewidth=1)
        ax3.plot(lon_grid[:, i], lat_grid[:, i], 'b-', alpha=0.5, linewidth=1)
    
    ax3.scatter(lon_pts, lat_pts, c='blue', s=30)
    ax3.scatter([lon_deg], [lat_deg], c='red', s=150, marker='*')
    
    ax3.set_xlabel('Longitude (deg)')
    ax3.set_ylabel('Latitude (deg)')
    ax3.set_title(f'Step 3: Geographic coords\nCenter: ({lon_deg:.0f}°, {lat_deg:.0f}°)\nStill curved',
                  fontsize=11, fontweight='bold')
    
    # Step 4: ERP Image
    ax4 = fig.add_subplot(gs[0, 3])
    
    erp_w, erp_h = 1024, 512
    erp_x = (lon_pts + 180) / 360 * erp_w
    erp_y = (90 - lat_pts) / 180 * erp_h
    
    center_x = (lon_deg + 180) / 360 * erp_w
    center_y = (90 - lat_deg) / 180 * erp_h
    
    ax4.set_xlim(center_x-80, center_x+80)
    ax4.set_ylim(center_y-60, center_y+60)
    ax4.set_aspect('equal')
    ax4.set_facecolor('#fce4ec')
    
    erp_x_grid = erp_x.reshape(grid_size, grid_size).T
    erp_y_grid = erp_y.reshape(grid_size, grid_size).T
    
    for i in range(grid_size):
        ax4.plot(erp_x_grid[i, :], erp_y_grid[i, :], 'r-', alpha=0.5, linewidth=1)
        ax4.plot(erp_x_grid[:, i], erp_y_grid[:, i], 'r-', alpha=0.5, linewidth=1)
    
    ax4.scatter(erp_x, erp_y, c='darkred', s=30)
    ax4.scatter([center_x], [center_y], c='red', s=150, marker='*')
    
    ax4.set_xlabel('ERP x (pixels)')
    ax4.set_ylabel('ERP y (pixels)')
    ax4.set_title('Step 4: ERP Image\nPixel coordinates\nStretched!', 
                  fontsize=11, fontweight='bold')
    
    # Step 5: Multiple BFOVs - 只绘制彩色点
    ax5 = fig.add_subplot(gs[1, :])
    
    ax5.set_xlim(0, erp_w)
    ax5.set_ylim(0, erp_h)
    ax5.set_aspect('equal')
    ax5.set_facecolor('#f5f5f5')
    
    for lon_bg in range(0, 361, 30):
        x_bg = lon_bg / 360 * erp_w
        ax5.axvline(x_bg, color='lightgray', linestyle='-', alpha=0.5, linewidth=0.5)
    for lat_bg in range(-90, 91, 30):
        y_bg = (90 - lat_bg) / 180 * erp_h
        ax5.axhline(y_bg, color='lightgray', linestyle='-', alpha=0.5, linewidth=0.5)
    
    bfov_list = [
        (60, 45, 60, 40, 'blue', 'BFOV 1'),
        (200, 70, 60, 40, 'green', 'BFOV 2'),
        (300, 0, 80, 50, 'purple', 'BFOV 3'),
        (500, 80, 60, 40, 'orange', 'BFOV 4'),
    ]
    
    for lon_c, lat_c, fov_u, fov_v, color, label in bfov_list:
        lon_c_rad = np.radians(lon_c)
        lat_c_rad = np.radians(lat_c)
        theta_c = np.pi/2 - lat_c_rad
        phi_c = lon_c_rad
        
        C_local = np.array([np.sin(theta_c)*np.cos(phi_c), 
                           np.sin(theta_c)*np.sin(phi_c), 
                           np.cos(theta_c)])
        e_u_local = np.array([-np.sin(phi_c), np.cos(phi_c), 0])
        e_v_local = np.array([-np.cos(theta_c)*np.cos(phi_c), 
                             -np.cos(theta_c)*np.sin(phi_c), 
                             np.sin(theta_c)])
        
        W_local = np.tan(np.radians(fov_u)/2)
        H_local = np.tan(np.radians(fov_v)/2)
        
        u_local = np.linspace(-W_local, W_local, 20)
        v_local = np.linspace(-H_local, H_local, 20)
        
        local_pts = []
        for ui in u_local:
            for vi in v_local:
                p = C_local + ui*e_u_local + vi*e_v_local
                p_s = p / np.linalg.norm(p)
                phi_p = np.arctan2(p_s[1], p_s[0])
                theta_p = np.arccos(np.clip(p_s[2], -1, 1))
                lat_p = 90 - np.degrees(theta_p)
                lon_p = np.degrees(phi_p)
                erp_x_p = (lon_p + 180) / 360 * erp_w
                erp_y_p = (90 - lat_p) / 180 * erp_h
                local_pts.append([erp_x_p, erp_y_p])
        
        local_pts = np.array(local_pts)
        
        ax5.scatter(local_pts[:,0], local_pts[:,1], c=color, s=20, alpha=0.7, 
                   edgecolors='none', label=label)
        
        cx, cy = local_pts[:,0].mean(), local_pts[:,1].mean()
        ax5.scatter([cx], [cy], c='black', s=200, marker='*', zorder=10)
        ax5.annotate(label, xy=(cx, cy), xytext=(cx+80, cy+50),
                    fontsize=10, color='black', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))
    
    ax5.set_xlabel('ERP X (pixels) → Longitude', fontsize=12)
    ax5.set_ylabel('ERP Y (pixels) → Latitude', fontsize=12)
    ax5.set_title('Multiple BFOVs on ERP Image: Colored points show tangent plane sampling coverage\n'
                  'High-latitude regions are stretched horizontally; tangent plane sampling avoids this distortion',
                  fontsize=13, fontweight='bold', pad=10)
    
    ax5.set_xticks([0, 256, 512, 768, 1024])
    ax5.set_xticklabels(['0°', '90°', '180°', '270°', '360°'])
    ax5.set_yticks([0, 128, 256, 384, 512])
    ax5.set_yticklabels(['90°N', '45°N', '0°', '45°S', '90°S'])
    
    plt.suptitle('Complete Projection Pipeline: Tangent Plane → Sphere → ERP',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('fig3_complete_pipeline.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("图3已保存: fig3_complete_pipeline.png")
    plt.close()


def main():
    print("=" * 50)
    print("开始生成切平面采样可视化图表")
    print("=" * 50)
    
    C, e_u, e_v, W, H, grid_size = create_figure_1()
    create_figure_2(C, e_u, e_v, W, H, grid_size)
    create_figure_3(C, e_u, e_v, W, H, grid_size)
    
    print("=" * 50)
    print("所有图表生成完成！")
    print("输出文件:")
    print("  - fig1_tangent_plane_core.png")
    print("  - fig2_tangent_vs_sphere.png")  
    print("  - fig3_complete_pipeline.png")
    print("=" * 50)


if __name__ == "__main__":
    main()