import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    """绘制实体球面（包含经纬线）"""
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=alpha, color='skyblue', 
                    edgecolor='none', rstride=2, cstride=2, shade=True, antialiased=True, linewidth=0)
    
    # 绘制纬线
    for theta in np.radians([0,30,60,90]):
        x_line = np.sin(theta) * np.cos(u)
        y_line = np.sin(theta) * np.sin(u)
        z_line = np.cos(theta) * np.ones_like(u)
        ax.plot(x_line, y_line, z_line, 'steelblue', alpha=0.3, linewidth=0.8)
    
    # 绘制经线
    for phi in np.radians([0,30,60,90,120,150,180,210,240,270]):
        x_line = np.sin(v) * np.cos(phi)
        y_line = np.sin(v) * np.sin(phi)
        z_line = np.cos(v)
        ax.plot(x_line, y_line, z_line, 'steelblue', alpha=0.3, linewidth=0.8)

def draw_wireframe_tangent_plane(ax, C, e_u, e_v, n, W, H, grid_size, plane_points, sphere_points, u_grid, v_grid, show_projection_lines=True, line_color='darkgreen'):
    """绘制透视化的切平面"""
    # 绘制切平面网格线 - 细线
    for i in range(grid_size):
        line_u = np.array([C + u_grid[j]*e_u + v_grid[i]*e_v for j in range(grid_size)])
        line_v = np.array([C + u_grid[i]*e_u + v_grid[j]*e_v for j in range(grid_size)])
        ax.plot(line_u[:,0], line_u[:,1], line_u[:,2], color=line_color, alpha=0.8, linewidth=1.0)
        ax.plot(line_v[:,0], line_v[:,1], line_v[:,2], color=line_color, alpha=0.8, linewidth=1.0)
    
    # 投影线
    if show_projection_lines:
        for i in range(0, len(plane_points), 3):
            ax.plot([plane_points[i,0], sphere_points[i,0]], 
                   [plane_points[i,1], sphere_points[i,1]], 
                   [plane_points[i,2], sphere_points[i,2]], 'm-', alpha=0.4, linewidth=1)
    
    # 球面投影区域
    sphere_grid = sphere_points.reshape(grid_size, grid_size, 3)
    X_s = sphere_grid[:,:,0]
    Y_s = sphere_grid[:,:,1]
    Z_s = sphere_grid[:,:,2]
    ax.plot_surface(X_s, Y_s, Z_s, color='lightsteelblue', alpha=0.7, edgecolor='steelblue', 
                    linewidth=0.5, shade=True, zorder=4)
    
    # 边界网格线
    for i in range(grid_size):
        ax.plot(sphere_grid[i,:,0], sphere_grid[i,:,1], sphere_grid[i,:,2], 
                'darkblue', alpha=0.6, linewidth=1.0, zorder=5)
        ax.plot(sphere_grid[:,i,0], sphere_grid[:,i,1], sphere_grid[:,i,2], 
                'darkblue', alpha=0.6, linewidth=1.0, zorder=5)
    
    # 投影点：藏青色实心小点
    ax.scatter(sphere_points[:,0], sphere_points[:,1], sphere_points[:,2], 
               c='navy', s=12, alpha=0.9, zorder=6)

def draw_annotations(ax, C, e_u, e_v, n, W, H):
    """绘制标注 - 调整坐标轴长度以适应紧凑视图"""
    # 1. 中心点C - 红色圆点
    ax.scatter(*C, color='red', s=200, marker='o', edgecolors='darkred', linewidths=2, zorder=10)
    ax.text(C[0]+0.08, C[1]+0.08, C[2]+0.08, 'C\ncenter', fontsize=11, color='darkred', fontweight='bold')
    
    # 2. 切平面标签 - 调整位置避免出界
    corner = C + W*e_u + H*e_v
    ax.text(corner[0]+0.05, corner[1]+0.05, corner[2]+0.05, 'Tangent\nPlane', 
            fontsize=10, color='darkgreen', fontweight='bold')
    
    # 3. 局部坐标轴 eu, ev（不绘制n轴）- 缩短长度以适应紧凑视图
    axis_color = 'crimson'
    axis_scale = max(W, H) * 0.9  # 稍微缩短，避免触及边界
    
    # eu - 小箭头
    ax.quiver(*C, *(axis_scale*e_u), color=axis_color, arrow_length_ratio=0.08, linewidth=2.0, alpha=0.95)
    # ev - 小箭头
    ax.quiver(*C, *(axis_scale*e_v), color=axis_color, arrow_length_ratio=0.08, linewidth=2.0, alpha=0.95)
    
    # 坐标轴标签
    offset = 0.06
    ax.text(C[0]+axis_scale*e_u[0]+offset, C[1]+axis_scale*e_u[1]+offset, C[2]+axis_scale*e_u[2]+offset, 
            r'$e_u$', fontsize=13, color=axis_color, fontweight='bold')
    ax.text(C[0]+axis_scale*e_v[0]+offset, C[1]+axis_scale*e_v[1]+offset, C[2]+axis_scale*e_v[2]+offset, 
            r'$e_v$', fontsize=13, color=axis_color, fontweight='bold')
    
    # 4. 原点O
    ax.scatter([0], [0], [0], color='black', s=100, marker='o', zorder=10)
    ax.text(0.06, 0.06, 0.08, 'O', fontsize=11, fontweight='bold')
    ax.plot([0, C[0]], [0, C[1]], [0, C[2]], 'k--', alpha=0.5, linewidth=2)
    
    # 5. XYZ全局坐标轴 - 缩短长度
    ax.quiver(0,0,0, 1.1,0,0, color='dimgray', arrow_length_ratio=0.08, linewidth=2, alpha=0.5)
    ax.quiver(0,0,0, 0,1.1,0, color='dimgray', arrow_length_ratio=0.08, linewidth=2, alpha=0.5)
    ax.quiver(0,0,0, 0,0,1.1, color='dimgray', arrow_length_ratio=0.08, linewidth=2, alpha=0.5)
    ax.text(1.15, 0, 0, 'X', fontsize=10, color='dimgray', fontweight='bold')
    ax.text(0, 1.15, 0, 'Y', fontsize=10, color='dimgray', fontweight='bold')
    ax.text(0, 0, 1.15, 'Z', fontsize=10, color='dimgray', fontweight='bold')

def remove_3d_background(ax):
    """移除3D坐标轴的背景平面"""
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.grid(False)
    ax.set_axis_off()

def save_view(C, e_u, e_v, n, W, H, grid_size, plane_points, sphere_points, u_grid, v_grid, 
              elev, azim, filename_prefix, roi_dir):
    """保存单个视角的图 - 紧凑布局，放大占比"""
    
    # 修改：减小figsize，让内容更紧凑
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    draw_solid_sphere(ax, alpha=0.25)
    draw_wireframe_tangent_plane(ax, C, e_u, e_v, n, W, H, grid_size, plane_points, sphere_points, u_grid, v_grid)
    draw_annotations(ax, C, e_u, e_v, n, W, H)
    remove_3d_background(ax)
    
    # 修改：收紧坐标范围，让切平面占满画面（空间占比大）
    # 基于C的位置(约0.44, 0.63, 0.64)，设置紧凑的边界
    ax.set_xlim([C[0]-0.55, C[0]+0.55])
    ax.set_ylim([C[1]-0.55, C[1]+0.55])
    ax.set_zlim([C[2]-0.55, C[2]+0.55])
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)
    
    # 移除刻度标签，减少视觉干扰
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # 保存 - 使用更小的边距
    pdf_path = os.path.join(roi_dir, f'{filename_prefix}.pdf')
    svg_path = os.path.join(roi_dir, f'{filename_prefix}.svg')
    png_path = os.path.join(roi_dir, f'{filename_prefix}.png')
    
    # 修改：tight_layout和更小的pad_inches
    plt.tight_layout(pad=0.2)
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white', pad_inches=0.05)
    plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white', pad_inches=0.05)
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.05)
    
    print(f"已保存: {pdf_path}")
    print(f"已保存: {svg_path}")
    print(f"已保存: {png_path}")
    plt.close()

def main():
    # 准备目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    roi_dir = os.path.join(script_dir, 'roi')
    os.makedirs(roi_dir, exist_ok=True)
    
    # 生成几何数据
    C, e_u, e_v, n, W, H, grid_size, plane_points, sphere_points, u_grid, v_grid = create_tangent_plane_geometry()
    lat = np.arccos(C[2])
    lon = np.arctan2(C[1], C[0])
    lat_deg, lon_deg = np.degrees(lat), np.degrees(lon)
    
    print("开始生成两个视角的切平面可视化...")
    print(f"切平面中心C: ({C[0]:.3f}, {C[1]:.3f}, {C[2]:.3f})")
    print(f"切平面尺寸: W={W}, H={H}")
    
    # 视角1：沿OC轴向外看
    save_view(C, e_u, e_v, n, W, H, grid_size, plane_points, sphere_points, u_grid, v_grid, 
              elev=90-lat_deg+10, azim=lon_deg, 
              filename_prefix='fig1a_view_outward', roi_dir=roi_dir)
    
    # 视角2：从左下方向上看
    save_view(C, e_u, e_v, n, W, H, grid_size, plane_points, sphere_points, u_grid, v_grid, 
              elev=90-lat_deg-35, azim=lon_deg+290, 
              filename_prefix='fig1b_view_lower_left', roi_dir=roi_dir)
    
    print("\n所有文件已保存到:", roi_dir)

if __name__ == "__main__":
    main()