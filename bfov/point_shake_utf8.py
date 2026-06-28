import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def spherical_to_cartesian(lam, phi, radius=1.0):
    """
    球坐标(经度λ, 纬度φ)转换为笛卡尔坐标(x, y, z)
    λ: longitude (-π to π), φ: latitude (-π/2 to π/2)
    """
    x = radius * np.cos(phi) * np.cos(lam)
    y = radius * np.cos(phi) * np.sin(lam)
    z = radius * np.sin(phi)
    return np.array([x, y, z])

def compute_shake_points(lambda0, phi0, fov_w, fov_h, ratio):
    """
    基于球面几何计算四个抖动方向的新坐标
    遵循原代码中的球面三角学公式
    """
    points = {}
    
    # 原始中心点
    points['center'] = (lambda0, phi0)
    
    # 1. 水平左抖动 (沿大圆航线向左/东)
    alpha = ratio * fov_w
    sin_phi_new = np.sin(phi0) * np.cos(alpha)
    sin_phi_new = np.clip(sin_phi_new, -1.0 + 1e-6, 1.0 - 1e-6)
    phi_h = np.arcsin(sin_phi_new)
    
    # 经度偏移量计算 (球面三角学)
    cos_phi0 = np.cos(phi0) + 1e-8
    delta_lambda = np.arctan2(np.sin(alpha), cos_phi0 * np.cos(alpha))
    
    lambda_left = lambda0 + delta_lambda
    points['left'] = (lambda_left, phi_h)
    
    # 2. 水平右抖动 (沿大圆航线向右/西)
    lambda_right = lambda0 - delta_lambda
    points['right'] = (lambda_right, phi_h)
    
    # 3. 垂直上抖动 (沿经线向北)
    phi_top = phi0 + ratio * fov_h
    # 约束在有效范围内
    phi_top = min(phi_top, np.pi/2 - 1e-6)
    points['up'] = (lambda0, phi_top)
    
    # 4. 垂直下抖动 (沿经线向南)
    phi_bottom = phi0 - ratio * fov_h
    phi_bottom = max(phi_bottom, -np.pi/2 + 1e-6)
    points['down'] = (lambda0, phi_bottom)
    
    return points

def get_tangent_basis(lambda0, phi0):
    """
    计算切平面上的标准正交基向量
    e_east: 沿纬线切线方向 (经度增加方向)
    e_north: 沿经线切线方向 (纬度增加方向)
    """
    # 东向基向量 (经度λ增加方向)
    e_east = np.array([-np.sin(lambda0), np.cos(lambda0), 0])
    
    # 北向基向量 (纬度φ增加方向)
    e_north = np.array([
        -np.sin(phi0) * np.cos(lambda0),
        -np.sin(phi0) * np.sin(lambda0),
        np.cos(phi0)
    ])
    
    return e_east, e_north

def draw_sphere_shake_3d(lambda0_deg=30, phi0_deg=30, fov_w_deg=40, 
                         fov_h_deg=30, shake_ratio=0.3, save_path='shake_3d.png'):
    """
    3D球面视图：展示切平面上的抖动箭头与球面实际位置的关系
    """
    # 转换为弧度
    lam0, phi0 = np.radians(lambda0_deg), np.radians(phi0_deg)
    fov_w, fov_h = np.radians(fov_w_deg), np.radians(fov_h_deg)
    R = 1.0
    
    # 计算球面坐标
    shakes = compute_shake_points(lam0, phi0, fov_w, fov_h, shake_ratio)
    cartesian = {k: spherical_to_cartesian(v[0], v[1], R) 
                for k, v in shakes.items()}
    
    # 计算切平面基向量
    e_east, e_north = get_tangent_basis(lam0, phi0)
    center = cartesian['center']
    
    # 计算切平面上的理想位置（平面近似）
    d_h = shake_ratio * fov_w * R  # 水平方向弧长
    d_v = shake_ratio * fov_h * R  # 垂直方向弧长
    
    tangent_pos = {
        'left': center + d_h * e_east,
        'right': center - d_h * e_east,
        'up': center + d_v * e_north,
        'down': center - d_v * e_north
    }
    
    # 创建图形
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制半透明球面
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(-np.pi/2, np.pi/2, 40)
    x = R * np.outer(np.cos(v), np.cos(u))
    y = R * np.outer(np.cos(v), np.sin(u))
    z = R * np.outer(np.sin(v), np.ones_like(u))
    ax.plot_surface(x, y, z, alpha=0.12, color='skyblue', 
                   rstride=4, cstride=4, edgecolor='none')
    
    # 绘制经纬网格（增强3D感）
    for lam in np.linspace(-np.pi, np.pi, 12):
        xv = R * np.cos(v) * np.cos(lam)
        yv = R * np.cos(v) * np.sin(lam)
        zv = R * np.sin(v)
        ax.plot(xv, yv, zv, 'gray', alpha=0.2, linewidth=0.5)
    
    # 绘制中心点
    ax.scatter(*center, color='red', s=200, marker='o', 
              label='原始中心点 $C(\\lambda_0, \\phi_0)$',
              edgecolors='black', linewidth=2, zorder=5)
    
    # 绘制切平面（半透明圆盘）
    disk_radius = max(d_h, d_v) * 1.3
    theta = np.linspace(0, 2*np.pi, 30)
    disk_pts = np.array([
        center + disk_radius * (np.cos(t)*e_east + np.sin(t)*e_north)
        for t in theta
    ])
    ax.add_collection3d(Poly3DCollection([disk_pts], alpha=0.15, 
                                        facecolor='yellow', edgecolor='orange'))
    
    # 绘制切平面坐标轴
    axis_len = 0.5
    ax.quiver(*center, *(axis_len*e_east), color='darkgreen', 
             arrow_length_ratio=0.25, linewidth=2.5, alpha=0.8)
    ax.text(*(center + 1.1*axis_len*e_east), r'东向 $\vec{e}_\lambda$', 
           fontsize=11, color='darkgreen', weight='bold')
    
    ax.quiver(*center, *(axis_len*e_north), color='darkblue', 
             arrow_length_ratio=0.25, linewidth=2.5, alpha=0.8)
    ax.text(*(center + 1.1*axis_len*e_north), r'北向 $\vec{e}_\phi$', 
           fontsize=11, color='darkblue', weight='bold')
    
    # 绘制抖动箭头和点
    colors = {'left': '#0066CC', 'right': '#CC6600', 
              'up': '#009900', 'down': '#CC0000'}
    labels = {'left': '左抖动 (经度+)', 'right': '右抖动 (经度-)', 
              'up': '上抖动 (纬度+)', 'down': '下抖动 (纬度-)'}
    
    for key in ['left', 'right', 'up', 'down']:
        tgt_sph = cartesian[key]    # 球面上的实际位置
        tgt_tan = tangent_pos[key]  # 切平面上的理想位置
        
        # 在切平面上绘制箭头（从中心指向切平面位置）
        direction = tgt_tan - center
        ax.quiver(center[0], center[1], center[2],
                 direction[0], direction[1], direction[2],
                 color=colors[key], arrow_length_ratio=0.3, 
                 linewidth=3, alpha=0.9, label=labels[key])
        
        # 绘制球面上的实际点（三角形）
        ax.scatter(*tgt_sph, color=colors[key], s=120, marker='^',
                  edgecolors='black', linewidth=1.5, zorder=5)
        
        # 绘制修正虚线（从切平面投影到球面）
        ax.plot([tgt_tan[0], tgt_sph[0]], 
               [tgt_tan[1], tgt_sph[1]], 
               [tgt_tan[2], tgt_sph[2]], 
               'k:', alpha=0.5, linewidth=1.5)
    
    # 设置视角
    ax.view_init(elev=20, azim=45)
    
    # 坐标轴设置
    ax.set_xlabel('X', fontsize=12, weight='bold')
    ax.set_ylabel('Y', fontsize=12, weight='bold')
    ax.set_zlabel('Z', fontsize=12, weight='bold')
    
    # 标题
    ax.set_title(f'球面提案中心点抖动机制示意图\n'
                f'中心: ({lambda0_deg}°, {phi0_deg}°), '
                f'视场角: ({fov_w_deg}°, {fov_h_deg}°), '
                f'抖动比例: {shake_ratio}',
                fontsize=13, weight='bold', pad=20)
    
    # 等比例设置
    limit = 1.4
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    ax.set_box_aspect([1,1,1])
    
    # 图例
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"已保存3D视图至: {save_path}")
    plt.show()

def draw_tangent_plane_2d(lambda0_deg=30, phi0_deg=30, fov_w_deg=40, 
                          fov_h_deg=30, shake_ratio=0.3, save_path='shake_2d.png'):
    """
    2D切平面局部视图：展示局部坐标系下的抖动方向（俯视图）
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 参数
    lam0, phi0 = np.radians(lambda0_deg), np.radians(phi0_deg)
    fov_w, fov_h = np.radians(fov_w_deg), np.radians(fov_h_deg)
    
    # 计算位移（在切平面上）
    d_h = shake_ratio * fov_w  # 水平位移
    d_v = shake_ratio * fov_h  # 垂直位移
    
    # 绘制坐标轴
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
    
    # 绘制中心点
    ax.scatter(0, 0, color='red', s=200, zorder=5, 
              label='原始中心点', marker='o', edgecolors='black', linewidth=2)
    
    # 四个抖动方向
    directions = {
        'left': (d_h, 0, '#0066CC', '左 (经度+)', r'$\lambda_0 + \Delta\lambda$'),
        'right': (-d_h, 0, '#CC6600', '右 (经度-)', r'$\lambda_0 - \Delta\lambda$'),
        'up': (0, d_v, '#009900', '上 (纬度+)', r'$\phi_0 + \alpha\cdot h$'),
        'down': (0, -d_v, '#CC0000', '下 (纬度-)', r'$\phi_0 - \alpha\cdot h$')
    }
    
    for key, (dx, dy, color, label, formula) in directions.items():
        # 绘制箭头
        ax.annotate('', xy=(dx, dy), xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color=color, lw=3))
        
        # 绘制点
        ax.scatter(dx, dy, color=color, s=150, zorder=5, 
                  marker='s', edgecolors='black', linewidth=1.5)
        
        # 添加标签
        offset = 0.02 if key in ['left', 'right'] else 0.03
        ax.text(dx + offset, dy + offset, 
               f'{label}\n{formula}', fontsize=10, color=color, weight='bold')
    
    # 绘制视场角示意框（原始提案范围）
    rect_w, rect_h = fov_w * shake_ratio * 2.5, fov_h * shake_ratio * 2.5
    rect = plt.Rectangle((-rect_w/2, -rect_h/2), rect_w, rect_h,
                        fill=False, linestyle='--', color='gray', 
                        alpha=0.5, linewidth=1.5, label='视场范围')
    ax.add_patch(rect)
    
    # 设置
    ax.set_xlabel('切平面东向 (经度方向) $\\vec{e}_\\lambda$', fontsize=12)
    ax.set_ylabel('切平面北向 (纬度方向) $\\vec{e}_\\phi$', fontsize=12)
    ax.set_title('切平面上的中心点抖动示意图 (局部坐标系)', fontsize=13, weight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    # 设置范围
    margin = max(d_h, d_v) * 1.5
    ax.set_xlim([-margin, margin])
    ax.set_ylim([-margin, margin])
    
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"已保存2D切平面视图至: {save_path}")
    plt.show()

if __name__ == "__main__":
    # 参数设置（可根据论文需要调整）
    center_lambda = 0      # 中心点经度 (度)
    center_phi = 30        # 中心点纬度 (度)
    fov_width = 45         # 水平视场角 (度)
    fov_height = 35        # 垂直视场角 (度)
    shake_ratio = 0.25     # 抖动比例
    
    # 生成3D球面视图（主图）
    draw_sphere_shake_3d(center_lambda, center_phi, fov_width, 
                        fov_height, shake_ratio, 'fig_spherical_shake_3d.png')
    
    # 生成2D切平面视图（辅助说明图）
    draw_tangent_plane_2d(center_lambda, center_phi, fov_width, 
                         fov_height, shake_ratio, 'fig_tangent_plane_2d.png')