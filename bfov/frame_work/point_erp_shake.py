import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

ERP_WIDTH = 1024
ERP_HEIGHT = 512
JITTER_ANGLE = 50
ETA = 1.0
output_dir = "./bfov/"
os.makedirs(output_dir, exist_ok=True)

def normalize_longitude(lon):
    """Normalize longitude to [-180, 180]"""
    return ((lon + 180) % 360) - 180

def spherical_jittering(lon0_deg, lat0_deg, theta_deg, eta=1.0):
    """Calculate spherical jittering with pole handling."""
    lon0 = np.radians(lon0_deg)
    lat0 = np.radians(lat0_deg)
    half_theta = np.radians(eta * theta_deg / 2)
    
    # Horizontal jittering (great circle)
    sin_lat_h = np.clip(np.sin(lat0) * np.cos(half_theta), -1, 1)
    lat_h = np.arcsin(sin_lat_h)
    delta_lon = np.arctan2(np.sin(half_theta), np.cos(lat0) * np.cos(half_theta))
    
    h_left = (normalize_longitude(lon0_deg - np.degrees(delta_lon)), np.degrees(lat_h))
    h_right = (normalize_longitude(lon0_deg + np.degrees(delta_lon)), np.degrees(lat_h))
    
    # Vertical jittering with pole wrapping
    raw_lat_up = lat0_deg + np.degrees(half_theta)
    raw_lat_down = lat0_deg - np.degrees(half_theta)
    
    # North Pole crossing
    if raw_lat_up > 90.0:
        lat_up = 180.0 - raw_lat_up
        lon_up = normalize_longitude(lon0_deg + 180.0)
    elif raw_lat_up < -90.0:
        lat_up = -180.0 - raw_lat_up
        lon_up = normalize_longitude(lon0_deg + 180.0)
    else:
        lat_up = raw_lat_up
        lon_up = lon0_deg
        
    # South Pole crossing
    if raw_lat_down < -90.0:
        lat_down = -180.0 - raw_lat_down
        lon_down = normalize_longitude(lon0_deg + 180.0)
    elif raw_lat_down > 90.0:
        lat_down = 180.0 - raw_lat_down
        lon_down = normalize_longitude(lon0_deg + 180.0)
    else:
        lat_down = raw_lat_down
        lon_down = lon0_deg
    
    return {
        'center': (lon0_deg, lat0_deg),
        'h_left': h_left,
        'h_right': h_right,
        'v_up': (lon_up, lat_up),
        'v_down': (lon_down, lat_down)
    }

def latlon_to_erp(lon, lat, width=1024, height=512):
    """Convert lat/lon to ERP pixel coordinates."""
    lat = np.clip(lat, -90, 90)
    lon = normalize_longitude(lon)
    x = (lon + 180.0) / 360.0 * width
    y = (90.0 - lat) / 180.0 * height
    return x, y

def crosses_pole(p1, p2):
    """检测是否跨越极点（经度差约180°且纬度同号）"""
    lon1, lat1 = p1
    lon2, lat2 = p2
    lon_diff = abs(normalize_longitude(lon2 - lon1))
    return (lon_diff > 179) and (lat1 * lat2 > 0)

def add_arrow_at_midpoint(ax, x_coords, y_coords, color, arrow_size=12):
    """
    在路径中间位置添加方向箭头
    x_coords, y_coords: 路径的坐标数组
    """
    if len(x_coords) < 2:
        return
    
    # 计算累计距离找到真正的中点
    distances = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
    total_dist = np.sum(distances)
    if total_dist < 1e-6:
        return
    
    cum_dist = np.cumsum(distances)
    mid_dist = total_dist / 2
    
    # 找到中点所在的线段
    mid_idx = np.searchsorted(cum_dist, mid_dist)
    if mid_idx >= len(x_coords) - 1:
        mid_idx = len(x_coords) - 2
    
    # 计算该线段内的插值位置
    if mid_idx == 0:
        t = mid_dist / cum_dist[0] if cum_dist[0] > 0 else 0.5
    else:
        segment_dist = cum_dist[mid_idx] - cum_dist[mid_idx-1]
        dist_into_segment = mid_dist - cum_dist[mid_idx-1]
        t = dist_into_segment / segment_dist if segment_dist > 0 else 0.5
    
    # 中点坐标
    x_mid = x_coords[mid_idx] + t * (x_coords[mid_idx + 1] - x_coords[mid_idx])
    y_mid = y_coords[mid_idx] + t * (y_coords[mid_idx + 1] - y_coords[mid_idx])
    
    # 方向向量（从中点指向下一个点）
    dx = x_coords[mid_idx + 1] - x_coords[mid_idx]
    dy = y_coords[mid_idx + 1] - y_coords[mid_idx]
    
    # 归一化
    norm = np.sqrt(dx**2 + dy**2)
    if norm < 1e-6:
        return
    
    # 箭头长度
    arrow_len = min(arrow_size, norm * 0.8)
    dx = dx / norm * arrow_len
    dy = dy / norm * arrow_len
    
    # 绘制箭头
    ax.annotate('', xy=(x_mid + dx/2, y_mid + dy/2), xytext=(x_mid - dx/2, y_mid - dy/2),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=2.5, 
                               mutation_scale=15, alpha=0.9),
                zorder=6)

def draw_great_circle_erp(ax, p1, p2, width=1024, height=512, color='blue', 
                          linestyle='--', linewidth=2, n_points=100):
    """绘制大圆路径，带中间箭头"""
    lon1, lat1 = np.radians(p1[0]), np.radians(p1[1])
    lon2, lat2 = np.radians(p2[0]), np.radians(p2[1])
    
    # 检查是否跨越极点
    if crosses_pole(p1, p2):
        pole_lat = 90 if p1[1] > 0 else -90
        # 分段绘制
        draw_great_circle_erp(ax, p1, (p1[0], pole_lat), width, height, 
                             color, linestyle, linewidth, n_points//2)
        draw_great_circle_erp(ax, (p2[0], pole_lat), p2, width, height, 
                             color, linestyle, linewidth, n_points//2)
        return
    
    # 3D插值计算大圆
    x1 = np.cos(lat1) * np.cos(lon1)
    y1 = np.cos(lat1) * np.sin(lon1)
    z1 = np.sin(lat1)
    x2 = np.cos(lat2) * np.cos(lon2)
    y2 = np.cos(lat2) * np.sin(lon2)
    z2 = np.sin(lat2)
    
    cos_omega = np.clip(x1*x2 + y1*y2 + z1*z2, -1, 1)
    omega = np.arccos(cos_omega)
    if omega < 0.001:
        return
        
    t_vals = np.linspace(0, 1, n_points)
    sin_omega = np.sin(omega)
    
    xs, ys = [], []
    for t in t_vals:
        coeff1 = np.sin((1-t)*omega) / sin_omega
        coeff2 = np.sin(t*omega) / sin_omega
        xi = coeff1*x1 + coeff2*x2
        yi = coeff1*y1 + coeff2*y2
        zi = coeff1*z1 + coeff2*z2
        norm = np.sqrt(xi**2 + yi**2 + zi**2)
        lat = np.degrees(np.arcsin(zi/norm))
        lon = np.degrees(np.arctan2(yi, xi))
        x, y = latlon_to_erp(lon, lat, width, height)
        xs.append(x)
        ys.append(y)
    
    xs, ys = np.array(xs), np.array(ys)
    
    # 检测经度边界跨越并分割绘制
    segments = []
    start_idx = 0
    for i in range(len(xs) - 1):
        if abs(xs[i+1] - xs[i]) > width / 2:
            segments.append((xs[start_idx:i+1], ys[start_idx:i+1]))
            start_idx = i + 1
    if start_idx < len(xs):
        segments.append((xs[start_idx:], ys[start_idx:]))
    
    # 绘制每个段并在中间加箭头
    for seg_x, seg_y in segments:
        if len(seg_x) > 1:
            ax.plot(seg_x, seg_y, color=color, linestyle=linestyle, 
                   linewidth=linewidth, alpha=0.8, zorder=5)
            add_arrow_at_midpoint(ax, seg_x, seg_y, color)

def draw_meridian_erp(ax, p1, p2, width=1024, height=512, color='red', 
                      linewidth=2, alpha=0.6):
    """绘制经线，带中间箭头，正确处理极点跨越和经度边界"""
    lon1, lat1 = p1
    lon2, lat2 = p2
    
    # 检查是否跨越极点
    if crosses_pole(p1, p2):
        pole_lat = 90 if lat1 > 0 else -90
        y_boundary = 0 if pole_lat == 90 else height
        
        x1, y1 = latlon_to_erp(lon1, lat1, width, height)
        x2, y2 = latlon_to_erp(lon2, lat2, width, height)
        x_pole1, _ = latlon_to_erp(lon1, pole_lat, width, height)
        x_pole2, _ = latlon_to_erp(lon2, pole_lat, width, height)
        
        # 绘制两段并各自在中间加箭头
        seg1_x, seg1_y = np.array([x1, x_pole1]), np.array([y1, y_boundary])
        ax.plot(seg1_x, seg1_y, color=color, linewidth=linewidth, alpha=alpha, zorder=4)
        add_arrow_at_midpoint(ax, seg1_x, seg1_y, color)
        
        seg2_x, seg2_y = np.array([x_pole2, x2]), np.array([y_boundary, y2])
        ax.plot(seg2_x, seg2_y, color=color, linewidth=linewidth, alpha=alpha, zorder=4)
        add_arrow_at_midpoint(ax, seg2_x, seg2_y, color)
        return
    
    x1, y1 = latlon_to_erp(lon1, lat1, width, height)
    x2, y2 = latlon_to_erp(lon2, lat2, width, height)
    
    # 检查经度边界跨越
    if abs(x2 - x1) > width / 2:
        # 跨越±180°经线，计算中间纬度
        if x1 > x2:  # 从右向左跨
            lat_mid = lat1 + (lat2 - lat1) * (width - x1) / ((width - x1) + x2)
            
            # 第一段
            seg1_x, seg1_y = np.array([x1, width]), np.array([y1, lat_mid])
            ax.plot(seg1_x, seg1_y, color=color, linewidth=linewidth, alpha=alpha, zorder=4)
            add_arrow_at_midpoint(ax, seg1_x, seg1_y, color)
            
            # 第二段
            seg2_x, seg2_y = np.array([0, x2]), np.array([lat_mid, y2])
            ax.plot(seg2_x, seg2_y, color=color, linewidth=linewidth, alpha=alpha, zorder=4)
            add_arrow_at_midpoint(ax, seg2_x, seg2_y, color)
        else:  # 从左向右跨
            lat_mid = lat1 + (lat2 - lat1) * x1 / (x1 + (width - x2))
            
            seg1_x, seg1_y = np.array([x1, 0]), np.array([y1, lat_mid])
            ax.plot(seg1_x, seg1_y, color=color, linewidth=linewidth, alpha=alpha, zorder=4)
            add_arrow_at_midpoint(ax, seg1_x, seg1_y, color)
            
            seg2_x, seg2_y = np.array([width, x2]), np.array([lat_mid, y2])
            ax.plot(seg2_x, seg2_y, color=color, linewidth=linewidth, alpha=alpha, zorder=4)
            add_arrow_at_midpoint(ax, seg2_x, seg2_y, color)
    else:
        # 普通情况
        seg_x, seg_y = np.array([x1, x2]), np.array([y1, y2])
        ax.plot(seg_x, seg_y, color=color, linewidth=linewidth, alpha=alpha, zorder=4)
        add_arrow_at_midpoint(ax, seg_x, seg_y, color)

def main():
    plt.rcParams['pdf.fonttype'] = 42
    fig, ax = plt.subplots(figsize=(14, 7))
    
    locations = [
        (-140, -70, 'blue', '-60°S'),
        (-45, -40, 'green', '-30°S'),
        (45, 0, 'red', '0° Equator'),
        (135, 85, 'purple', '85°N')
    ]
    
    # ERP网格
    for lon in range(-180, 181, 30):
        x = (lon + 180) / 360 * ERP_WIDTH
        ax.axvline(x, color='lightgray', alpha=0.3, linewidth=0.5)
    for lat in range(-90, 91, 30):
        y = (90 - lat) / 180 * ERP_HEIGHT
        ax.axhline(y, color='lightgray', alpha=0.3, linewidth=0.5)
    
    # 绘制每个位置
    for lon0, lat0, color, label in locations:
        jitter = spherical_jittering(lon0, lat0, JITTER_ANGLE, ETA)
        
        # 中心点
        cx, cy = latlon_to_erp(lon0, lat0, ERP_WIDTH, ERP_HEIGHT)
        ax.scatter([cx], [cy], c=color, s=300, marker='o', 
                  edgecolors='black', linewidths=2, zorder=10)
        ax.text(cx+35, cy+25, label, ha='center', fontsize=10, 
               fontweight='bold', color=color)
        
        # 水平抖动（大圆）- 带中间箭头
        for key in ['h_left', 'h_right']:
            lon_j, lat_j = jitter[key]
            xj, yj = latlon_to_erp(lon_j, lat_j, ERP_WIDTH, ERP_HEIGHT)
            ax.scatter([xj], [yj], c=color, s=200, marker='s', 
                      edgecolors='black', linewidths=1.5, zorder=9, alpha=0.9)
            draw_great_circle_erp(ax, (lon0, lat0), (lon_j, lat_j), 
                                 ERP_WIDTH, ERP_HEIGHT, color=color)
        
        # 垂直抖动（经线）- 带中间箭头
        for key in ['v_up', 'v_down']:
            lon_j, lat_j = jitter[key]
            xj, yj = latlon_to_erp(lon_j, lat_j, ERP_WIDTH, ERP_HEIGHT)
            ax.scatter([xj], [yj], c=color, s=200, marker='^', 
                      edgecolors='black', linewidths=1.5, zorder=9, alpha=0.9)
            draw_meridian_erp(ax, (lon0, lat0), (lon_j, lat_j), 
                             ERP_WIDTH, ERP_HEIGHT, color=color)
    
    # 坐标轴设置
    ax.set_xlim(0, ERP_WIDTH)
    ax.set_ylim(ERP_HEIGHT, 0)
    
    x_ticks = np.arange(-180, 181, 60)
    ax.set_xticks([(lon + 180)/360*ERP_WIDTH for lon in x_ticks])
    ax.set_xticklabels([f'{lon}°' for lon in x_ticks], fontsize=10)
    
    y_ticks = np.arange(-90, 91, 30)
    ax.set_yticks([(90 - lat)/180*ERP_HEIGHT for lat in y_ticks])
    ax.set_yticklabels([f'{lat}°' for lat in y_ticks], fontsize=10)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # 图例 - 放在图像内部右侧中间
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=12, label='Center', markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
               markersize=12, label='Horizontal', markeredgecolor='black'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', 
               markersize=12, label='Vertical', markeredgecolor='black'),
        Line2D([0], [0], color='gray', linestyle='--', label='Great Circle'),
        Line2D([0], [0], color='gray', linestyle='-', label='Meridian')
    ]
    
    # 放在右侧中间，带半透明背景
    legend = ax.legend(handles=legend_elements, loc='center right', 
                      bbox_to_anchor=(0.98, 0.5), fontsize=10, 
                      framealpha=0.9, edgecolor='gray')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, "spherical_jittering_arrows.pdf")
    plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved to: {output_file}")
    plt.show()

if __name__ == "__main__":
    main()