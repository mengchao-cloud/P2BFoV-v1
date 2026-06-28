#!/usr/bin/env python3
"""
生成沙发切平面图（学术风格）
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import cv2
import os

# 设置英文学术风格
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

# ====== 配置 ======
ERP_WIDTH = 1920
ERP_HEIGHT = 960
IMAGE_FILENAME = '29803495916_be313c45bb_o.jpg'
DATA_ROOT = '/home/mengchao/workspace/P2BFoV/P2BNet-main/360indoor/'
ANN_FILE = DATA_ROOT + 'ann/train_coco.json'
IMAGE_PATH = DATA_ROOT + 'images/' + IMAGE_FILENAME
OUTPUT_DIR = '/home/mengchao/workspace/P2BFoV/P2BNet-main/TOV_mmdetection/bfov/eswa_frame/sphroi/'

# 只绘制沙发（category_id: 18）
TARGET_CATEGORY_ID = 18

def load_annotations_from_json(json_path, image_filename):
    """从JSON文件加载指定图像的标注"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    target_image = None
    for img in data['images']:
        if img['file_name'] == image_filename:
            target_image = img
            break
    
    if not target_image:
        raise ValueError(f"未找到图像: {image_filename}")
    
    img_id = target_image['id']
    annotations = [ann for ann in data['annotations'] if ann['image_id'] == img_id and 'bfov' in ann]
    
    return target_image, annotations

def project_bfov_to_plane(img, bfov_params):
    """将BFOV投影成切平面无畸变图"""
    lon, lat, fov_u, fov_v = bfov_params
    PI = np.pi

    sphere_circumference = ERP_WIDTH
    plane_width = int(sphere_circumference * (fov_u / (2 * PI)))
    aspect_ratio = fov_v / fov_u
    plane_height = int(plane_width * aspect_ratio)
    
    plane_width = max(16, plane_width)
    plane_height = max(16, plane_height)

    phi0 = lon
    theta0 = PI/2 - lat

    c_x = np.sin(theta0) * np.cos(phi0)
    c_y = np.sin(theta0) * np.sin(phi0)
    c_z = np.cos(theta0)
    c = np.array([c_x, c_y, c_z])
    c = c / np.linalg.norm(c)

    sin_phi = np.sin(phi0)
    cos_phi = np.cos(phi0)
    sin_theta = np.sin(theta0)
    cos_theta = np.cos(theta0)

    lon_dir = np.array([-sin_phi, cos_phi, 0])
    lon_dir = lon_dir / np.linalg.norm(lon_dir)

    lat_dir = np.array([-cos_theta*cos_phi, -cos_theta*sin_phi, sin_theta])
    lat_dir = lat_dir / np.linalg.norm(lat_dir)

    dot_product = np.dot(lon_dir, lat_dir)
    if abs(dot_product) > 1e-6:
        lon_dir_ortho = lon_dir - np.dot(lon_dir, c) * c
        lon_dir_ortho = lon_dir_ortho / np.linalg.norm(lon_dir_ortho)
        lat_dir_ortho = np.cross(c, lon_dir_ortho)
        lat_dir_ortho = lat_dir_ortho / np.linalg.norm(lat_dir_ortho)
        lon_dir, lat_dir = lon_dir_ortho, lat_dir_ortho

    max_angle = PI/2 - 1e-4
    fov_u_half = np.clip(fov_u / 2.0, -max_angle, max_angle)
    fov_v_half = np.clip(fov_v / 2.0, -max_angle, max_angle)

    U = np.tan(fov_u_half)
    V = np.tan(fov_v_half)

    delta_u = 2 * U / plane_width
    delta_v = 2 * V / plane_height

    j_grid, i_grid = np.meshgrid(np.arange(plane_width), np.arange(plane_height))

    j_plus_half = j_grid + 0.5
    vertical_idx = (plane_height - 1 - i_grid) + 0.5

    u_center = -U + j_plus_half * delta_u
    v_center = -V + vertical_idx * delta_v

    c_expand = np.tile(c, (plane_height, plane_width, 1))
    e_u_expand = np.tile(lon_dir, (plane_height, plane_width, 1))
    e_v_expand = np.tile(lat_dir, (plane_height, plane_width, 1))

    u_center_expand = np.expand_dims(u_center, axis=2)
    v_center_expand = np.expand_dims(v_center, axis=2)
    x_plane = c_expand + u_center_expand * e_u_expand + v_center_expand * e_v_expand

    norm_x_plane = np.linalg.norm(x_plane, axis=2, keepdims=True)
    x_sphere = x_plane / norm_x_plane

    x = x_sphere[..., 0]
    y = x_sphere[..., 1]
    z = x_sphere[..., 2]

    z_clamped = np.clip(z, -1.0 + 1e-6, 1.0 - 1e-6)

    theta = np.arccos(z_clamped)
    phi = np.arctan2(y, x)

    lat_deg = 90.0 - (theta * 180.0 / PI)
    lon_deg = phi * 180.0 / PI

    erp_x = (lon_deg + 180.0) / 360.0 * ERP_WIDTH
    erp_y = (90.0 - lat_deg) / 180.0 * ERP_HEIGHT

    erp_x = np.clip(erp_x, 0, ERP_WIDTH - 1)
    erp_y = np.clip(erp_y, 0, ERP_HEIGHT - 1)

    plane_img = np.zeros((plane_height, plane_width, 3), dtype=np.uint8)
    
    for i in range(plane_height):
        for j in range(plane_width):
            px = int(erp_x[i, j])
            py = int(erp_y[i, j])
            plane_img[i, j] = img[py, px]

    return plane_img

def main():
    print(f"正在加载标注文件: {ANN_FILE}")
    target_image, annotations = load_annotations_from_json(ANN_FILE, IMAGE_FILENAME)
    
    sofa_annotations = [ann for ann in annotations if ann['category_id'] == TARGET_CATEGORY_ID]
    
    if not sofa_annotations:
        print(f"未找到类别ID为 {TARGET_CATEGORY_ID} (sofa) 的标注")
        return
    
    max_area = 0
    largest_sofa = None
    for ann in sofa_annotations:
        theta, phi, fov_x, fov_y = ann['bfov']
        area = fov_x * fov_y
        if area > max_area:
            max_area = area
            largest_sofa = ann
    
    print(f"正在读取图像: {IMAGE_PATH}")
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise ValueError(f"无法读取图像: {IMAGE_PATH}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    theta, phi, fov_x, fov_y = largest_sofa['bfov']
    bfov_params = [theta, phi, fov_x, fov_y]
    
    print(f"\n正在生成切平面无畸变图...")
    plane_img = project_bfov_to_plane(img, bfov_params)
    plane_img_rgb = cv2.cvtColor(plane_img, cv2.COLOR_BGR2RGB)
    
    plane_h, plane_w = plane_img.shape[:2]
    
    # ====== 使用matplotlib绘制专业学术图 ======
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 显示切平面图像
    ax.imshow(plane_img_rgb, extent=[0, plane_w, 0, plane_h])
    
    # 绘制7x7网格
    for i in range(8):
        x = i * plane_w / 7
        ax.axvline(x=x, color='blue', linewidth=0.8, linestyle='-', alpha=0.8)
    for i in range(8):
        y = i * plane_h / 7
        ax.axhline(y=y, color='blue', linewidth=0.8, linestyle='-', alpha=0.8)
    
    # 绘制49个采样点
    cell_w = plane_w / 7
    cell_h = plane_h / 7
    for i in range(7):
        for j in range(7):
            center_x = j * cell_w + cell_w / 2
            center_y = i * cell_h + cell_h / 2
            ax.plot(center_x, center_y, 'go', markersize=4, markeredgecolor='darkgreen', markeredgewidth=0.5)
    
    # 绘制坐标轴（从中心出发）
    center_x = plane_w / 2
    center_y = plane_h / 2
    arrow_len = min(plane_w, plane_h) * 0.25
    
    # e_u轴（水平向右）
    ax.annotate('', xy=(center_x + arrow_len, center_y), xytext=(center_x, center_y),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
    
    # e_v轴（垂直向上）
    ax.annotate('', xy=(center_x, center_y + arrow_len), xytext=(center_x, center_y),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
    
    # 坐标轴标签
    ax.text(center_x + arrow_len + 5, center_y + 3, r'$\mathbf{e}_u$', 
            fontsize=16, fontweight='bold', color='red', va='center')
    ax.text(center_x - 12, center_y + arrow_len + 8, r'$\mathbf{e}_v$', 
            fontsize=16, fontweight='bold', color='red', va='center')
    
    # 原点标记
    ax.plot(center_x, center_y, 'r.', markersize=10)
    
    # 设置坐标轴
    ax.set_xlim(-5, plane_w + 5)
    ax.set_ylim(-5, plane_h + 5)
    ax.set_xlabel('u (pixels)', fontsize=12)
    ax.set_ylabel('v (pixels)', fontsize=12)
    ax.set_title('Tangent Plane with Sampling Grid and Coordinate System', fontsize=14, fontweight='bold')
    
    # 添加图例
    grid_patch = mpatches.Patch(color='blue', label='7×7 Sampling Grid')
    point_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                             markersize=8, label='Sampling Points')
    arrow_eu = plt.Line2D([0], [0], color='red', linewidth=2, label=r'$\mathbf{e}_u$ axis')
    arrow_ev = plt.Line2D([0], [0], color='red', linewidth=2, label=r'$\mathbf{e}_v$ axis')
    ax.legend(handles=[grid_patch, point_patch, arrow_eu, arrow_ev], 
              loc='upper right', fontsize=10, framealpha=0.9)
    
    # 添加边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(OUTPUT_DIR, 'plane_sofa_grid_' + IMAGE_FILENAME.replace('.jpg', '.png'))
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n切平面图已保存到: {output_path}")
    print(f"切平面尺寸: {plane_w} x {plane_h}")

if __name__ == '__main__':
    main()
