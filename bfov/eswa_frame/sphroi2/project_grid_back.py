#!/usr/bin/env python3
"""
将切平面上的网格和点投影回原ERP图像上（风扇）
"""

import json
import numpy as np
import cv2
import os

ERP_WIDTH = 1920
ERP_HEIGHT = 960
IMAGE_FILENAME = '7l0Hp.jpg'
DATA_ROOT = '/home/mengchao/workspace/P2BFoV/P2BNet-main/360indoor/'
ANN_FILE = DATA_ROOT + 'ann/train_coco.json'
IMAGE_PATH = DATA_ROOT + 'images/' + IMAGE_FILENAME
OUTPUT_DIR = '/home/mengchao/workspace/P2BFoV/P2BNet-main/TOV_mmdetection/bfov/eswa_frame/sphroi2/'

TARGET_CATEGORY_ID = 10

def load_annotations_from_json(json_path, image_filename):
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

def plane_point_to_erp(plane_x, plane_y, bfov_params, plane_width, plane_height):
    lon, lat, fov_u, fov_v = bfov_params
    PI = np.pi

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

    j_plus_half = plane_x + 0.5
    vertical_idx = (plane_height - 1 - plane_y) + 0.5
    
    u_center = -U + j_plus_half * delta_u
    v_center = -V + vertical_idx * delta_v

    x_plane = c + u_center * lon_dir + v_center * lat_dir

    norm_x_plane = np.linalg.norm(x_plane)
    x_sphere = x_plane / norm_x_plane

    x, y, z = x_sphere
    z_clamped = np.clip(z, -1.0 + 1e-6, 1.0 - 1e-6)

    theta = np.arccos(z_clamped)
    phi = np.arctan2(y, x)

    lat_deg = 90.0 - (theta * 180.0 / PI)
    lon_deg = phi * 180.0 / PI

    erp_x = (lon_deg + 180.0) / 360.0 * ERP_WIDTH
    erp_y = (90.0 - lat_deg) / 180.0 * ERP_HEIGHT

    return erp_x, erp_y

def main():
    print(f"正在加载标注文件: {ANN_FILE}")
    target_image, annotations = load_annotations_from_json(ANN_FILE, IMAGE_FILENAME)
    
    fan_annotations = [ann for ann in annotations if ann['category_id'] == TARGET_CATEGORY_ID]
    
    if not fan_annotations:
        print(f"未找到类别ID为 {TARGET_CATEGORY_ID} (fan) 的标注")
        return
    
    max_area = 0
    largest_fan = None
    for ann in fan_annotations:
        theta, phi, fov_x, fov_y = ann['bfov']
        area = fov_x * fov_y
        if area > max_area:
            max_area = area
            largest_fan = ann
    
    print(f"正在读取图像: {IMAGE_PATH}")
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise ValueError(f"无法读取图像: {IMAGE_PATH}")
    
    theta, phi, fov_x, fov_y = largest_fan['bfov']
    bfov_params = [theta, phi, fov_x, fov_y]
    
    sphere_circumference = ERP_WIDTH
    plane_width = int(sphere_circumference * (fov_x / (2 * np.pi)))
    aspect_ratio = fov_y / fov_x
    plane_height = int(plane_width * aspect_ratio)
    
    print(f"切平面分辨率: {plane_width} x {plane_height}")
    print(f"风扇BFoV: theta={theta:.4f}, phi={phi:.4f}, fov_x={fov_x:.4f}, fov_y={fov_y:.4f}")
    
    result_img = img.copy()
    
    grid_color = (255, 0, 0)
    grid_thickness = 2
    point_color = (0, 255, 0)
    point_radius = 3
    
    cell_w = plane_width / 7
    cell_h = plane_height / 7
    
    for i in range(7):
        for j in range(7):
            plane_x = int(j * cell_w + cell_w / 2)
            plane_y = int(i * cell_h + cell_h / 2)
            
            erp_x, erp_y = plane_point_to_erp(plane_x, plane_y, bfov_params, plane_width, plane_height)
            
            erp_x = max(0, min(ERP_WIDTH - 1, erp_x))
            erp_y = max(0, min(ERP_HEIGHT - 1, erp_y))
            
            cv2.circle(result_img, (int(erp_x), int(erp_y)), point_radius, point_color, -1)
    
    num_points_per_line = 50
    
    for i in range(8):
        plane_x = int(i * plane_width / 7)
        points = []
        for j in range(num_points_per_line):
            plane_y = int(j * (plane_height - 1) / (num_points_per_line - 1))
            erp_x, erp_y = plane_point_to_erp(plane_x, plane_y, bfov_params, plane_width, plane_height)
            erp_x = max(0, min(ERP_WIDTH - 1, erp_x))
            erp_y = max(0, min(ERP_HEIGHT - 1, erp_y))
            points.append((int(erp_x), int(erp_y)))
        
        for j in range(len(points) - 1):
            cv2.line(result_img, points[j], points[j+1], grid_color, grid_thickness)
    
    for i in range(8):
        plane_y = int(i * plane_height / 7)
        points = []
        for j in range(num_points_per_line):
            plane_x = int(j * (plane_width - 1) / (num_points_per_line - 1))
            erp_x, erp_y = plane_point_to_erp(plane_x, plane_y, bfov_params, plane_width, plane_height)
            erp_x = max(0, min(ERP_WIDTH - 1, erp_x))
            erp_y = max(0, min(ERP_HEIGHT - 1, erp_y))
            points.append((int(erp_x), int(erp_y)))
        
        for j in range(len(points) - 1):
            cv2.line(result_img, points[j], points[j+1], grid_color, grid_thickness)
    
    output_path = os.path.join(OUTPUT_DIR, 'erp_with_grid_' + IMAGE_FILENAME)
    cv2.imwrite(output_path, result_img)
    print(f"\n投影结果已保存到: {output_path}")

if __name__ == '__main__':
    main()
