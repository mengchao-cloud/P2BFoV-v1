#!/usr/bin/env python3
"""
从JSON标注文件中读取最大的风扇，生成切平面无畸变图
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
    
    print(f"找到图像: {image_filename} (id: {img_id})")
    print(f"该图像有 {len(annotations)} 个BFoV标注")
    
    return target_image, annotations

def project_bfov_to_plane(img, bfov_params):
    lon, lat, fov_u, fov_v = bfov_params
    PI = np.pi

    sphere_circumference = ERP_WIDTH
    
    plane_width = int(sphere_circumference * (fov_u / (2 * PI)))
    aspect_ratio = fov_v / fov_u
    plane_height = int(plane_width * aspect_ratio)
    
    plane_width = max(16, plane_width)
    plane_height = max(16, plane_height)

    print(f"切平面分辨率: {plane_width} x {plane_height}")

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
    
    fan_annotations = [ann for ann in annotations if ann['category_id'] == TARGET_CATEGORY_ID]
    
    if not fan_annotations:
        print(f"未找到类别ID为 {TARGET_CATEGORY_ID} (fan) 的标注")
        return
    
    print(f"找到 {len(fan_annotations)} 个风扇标注")
    
    max_area = 0
    largest_fan = None
    for ann in fan_annotations:
        theta, phi, fov_x, fov_y = ann['bfov']
        area = fov_x * fov_y
        if area > max_area:
            max_area = area
            largest_fan = ann
    
    print(f"最大的风扇面积: {max_area:.6f} rad² ({max_area * (180/np.pi)**2:.2f} deg²)")
    
    print(f"正在读取图像: {IMAGE_PATH}")
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise ValueError(f"无法读取图像: {IMAGE_PATH}")
    
    img_h, img_w = img.shape[:2]
    print(f"图像尺寸: {img_w} x {img_h}")
    
    theta, phi, fov_x, fov_y = largest_fan['bfov']
    
    print(f"\n风扇BFoV参数:")
    print(f"  theta: {theta:.6f} rad ({theta * 180/np.pi:.2f}°)")
    print(f"  phi: {phi:.6f} rad ({phi * 180/np.pi:.2f}°)")
    print(f"  fov_x: {fov_x:.6f} rad ({fov_x * 180/np.pi:.2f}°)")
    print(f"  fov_y: {fov_y:.6f} rad ({fov_y * 180/np.pi:.2f}°)")
    
    print("\n正在生成切平面无畸变图...")
    plane_img = project_bfov_to_plane(img, [theta, phi, fov_x, fov_y])
    
    output_path = os.path.join(OUTPUT_DIR, 'plane_fan_' + IMAGE_FILENAME)
    cv2.imwrite(output_path, plane_img)
    print(f"\n切平面图已保存到: {output_path}")
    print(f"切平面尺寸: {plane_img.shape[1]} x {plane_img.shape[0]}")
    
    plane_img_with_grid = plane_img.copy()
    plane_h, plane_w = plane_img.shape[:2]
    
    grid_color = (255, 0, 0)
    grid_thickness = 1
    point_color = (0, 255, 0)
    point_radius = 2
    
    for i in range(8):
        x = int(i * plane_w / 7)
        cv2.line(plane_img_with_grid, (x, 0), (x, plane_h), grid_color, grid_thickness)
    
    for i in range(8):
        y = int(i * plane_h / 7)
        cv2.line(plane_img_with_grid, (0, y), (plane_w, y), grid_color, grid_thickness)
    
    cell_w = plane_w / 7
    cell_h = plane_h / 7
    for i in range(7):
        for j in range(7):
            center_x = int(j * cell_w + cell_w / 2)
            center_y = int(i * cell_h + cell_h / 2)
            cv2.circle(plane_img_with_grid, (center_x, center_y), point_radius, point_color, -1)
    
    center_x = plane_w // 2
    center_y = plane_h // 2
    axis_color = (0, 0, 255)
    
    arrow_len = int(min(plane_w, plane_h) * 0.20)
    
    end_x = center_x + arrow_len
    end_y = center_y
    
    cv2.line(plane_img_with_grid, (center_x, center_y), (end_x, end_y), axis_color, 1)
    arrow_head_size = 6
    triangle_points = np.array([
        [end_x, end_y],
        [end_x - arrow_head_size, end_y - arrow_head_size // 2],
        [end_x - arrow_head_size, end_y + arrow_head_size // 2]
    ], np.int32)
    cv2.fillConvexPoly(plane_img_with_grid, triangle_points, axis_color)
    
    end_x = center_x
    end_y = center_y - arrow_len
    
    cv2.line(plane_img_with_grid, (center_x, center_y), (end_x, end_y), axis_color, 1)
    triangle_points = np.array([
        [end_x, end_y],
        [end_x - arrow_head_size // 2, end_y + arrow_head_size],
        [end_x + arrow_head_size // 2, end_y + arrow_head_size]
    ], np.int32)
    cv2.fillConvexPoly(plane_img_with_grid, triangle_points, axis_color)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    cv2.putText(plane_img_with_grid, 'e_u', 
                 (center_x + arrow_len + 3, center_y + 12), 
                 font, font_scale, axis_color, font_thickness)
    
    cv2.putText(plane_img_with_grid, 'e_v', 
                 (center_x + 8, center_y - arrow_len - 3), 
                 font, font_scale, axis_color, font_thickness)
    
    output_path_grid = os.path.join(OUTPUT_DIR, 'plane_fan_grid_' + IMAGE_FILENAME)
    cv2.imwrite(output_path_grid, plane_img_with_grid)
    print(f"带网格的切平面图已保存到: {output_path_grid}")
    
    return largest_fan

if __name__ == '__main__':
    main()
