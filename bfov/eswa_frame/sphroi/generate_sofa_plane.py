#!/usr/bin/env python3
"""
从JSON标注文件中读取最大的沙发，生成切平面无畸变图
"""

import json
import numpy as np
import cv2
import os

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
    
    # 查找目标图像
    target_image = None
    for img in data['images']:
        if img['file_name'] == image_filename:
            target_image = img
            break
    
    if not target_image:
        raise ValueError(f"未找到图像: {image_filename}")
    
    img_id = target_image['id']
    # 筛选出有bfov字段的标注
    annotations = [ann for ann in data['annotations'] if ann['image_id'] == img_id and 'bfov' in ann]
    
    print(f"找到图像: {image_filename} (id: {img_id})")
    print(f"该图像有 {len(annotations)} 个BFoV标注")
    
    return target_image, annotations

def project_bfov_to_plane(img, bfov_params):
    """
    将BFOV投影成切平面无畸变图
    基于球面坐标和局部正交基进行切平面投影

    参数:
    img: 输入ERP图像
    bfov_params: BFOV参数 [theta, phi, fov_x, fov_y]，弧度制

    返回:
    plane_img: 切平面无畸变图
    """
    # 提取BFOV参数（弧度制）
    lon, lat, fov_u, fov_v = bfov_params
    PI = np.pi

    # 球的周长等于原始图像宽度
    sphere_circumference = ERP_WIDTH
    
    # 根据FOV角度计算切平面分辨率
    # 水平分辨率：基于水平FOV占球周长的比例
    plane_width = int(sphere_circumference * (fov_u / (2 * PI)))
    # 垂直分辨率：基于垂直FOV与水平FOV的比例
    aspect_ratio = fov_v / fov_u
    plane_height = int(plane_width * aspect_ratio)
    
    # 确保分辨率合理
    plane_width = max(16, plane_width)
    plane_height = max(16, plane_height)

    print(f"切平面分辨率: {plane_width} x {plane_height}")

    # 转换经度/纬度为极角/方位角
    phi0 = lon  # 方位角
    theta0 = PI/2 - lat  # 极角

    # 1. 计算BFOV中心单位向量c (x,y,z)
    c_x = np.sin(theta0) * np.cos(phi0)
    c_y = np.sin(theta0) * np.sin(phi0)
    c_z = np.cos(theta0)
    c = np.array([c_x, c_y, c_z])
    
    # 归一化确保单位向量精度
    c = c / np.linalg.norm(c)

    # 2. 构造局部正交基 (e_u, e_v)
    sin_phi = np.sin(phi0)
    cos_phi = np.cos(phi0)
    sin_theta = np.sin(theta0)
    cos_theta = np.cos(theta0)

    # 经度方向向量（水平）
    lon_dir = np.array([-sin_phi, cos_phi, 0])
    lon_dir = lon_dir / np.linalg.norm(lon_dir)

    # 纬度方向向量（垂直）
    lat_dir = np.array([-cos_theta*cos_phi, -cos_theta*sin_phi, sin_theta])
    lat_dir = lat_dir / np.linalg.norm(lat_dir)

    # 确保正交性
    dot_product = np.dot(lon_dir, lat_dir)
    if abs(dot_product) > 1e-6:
        # 重新计算正交基
        lon_dir_ortho = lon_dir - np.dot(lon_dir, c) * c
        lon_dir_ortho = lon_dir_ortho / np.linalg.norm(lon_dir_ortho)
        lat_dir_ortho = np.cross(c, lon_dir_ortho)
        lat_dir_ortho = lat_dir_ortho / np.linalg.norm(lat_dir_ortho)
        lon_dir, lat_dir = lon_dir_ortho, lat_dir_ortho

    # 3. 计算切平面半宽 (U, V)
    max_angle = PI/2 - 1e-4  # 接近但小于π/2的值
    fov_u_half = np.clip(fov_u / 2.0, -max_angle, max_angle)
    fov_v_half = np.clip(fov_v / 2.0, -max_angle, max_angle)

    U = np.tan(fov_u_half)  # 水平半宽
    V = np.tan(fov_v_half)  # 垂直半宽

    # 4. 计算网格步长
    delta_u = 2 * U / plane_width  # 水平步长
    delta_v = 2 * V / plane_height  # 垂直步长

    # 5. 生成网格坐标
    j_grid, i_grid = np.meshgrid(np.arange(plane_width), np.arange(plane_height))

    # 6. 计算u_center和v_center
    # 计算中心坐标
    j_plus_half = j_grid + 0.5  # 水平方向网格中心
    vertical_idx = (plane_height - 1 - i_grid) + 0.5  # 垂直方向网格中心（已反转）

    # 水平方向(u): 从左到右
    u_center = -U + j_plus_half * delta_u
    # 垂直方向(v): 从上到下
    v_center = -V + vertical_idx * delta_v

    # 7. 计算切平面上的点
    # 扩展基向量到网格大小
    c_expand = np.tile(c, (plane_height, plane_width, 1))
    e_u_expand = np.tile(lon_dir, (plane_height, plane_width, 1))
    e_v_expand = np.tile(lat_dir, (plane_height, plane_width, 1))

    # 计算切平面上的点
    u_center_expand = np.expand_dims(u_center, axis=2)
    v_center_expand = np.expand_dims(v_center, axis=2)
    x_plane = c_expand + u_center_expand * e_u_expand + v_center_expand * e_v_expand

    # 8. 投影回单位球面
    norm_x_plane = np.linalg.norm(x_plane, axis=2, keepdims=True)
    x_sphere = x_plane / norm_x_plane

    # 9. 转换球面坐标到经纬度（度数）
    x = x_sphere[..., 0]
    y = x_sphere[..., 1]
    z = x_sphere[..., 2]

    # 添加数值稳定性处理
    z_clamped = np.clip(z, -1.0 + 1e-6, 1.0 - 1e-6)

    theta = np.arccos(z_clamped)  # 极角
    phi = np.arctan2(y, x)  # 方位角

    # 转换为经纬度（度数）
    lat_deg = 90.0 - (theta * 180.0 / PI)
    lon_deg = phi * 180.0 / PI

    # 10. 转换经纬度到ERP坐标
    erp_x = (lon_deg + 180.0) / 360.0 * ERP_WIDTH
    erp_y = (90.0 - lat_deg) / 180.0 * ERP_HEIGHT

    # 11. 确保坐标在图像范围内
    erp_x = np.clip(erp_x, 0, ERP_WIDTH - 1)
    erp_y = np.clip(erp_y, 0, ERP_HEIGHT - 1)

    # 12. 填充切平面图像
    plane_img = np.zeros((plane_height, plane_width, 3), dtype=np.uint8)
    
    # 使用双线性插值获取像素值（这里简化为最近邻）
    for i in range(plane_height):
        for j in range(plane_width):
            px = int(erp_x[i, j])
            py = int(erp_y[i, j])
            plane_img[i, j] = img[py, px]

    return plane_img

def main():
    print(f"正在加载标注文件: {ANN_FILE}")
    target_image, annotations = load_annotations_from_json(ANN_FILE, IMAGE_FILENAME)
    
    # 筛选出沙发的标注
    sofa_annotations = [ann for ann in annotations if ann['category_id'] == TARGET_CATEGORY_ID]
    
    if not sofa_annotations:
        print(f"未找到类别ID为 {TARGET_CATEGORY_ID} (sofa) 的标注")
        return
    
    print(f"找到 {len(sofa_annotations)} 个沙发标注")
    
    # 找到面积最大的沙发
    max_area = 0
    largest_sofa = None
    for ann in sofa_annotations:
        theta, phi, fov_x, fov_y = ann['bfov']
        area = fov_x * fov_y
        if area > max_area:
            max_area = area
            largest_sofa = ann
    
    print(f"最大的沙发面积: {max_area:.6f} rad² ({max_area * (180/np.pi)**2:.2f} deg²)")
    
    print(f"正在读取图像: {IMAGE_PATH}")
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise ValueError(f"无法读取图像: {IMAGE_PATH}")
    
    img_h, img_w = img.shape[:2]
    print(f"图像尺寸: {img_w} x {img_h}")
    
    # 提取BFoV参数
    theta, phi, fov_x, fov_y = largest_sofa['bfov']
    
    print(f"\n沙发BFoV参数:")
    print(f"  theta: {theta:.6f} rad ({theta * 180/np.pi:.2f}°)")
    print(f"  phi: {phi:.6f} rad ({phi * 180/np.pi:.2f}°)")
    print(f"  fov_x: {fov_x:.6f} rad ({fov_x * 180/np.pi:.2f}°)")
    print(f"  fov_y: {fov_y:.6f} rad ({fov_y * 180/np.pi:.2f}°)")
    
    print("\n正在生成切平面无畸变图...")
    plane_img = project_bfov_to_plane(img, [theta, phi, fov_x, fov_y])
    
    # 保存正常切平面图
    output_path = os.path.join(OUTPUT_DIR, 'plane_sofa_' + IMAGE_FILENAME)
    cv2.imwrite(output_path, plane_img)
    print(f"\n切平面图已保存到: {output_path}")
    print(f"切平面尺寸: {plane_img.shape[1]} x {plane_img.shape[0]}")
    
    # 生成带有7x7网格的切平面图
    plane_img_with_grid = plane_img.copy()
    plane_h, plane_w = plane_img.shape[:2]
    
    # 绘制7x7网格线
    grid_color = (255, 0, 0)  # 蓝色
    grid_thickness = 1  # 细线
    point_color = (0, 255, 0)  # 绿色点
    point_radius = 2
    
    # 垂直线（8条，分为7列）
    for i in range(8):
        x = int(i * plane_w / 7)
        cv2.line(plane_img_with_grid, (x, 0), (x, plane_h), grid_color, grid_thickness)
    
    # 水平线（8条，分为7行）
    for i in range(8):
        y = int(i * plane_h / 7)
        cv2.line(plane_img_with_grid, (0, y), (plane_w, y), grid_color, grid_thickness)
    
    # 在每个网格单元中心绘制点
    cell_w = plane_w / 7
    cell_h = plane_h / 7
    for i in range(7):
        for j in range(7):
            # 计算网格单元中心坐标
            center_x = int(j * cell_w + cell_w / 2)
            center_y = int(i * cell_h + cell_h / 2)
            cv2.circle(plane_img_with_grid, (center_x, center_y), point_radius, point_color, -1)
    
    # 绘制坐标轴 e_u 和 e_v（从切平面中心出发）
    center_x = plane_w // 2
    center_y = plane_h // 2
    axis_color = (0, 0, 255)  # 红色
    
    # 计算箭头长度（短一点，更协调）
    arrow_len = int(min(plane_w, plane_h) * 0.20)
    
    # e_u 方向（水平轴，正方向向右）
    end_x = center_x + arrow_len
    end_y = center_y
    
    # 绘制e_u箭头（细线条）
    cv2.line(plane_img_with_grid, (center_x, center_y), (end_x, end_y), axis_color, 1)
    # 绘制箭头头部
    arrow_head_size = 6
    triangle_points = np.array([
        [end_x, end_y],
        [end_x - arrow_head_size, end_y - arrow_head_size // 2],
        [end_x - arrow_head_size, end_y + arrow_head_size // 2]
    ], np.int32)
    cv2.fillConvexPoly(plane_img_with_grid, triangle_points, axis_color)
    
    # e_v 方向（垂直轴，正方向向上）
    end_x = center_x
    end_y = center_y - arrow_len
    
    # 绘制e_v箭头（细线条）
    cv2.line(plane_img_with_grid, (center_x, center_y), (end_x, end_y), axis_color, 1)
    # 绘制箭头头部
    triangle_points = np.array([
        [end_x, end_y],
        [end_x - arrow_head_size // 2, end_y + arrow_head_size],
        [end_x + arrow_head_size // 2, end_y + arrow_head_size]
    ], np.int32)
    cv2.fillConvexPoly(plane_img_with_grid, triangle_points, axis_color)
    
    # 添加坐标轴标签（简洁风格）
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    # e_u 标签
    cv2.putText(plane_img_with_grid, 'e_u', 
                 (center_x + arrow_len + 3, center_y + 12), 
                 font, font_scale, axis_color, font_thickness)
    
    # e_v 标签
    cv2.putText(plane_img_with_grid, 'e_v', 
                 (center_x + 8, center_y - arrow_len - 3), 
                 font, font_scale, axis_color, font_thickness)
    
    # 保存带网格的切平面图
    output_path_grid = os.path.join(OUTPUT_DIR, 'plane_sofa_grid_' + IMAGE_FILENAME)
    cv2.imwrite(output_path_grid, plane_img_with_grid)
    print(f"带网格的切平面图已保存到: {output_path_grid}")
    
    # 生成7x7采样组合图
    # 将切平面划分为7x7区域，从每个区域中心采集一小块像素，然后拼接成新图
    grid_size = 7
    cell_w = plane_w // grid_size
    cell_h = plane_h // grid_size
    
    # 从每个网格中心采集的小块大小（取每个网格的1/4大小）
    patch_size_w = cell_w // 2
    patch_size_h = cell_h // 2
    
    # 创建输出图像
    output_w = patch_size_w * grid_size
    output_h = patch_size_h * grid_size
    combined_img = np.zeros((output_h, output_w, 3), dtype=np.uint8)
    
    for i in range(grid_size):
        for j in range(grid_size):
            # 计算每个网格的中心位置
            cell_center_x = j * cell_w + cell_w // 2
            cell_center_y = i * cell_h + cell_h // 2
            
            # 计算采样区域
            patch_start_x = cell_center_x - patch_size_w // 2
            patch_start_y = cell_center_y - patch_size_h // 2
            patch_end_x = patch_start_x + patch_size_w
            patch_end_y = patch_start_y + patch_size_h
            
            # 确保在图像范围内
            patch_start_x = max(0, patch_start_x)
            patch_start_y = max(0, patch_start_y)
            patch_end_x = min(plane_w, patch_end_x)
            patch_end_y = min(plane_h, patch_end_y)
            
            # 提取小块
            patch = plane_img[patch_start_y:patch_end_y, patch_start_x:patch_end_x]
            
            # 计算在输出图像中的位置
            output_start_x = j * patch_size_w
            output_start_y = i * patch_size_h
            output_end_x = output_start_x + patch_size_w
            output_end_y = output_start_y + patch_size_h
            
            # 将小块复制到输出图像
            combined_img[output_start_y:output_end_y, output_start_x:output_end_x] = patch
    
    # 保存7x7采样组合图
    output_path_combined = os.path.join(OUTPUT_DIR, 'plane_sofa_7x7_patches_' + IMAGE_FILENAME)
    cv2.imwrite(output_path_combined, combined_img)
    print(f"7x7采样组合图已保存到: {output_path_combined}")

if __name__ == '__main__':
    main()