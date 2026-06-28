#!/usr/bin/env python3
"""
从COCO格式的JSON标注文件中读取BFoV标注并绘制，同时生成抖动效果图

使用方法：
python draw_bfov_from_json.py
"""

import json
import numpy as np
import cv2
import os
import sys

# 添加sphdet路径到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../sphdet/visualizers'))
from ImageRecorder import ImageRecorder as BFoV

# 抖动参数（参考P2BFoV.py）
SHAKE_RATIO = 0.4  # 抖动比例
SCALE_FACTORS = [0.5, 1.2]  # 面积尺度因子
ASPECT_RATIOS = [0.6, 1.0, 1.3]  # fovx/fovy比例

# 类别名称映射（根据360indoor数据集标注文件）
CATEGORY_NAMES = {
    0: 'toilet',
    1: 'board',
    2: 'mirror',
    3: 'bed',
    4: 'potted plant',
    5: 'book',
    6: 'clock',
    7: 'phone',
    8: 'keyboard',
    9: 'tv',
    10: 'fan',
    11: 'backpack',
    12: 'light',
    13: 'refrigerator',
    14: 'bathtub',
    15: 'wine glass',
    16: 'airconditioner',
    17: 'cabinet',
    18: 'sofa',
    19: 'bowl',
    20: 'sink',
    21: 'computer',
    22: 'cup',
    23: 'bottle',
    24: 'washer',
    25: 'chair',
    26: 'picture',
    27: 'window',
    28: 'door',
    29: 'heater',
    30: 'fireplace',
    31: 'mouse',
    32: 'oven',
    33: 'microwave',
    34: 'person',
    35: 'vase',
    36: 'table',
}

# 类别颜色映射
CATEGORY_COLORS = {
    1: (0, 255, 0),     # board - 绿色
    4: (0, 255, 128),   # potted plant - 浅绿色
    5: (0, 128, 255),   # book - 天蓝色
    8: (255, 0, 0),     # keyboard - 红色
    9: (255, 128, 0),   # tv - 橙色
    12: (0, 0, 255),    # light - 蓝色
    13: (128, 0, 255),  # refrigerator - 紫色
    16: (255, 0, 128),  # airconditioner - 粉色
    17: (128, 255, 0),  # cabinet - 黄绿色
    18: (255, 255, 0),  # sofa - 黄色
    21: (0, 255, 255),  # computer - 青色
    25: (128, 0, 0),    # chair - 深红色
    27: (0, 128, 128),  # window - 蓝绿色
    28: (128, 128, 128),# door - 灰色
    34: (255, 0, 255),  # person - 品红色
    35: (0, 64, 128),   # vase - 深蓝色
    36: (64, 128, 64),  # table - 深绿色
}

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

def constrain_spherical_coords(coords):
    """
    约束球面坐标在有效范围内
    参考 P2BFoV.py 中的实现
    
    参数:
    coords: 坐标数组，形状 [..., 2]，其中[...,0]是经度(lambda), [...,1]是纬度(phi)
    
    返回:
    约束后的坐标
    """
    lambda_coords = coords[..., 0]
    phi_coords = coords[..., 1]
    
    # 约束纬度 phi: [-π/2, π/2]
    phi_coords = np.mod(phi_coords + np.pi/2, np.pi) - np.pi/2
    
    # 约束经度 lambda: [-π, π]
    lambda_coords = np.mod(lambda_coords + np.pi, 2*np.pi) - np.pi
    
    constrained_coords = coords.copy()
    constrained_coords[..., 0] = lambda_coords
    constrained_coords[..., 1] = phi_coords
    
    return constrained_coords

def expand_proposals(proposals, scale_factors, aspect_ratios):
    """
    对BFoV提案进行尺度扩展
    
    参数:
    proposals: 提案列表，每个元素为 [theta, phi, fov_x, fov_y]（不含category_id）
    scale_factors: 面积尺度因子列表
    aspect_ratios: fovx/fovy比例列表
    
    返回:
    expanded_proposals: 扩展后的提案列表
    """
    expanded_proposals = []
    
    for proposal in proposals:
        theta, phi, fov_x, fov_y = proposal
        
        current_area = fov_x * fov_y
        
        for scale in scale_factors:
            for aspect_ratio in aspect_ratios:
                new_area = scale * current_area
                new_fov_x = np.sqrt(aspect_ratio * new_area)
                new_fov_y = new_fov_x / aspect_ratio
                
                # 确保FOV不超过pi
                new_fov_x = min(new_fov_x, np.pi - 1e-6)
                new_fov_y = min(new_fov_y, np.pi - 1e-6)
                
                expanded_proposals.append([theta, phi, new_fov_x, new_fov_y])
    
    return expanded_proposals

def shake_bfov_proposals_horizontal(proposals, ratio):
    """
    对BFoV提案进行水平抖动（左/右）
    
    参数:
    proposals: 提案列表，每个元素为 [theta, phi, fov_x, fov_y]（不含category_id）
    ratio: 抖动比例
    
    返回:
    shaken_proposals: 抖动后的提案列表
    """
    shaken_proposals = []
    
    for proposal in proposals:
        lambda0, phi0, fov_x, fov_y = proposal
        
        alpha = ratio * fov_x
        
        sin_phi2 = np.sin(phi0) * np.cos(alpha)
        sin_phi2 = np.clip(sin_phi2, -1.0 + 1e-6, 1.0 - 1e-6)
        phi2 = np.arcsin(sin_phi2)
        
        cos_phi0 = np.cos(phi0) + 1e-8
        delta_lambda = np.arctan2(np.sin(alpha), cos_phi0 * np.cos(alpha))
        
        lambda2_l = lambda0 + delta_lambda
        lambda2_r = lambda0 - delta_lambda
        
        shaken_centers = [
            [lambda2_l, phi2],
            [lambda2_r, phi2],
        ]
        
        for shaken_center in shaken_centers:
            shaken_center = constrain_spherical_coords(np.array([shaken_center]))[0]
            shaken_proposals.append([shaken_center[0], shaken_center[1], fov_x, fov_y])
    
    return shaken_proposals

def shake_bfov_proposals_vertical(proposals, ratio):
    """
    对BFoV提案进行垂直抖动（上/下）
    
    参数:
    proposals: 提案列表，每个元素为 [theta, phi, fov_x, fov_y]（不含category_id）
    ratio: 抖动比例
    
    返回:
    shaken_proposals: 抖动后的提案列表
    """
    shaken_proposals = []
    
    for proposal in proposals:
        lambda0, phi0, fov_x, fov_y = proposal
        
        phi_t = phi0 + ratio * fov_y
        phi_d = phi0 - ratio * fov_y
        
        shaken_centers = [
            [lambda0, phi_t],
            [lambda0, phi_d],
        ]
        
        for shaken_center in shaken_centers:
            shaken_center = constrain_spherical_coords(np.array([shaken_center]))[0]
            shaken_proposals.append([shaken_center[0], shaken_center[1], fov_x, fov_y])
    
    return shaken_proposals

def draw_bfov_on_image(img, annotations, img_w, img_h):
    """
    使用专业BFoV绘制工具在图像上绘制BFoV
    
    参数:
    img: 输入图像
    annotations: BFoV标注列表，每个元素包含 'bfov' 和 'category_id'
    img_w: 图像宽度
    img_h: 图像高度
    
    返回:
    img: 绘制后的图像
    """
    img_copy = img.copy()
    
    for i, ann in enumerate(annotations):
        if 'bfov' not in ann:
            continue
        
        theta, phi, fov_x, fov_y = ann['bfov']
        category_id = ann['category_id']
        
        # 转换为角度制
        theta_deg = theta * 180 / np.pi
        phi_deg = phi * 180 / np.pi
        fov_x_deg = fov_x * 180 / np.pi
        fov_y_deg = fov_y * 180 / np.pi
        
        # 检查FOV参数是否有效
        if fov_x_deg <= 0 or fov_y_deg <= 0:
            print(f"  跳过无效BFoV {i+1}: FOV参数为负")
            continue
        
        # 获取类别名称
        category_name = CATEGORY_NAMES.get(category_id, f'category_{category_id}')
        
        # 获取颜色
        color = CATEGORY_COLORS.get(category_id, (128, 128, 128))  # 默认灰色
        
        print(f"  绘制BFoV {i+1}: {category_name}")
        print(f"    中心: ({theta_deg:.2f}°, {phi_deg:.2f}°)")
        print(f"    FOV:  ({fov_x_deg:.2f}° x {fov_y_deg:.2f}°)")
        
        # 使用专业BFoV绘制工具
        try:
            # 创建BFoV实例
            bfov = BFoV(img_w, img_h, 
                       view_angle_w=fov_x_deg, 
                       view_angle_h=fov_y_deg, 
                       long_side=img_w)
            
            # 获取BFoV的边界点（使用弧度制坐标）
            px, py = bfov._sample_points(theta, phi, border_only=True)
            
            # 逐点绘制BFoV边界
            for j in range(px.shape[0]):
                x, y = int(px[j]), int(py[j])
                if 0 <= y < img_h and 0 <= x < img_w:
                    cv2.circle(img_copy, (x, y), 1, color, -1, lineType=cv2.LINE_AA)
            
            # 绘制中心点
            cx = int(((theta + np.pi) / (2 * np.pi)) * img_w)
            cy = int(((np.pi / 2 - phi) / np.pi) * img_h)
            cv2.circle(img_copy, (cx, cy), 3, color, -1, lineType=cv2.LINE_AA)
            
        except Exception as e:
            print(f"  绘制BFoV {i+1}失败: {e}")
            import traceback
            traceback.print_exc()
    
    return img_copy

def draw_bfov_proposals(img, proposals, img_w, img_h, color=(255, 255, 0)):
    """
    绘制多个BFoV提案（不带category_id的简化格式）
    
    参数:
    img: 输入图像
    proposals: BFoV提案列表，每个元素为 [theta, phi, fov_x, fov_y]
    img_w: 图像宽度
    img_h: 图像高度
    color: 绘制颜色
    
    返回:
    img: 绘制后的图像
    """
    img_copy = img.copy()
    
    for i, proposal in enumerate(proposals):
        theta, phi, fov_x, fov_y = proposal
        
        fov_x_deg = fov_x * 180 / np.pi
        fov_y_deg = fov_y * 180 / np.pi
        
        try:
            bfov = BFoV(img_w, img_h, 
                       view_angle_w=fov_x_deg, 
                       view_angle_h=fov_y_deg, 
                       long_side=img_w)
            
            px, py = bfov._sample_points(theta, phi, border_only=True)
            
            for j in range(px.shape[0]):
                x, y = int(px[j]), int(py[j])
                if 0 <= y < img_h and 0 <= x < img_w:
                    cv2.circle(img_copy, (x, y), 1, color, -1, lineType=cv2.LINE_AA)
        
        except Exception as e:
            print(f"  绘制提案 {i+1}失败: {e}")
    
    return img_copy

def draw_all_shaken_proposals(img, center_proposals, horizontal_proposals, vertical_proposals, img_w, img_h):
    """
    将所有抖动结果绘制在同一张图上，用三种颜色区分
    
    参数:
    img: 输入图像
    center_proposals: 中心扩展提案列表（带尺度扩展）
    horizontal_proposals: 水平抖动提案列表
    vertical_proposals: 垂直抖动提案列表
    img_w: 图像宽度
    img_h: 图像高度
    
    返回:
    img: 绘制后的图像
    """
    img_copy = img.copy()
    
    # 定义三种颜色
    center_color = (0, 255, 0)      # 中心：绿色
    horizontal_color = (0, 0, 255)   # 左右抖动：蓝色
    vertical_color = (255, 0, 0)     # 上下抖动：红色
    
    # 绘制水平抖动提案（蓝色）
    for proposal in horizontal_proposals:
        theta, phi, fov_x, fov_y = proposal
        fov_x_deg = fov_x * 180 / np.pi
        fov_y_deg = fov_y * 180 / np.pi
        
        try:
            bfov = BFoV(img_w, img_h, 
                       view_angle_w=fov_x_deg, 
                       view_angle_h=fov_y_deg, 
                       long_side=img_w)
            
            px, py = bfov._sample_points(theta, phi, border_only=True)
            
            for j in range(px.shape[0]):
                x, y = int(px[j]), int(py[j])
                if 0 <= y < img_h and 0 <= x < img_w:
                    cv2.circle(img_copy, (x, y), 1, horizontal_color, -1, lineType=cv2.LINE_AA)
        except Exception as e:
            pass
    
    # 绘制垂直抖动提案（红色）
    for proposal in vertical_proposals:
        theta, phi, fov_x, fov_y = proposal
        fov_x_deg = fov_x * 180 / np.pi
        fov_y_deg = fov_y * 180 / np.pi
        
        try:
            bfov = BFoV(img_w, img_h, 
                       view_angle_w=fov_x_deg, 
                       view_angle_h=fov_y_deg, 
                       long_side=img_w)
            
            px, py = bfov._sample_points(theta, phi, border_only=True)
            
            for j in range(px.shape[0]):
                x, y = int(px[j]), int(py[j])
                if 0 <= y < img_h and 0 <= x < img_w:
                    cv2.circle(img_copy, (x, y), 1, vertical_color, -1, lineType=cv2.LINE_AA)
        except Exception as e:
            pass
    
    # 绘制中心扩展提案组（绿色）
    for proposal in center_proposals:
        theta, phi, fov_x, fov_y = proposal
        fov_x_deg = fov_x * 180 / np.pi
        fov_y_deg = fov_y * 180 / np.pi
        
        try:
            bfov = BFoV(img_w, img_h, 
                       view_angle_w=fov_x_deg, 
                       view_angle_h=fov_y_deg, 
                       long_side=img_w)
            
            px, py = bfov._sample_points(theta, phi, border_only=True)
            
            for j in range(px.shape[0]):
                x, y = int(px[j]), int(py[j])
                if 0 <= y < img_h and 0 <= x < img_w:
                    cv2.circle(img_copy, (x, y), 1, center_color, -1, lineType=cv2.LINE_AA)
        except Exception as e:
            pass
    
    return img_copy

def generate_high_latitude_bfovs(num_proposals=5):
    """
    在高纬度区域随机生成BFoV提案
    
    参数:
    num_proposals: 生成的提案数量
    
    返回:
    proposals: BFoV提案列表，每个元素为 [theta, phi, fov_x, fov_y]
    """
    proposals = []
    
    for _ in range(num_proposals):
        # 随机选择北极或南极区域
        if np.random.random() > 0.5:
            # 北极区域: phi ∈ [π/3, π/2 - 0.1] (约60°到84°N)
            phi = np.random.uniform(np.pi/3, np.pi/2 - 0.1)
        else:
            # 南极区域: phi ∈ [-π/2 + 0.1, -π/3] (约84°到60°S)
            phi = np.random.uniform(-np.pi/2 + 0.1, -np.pi/3)
        
        # 随机经度: theta ∈ [-π, π]
        theta = np.random.uniform(-np.pi, np.pi)
        
        # 随机FOV: fov_x ∈ [30°, 60°], fov_y ∈ [20°, 40°] (更小的视场)
        fov_x = np.random.uniform(np.pi/6, np.pi/3)   # 30°到60°
        fov_y = np.random.uniform(np.pi/9, 2*np.pi/9)  # 20°到40°
        
        proposals.append([theta, phi, fov_x, fov_y])
    
    return proposals

def draw_high_latitude_bfovs(img, proposals, img_w, img_h):
    """
    绘制高纬度区域的BFoV提案
    
    参数:
    img: 输入图像
    proposals: BFoV提案列表
    img_w: 图像宽度
    img_h: 图像高度
    
    返回:
    img: 绘制后的图像
    """
    img_copy = img.copy()
    color = (255, 165, 0)  # 橙色
    
    for i, proposal in enumerate(proposals):
        theta, phi, fov_x, fov_y = proposal
        fov_x_deg = fov_x * 180 / np.pi
        fov_y_deg = fov_y * 180 / np.pi
        
        try:
            bfov = BFoV(img_w, img_h, 
                       view_angle_w=fov_x_deg, 
                       view_angle_h=fov_y_deg, 
                       long_side=img_w)
            
            px, py = bfov._sample_points(theta, phi, border_only=True)
            
            for j in range(px.shape[0]):
                x, y = int(px[j]), int(py[j])
                if 0 <= y < img_h and 0 <= x < img_w:
                    cv2.circle(img_copy, (x, y), 2, color, -1, lineType=cv2.LINE_AA)
            
            # 绘制中心点
            cx = int(((theta + np.pi) / (2 * np.pi)) * img_w)
            cy = int(((np.pi / 2 - phi) / np.pi) * img_h)
            cv2.circle(img_copy, (cx, cy), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            
        except Exception as e:
            print(f"  绘制高纬度BFoV {i+1}失败: {e}")
    
    return img_copy

def project_bfov_to_plane(img, bfov_params, erp_width, erp_height):
    """
    将BFoV投影成切平面无畸变图
    
    参数:
    img: 输入ERP图像
    bfov_params: BFoV参数 [lon, lat, fov_u, fov_v]（弧度）
    erp_width: ERP图像宽度
    erp_height: ERP图像高度
    
    返回:
    plane_img: 切平面图像
    """
    lon, lat, fov_u, fov_v = bfov_params
    PI = np.pi

    sphere_circumference = erp_width
    
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

    erp_x = (lon_deg + 180.0) / 360.0 * erp_width
    erp_y = (90.0 - lat_deg) / 180.0 * erp_height

    erp_x = np.clip(erp_x, 0, erp_width - 1)
    erp_y = np.clip(erp_y, 0, erp_height - 1)

    plane_img = np.zeros((plane_height, plane_width, 3), dtype=np.uint8)
    
    for i in range(plane_height):
        for j in range(plane_width):
            px = int(erp_x[i, j])
            py = int(erp_y[i, j])
            plane_img[i, j] = img[py, px]

    return plane_img

def generate_high_latitude_bfovs_with_planes(img, proposals, img_w, img_h, output_dir):
    """
    绘制高纬度区域的BFoV提案并生成切平面投影图
    
    参数:
    img: 输入图像
    proposals: BFoV提案列表
    img_w: 图像宽度
    img_h: 图像高度
    output_dir: 切平面输出目录
    
    返回:
    img: 绘制后的图像
    """
    img_copy = img.copy()
    color = (255, 165, 0)  # 橙色
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    for i, proposal in enumerate(proposals):
        theta, phi, fov_x, fov_y = proposal
        fov_x_deg = fov_x * 180 / np.pi
        fov_y_deg = fov_y * 180 / np.pi
        
        try:
            bfov = BFoV(img_w, img_h, 
                       view_angle_w=fov_x_deg, 
                       view_angle_h=fov_y_deg, 
                       long_side=img_w)
            
            # 绘制BFoV边框
            px, py = bfov._sample_points(theta, phi, border_only=True)
            
            for j in range(px.shape[0]):
                x, y = int(px[j]), int(py[j])
                if 0 <= y < img_h and 0 <= x < img_w:
                    cv2.circle(img_copy, (x, y), 2, color, -1, lineType=cv2.LINE_AA)
            
            # 绘制中心点
            cx = int(((theta + np.pi) / (2 * np.pi)) * img_w)
            cy = int(((np.pi / 2 - phi) / np.pi) * img_h)
            cv2.circle(img_copy, (cx, cy), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            
            # 生成切平面投影图
            plane_img = project_bfov_to_plane(img, [theta, phi, fov_x, fov_y], img_w, img_h)
            if plane_img is not None:
                plane_filename = f'high_lat_bfov_{i+1}_plane.jpg'
                plane_path = os.path.join(output_dir, plane_filename)
                cv2.imwrite(plane_path, plane_img)
                print(f"    切平面投影图 {i+1} 保存到: {plane_path}")
            
        except Exception as e:
            print(f"  处理高纬度BFoV {i+1}失败: {e}")
    
    return img_copy

def main():
    # 配置参数
    IMAGE_FILENAME = '29803495916_be313c45bb_o.jpg'
    DATA_ROOT = '/home/mengchao/workspace/P2BFoV/P2BNet-main/360indoor/'
    ANN_FILE = DATA_ROOT + 'ann/train_coco.json'
    IMAGE_PATH = DATA_ROOT + 'images/' + IMAGE_FILENAME
    OUTPUT_DIR = '/home/mengchao/workspace/P2BFoV/P2BNet-main/TOV_mmdetection/bfov/eswa_frame/sphroi/'
    
    # 只绘制沙发（category_id: 18）
    TARGET_CATEGORY_ID = 18
    
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
    
    # 获取沙发BFoV参数
    theta, phi, fov_x, fov_y = largest_sofa['bfov']
    category_id = largest_sofa['category_id']
    category_name = CATEGORY_NAMES.get(category_id, f'category_{category_id}')
    
    print("\n标注详情:")
    print(f"category_id: {category_id} ({category_name})")
    print(f"theta: {theta:.6f} rad ({theta * 180/np.pi:.2f}°)")
    print(f"phi: {phi:.6f} rad ({phi * 180/np.pi:.2f}°)")
    print(f"fov_x: {fov_x:.6f} rad ({fov_x * 180/np.pi:.2f}°)")
    print(f"fov_y: {fov_y:.6f} rad ({fov_y * 180/np.pi:.2f}°)")
    print(f"面积: {fov_x * fov_y:.6f} rad² ({fov_x * fov_y * (180/np.pi)**2:.2f} deg²)")
    
    # ========== 1. 绘制原始BFoV（带扩展提案）==========
    print("\n=== 绘制原始BFoV（带扩展提案）===")
    original_proposal = [theta, phi, fov_x, fov_y]
    expanded_orig_proposals = expand_proposals([original_proposal], SCALE_FACTORS, ASPECT_RATIOS)
    print(f"扩展后提案数: {len(expanded_orig_proposals)}")
    
    color = CATEGORY_COLORS.get(category_id, (255, 255, 0))
    img_with_orig = draw_bfov_proposals(img.copy(), expanded_orig_proposals, img_w, img_h, color)
    
    # 绘制原始BFoV中心点
    cx = int(((theta + np.pi) / (2 * np.pi)) * img_w)
    cy = int(((np.pi / 2 - phi) / np.pi) * img_h)
    cv2.circle(img_with_orig, (cx, cy), 3, color, -1, lineType=cv2.LINE_AA)
    
    output_path_orig = os.path.join(OUTPUT_DIR, 'bfov_shaken_orig_' + IMAGE_FILENAME)
    cv2.imwrite(output_path_orig, img_with_orig)
    print(f"原始BFoV扩展图像保存到: {output_path_orig}")
    
    # ========== 2. 绘制水平抖动后的BFoV ==========
    print("\n=== 绘制水平抖动后的BFoV ===")
    horizontal_shaken = shake_bfov_proposals_horizontal([original_proposal], SHAKE_RATIO)
    
    all_horizontal_proposals = []
    for shaken in horizontal_shaken:
        expanded = expand_proposals([shaken], SCALE_FACTORS, ASPECT_RATIOS)
        all_horizontal_proposals.extend(expanded)
    
    print(f"水平抖动提案数: {len(all_horizontal_proposals)}")
    img_with_horizontal = draw_bfov_proposals(img.copy(), all_horizontal_proposals, img_w, img_h, color)
    
    output_path_horizontal = os.path.join(OUTPUT_DIR, 'bfov_shaken_horizontal_' + IMAGE_FILENAME)
    cv2.imwrite(output_path_horizontal, img_with_horizontal)
    print(f"水平抖动BFoV图像保存到: {output_path_horizontal}")
    
    # ========== 3. 绘制垂直抖动后的BFoV ==========
    print("\n=== 绘制垂直抖动后的BFoV ===")
    vertical_shaken = shake_bfov_proposals_vertical([original_proposal], SHAKE_RATIO)
    
    all_vertical_proposals = []
    for shaken in vertical_shaken:
        expanded = expand_proposals([shaken], SCALE_FACTORS, ASPECT_RATIOS)
        all_vertical_proposals.extend(expanded)
    
    print(f"垂直抖动提案数: {len(all_vertical_proposals)}")
    img_with_vertical = draw_bfov_proposals(img.copy(), all_vertical_proposals, img_w, img_h, color)
    
    output_path_vertical = os.path.join(OUTPUT_DIR, 'bfov_shaken_vertical_' + IMAGE_FILENAME)
    cv2.imwrite(output_path_vertical, img_with_vertical)
    print(f"垂直抖动BFoV图像保存到: {output_path_vertical}")
    
    # ========== 4. 绘制原始单一BFoV（原功能）==========
    print("\n=== 绘制原始单一BFoV ===")
    img_with_bfov = draw_bfov_on_image(img, [largest_sofa], img_w, img_h)
    
    output_path = os.path.join(OUTPUT_DIR, 'bfov_sofa_' + IMAGE_FILENAME)
    cv2.imwrite(output_path, img_with_bfov)
    print(f"原始BFoV图像保存到: {output_path}")
    
    # ========== 5. 绘制所有抖动结果叠加图（三种颜色）==========
    print("\n=== 绘制所有抖动结果叠加图 ===")
    # 获取扩展后的原始提案（中心）
    expanded_center = expand_proposals([original_proposal], SCALE_FACTORS, ASPECT_RATIOS)
    
    img_all_shaken = draw_all_shaken_proposals(
        img.copy(), 
        expanded_center,  # 传入扩展后的中心提案组
        all_horizontal_proposals, 
        all_vertical_proposals, 
        img_w, 
        img_h
    )
    
    output_path_all = os.path.join(OUTPUT_DIR, 'bfov_shaken_all_' + IMAGE_FILENAME)
    cv2.imwrite(output_path_all, img_all_shaken)
    print(f"所有抖动结果叠加图像保存到: {output_path_all}")
    print("  颜色说明: 中心(绿色), 左右抖动(蓝色), 上下抖动(红色)")
    
    # ========== 6. 在高纬度区域随机生成BFoV并绘制（含切平面投影）==========
    print("\n=== 在高纬度区域随机生成BFoV ===")
    high_latitude_proposals = generate_high_latitude_bfovs(num_proposals=5)
    
    # 切平面输出目录
    plane_output_dir = '/home/mengchao/workspace/P2BFoV/P2BNet-main/TOV_mmdetection/bfov/eswa_frame/sphroi/panel_neg'
    
    # 绘制BFoV并生成切平面投影图
    img_high_latitude = generate_high_latitude_bfovs_with_planes(
        img.copy(), 
        high_latitude_proposals, 
        img_w, 
        img_h, 
        plane_output_dir
    )
    
    output_path_high_lat = os.path.join(OUTPUT_DIR, 'bfov_high_latitude_' + IMAGE_FILENAME)
    cv2.imwrite(output_path_high_lat, img_high_latitude)
    print(f"高纬度BFoV图像保存到: {output_path_high_lat}")
    print("  生成了 5 个高纬度BFoV（橙色边框，红色中心点）")
    print("  纬度范围: 60°-84°N 或 60°-84°S")
    
    print("\n处理完成！")

if __name__ == '__main__':
    main()
