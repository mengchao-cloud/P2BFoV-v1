#!/usr/bin/env python3
"""
绘制多尺度多比例BFoV并生成三组切平面叠加图

功能：
1. 绘制原始视角和旋转视角的多尺度BFoV
2. 为三个尺度（0.67、1.0、1.5）分别生成切平面
3. 将三组切平面按要求叠放（中心对齐，中间的放最上面）
"""

import json
import numpy as np
import cv2
import os
import sys

# 添加sphdet路径到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../sphdet/visualizers'))
from ImageRecorder import ImageRecorder as BFoV

# 添加PANDORA路径用于三维旋转
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../PANDORA/RoIoU/libs'))
try:
    from tools import rotate_image, tools
    print("成功导入 rotate_image 和 tools")
except ImportError as e:
    print(f"导入失败: {e}")
    raise

# 类别名称映射
CATEGORY_NAMES = {
    18: 'sofa',
}

# 不同尺度的颜色映射
SCALE_COLORS = {
    0.67: (255, 255, 0),     # 黄色系
    1.0: (0, 255, 0),        # 绿色系
    1.5: (255, 0, 0),        # 红色系
}

# 图像尺寸配置
ERP_WIDTH = 1920
ERP_HEIGHT = 960

def random_rotate_image_3d(img, max_angles=[30, 30, 30], rotate_ratio=1.0):
    """对ERP图像进行随机三维旋转"""
    roll = np.random.uniform(-max_angles[0], max_angles[0]) * rotate_ratio
    pitch = np.random.uniform(-max_angles[1], max_angles[1]) * rotate_ratio
    yaw = np.random.uniform(-max_angles[2], max_angles[2]) * rotate_ratio
    
    print(f"随机旋转参数: Roll={roll:.2f}°, Pitch={pitch:.2f}°, Yaw={yaw:.2f}°")
    
    rotated_img = img.copy()
    if abs(roll) > 1e-6:
        rotated_img = rotate_image(rotated_img, roll, np.array([1, 0, 0], dtype=np.float64))
    if abs(pitch) > 1e-6:
        rotated_img = rotate_image(rotated_img, pitch, np.array([0, 1, 0], dtype=np.float64))
    if abs(yaw) > 1e-6:
        rotated_img = rotate_image(rotated_img, yaw, np.array([0, 0, 1], dtype=np.float64))
    
    return rotated_img, [roll, pitch, yaw]

def transform_bfov_params(bfov_params, rotation_angles, erp_w, erp_h):
    """将BFoV参数进行三维旋转变换"""
    theta, phi, fov_x, fov_y = bfov_params
    roll, pitch, yaw = rotation_angles
    
    t = tools(erp_w, erp_h)
    
    center_x = theta * 180 / np.pi
    center_y = phi * 180 / np.pi
    
    roll_t = roll
    pitch_t = -pitch
    yaw_t = yaw
    
    center_px = (center_x + 180) / 360 * erp_w
    center_py = (90 - center_y) / 180 * erp_h
    
    center_xyz = np.array(t.pxpy2xyz([center_px + 0.5, center_py + 0.5]))
    center_xyz = center_xyz / np.linalg.norm(center_xyz)
    
    if abs(roll_t) > 1e-6:
        center_xyz = np.array(t.roll_T(np.array([1, 0, 0]), center_xyz, roll_t))
    if abs(pitch_t) > 1e-6:
        center_xyz = np.array(t.roll_T(np.array([0, 1, 0]), center_xyz, pitch_t))
    if abs(yaw_t) > 1e-6:
        center_xyz = np.array(t.roll_T(np.array([0, 0, 1]), center_xyz, yaw_t))
        
    center_xyz = center_xyz / np.linalg.norm(center_xyz)
    
    px_new, py_new = t.xyz2pxpy(center_xyz)
    new_center_x = (px_new / erp_w) * 360 - 180
    new_center_y = 90 - (py_new / erp_h) * 180
    
    new_theta = new_center_x * np.pi / 180
    new_phi = new_center_y * np.pi / 180
    
    return [new_theta, new_phi, fov_x, fov_y]

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
    
    print(f"找到图像: {image_filename} (id: {img_id})")
    print(f"该图像有 {len(annotations)} 个BFoV标注")
    
    return target_image, annotations

def project_bfov_to_plane(img, bfov_params, max_fov_deg=175.0):
    """
    将BFoV投影成切平面无畸变图
    
    参数:
    img: 输入ERP图像
    bfov_params: BFOV参数 [theta, phi, fov_x, fov_y]，弧度制
    max_fov_deg: FOV最大值（度数），防止FOV过大导致投影异常
    """
    lon, lat, fov_u, fov_v = bfov_params
    PI = np.pi

    # 将FOV限制在最大范围内
    fov_u_deg = fov_u * 180 / PI
    fov_v_deg = fov_v * 180 / PI
    fov_u_deg_clamped = min(fov_u_deg, max_fov_deg)
    fov_v_deg_clamped = min(fov_v_deg, max_fov_deg)
    
    # 转换回弧度
    fov_u = fov_u_deg_clamped * PI / 180
    fov_v = fov_v_deg_clamped * PI / 180

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

def add_blue_border(img, border_thickness=2):
    """给图片添加蓝色边框（带透明背景支持）"""
    h, w = img.shape[:2]
    
    new_h = h + border_thickness * 2
    new_w = w + border_thickness * 2
    
    img_with_border = np.zeros((new_h, new_w, 4), dtype=np.uint8)
    img_with_border[:, :, :3] = (255, 0, 0)  # 蓝色边框
    img_with_border[:, :, 3] = 255  # 完全不透明
    
    img_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    img_rgba[:, :, :3] = img
    img_rgba[:, :, 3] = 255
    
    img_with_border[border_thickness:border_thickness+h, border_thickness:border_thickness+w] = img_rgba
    
    return img_with_border

def stack_images_center_aligned(image_list, add_border=True):
    """将多张图片中心对齐叠加（透明背景）"""
    images = []
    for img in image_list:
        if add_border:
            img = add_blue_border(img, 2)
        else:
            h, w = img.shape[:2]
            img_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            img_rgba[:, :, :3] = img
            img_rgba[:, :, 3] = 255
            img = img_rgba
        images.append(img)
    
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)
    
    canvas_height = max_height
    canvas_width = max_width
    stacked_img = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
    stacked_img[:, :, 3] = 0
    
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        
        offset_x = (canvas_width - w) // 2
        offset_y = (canvas_height - h) // 2
        
        alpha_mask = img[:, :, 3] > 0
        
        for c in range(4):
            stacked_img[offset_y:offset_y+h, offset_x:offset_x+w, c][alpha_mask] = img[:, :, c][alpha_mask]
    
    return stacked_img

def draw_multi_scale_bfov_on_image(img, theta, phi, fov_x, fov_y, img_w, img_h,
                                     scales=[0.67, 1.0, 1.5],
                                     aspect_ratios=[[0.5, 2], [1, 1], [2, 0.5]]):
    """绘制多尺度多比例BFoV"""
    img_copy = img.copy()
    
    for scale in scales:
        for scale_x, scale_y in aspect_ratios:
            scaled_fov_x = fov_x * scale * scale_x
            scaled_fov_y = fov_y * scale * scale_y
            
            fov_x_deg = scaled_fov_x * 180 / np.pi
            fov_y_deg = scaled_fov_y * 180 / np.pi
            
            fov_x_deg_clamped = min(fov_x_deg, 175.0)
            fov_y_deg_clamped = min(fov_y_deg, 175.0)
            
            color = SCALE_COLORS.get(scale, (128, 128, 128))
            
            try:
                bfov = BFoV(img_w, img_h, 
                           view_angle_w=fov_x_deg_clamped, 
                           view_angle_h=fov_y_deg_clamped, 
                           long_side=img_w)
                
                px, py = bfov._sample_points(theta, phi, border_only=True)
                
                for j in range(px.shape[0]):
                    x, y = int(px[j]), int(py[j])
                    if 0 <= y < img_h and 0 <= x < img_w:
                        cv2.circle(img_copy, (x, y), 1, color, -1, lineType=cv2.LINE_AA)
                
            except Exception as e:
                print(f"  绘制失败: {e}")
    
    return img_copy

def main():
    # 配置参数
    IMAGE_FILENAME = '29803495916_be313c45bb_o.jpg'
    DATA_ROOT = '/home/mengchao/workspace/P2BFoV/P2BNet-main/360indoor/'
    ANN_FILE = DATA_ROOT + 'ann/train_coco.json'
    IMAGE_PATH = DATA_ROOT + 'images/' + IMAGE_FILENAME
    OUTPUT_DIR = '/home/mengchao/workspace/P2BFoV/P2BNet-main/TOV_mmdetection/bfov/eswa_frame/sphroi/'
    PLANES_STACK_DIR = os.path.join(OUTPUT_DIR, 'planes_stacks')
    
    # 创建输出目录
    os.makedirs(PLANES_STACK_DIR, exist_ok=True)
    
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
    
    theta, phi, fov_x, fov_y = largest_sofa['bfov']
    print(f"原始BFoV: theta={theta:.4f}, phi={phi:.4f}, fov_x={fov_x:.4f}, fov_y={fov_y:.4f}")
    
    print(f"正在读取图像: {IMAGE_PATH}")
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise ValueError(f"无法读取图像: {IMAGE_PATH}")
    
    img_h, img_w = img.shape[:2]
    print(f"图像尺寸: {img_w} x {img_h}")
    
    # ========== 绘制原始视角多尺度BFoV ==========
    print("\n正在绘制多尺度BFoV（原始视角）...")
    img_with_multi_bfov = draw_multi_scale_bfov_on_image(img, theta, phi, fov_x, fov_y, img_w, img_h)
    
    output_path = os.path.join(OUTPUT_DIR, 'multi_scale_bfov_' + IMAGE_FILENAME)
    cv2.imwrite(output_path, img_with_multi_bfov)
    print(f"原始视角图像已保存到: {output_path}")
    
    # ========== 绘制旋转视角多尺度BFoV ==========
    print("\n=== 开始随机三维旋转 ===")
    max_angles = [60, 60, 60]
    rotate_ratio = 1.0
    rotated_img, rotation_params = random_rotate_image_3d(img.copy(), max_angles, rotate_ratio)
    
    print("\n=== 变换BFoV参数 ===")
    transformed_bfov = transform_bfov_params([theta, phi, fov_x, fov_y], rotation_params, img_w, img_h)
    new_theta, new_phi, new_fov_x, new_fov_y = transformed_bfov
    
    print(f"变换后的BFoV: theta={new_theta:.4f}, phi={new_phi:.4f}, fov_x={new_fov_x:.4f}, fov_y={new_fov_y:.4f}")
    
    print("\n正在绘制多尺度BFoV（旋转视角）...")
    rotated_img_with_multi_bfov = draw_multi_scale_bfov_on_image(rotated_img, new_theta, new_phi, new_fov_x, new_fov_y, img_w, img_h)
    
    rotated_output_path = os.path.join(OUTPUT_DIR, 'multi_scale_bfov_rotated_' + IMAGE_FILENAME)
    cv2.imwrite(rotated_output_path, rotated_img_with_multi_bfov)
    print(f"旋转视角图像已保存到: {rotated_output_path}")
    
    # ========== 生成三组切平面并叠加 ==========
    print("\n=== 生成三组切平面 ===")
    scales = [0.67, 1.0, 1.5]
    aspect_ratios = [[0.5, 2], [1, 1], [2, 0.5]]
    
    # 为每个尺度生成三组切平面（对应三种比例）
    scale_planes = {}
    for scale in scales:
        print(f"\n处理尺度 {scale}:")
        scale_planes[scale] = []
        
        for scale_x, scale_y in aspect_ratios:
            scaled_fov_x = fov_x * scale * scale_x
            scaled_fov_y = fov_y * scale * scale_y
            
            print(f"  比例 [{scale_x}, {scale_y}]")
            plane_img = project_bfov_to_plane(img, [theta, phi, scaled_fov_x, scaled_fov_y])
            scale_planes[scale].append(plane_img)
            
            # 保存单独的切平面
            plane_name = f'plane_scale{scale}_ratio{scale_x}-{scale_y}.jpg'
            cv2.imwrite(os.path.join(PLANES_STACK_DIR, plane_name), plane_img)
    
    # 将每组切平面叠加（中心对齐，中间的放最上面）
    print("\n=== 叠加切平面 ===")
    for i, scale in enumerate(scales, 1):
        planes = scale_planes[scale]
        if len(planes) >= 3:
            # 叠加顺序：底层, 中层, 顶层（中间的放最上面）
            stacked = stack_images_center_aligned([planes[0], planes[2], planes[1]], add_border=True)
            output_path = os.path.join(PLANES_STACK_DIR, f'stack_{i:02d}_border.png')
            cv2.imwrite(output_path, stacked)
            print(f"  scale={scale} 叠加完成，保存到: {output_path}")
    
    # 额外生成九图叠加（所有尺度的切平面叠加）
    print("\n=== 生成九图叠加 ===")
    all_planes = []
    for scale in scales:
        all_planes.extend(scale_planes[scale])
    
    if len(all_planes) >= 9:
        # 叠加顺序：最大的在最底层，最小的在最顶层
        nine_stack = stack_images_center_aligned([
            all_planes[6], all_planes[8], all_planes[7],  # scale=1.5 组
            all_planes[3], all_planes[5], all_planes[4],  # scale=1.0 组
            all_planes[0], all_planes[2], all_planes[1],  # scale=0.67 组
        ], add_border=True)
        output_path = os.path.join(PLANES_STACK_DIR, 'stack_nine_border.png')
        cv2.imwrite(output_path, nine_stack)
        print(f"  九图叠加完成，保存到: {output_path}")
    
    print("\n=== 所有处理完成 ===")
    print(f"多尺度BFoV图像已保存到: {OUTPUT_DIR}")
    print(f"切平面叠加图像已保存到: {PLANES_STACK_DIR}")

if __name__ == '__main__':
    main()
