#!/usr/bin/env python3
"""
绘制多尺度多比例BFoV：3个尺度 × 3种比例 = 9个BFoV
支持绘制原始视角和旋转视角两个图

尺度：0.67、1、1.5
比例组合：
  - [0.5, 2] - fov_x缩放0.5，fov_y缩放2
  - [1, 1] - fov_x缩放1，fov_y缩放1（原始比例）
  - [2, 0.5] - fov_x缩放2，fov_y缩放0.5

使用方法：
python draw_multi_scale_bfov.py
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

# 不同尺度的颜色映射（同一尺度使用相同颜色）
SCALE_COLORS = {
    0.67: (255, 255, 0),     # 黄色系
    1.0: (0, 255, 0),        # 绿色系
    1.5: (255, 0, 0),        # 红色系
}

def random_rotate_image_3d(img, max_angles=[30, 30, 30], rotate_ratio=1.0):
    """
    对ERP图像进行随机三维旋转
    
    参数:
    img: 输入图像
    max_angles: 三个旋转角度的最大值 [roll_max, pitch_max, yaw_max]，单位为度
    rotate_ratio: 比例系数，随机值范围为 [-max*ratio, max*ratio]
    
    返回:
    rotated_img: 旋转后的图像
    rotation_params: 使用的旋转参数 [roll, pitch, yaw]，单位为度
    """
    # 生成随机旋转角度
    roll = np.random.uniform(-max_angles[0], max_angles[0]) * rotate_ratio
    pitch = np.random.uniform(-max_angles[1], max_angles[1]) * rotate_ratio
    yaw = np.random.uniform(-max_angles[2], max_angles[2]) * rotate_ratio
    
    print(f"随机旋转参数: Roll={roll:.2f}°, Pitch={pitch:.2f}°, Yaw={yaw:.2f}°")
    
    # 应用三维旋转
    rotated_img = img.copy()
    
    if abs(roll) > 1e-6:
        rotated_img = rotate_image(rotated_img, roll, np.array([1, 0, 0], dtype=np.float64))
    if abs(pitch) > 1e-6:
        rotated_img = rotate_image(rotated_img, pitch, np.array([0, 1, 0], dtype=np.float64))
    if abs(yaw) > 1e-6:
        rotated_img = rotate_image(rotated_img, yaw, np.array([0, 0, 1], dtype=np.float64))
    
    return rotated_img, [roll, pitch, yaw]

def transform_bfov_params(bfov_params, rotation_angles, erp_w, erp_h):
    """
    将BFoV参数进行三维旋转变换（应用 Case 5 逻辑: Roll+, Pitch-, Yaw+）
    
    参数:
    bfov_params: [theta_rad, phi_rad, fov_x_rad, fov_y_rad]
                 theta: -π~π (弧度), phi: -π/2~π/2 (弧度)
    rotation_angles: [roll, pitch, yaw] 单位为度
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    
    返回:
    transformed_bfov: [new_theta_rad, new_phi_rad, fov_x_rad, fov_y_rad]
    """
    theta, phi, fov_x, fov_y = bfov_params
    roll, pitch, yaw = rotation_angles
    
    t = tools(erp_w, erp_h)
    
    # 将弧度转换为角度
    center_x = theta * 180 / np.pi  # -180 ~ 180
    center_y = phi * 180 / np.pi     # -90 ~ 90
    
    # ========== 1. 确定变换角度（Case 5: Pitch 取反）==========
    roll_t = roll
    pitch_t = -pitch
    yaw_t = yaw
    
    # ========== 2. 变换中心点坐标 ==========
    # 计算原始中心点的像素坐标
    center_px = (center_x + 180) / 360 * erp_w
    center_py = (90 - center_y) / 180 * erp_h
    
    # 转为三维单位向量
    center_xyz = np.array(t.pxpy2xyz([center_px + 0.5, center_py + 0.5]))
    center_xyz = center_xyz / np.linalg.norm(center_xyz)
    
    # 应用旋转矩阵
    if abs(roll_t) > 1e-6:
        center_xyz = np.array(t.roll_T(np.array([1, 0, 0]), center_xyz, roll_t))
    if abs(pitch_t) > 1e-6:
        center_xyz = np.array(t.roll_T(np.array([0, 1, 0]), center_xyz, pitch_t))
    if abs(yaw_t) > 1e-6:
        center_xyz = np.array(t.roll_T(np.array([0, 0, 1]), center_xyz, yaw_t))
        
    center_xyz = center_xyz / np.linalg.norm(center_xyz)
    
    # 转回经纬度（角度）
    px_new, py_new = t.xyz2pxpy(center_xyz)
    new_center_x = (px_new / erp_w) * 360 - 180
    new_center_y = 90 - (py_new / erp_h) * 180
    
    # 转换回弧度
    new_theta = new_center_x * np.pi / 180
    new_phi = new_center_y * np.pi / 180
    
    return [new_theta, new_phi, fov_x, fov_y]

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

def draw_multi_scale_bfov_on_image(img, theta, phi, fov_x, fov_y, img_w, img_h, 
                                     scales=[0.67, 1.0, 1.5], 
                                     aspect_ratios=[[0.5, 2], [1, 1], [2, 0.5]]):
    """
    绘制多尺度多比例BFoV
    
    参数:
    img: 输入图像
    theta, phi: BFoV中心坐标（弧度）
    fov_x, fov_y: 原始FOV尺寸（弧度）
    img_w: 图像宽度
    img_h: 图像高度
    scales: 缩放比例列表 [0.5, 1.0, 1.5]
    aspect_ratios: FOV比例调整列表 [[scale_x, scale_y], ...]
    
    返回:
    img: 绘制后的图像
    """
    img_copy = img.copy()
    
    for scale in scales:
        for scale_x, scale_y in aspect_ratios:
            # 计算缩放后的FOV
            scaled_fov_x = fov_x * scale * scale_x
            scaled_fov_y = fov_y * scale * scale_y
            
            # 转换为角度制
            theta_deg = theta * 180 / np.pi
            phi_deg = phi * 180 / np.pi
            fov_x_deg = scaled_fov_x * 180 / np.pi
            fov_y_deg = scaled_fov_y * 180 / np.pi
            
            # 限制FOV不超过175°
            fov_x_deg_clamped = min(fov_x_deg, 175.0)
            fov_y_deg_clamped = min(fov_y_deg, 175.0)
            
            # 获取颜色（同一尺度使用相同颜色）
            color = SCALE_COLORS.get(scale, (128, 128, 128))
            
            print(f"  绘制 scale={scale}, ratio=[{scale_x}, {scale_y}] BFoV:")
            print(f"    中心: ({theta_deg:.2f}°, {phi_deg:.2f}°)")
            print(f"    FOV:  ({fov_x_deg:.2f}° x {fov_y_deg:.2f}°)")
            if fov_x_deg > 175.0 or fov_y_deg > 175.0:
                print(f"    FOV限制后: ({fov_x_deg_clamped:.2f}° x {fov_y_deg_clamped:.2f}°)")
            
            try:
                # 创建BFoV实例
                bfov = BFoV(img_w, img_h, 
                           view_angle_w=fov_x_deg_clamped, 
                           view_angle_h=fov_y_deg_clamped, 
                           long_side=img_w)
                
                # 获取BFoV的边界点
                px, py = bfov._sample_points(theta, phi, border_only=True)
                
                # 逐点绘制BFoV边界
                for j in range(px.shape[0]):
                    x, y = int(px[j]), int(py[j])
                    if 0 <= y < img_h and 0 <= x < img_w:
                        cv2.circle(img_copy, (x, y), 1, color, -1, lineType=cv2.LINE_AA)
                
            except Exception as e:
                print(f"  绘制失败: {e}")
                import traceback
                traceback.print_exc()
    
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
    
    theta, phi, fov_x, fov_y = largest_sofa['bfov']
    print(f"原始BFoV: theta={theta:.4f}, phi={phi:.4f}, fov_x={fov_x:.4f}, fov_y={fov_y:.4f}")
    
    print(f"正在读取图像: {IMAGE_PATH}")
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise ValueError(f"无法读取图像: {IMAGE_PATH}")
    
    img_h, img_w = img.shape[:2]
    print(f"图像尺寸: {img_w} x {img_h}")
    
    print("正在绘制多尺度BFoV（原始视角）...")
    img_with_multi_bfov = draw_multi_scale_bfov_on_image(img, theta, phi, fov_x, fov_y, img_w, img_h)
    
    # 保存原始视角图像
    output_path = os.path.join(OUTPUT_DIR, 'multi_scale_bfov_' + IMAGE_FILENAME)
    cv2.imwrite(output_path, img_with_multi_bfov)
    print(f"\n原始视角图像已保存到: {output_path}")
    
    # ========== 绘制旋转视角 ==========
    print("\n=== 开始随机三维旋转 ===")
    max_angles = [60, 60, 60]  # Roll, Pitch, Yaw 的最大角度（度）
    rotate_ratio = 1.0  # 比例系数
    rotated_img, rotation_params = random_rotate_image_3d(img.copy(), max_angles, rotate_ratio)
    
    # 变换BFoV参数（应用同样的旋转）
    print("\n=== 变换BFoV参数 ===")
    transformed_bfov = transform_bfov_params([theta, phi, fov_x, fov_y], rotation_params, img_w, img_h)
    new_theta, new_phi, new_fov_x, new_fov_y = transformed_bfov
    
    print(f"变换后的BFoV: theta={new_theta:.4f}, phi={new_phi:.4f}, fov_x={new_fov_x:.4f}, fov_y={new_fov_y:.4f}")
    
    # 在旋转后的图像上绘制变换后的多尺度BFoV
    print("\n正在绘制多尺度BFoV（旋转视角）...")
    rotated_img_with_multi_bfov = draw_multi_scale_bfov_on_image(rotated_img, new_theta, new_phi, new_fov_x, new_fov_y, img_w, img_h)
    
    # 保存旋转视角图像
    rotated_output_path = os.path.join(OUTPUT_DIR, 'multi_scale_bfov_rotated_' + IMAGE_FILENAME)
    cv2.imwrite(rotated_output_path, rotated_img_with_multi_bfov)
    print(f"旋转视角图像已保存到: {rotated_output_path}")
    
    print("\n标注详情:")
    category_id = largest_sofa['category_id']
    category_name = CATEGORY_NAMES.get(category_id, f'category_{category_id}')
    print(f"category_id: {category_id} ({category_name})")
    print(f"theta: {theta:.6f} rad ({theta * 180/np.pi:.2f}°)")
    print(f"phi: {phi:.6f} rad ({phi * 180/np.pi:.2f}°)")
    print(f"fov_x: {fov_x:.6f} rad ({fov_x * 180/np.pi:.2f}°)")
    print(f"fov_y: {fov_y:.6f} rad ({fov_y * 180/np.pi:.2f}°)")
    
    print("\n多尺度多比例信息:")
    color_names = {0.67: '黄色', 1.0: '绿色', 1.5: '红色'}
    for scale in [0.67, 1.0, 1.5]:
        for scale_x, scale_y in [[0.5, 2], [1, 1], [2, 0.5]]:
            scaled_fov_x = fov_x * scale * scale_x
            scaled_fov_y = fov_y * scale * scale_y
            scaled_area = scaled_fov_x * scaled_fov_y
            fov_x_deg = scaled_fov_x * 180 / np.pi
            fov_y_deg = scaled_fov_y * 180 / np.pi
            
            # 显示限制信息
            clamped_info = ""
            if fov_x_deg > 175.0 or fov_y_deg > 175.0:
                clamped_info = f" (限制为175°)"
            
            print(f"  scale={scale}, ratio=[{scale_x}, {scale_y}]: FOV={fov_x_deg:.2f}°x{fov_y_deg:.2f}°{clamped_info}, 面积={scaled_area*(180/np.pi)**2:.2f} deg² ({color_names[scale]})")

if __name__ == '__main__':
    main()