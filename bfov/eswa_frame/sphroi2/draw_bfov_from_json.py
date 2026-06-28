#!/usr/bin/env python3
"""
从COCO格式的JSON标注文件中读取BFoV标注并绘制风扇
"""

import json
import numpy as np
import cv2
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../sphdet/visualizers'))
from ImageRecorder import ImageRecorder as BFoV

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

CATEGORY_COLORS = {
    10: (0, 255, 255),  # fan - 青色
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

def draw_bfov_on_image(img, annotations, img_w, img_h):
    img_copy = img.copy()
    
    for i, ann in enumerate(annotations):
        if 'bfov' not in ann:
            continue
        
        theta, phi, fov_x, fov_y = ann['bfov']
        category_id = ann['category_id']
        
        theta_deg = theta * 180 / np.pi
        phi_deg = phi * 180 / np.pi
        fov_x_deg = fov_x * 180 / np.pi
        fov_y_deg = fov_y * 180 / np.pi
        
        if fov_x_deg <= 0 or fov_y_deg <= 0:
            print(f"  跳过无效BFoV {i+1}: FOV参数为负")
            continue
        
        category_name = CATEGORY_NAMES.get(category_id, f'category_{category_id}')
        color = CATEGORY_COLORS.get(category_id, (128, 128, 128))
        
        print(f"  绘制BFoV {i+1}: {category_name}")
        print(f"    中心: ({theta_deg:.2f}°, {phi_deg:.2f}°)")
        print(f"    FOV:  ({fov_x_deg:.2f}° x {fov_y_deg:.2f}°)")
        
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
            
            cx = int(((theta + np.pi) / (2 * np.pi)) * img_w)
            cy = int(((np.pi / 2 - phi) / np.pi) * img_h)
            cv2.circle(img_copy, (cx, cy), 3, color, -1, lineType=cv2.LINE_AA)
            
        except Exception as e:
            print(f"  绘制BFoV {i+1}失败: {e}")
            import traceback
            traceback.print_exc()
    
    return img_copy

def main():
    IMAGE_FILENAME = '7l0Hp.jpg'
    DATA_ROOT = '/home/mengchao/workspace/P2BFoV/P2BNet-main/360indoor/'
    ANN_FILE = DATA_ROOT + 'ann/train_coco.json'
    IMAGE_PATH = DATA_ROOT + 'images/' + IMAGE_FILENAME
    OUTPUT_DIR = '/home/mengchao/workspace/P2BFoV/P2BNet-main/TOV_mmdetection/bfov/eswa_frame/sphroi2/'
    
    TARGET_CATEGORY_ID = 10
    
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
    
    print("正在绘制BFoV标注...")
    img_with_bfov = draw_bfov_on_image(img, [largest_fan], img_w, img_h)
    
    output_path = os.path.join(OUTPUT_DIR, 'bfov_fan_' + IMAGE_FILENAME)
    cv2.imwrite(output_path, img_with_bfov)
    print(f"\n结果已保存到: {output_path}")
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, IMAGE_FILENAME), img)
    print(f"原图已复制到: {os.path.join(OUTPUT_DIR, IMAGE_FILENAME)}")
    
    print("\n标注详情:")
    theta, phi, fov_x, fov_y = largest_fan['bfov']
    category_id = largest_fan['category_id']
    category_name = CATEGORY_NAMES.get(category_id, f'category_{category_id}')
    print(f"category_id: {category_id} ({category_name})")
    print(f"theta: {theta:.6f} rad ({theta * 180/np.pi:.2f}°)")
    print(f"phi: {phi:.6f} rad ({phi * 180/np.pi:.2f}°)")
    print(f"fov_x: {fov_x:.6f} rad ({fov_x * 180/np.pi:.2f}°)")
    print(f"fov_y: {fov_y:.6f} rad ({fov_y * 180/np.pi:.2f}°)")
    print(f"面积: {fov_x * fov_y:.6f} rad² ({fov_x * fov_y * (180/np.pi)**2:.2f} deg²)")
    
    return largest_fan

if __name__ == '__main__':
    main()
