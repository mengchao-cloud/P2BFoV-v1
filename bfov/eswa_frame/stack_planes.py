#!/usr/bin/env python3
"""
将person的切平面图片每三张叠加在一起，中心对齐，中间的图片放在最上面
叠加前给每张图片添加蓝色边框
额外生成一个九图叠加的图片
背景为透明（PNG格式）
"""

import os
import cv2
import numpy as np

# 配置
INPUT_DIR = './planes'
OUTPUT_DIR = './planes/stacks'
IMAGE_PREFIX = 'person_'
IMAGE_SUFFIX = '.jpg'
BORDER_COLOR = (255, 0, 0)  # 蓝色边框 (BGR格式)
BORDER_THICKNESS = 2  # 边框厚度

def add_blue_border(img, border_thickness=2):
    """
    给图片添加蓝色边框（带透明背景支持）
    
    参数:
    img: 输入图像 (BGR格式)
    border_thickness: 边框厚度（像素）
    
    返回:
    img_with_border: 添加边框后的图像 (RGBA格式)
    """
    h, w = img.shape[:2]
    
    # 创建一个比原图大的画布，用于放置带边框的图片 (RGBA格式)
    new_h = h + border_thickness * 2
    new_w = w + border_thickness * 2
    
    # 使用RGBA格式，初始全透明
    img_with_border = np.zeros((new_h, new_w, 4), dtype=np.uint8)
    
    # 设置蓝色边框（带完全不透明alpha）
    img_with_border[:, :, :3] = BORDER_COLOR  # 蓝色
    img_with_border[:, :, 3] = 255  # 完全不透明
    
    # 将原图复制到中心（添加alpha通道）
    # 创建带alpha通道的原图
    img_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    img_rgba[:, :, :3] = img
    img_rgba[:, :, 3] = 255  # 完全不透明
    
    # 复制到边框内部
    img_with_border[border_thickness:border_thickness+h, border_thickness:border_thickness+w] = img_rgba
    
    return img_with_border

def stack_images_center_aligned(image_paths, add_border=True):
    """
    将多张图片中心对齐叠加（透明背景）
    
    参数:
    image_paths: 图片路径列表，顺序为[底层, 中层, 顶层]
    add_border: 是否添加蓝色边框
    
    返回:
    stacked_img: 叠加后的图像 (RGBA格式)
    """
    # 读取所有图片并添加边框
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"警告：无法读取图像 {path}")
            return None
        
        # 添加蓝色边框（转换为RGBA）
        if add_border:
            img = add_blue_border(img, BORDER_THICKNESS)
        else:
            # 转换为RGBA
            h, w = img.shape[:2]
            img_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            img_rgba[:, :, :3] = img
            img_rgba[:, :, 3] = 255
            img = img_rgba
        
        images.append(img)
    
    # 计算所有图片的最大尺寸
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)
    
    # 创建一个足够大的画布（RGBA格式，透明背景）
    canvas_height = max_height
    canvas_width = max_width
    stacked_img = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
    # alpha通道初始化为0（完全透明）
    stacked_img[:, :, 3] = 0
    
    # 将每张图片叠加到画布上（中心对齐）
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        
        # 计算偏移量（中心对齐）
        offset_x = (canvas_width - w) // 2
        offset_y = (canvas_height - h) // 2
        
        # 获取alpha掩码（只复制非透明区域）
        alpha_mask = img[:, :, 3] > 0
        
        # 将图片复制到画布上（只复制非透明部分）
        for c in range(4):
            stacked_img[offset_y:offset_y+h, offset_x:offset_x+w, c][alpha_mask] = img[:, :, c][alpha_mask]
    
    return stacked_img

def main():
    print("开始叠加切平面图片（带蓝色边框，透明背景）...")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 定义三组图片
    groups = [
        ['person_01.jpg', 'person_02.jpg', 'person_03.jpg'],
        ['person_04.jpg', 'person_05.jpg', 'person_06.jpg'],
        ['person_07.jpg', 'person_08.jpg', 'person_09.jpg'],
    ]
    
    # 处理每组（三张叠加）
    for i, group in enumerate(groups, 1):
        print(f"\n处理第 {i} 组: {group}")
        
        # 获取完整路径
        image_paths = [os.path.join(INPUT_DIR, img_name) for img_name in group]
        
        # 检查所有文件是否存在
        all_exists = all(os.path.exists(path) for path in image_paths)
        if not all_exists:
            print(f"警告：第 {i} 组的某些文件不存在，跳过")
            continue
        
        # 叠加图片（顺序：底层, 中层, 顶层）
        # 用户要求中间的图片在最上面，所以顺序是 [0, 2, 1]
        stacked_img = stack_images_center_aligned([
            image_paths[0],  # 底层
            image_paths[2],  # 中层
            image_paths[1],  # 顶层（中间的图片）
        ], add_border=True)
        
        if stacked_img is not None:
            output_path = os.path.join(OUTPUT_DIR, f'stack_{i:02d}_border.png')
            cv2.imwrite(output_path, stacked_img)
            print(f"  叠加完成，保存到: {output_path}")
        else:
            print(f"  叠加失败")
    
    # 额外生成一个九图叠加的图片
    print("\n处理九图叠加...")
    all_person_images = [f'person_{i:02d}.jpg' for i in range(1, 10)]
    all_image_paths = [os.path.join(INPUT_DIR, img_name) for img_name in all_person_images]
    
    # 检查所有文件是否存在
    all_exists = all(os.path.exists(path) for path in all_image_paths)
    if all_exists:
        # 九图叠加：按大小顺序，最大的在最底层，最小的在最顶层
        # person_01~03 是小图，04~06 是中图，07~09 是大图
        # 叠加顺序：07, 09, 08, 04, 06, 05, 01, 03, 02（每组中间的放该组最上面）
        nine_stack_img = stack_images_center_aligned([
            all_image_paths[6],   # person_07 - 底层（最大）
            all_image_paths[8],   # person_09
            all_image_paths[7],   # person_08
            all_image_paths[3],   # person_04（中等）
            all_image_paths[5],   # person_06
            all_image_paths[4],   # person_05
            all_image_paths[0],   # person_01（最小）
            all_image_paths[2],   # person_03
            all_image_paths[1],   # person_02 - 顶层（最小）
        ], add_border=True)
        
        if nine_stack_img is not None:
            output_path = os.path.join(OUTPUT_DIR, 'stack_nine_border.png')
            cv2.imwrite(output_path, nine_stack_img)
            print(f"  九图叠加完成，保存到: {output_path}")
        else:
            print(f"  九图叠加失败")
    else:
        print("警告：某些图片不存在，跳过九图叠加")
    
    print("\n所有组处理完成！")

if __name__ == '__main__':
    main()
