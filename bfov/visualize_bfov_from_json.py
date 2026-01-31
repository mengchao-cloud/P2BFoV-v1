import numpy as np
import cv2
import json
import os
import sys
from pathlib import Path

# 添加本地lib目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
from ImageRecorder import ImageRecorder

def visualize_bfov_from_json(json_path, images_dir, output_dir='visual-bfov'):
    """
    从JSON文件读取BFOV参数并在对应图像上可视化
    
    参数:
    json_path: JSON文件路径
    images_dir: 图像文件夹路径
    output_dir: 输出文件夹路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取JSON文件
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 创建图像ID到文件名的映射
    image_id_to_filename = {}
    for image_info in data['images']:
        image_id_to_filename[image_info['id']] = image_info['file_name']
    
    # 按图像ID分组注释
    annotations_by_image = {}
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(annotation)
    
    # 处理每个图像
    for image_id, annotations in annotations_by_image.items():
        if image_id not in image_id_to_filename:
            print(f"警告: 图像ID {image_id} 在images列表中不存在")
            continue
            
        filename = image_id_to_filename[image_id]
        image_path = os.path.join(images_dir, filename)
        
        # 检查图像文件是否存在
        if not os.path.exists(image_path):
            print(f"警告: 图像文件不存在: {image_path}")
            continue
        
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"警告: 无法读取图像: {image_path}")
            continue
        
        # 获取图像尺寸
        img_h, img_w = img.shape[:2]
        
        print(f"处理图像: {filename} (ID: {image_id}), 包含 {len(annotations)} 个BFOV")
        
        # 为每个BFOV分配不同的颜色
        colors = [
            (0, 255, 0),   # 绿色
            (255, 0, 0),   # 蓝色
            (0, 0, 255),   # 红色
            (255, 255, 0), # 青色
            (255, 0, 255), # 品红色
            (0, 255, 255), # 黄色
            (128, 0, 0),   # 深蓝色
            (0, 128, 0),   # 深绿色
            (0, 0, 128),   # 深红色
            (128, 128, 0), # 橄榄色
        ]
        
        # 绘制每个BFOV
        for i, annotation in enumerate(annotations):
            bfov_params = annotation['bfov']
            category_id = annotation['category_id']
            
            # 提取BFOV参数 [center_x, center_y, fov_x, fov_y] (弧度制)
            center_x_rad, center_y_rad, fov_x_rad, fov_y_rad = bfov_params
            
            # 转换为角度制用于显示
            center_x_deg = center_x_rad * 180 / np.pi
            center_y_deg = center_y_rad * 180 / np.pi
            fov_x_deg = fov_x_rad * 180 / np.pi
            fov_y_deg = fov_y_rad * 180 / np.pi
            
            # 检查FOV参数是否为负值，如果是则跳过绘制
            if fov_x_deg <= 0 or fov_y_deg <= 0:
                print(f"  BFOV {i+1}: 中心({center_x_deg:.2f}°, {center_y_deg:.2f}°), "
                      f"FOV({fov_x_deg:.2f}°x{fov_y_deg:.2f}°) - FOV参数为负，跳过绘制")
                continue
            
            print(f"  BFOV {i+1}: 中心({center_x_deg:.2f}°, {center_y_deg:.2f}°), "
                  f"FOV({fov_x_deg:.2f}°x{fov_y_deg:.2f}°), 类别: {category_id}")
            
            # 使用统一的绿色细线
            color = (0, 255, 0)  # 绿色
            
            # 使用ImageRecorder绘制BFOV
            try:
                # 为每个BFOV创建新的ImageRecorder实例（使用角度制FOV参数）
                BFoV = ImageRecorder(img_w, img_h, 
                                   view_angle_w=fov_x_deg, 
                                   view_angle_h=fov_y_deg, 
                                   long_side=img_w)
                
                # 1. 获取BFOV的边界点
                Px, Py = BFoV._sample_points(center_x_rad, center_y_rad, border_only=True)
                
                # 2. 使用原始的draw_BFoV方法绘制BFOV
                img = BFoV.draw_BFoV(img, Px, Py, border_only=True, color=color)
                
                # 3. 绘制BFOV中心点
                center_px = int((center_x_rad + np.pi) / (2 * np.pi) * img_w)
                center_py = int((np.pi/2 - center_y_rad) / np.pi * img_h)
                cv2.circle(img, (center_px, center_py), 3, color, -1)
                
                # 4. 添加类别标签
                label = f"C{category_id}"
                cv2.putText(img, label, (center_px + 5, center_py), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
            except Exception as e:
                print(f"    警告: 绘制BFOV {i+1} 时出错: {e}")
                continue
        
        # 保存结果图像
        output_path = os.path.join(output_dir, f"visual_{filename}")
        cv2.imwrite(output_path, img)
        print(f"    结果已保存到: {output_path}")
        print()

def main():
    # 设置路径
    json_path = '/mnt/c/mengchao/shared/wsl/P2BFoV-v4几乎解决了所有问题目前进行测试的最终版本/P2BNet-main/TOV_mmdetection/360indoor-short/ann/train_coco_short.json'
    images_dir = '/mnt/c/mengchao/shared/wsl/P2BFoV-v4几乎解决了所有问题目前进行测试的最终版本/P2BNet-main/TOV_mmdetection/360indoor-short/images_short'
    output_dir = 'visual-bfov'
    
    # 检查路径是否存在
    if not os.path.exists(json_path):
        print(f"错误: JSON文件不存在: {json_path}")
        return
    
    if not os.path.exists(images_dir):
        print(f"错误: 图像文件夹不存在: {images_dir}")
        return
    
    print("开始可视化BFOV...")
    print(f"JSON文件: {json_path}")
    print(f"图像文件夹: {images_dir}")
    print(f"输出文件夹: {output_dir}")
    print("-" * 50)
    
    # 执行可视化
    visualize_bfov_from_json(json_path, images_dir, output_dir)
    
    print("-" * 50)
    print("BFOV可视化完成!")

if __name__ == '__main__':
    main()