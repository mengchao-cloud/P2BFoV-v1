import numpy as np
import cv2
import json
import os
import sys
from pathlib import Path

# 添加本地lib目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
from ImageRecorder import ImageRecorder

def visualize_comparison(inference_json_path, gt_visual_dir, output_dir='visual-comparance'):
    """
    将推理结果中的BFoV参数绘制到GT可视化图像上进行比较
    
    参数:
    inference_json_path: 推理结果JSON文件路径
    gt_visual_dir: GT可视化图像文件夹路径
    output_dir: 输出文件夹路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取推理JSON文件
    with open(inference_json_path, 'r') as f:
        inference_data = json.load(f)
    
    # 按图像ID分组推理结果
    inference_by_image = {}
    for detection in inference_data:
        image_id = detection['image_id']
        if image_id not in inference_by_image:
            inference_by_image[image_id] = []
        inference_by_image[image_id].append(detection)
    
    # 获取GT可视化图像文件列表
    gt_images = [f for f in os.listdir(gt_visual_dir) if f.startswith('visual_') and f.endswith('.jpg')]
    
    print(f"找到 {len(gt_images)} 个GT可视化图像")
    print("开始叠加推理结果...")
    print("-" * 50)
    
    # 处理每个图像
    for gt_image_name in gt_images:
        gt_image_path = os.path.join(gt_visual_dir, gt_image_name)
        
        # 从文件名提取图像ID
        # 假设文件名为 visual_原文件名.jpg，需要映射到image_id
        # 这里需要根据实际情况建立文件名到image_id的映射
        
        # 读取GT可视化图像
        img = cv2.imread(gt_image_path)
        if img is None:
            print(f"警告: 无法读取GT图像: {gt_image_path}")
            continue
        
        # 获取图像尺寸
        img_h, img_w = img.shape[:2]
        
        # 尝试根据文件名推断image_id
        # 这里需要根据您的实际文件名格式进行调整
        image_id = None
        if '37936706382_45b3fbc711_f' in gt_image_name:
            image_id = 1
        elif '7fzx6' in gt_image_name:
            image_id = 2
        elif '7PhJB' in gt_image_name:
            image_id = 3
        
        if image_id is None:
            print(f"警告: 无法确定图像 {gt_image_name} 的ID，跳过")
            continue
        
        # 检查该图像是否有推理结果
        if image_id not in inference_by_image:
            print(f"图像ID {image_id} 没有推理结果，跳过")
            continue
        
        detections = inference_by_image[image_id]
        print(f"处理图像: {gt_image_name} (ID: {image_id}), 包含 {len(detections)} 个推理BFoV")
        
        # 绘制每个推理BFoV（使用红色虚线）
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            score = detection['score']
            category_id = detection['category_id']
            ann_id = detection['ann_id']
            
            # 提取BFoV参数 [center_x, center_y, fov_x, fov_y] (弧度制)
            center_x_rad, center_y_rad, fov_x_rad, fov_y_rad = bbox
            
            # 转换为角度制用于显示
            center_x_deg = center_x_rad * 180 / np.pi
            center_y_deg = center_y_rad * 180 / np.pi
            fov_x_deg = fov_x_rad * 180 / np.pi
            fov_y_deg = fov_y_rad * 180 / np.pi
            
            # 检查FOV参数是否为负值，如果是则跳过绘制
            if fov_x_deg <= 0 or fov_y_deg <= 0:
                print(f"  推理BFoV {i+1}: 中心({center_x_deg:.2f}°, {center_y_deg:.2f}°), "
                      f"FOV({fov_x_deg:.2f}°x{fov_y_deg:.2f}°) - FOV参数为负，跳过绘制")
                continue
            
            print(f"  推理BFoV {i+1}: 中心({center_x_deg:.2f}°, {center_y_deg:.2f}°), "
                  f"FOV({fov_x_deg:.2f}°x{fov_y_deg:.2f}°), 类别: {category_id}, 得分: {score:.4f}")
            
            # 使用蓝色细线绘制推理结果（与GT的绿色区分）
            color = (255, 0, 0)  # 蓝色
            
            # 使用ImageRecorder绘制推理BFoV
            try:
                # 为每个BFoV创建新的ImageRecorder实例（使用角度制FOV参数）
                BFoV = ImageRecorder(img_w, img_h, 
                                   view_angle_w=fov_x_deg, 
                                   view_angle_h=fov_y_deg, 
                                   long_side=img_w)
                
                # 1. 获取BFoV的边界点
                Px, Py = BFoV._sample_points(center_x_rad, center_y_rad, border_only=True)
                
                # 2. 使用原始的draw_BFoV方法绘制BFoV
                img = BFoV.draw_BFoV(img, Px, Py, border_only=True, color=color)
                
                # 3. 绘制BFoV中心点
                center_px = int((center_x_rad + np.pi) / (2 * np.pi) * img_w)
                center_py = int((np.pi/2 - center_y_rad) / np.pi * img_h)
                cv2.circle(img, (center_px, center_py), 3, color, -1)
                
                # 4. 添加推理标签（带得分）
                label = f"P{category_id}({score:.3f})"
                cv2.putText(img, label, (center_px + 8, center_py), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
            except Exception as e:
                print(f"    警告: 绘制推理BFoV {i+1} 时出错: {e}")
                continue
        
        # 保存比较结果图像
        output_path = os.path.join(output_dir, f"comparison_{gt_image_name}")
        cv2.imwrite(output_path, img)
        print(f"    比较结果已保存到: {output_path}")
        print()

def main():
    # 设置路径
    inference_json_path = '/mnt/c/mengchao/shared/wsl/P2BFoV-v4几乎解决了所有问题目前进行测试的最终版本/P2BNet-main/TOV_mmdetection/bfov/inference-json/_1200_latest_result.json'
    gt_visual_dir = '/mnt/c/mengchao/shared/wsl/P2BFoV-v4几乎解决了所有问题目前进行测试的最终版本/P2BNet-main/TOV_mmdetection/bfov/visual-bfov'
    output_dir = 'visual-comparance'
    
    # 检查路径是否存在
    if not os.path.exists(inference_json_path):
        print(f"错误: 推理JSON文件不存在: {inference_json_path}")
        return
    
    if not os.path.exists(gt_visual_dir):
        print(f"错误: GT可视化文件夹不存在: {gt_visual_dir}")
        return
    
    print("开始生成推理与GT的比较可视化...")
    print(f"推理文件: {inference_json_path}")
    print(f"GT可视化文件夹: {gt_visual_dir}")
    print(f"输出文件夹: {output_dir}")
    print("-" * 50)
    
    # 执行比较可视化
    visualize_comparison(inference_json_path, gt_visual_dir, output_dir)
    
    print("-" * 50)
    print("比较可视化完成!")
    print("说明:")
    print("- 绿色细线边界框: GT标注")
    print("- 蓝色细线边界框: 推理结果")
    print("- 蓝色细十字: 推理中心点")
    print("- P{类别}(得分): 推理标签")

if __name__ == '__main__':
    main()