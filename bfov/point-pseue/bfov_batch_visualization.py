import json
import os
import cv2
import numpy as np
import torch

from PANDORA.PRDA.lib.ReuseGPUImageRecorder import ReuseGPUImageRecorder

# 输入输出配置
MERGED_COCO_FILE = './bfov/point-pseue/test_coco.json'
# 指定要处理的图像文件
TARGET_IMAGES = [
    '7lBhd.jpg',
    '7l8xM.jpg', 
    '7l8ds.jpg'
]
# 图像目录（使用脚本所在目录）
IMAGE_DIR = '../360indoor/images/'
# 输出目录
OUTPUT_DIR = './bfov/point-pseue/result_image/'

# 可视化配置
ERP_WIDTH = 1920
ERP_HEIGHT = 960
MIN_SCORE = 0  # 过滤低分数检测结果

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

def visualize_bbox_center(img, bbox_params, category_id=None):
    """
    在图像上只绘制bbox的中心坐标
    
    参数:
    img: 输入图像
    bbox_params: bbox参数 [x1, y1, width, height]，像素坐标
    category_id: 类别ID
    
    返回:
    img: 绘制后的图像
    """
    # 提取bbox参数（正常矩形框格式）
    x1, y1, width, height = bbox_params
    
    # 根据类别ID选择颜色
    colors = {
        0: (0, 255, 0),    # 绿色
        1: (255, 0, 0),    # 蓝色
        2: (0, 0, 255),    # 红色
        3: (255, 255, 0),  # 黄色
        4: (255, 0, 255),  # 紫色
        5: (0, 255, 255)   # 青色
    }
    color = colors.get(category_id % len(colors), (128, 128, 128))  # 默认灰色
    
    # 计算中心坐标（从矩形框坐标计算）
    center_px = int(x1 + width / 2)
    center_py = int(y1 + height / 2)
    
    # 添加随机抖动，抖动幅度为box宽高的0.1倍，最小为2像素
    max_jitter_x = max(2, int(width * 0.1))
    max_jitter_y = max(2, int(height * 0.1))
    jitter_x = np.random.randint(-max_jitter_x, max_jitter_x + 1)
    jitter_y = np.random.randint(-max_jitter_y, max_jitter_y + 1)
    center_px += jitter_x
    center_py += jitter_y
    
    # 确保点在图像范围内
    center_px = max(5, min(ERP_WIDTH - 5, center_px))
    center_py = max(5, min(ERP_HEIGHT - 5, center_py))
    
    # 绘制中心点（使用较大的点以便于观察）
    cv2.circle(img, (center_px, center_py), 10, color, -1)
    
    return img

def main():
    """
    处理指定的三个图像，只绘制bbox中心坐标
    """
    print("开始处理指定图像，只绘制bbox中心坐标...")
    
    # 加载COCO文件
    with open(MERGED_COCO_FILE, 'r') as f:
        coco_data = json.load(f)
    
    # 创建文件名到图像ID的映射
    file_to_image_id = {img['file_name']: img['id'] for img in coco_data['images']}
    
    # 创建图像ID到标注的映射
    image_id_to_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_id_to_annotations:
            image_id_to_annotations[image_id] = []
        image_id_to_annotations[image_id].append(ann)
    
    # 处理指定的三个图像
    processed_count = 0
    error_count = 0
    
    for target_image in TARGET_IMAGES:
        # 检查图像是否在COCO文件中
        if target_image not in file_to_image_id:
            print(f"警告：图像 {target_image} 不在COCO文件中")
            error_count += 1
            continue
        
        # 获取图像ID
        image_id = file_to_image_id[target_image]
        
        # 构建图像路径
        image_path = os.path.join(IMAGE_DIR, target_image)
        
        # 检查图像是否存在
        if not os.path.exists(image_path):
            print(f"警告：图像不存在 {image_path}")
            error_count += 1
            continue
        
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"警告：无法读取图像 {image_path}")
            error_count += 1
            continue
        
        # 调整图像尺寸
        img = cv2.resize(img, (ERP_WIDTH, ERP_HEIGHT))
        
        # 获取该图像的所有标注
        annotations = image_id_to_annotations.get(image_id, [])
        
        # 绘制所有bbox中心坐标
        drawn_count = 0
        for ann in annotations:
            # 获取bbox参数
            if 'bbox' in ann:
                bbox_params = ann['bbox']
            elif 'bfov' in ann:
                bbox_params = ann['bfov']
            else:
                continue
            
            # 检查bbox参数长度
            if len(bbox_params) < 4:
                continue
            
            # 过滤低分数结果
            score = ann.get('score', 1.0)
            if score < MIN_SCORE:
                continue
            
            # 绘制bbox中心坐标
            category_id = ann['category_id']
            img = visualize_bbox_center(img, bbox_params, category_id=category_id)
            drawn_count += 1
        
        # 保存结果图像
        output_path = os.path.join(OUTPUT_DIR, f"result_{target_image}")
        cv2.imwrite(output_path, img)
        
        processed_count += 1
        print(f"已处理图像 {target_image}，绘制了 {drawn_count} 个bbox中心，保存到 {output_path}")
    
    print(f"\n处理完成！")
    print(f"- 处理图像数: {processed_count}")
    print(f"- 错误图像数: {error_count}")
    print(f"- 输出目录: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
