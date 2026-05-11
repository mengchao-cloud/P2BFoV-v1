import json
import os
import cv2
import numpy as np
import torch

from PANDORA.PRDA.lib.ReuseGPUImageRecorder import ReuseGPUImageRecorder

# 输入输出配置
MERGED_COCO_FILE = './bfov/temp_json/merge_result/lunwen/test_coco.json'
# MERGED_COCO_FILE = '../360indoor/ann/test_coco.json'
IMAGE_DIR = '../360indoor/images/'
OUTPUT_DIR = './bfov/temp_json/result_image/'

# 可视化配置
ERP_WIDTH = 1920
ERP_HEIGHT = 960
MIN_SCORE = 0  # 过滤低分数检测结果

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

def visualize_bfov_on_image(img, bfov_params, score=None, category_id=None):
    """
    在图像上可视化单个BFOV
    
    参数:
    img: 输入图像
    bfov_params: BFOV参数 [center_x, center_y, fov_x, fov_y]，弧度制
    score: 检测分数
    category_id: 类别ID
    
    返回:
    img: 绘制后的图像
    """
    # 提取BFOV参数并转换为角度制
    center_x, center_y, fov_x, fov_y = bfov_params
    center_x_deg = center_x * 180 / np.pi
    center_y_deg = center_y * 180 / np.pi
    fov_x_deg = fov_x * 180 / np.pi
    fov_y_deg = fov_y * 180 / np.pi
    
    # 验证BFOV参数，确保视场角在合理范围内
    # 限制视场角最大为170度，避免tan函数计算问题
    max_fov_deg = 170
    fov_x_deg = max(-max_fov_deg, min(max_fov_deg, fov_x_deg))
    fov_y_deg = max(-max_fov_deg, min(max_fov_deg, fov_y_deg))
    
    # 确保视场角大于0
    fov_x_deg = max(1, abs(fov_x_deg))
    fov_y_deg = max(1, abs(fov_y_deg))
    
    try:
        # 创建ReuseGPUImageRecorder实例
        BFoV = ReuseGPUImageRecorder(ERP_WIDTH, ERP_HEIGHT, 
                                    view_angle_w=fov_x_deg, view_angle_h=fov_y_deg, 
                                    long_side=ERP_WIDTH, device='cuda')
        
        # 将中心点角度转换为弧度制（ImageRecorder需要）
        center_x_rad = center_x
        center_y_rad = center_y
        
        # 获取BFOV的边界点
        Px, Py = BFoV._sample_points(center_x_rad, center_y_rad, border_only=True)
        
        # 将GPU张量转换为numpy数组
        Px = Px.cpu().numpy()
        Py = Py.cpu().numpy()
        
        # 绘制BFOV边界
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
        
        # 绘制边界点
        if Px.shape[0] > 0:
            # 将坐标转换为整数
            points = np.array([[int(px), int(py)] for px, py in zip(Px, Py)])
            # 绘制点
            for point in points:
                cv2.circle(img, tuple(point), 2, color, -1)
        
        # 绘制BFOV中心点
        center_px = int((center_x_rad + np.pi) / (2 * np.pi) * ERP_WIDTH)
        center_py = int((np.pi/2 - center_y_rad) / np.pi * ERP_HEIGHT)
        cv2.circle(img, (center_px, center_py), 3, color, -1)
        
        # 不绘制分数和类别信息
    except Exception as e:
        # 处理异常，记录错误但不中断程序
        print(f"可视化BFOV时发生错误: {e}")
        print(f"弧度制BFOV参数: center_x={bfov_params[0]:.4f}, center_y={bfov_params[1]:.4f}, fov_x={bfov_params[2]:.4f}, fov_y={bfov_params[3]:.4f}")
        # 转换为角度制打印
        center_x_deg = bfov_params[0] * 180 / np.pi
        center_y_deg = bfov_params[1] * 180 / np.pi
        fov_x_deg = bfov_params[2] * 180 / np.pi
        fov_y_deg = bfov_params[3] * 180 / np.pi
        print(f"角度制BFOV参数: center_x={center_x_deg:.2f}°, center_y={center_y_deg:.2f}°, fov_x={fov_x_deg:.2f}°, fov_y={fov_y_deg:.2f}°")
        # 使用简单的矩形绘制作为备选方案
        center_x, center_y, fov_x, fov_y = bfov_params
        # 将球面坐标转换为平面坐标
        center_px = int((center_x + np.pi) / (2 * np.pi) * ERP_WIDTH)
        center_py = int((np.pi/2 - center_y) / np.pi * ERP_HEIGHT)
        width_px = int(fov_x / (2 * np.pi) * ERP_WIDTH)
        height_px = int(fov_y / np.pi * ERP_HEIGHT)
        
        # 确保坐标在图像范围内
        x1 = max(0, center_px - width_px // 2)
        y1 = max(0, center_py - height_px // 2)
        x2 = min(ERP_WIDTH, center_px + width_px // 2)
        y2 = min(ERP_HEIGHT, center_py + height_px // 2)
        
        # 绘制简单矩形
        colors = {
            0: (0, 255, 0),    # 绿色
            1: (255, 0, 0),    # 蓝色
            2: (0, 0, 255),    # 红色
            3: (255, 255, 0),  # 黄色
            4: (255, 0, 255),  # 紫色
            5: (0, 255, 255)   # 青色
        }
        color = colors.get(category_id % len(colors), (128, 128, 128))  # 默认灰色
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)  # 线宽从 2 改为 1
        
        # 绘制中心点
        cv2.circle(img, (center_px, center_py), 3, color, -1)
        
        # 不绘制分数和类别信息
    
    return img

def main():
    """
    批量可视化BFOV检测结果
    """
    print("开始批量可视化BFOV检测结果...")
    
    # 加载合并后的COCO文件
    with open(MERGED_COCO_FILE, 'r') as f:
        coco_data = json.load(f)
    
    # 创建图像ID到文件名的映射
    image_id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # 创建图像ID到标注的映射
    image_id_to_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_id_to_annotations:
            image_id_to_annotations[image_id] = []
        image_id_to_annotations[image_id].append(ann)
    
    # 处理每个图像
    processed_count = 0
    error_count = 0
    
    for image_id, file_name in image_id_to_file.items():
        # 构建图像路径
        image_path = os.path.join(IMAGE_DIR, file_name)
        
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
        
        # 绘制所有BFOV
        drawn_count = 0
        for ann in annotations:
            # 获取BFOV参数（优先使用bfov字段，其次使用bbox字段）
            if 'bfov' in ann:
                bfov_params = ann['bfov']
            else:
                bfov_params = ann['bbox']
            
            # 检查BFOV参数长度
            if len(bfov_params) < 4:
                continue
            
            # 过滤低分数结果
            score = ann.get('score', 1.0)
            if score < MIN_SCORE:
                continue
            
            # 绘制BFOV
            category_id = ann['category_id']
            img = visualize_bfov_on_image(img, bfov_params, score=score, category_id=category_id)
            drawn_count += 1
        
        # 保存结果图像
        output_path = os.path.join(OUTPUT_DIR, f"result_{file_name}")
        cv2.imwrite(output_path, img)
        
        processed_count += 1
        print(f"已处理图像 {file_name}，绘制了 {drawn_count} 个BFOV，保存到 {output_path}")
    
    print(f"\n批量可视化完成！")
    print(f"- 处理图像数: {processed_count}")
    print(f"- 错误图像数: {error_count}")
    print(f"- 输出目录: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
