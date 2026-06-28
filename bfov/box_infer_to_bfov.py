import json
import os
import cv2
import numpy as np

# 导入ImageRecorder库
import sys
sys.path.append('..')
from PANDORA.PRDA.lib.ImageRecorder import ImageRecorder

# 输入输出配置
INPUT_FILE = './box_infer/box_infer.json'
OUTPUT_FILE = './box_to_bfov.json'
OUTPUT_DIR = './box_to_bfov_visualization/'

# 图像配置
IMG_WIDTH = 1920
IMG_HEIGHT = 960

# 可视化前5个结果
VISUALIZE_COUNT = 5

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

def box_to_bfov(box, img_width, img_height):
    """
    将2D边界框(box)转换为BFOV参数
    
    参数:
    box: 边界框 [x, y, width, height]，像素坐标
    img_width: 图像宽度
    img_height: 图像高度
    
    返回:
    bfov_params: BFOV参数 [center_x, center_y, fov_x, fov_y]，弧度制
    """
    # 提取box参数
    x, y, width, height = box
    
    # 计算中心点坐标（像素）
    center_px = x + width / 2
    center_py = y + height / 2
    
    # 将像素坐标转换为球面坐标（弧度）
    # 经度范围: [-π, π]
    # 纬度范围: [-π/2, π/2]
    center_x = (center_px / img_width) * 2 * np.pi - np.pi
    center_y = np.pi/2 - (center_py / img_height) * np.pi
    
    # 计算视场角（弧度）
    # 水平视场角: 基于box宽度
    fov_x = (width / img_width) * 2 * np.pi
    # 垂直视场角: 基于box高度
    fov_y = (height / img_height) * np.pi
    
    # 确保视场角在合理范围内
    max_fov = np.pi * 0.9  # 最大视场角为162度
    fov_x = min(fov_x, max_fov)
    fov_y = min(fov_y, max_fov)
    
    return [center_x, center_y, fov_x, fov_y]

def visualize_box_and_bfov(box, bfov_params, output_path):
    """
    在空白背景上可视化box和BFOV
    
    参数:
    box: 边界框 [x, y, width, height]，像素坐标
    bfov_params: BFOV参数 [center_x, center_y, fov_x, fov_y]，弧度制
    output_path: 输出图像路径
    """
    # 创建白色背景
    img = np.ones((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8) * 255
    
    # 绘制box
    x, y, width, height = box
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + width), int(y + height)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框
    
    # 绘制BFOV边界
    center_x, center_y, fov_x, fov_y = bfov_params
    
    # 将弧度转换为角度
    fov_x_deg = fov_x * 180 / np.pi
    fov_y_deg = fov_y * 180 / np.pi
    
    # 确保视场角在合理范围内
    max_fov_deg = 170
    fov_x_deg = max(-max_fov_deg, min(max_fov_deg, fov_x_deg))
    fov_y_deg = max(-max_fov_deg, min(max_fov_deg, fov_y_deg))
    
    # 确保视场角大于0
    fov_x_deg = max(1, abs(fov_x_deg))
    fov_y_deg = max(1, abs(fov_y_deg))
    
    try:
        # 创建ImageRecorder实例
        BFoV = ImageRecorder(IMG_WIDTH, IMG_HEIGHT, 
                            view_angle_w=fov_x_deg, view_angle_h=fov_y_deg, 
                            long_side=IMG_WIDTH)
        
        # 获取BFOV的边界点
        Px, Py = BFoV._sample_points(center_x, center_y, border_only=True)
        
        # 绘制BFOV边界
        img = BFoV.draw_BFoV(img, Px, Py, border_only=True, color=(255, 0, 0))  # 蓝色边界
    except Exception as e:
        # 处理异常，记录错误但不中断程序
        print(f"可视化BFOV时发生错误: {e}")
        # 使用简单的矩形绘制作为备选方案
        # 将BFOV中心点转换为像素坐标
        center_px = int((center_x + np.pi) / (2 * np.pi) * IMG_WIDTH)
        center_py = int((np.pi/2 - center_y) / np.pi * IMG_HEIGHT)
        # 计算BFOV在图像上的大致范围
        width_px = int(fov_x / (2 * np.pi) * IMG_WIDTH)
        height_px = int(fov_y / np.pi * IMG_HEIGHT)
        # 绘制BFOV边界
        bfov_x1 = max(0, center_px - width_px // 2)
        bfov_y1 = max(0, center_py - height_px // 2)
        bfov_x2 = min(IMG_WIDTH, center_px + width_px // 2)
        bfov_y2 = min(IMG_HEIGHT, center_py + height_px // 2)
        # 绘制BFOV边界（蓝色虚线）
        cv2.rectangle(img, (bfov_x1, bfov_y1), (bfov_x2, bfov_y2), (255, 0, 0), 2, cv2.LINE_8)
    
    # 绘制中心点
    center_px = int((center_x + np.pi) / (2 * np.pi) * IMG_WIDTH)
    center_py = int((np.pi/2 - center_y) / np.pi * IMG_HEIGHT)
    cv2.circle(img, (center_px, center_py), 3, (0, 0, 255), -1)  # 红色中心点
    
    # 添加文字说明
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Box', (x1 + 10, y1 - 10), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img, 'BFOV', (center_px + 10, center_py - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    
    # 保存图像
    cv2.imwrite(output_path, img)
    print(f"可视化结果已保存到: {output_path}")

def main():
    """
    从box_infer.json读取bbox信息，生成bfov，替换原有的bfov，可视化前5个结果，输出为box_to_bfov.json
    """
    print("从box_infer.json生成bfov...")
    print("=" * 50)
    
    # 加载输入文件
    if not os.path.exists(INPUT_FILE):
        print(f"错误：文件不存在 {INPUT_FILE}")
        return
    
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    # 处理每个标注
    annotations = data.get('annotations', [])
    processed_count = 0
    visualized_count = 0
    
    for ann in annotations:
        if 'bbox' in ann:
            # 读取bbox信息
            box = ann['bbox']
            
            # 转换box为bfov
            bfov_params = box_to_bfov(box, IMG_WIDTH, IMG_HEIGHT)
            
            # 替换原有的bfov
            ann['bfov'] = bfov_params
            
            # 替换point字段（使用bfov的前两个值）
            if 'point' in ann:
                ann['point'] = [bfov_params[0], bfov_params[1]]
            
            processed_count += 1
            
            # 可视化前5个结果
            if visualized_count < VISUALIZE_COUNT:
                output_path = os.path.join(OUTPUT_DIR, f"box_to_bfov_{ann['id']}.png")
                visualize_box_and_bfov(box, bfov_params, output_path)
                visualized_count += 1
    
    # 保存输出文件
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n处理完成！")
    print(f"- 处理标注数: {processed_count}")
    print(f"- 可视化结果数: {visualized_count}")
    print(f"- 输出文件: {OUTPUT_FILE}")
    print(f"- 可视化输出目录: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()