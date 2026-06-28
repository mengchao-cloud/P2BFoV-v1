import cv2
import numpy as np

# 导入ImageRecorder库
import sys
sys.path.append('..')
from PANDORA.PRDA.lib.ImageRecorder import ImageRecorder

# 图像配置
IMG_WIDTH = 1920
IMG_HEIGHT = 960
OUTPUT_PATH = './bfov/box_to_bfov_visualization.png'

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

def visualize_box_and_bfov(box, bfov_params, img_width, img_height, output_path):
    """
    在空白背景上可视化box和BFOV
    
    参数:
    box: 边界框 [x, y, width, height]，像素坐标
    bfov_params: BFOV参数 [center_x, center_y, fov_x, fov_y]，弧度制
    img_width: 图像宽度
    img_height: 图像高度
    output_path: 输出图像路径
    """
    # 创建白色背景
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    
    # 绘制box
    x, y, width, height = box
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + width), int(y + height)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框
    
    # 绘制BFOV边界
    center_x, center_y, fov_x, fov_y = bfov_params
    
    # 将弧度转换为角度
    center_x_deg = center_x * 180 / np.pi
    center_y_deg = center_y * 180 / np.pi
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
        BFoV = ImageRecorder(img_width, img_height, 
                            view_angle_w=fov_x_deg, view_angle_h=fov_y_deg, 
                            long_side=img_width)
        
        # 获取BFOV的边界点
        Px, Py = BFoV._sample_points(center_x, center_y, border_only=True)
        
        # 绘制BFOV边界
        img = BFoV.draw_BFoV(img, Px, Py, border_only=True, color=(255, 0, 0))  # 蓝色边界
    except Exception as e:
        # 处理异常，记录错误但不中断程序
        print(f"可视化BFOV时发生错误: {e}")
        # 使用简单的矩形绘制作为备选方案
        # 将BFOV中心点转换为像素坐标
        center_px = int((center_x + np.pi) / (2 * np.pi) * img_width)
        center_py = int((np.pi/2 - center_y) / np.pi * img_height)
        # 计算BFOV在图像上的大致范围
        width_px = int(fov_x / (2 * np.pi) * img_width)
        height_px = int(fov_y / np.pi * img_height)
        # 绘制BFOV边界
        bfov_x1 = max(0, center_px - width_px // 2)
        bfov_y1 = max(0, center_py - height_px // 2)
        bfov_x2 = min(img_width, center_px + width_px // 2)
        bfov_y2 = min(img_height, center_py + height_px // 2)
        # 绘制BFOV边界（蓝色虚线）
        cv2.rectangle(img, (bfov_x1, bfov_y1), (bfov_x2, bfov_y2), (255, 0, 0), 2, cv2.LINE_8)
    
    # 绘制中心点
    center_px = int((center_x + np.pi) / (2 * np.pi) * img_width)
    center_py = int((np.pi/2 - center_y) / np.pi * img_height)
    cv2.circle(img, (center_px, center_py), 3, (0, 0, 255), -1)  # 红色中心点
    
    # 添加文字说明
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Box', (x1 + 10, y1 - 10), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img, 'BFOV', (center_px + 10, center_py - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    
    # 添加BFOV参数信息
    bfov_text = f"BFOV: center=({center_x:.2f}, {center_y:.2f}), fov=({fov_x:.2f}, {fov_y:.2f})"
    cv2.putText(img, bfov_text, (10, 30), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    # 保存图像
    cv2.imwrite(output_path, img)
    print(f"可视化结果已保存到: {output_path}")

def main():
    """
    主函数：给定一个box计算bfov并可视化
    """
    print("Box转BFOV并可视化")
    print("=" * 50)
    
    # 示例box：[x, y, width, height]，像素坐标
    # 你可以修改这里的box值来测试不同的情况
    box = [200, 100, 300, 200]  # 示例box
    
    print(f"输入Box: {box}")
    
    # 转换box为bfov
    bfov_params = box_to_bfov(box, IMG_WIDTH, IMG_HEIGHT)
    print(f"输出BFOV参数: {bfov_params}")
    
    # 可视化
    visualize_box_and_bfov(box, bfov_params, IMG_WIDTH, IMG_HEIGHT, OUTPUT_PATH)
    
    print("\n处理完成！")

if __name__ == '__main__':
    main()