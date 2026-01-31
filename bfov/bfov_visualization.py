import numpy as np
import cv2
import sys
import os

# 添加PRDA/lib目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'PRDA', 'lib'))

from ImageRecorder import ImageRecorder


def visualize_bfov(bfov_params, erp_w=1920, erp_h=960, output_path='./bfov_visualization_result.jpg'):
    """
    可视化BFOV在空白图像上，正确处理曲边
    
    参数:
    bfov_params: BFOV参数 [center_x, center_y, fov_x, fov_y, angle]，角度制
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    output_path: 输出图像路径
    """
    # 读取用户自己的图片
    img_path = './37936706382_45b3fbc711_f.jpg'
    img = cv2.imread(img_path)
    
    # 检查图片是否成功读取
    if img is None:
        print(f"无法读取图片: {img_path}")
        # 如果无法读取，创建空白图像作为备选
        img = np.zeros((erp_h, erp_w, 3), dtype=np.uint8)
        img[:] = (255, 255, 255)  # 设置为白色背景
    else:
        # 调整图片尺寸以匹配erp_w和erp_h
        img = cv2.resize(img, (erp_w, erp_h))
    
    # 提取BFOV参数
    center_x, center_y, fov_x, fov_y, angle = bfov_params
    print(f"BFOV参数: center_x={center_x}, center_y={center_y}, fov_x={fov_x}, fov_y={fov_y}, angle={angle}")
    
    # 创建ImageRecorder实例
    BFoV = ImageRecorder(erp_w, erp_h, view_angle_w=fov_x, view_angle_h=fov_y, long_side=erp_w)
    
    # 注意：ImageRecorder的catch方法和draw_BFoV方法使用的是弧度制参数
    # 将中心点角度转换为弧度制
    center_x_rad = center_x / 180 * np.pi
    center_y_rad = center_y / 180 * np.pi
    
    # 使用ImageRecorder的内置方法draw_BFoV来绘制BFOV
    # 1. 获取BFOV的边界点
    Px, Py = BFoV._sample_points(center_x_rad, center_y_rad, border_only=True)
    
    # 2. 使用draw_BFoV方法绘制BFOV
    img = BFoV.draw_BFoV(img, Px, Py, border_only=True, color=(0, 255, 0))
    
    # 3. 绘制BFOV中心点
    center_px = int((center_x_rad + np.pi) / (2 * np.pi) * erp_w)
    center_py = int((np.pi/2 - center_y_rad) / np.pi * erp_h)
    cv2.circle(img, (center_px, center_py), 2, (0, 0, 255), -1)
    
    print(f"中心点坐标: ({center_px}, {center_py})")
    
    # 保存结果图像
    cv2.imwrite(output_path, img)
    print(f"BFOV可视化结果已保存到: {output_path}")
    
    return img


if __name__ == '__main__':
    # BFOV参数 [center_x, center_y, fov_x, fov_y, angle]，角度制
    bfov_params = [-123.4, -38.27, 13, 20, 0]
    
    # 调用可视化函数
    visualize_bfov(bfov_params)
