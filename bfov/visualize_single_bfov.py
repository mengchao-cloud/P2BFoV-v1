import numpy as np
import cv2
import sys
import os

# 添加本地lib目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
from ImageRecorder import ImageRecorder


def draw_single_bfov(bfov_params, erp_w=1920, erp_h=960, output_path='bfov_visualization.jpg'):
    """
    绘制单个BFOV到白色背景图像上
    
    参数:
    bfov_params: BFOV参数 [center_x, center_y, fov_x, fov_y] (弧度制)
    erp_w: ERP图像宽度，默认1920
    erp_h: ERP图像高度，默认960
    output_path: 输出图像路径，默认'bfov_visualization.jpg'
    """
    # 提取BFOV参数
    center_x_rad, center_y_rad, fov_x_rad, fov_y_rad = bfov_params
    
    # 转换为角度制
    center_x_deg = center_x_rad * 180 / np.pi
    center_y_deg = center_y_rad * 180 / np.pi
    fov_x_deg = fov_x_rad * 180 / np.pi
    fov_y_deg = fov_y_rad * 180 / np.pi
    
    print(f"BFOV参数:")
    print(f"  中心: ({center_x_deg:.2f}°, {center_y_deg:.2f}°)")
    print(f"  FOV: {fov_x_deg:.2f}° x {fov_y_deg:.2f}°")
    
    # 创建白色背景图像
    img = np.zeros((erp_h, erp_w, 3), dtype=np.uint8)
    img[:] = (255, 255, 255)
    
    # 绘制分割线（ERP图像的左右边界）
    threshold = erp_w // 2
    cv2.line(img, (threshold, 0), (threshold, erp_h), (0, 0, 0), 2)
    
    # 使用绿色绘制BFOV
    color = (0, 255, 0)
    
    try:
        # 创建ImageRecorder实例（使用角度制FOV参数）
        BFoV = ImageRecorder(erp_w, erp_h, 
                           view_angle_w=fov_x_deg, 
                           view_angle_h=fov_y_deg, 
                           long_side=erp_w)
        
        # 获取BFOV的边界点
        Px, Py = BFoV._sample_points(center_x_rad, center_y_rad, border_only=True)
        
        # 绘制BFOV边界
        img = BFoV.draw_BFoV(img, Px, Py, border_only=True, color=color)
        
        # 绘制BFOV中心点
        center_px = int((center_x_rad + np.pi) / (2 * np.pi) * erp_w)
        center_py = int((np.pi/2 - center_y_rad) / np.pi * erp_h)
        cv2.circle(img, (center_px, center_py), 5, color, -1)
        
        # 添加标签
        label = f"({center_x_deg:.1f}°, {center_y_deg:.1f}°)"
        cv2.putText(img, label, (center_px + 10, center_py), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 添加FOV信息
        fov_label = f"FOV: {fov_x_deg:.1f}°x{fov_y_deg:.1f}°"
        cv2.putText(img, fov_label, (center_px + 10, center_py + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        print(f"BFOV绘制成功")
        
    except Exception as e:
        print(f"绘制BFOV时出错: {e}")
        return False
    
    # 保存图像
    cv2.imwrite(output_path, img)
    print(f"图像已保存到: {output_path}")
    
    return True


def main():
    """
    主函数：读取BFOV参数并绘制
    """
    # BFOV参数 [center_x, center_y, fov_x, fov_y] (弧度制)
    # 示例：中心在(0, 0)，FOV为64度x64度
    # 示例3：中心在(-30°, -15°)，FOV为120度×80度
    bfov_params = [np.deg2rad(-30), np.deg2rad(-15), np.deg2rad(120), np.deg2rad(80)]
    # 图像尺寸
    erp_w = 1920
    erp_h = 960
    
    # 输出路径
    output_path = 'bfov_visualization.jpg'
    
    print("开始绘制BFOV...")
    print(f"图像尺寸: {erp_w} x {erp_h}")
    print(f"输出路径: {output_path}")
    print("-" * 50)
    
    # 绘制BFOV
    success = draw_single_bfov(bfov_params, erp_w, erp_h, output_path)
    
    if success:
        print("-" * 50)
        print("BFOV绘制完成!")
    else:
        print("-" * 50)
        print("BFOV绘制失败!")


if __name__ == '__main__':
    main()
