import numpy as np
import cv2
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入RBFoV相关的ImageRecorder和旋转函数
from PANDORA.RoIoU.libs.ImageRecorder import ImageRecorder
from PANDORA.RoIoU.libs.tools import ro_Shpbbox


def generate_random_bfov(num_bfov=10):
    """
    随机生成BFOV参数
    
    参数:
    num_bfov: 生成的BFOV数量
    
    返回:
    bfov_list: BFOV参数列表，每个元素为 [center_x, center_y, fov_x, fov_y, angle]（角度制）
    """
    bfov_list = []
    
    for _ in range(num_bfov):
        # 随机生成BFOV参数
        # center_x: 经度，范围 [-180, 180] 度
        center_x = np.random.uniform(-180, 180)
        # center_y: 纬度，范围 [-90, 90] 度
        center_y = np.random.uniform(-90, 90)
        # fov_x: 水平视场角，范围 [5, 120] 度
        fov_x = np.random.uniform(5, 120)
        # fov_y: 垂直视场角，范围 [5, 120] 度
        fov_y = np.random.uniform(5, 120)
        # angle: 旋转角度，范围 [-160, 160] 度（当前脚本未使用旋转）
        angle = np.random.uniform(-160, 160)  # np.random.uniform(-180, 180)
        
        bfov_list.append([center_x, center_y, fov_x, fov_y, angle])
    
    return bfov_list


def visualize_random_bfovs(num_bfov=5, erp_w=1024, erp_h=512, output_path='./bfov/random_rbfov_visualization.jpg'):
    """
    在空白图像上随机生成并绘制多个BFOV
    
    参数:
    num_bfov: 生成的BFOV数量
    erp_w: ERP图像宽度（默认1024）
    erp_h: ERP图像高度（默认512）
    output_path: 输出图像路径
    """
    # 创建空白图像（白色背景）
    img = np.zeros((erp_h, erp_w, 3), dtype=np.uint8)
    img[:] = (255, 255, 255)  # 白色背景
    
    # 生成随机BFOV参数
    bfov_list = generate_random_bfov(num_bfov)
    
    # 预定义颜色列表（用于区分不同BFOV）
    colors = [
        (0, 0, 255),     # 红色
        (0, 255, 0),     # 绿色
        (255, 0, 0),     # 蓝色
        (0, 255, 255),   # 青色
        (255, 0, 255),   # 品红色
        (255, 255, 0),   # 黄色
        (128, 0, 128),   # 紫色
        (0, 128, 128),   # 深青色
    ]
    
    print(f"生成了 {num_bfov} 个随机BFOV:")
    
    # 遍历每个BFOV并绘制
    for i, bfov_params in enumerate(bfov_list):
        center_x, center_y, fov_x, fov_y, angle = bfov_params
        
        # 打印BFOV参数
        print(f"\nBFOV[{i}]:")
        print(f"  中心点: ({center_x:.2f}°, {center_y:.2f}°)")
        print(f"  视场角: {fov_x:.2f}° × {fov_y:.2f}°")
        print(f"  旋转角度: {angle:.2f}°")
        
        # 创建ImageRecorder实例
        BFoV = ImageRecorder(erp_w, erp_h, view_angle_w=fov_x, view_angle_h=fov_y, long_side=erp_w)
        
        # 将中心点角度转换为弧度制
        center_x_rad = center_x / 180 * np.pi
        center_y_rad = center_y / 180 * np.pi
        
        # 获取RBFoV的边界点
        Px, Py = BFoV._sample_points(center_x_rad, center_y_rad, border_only=True)
        
        # 使用ro_Shpbbox对边界点进行旋转变换（使用第五个参数angle）
        # RBFoV参数格式: [center_x, center_y, fov_x, fov_y, angle]
        rbfov_params = np.array([center_x, center_y, fov_x, fov_y, angle])
        Px, Py = ro_Shpbbox(rbfov_params, Px, Py, erp_w=erp_w, erp_h=erp_h)
        
        # 使用draw_Sphbbox绘制RBFoV边界
        color = colors[i % len(colors)]
        BFoV.draw_Sphbbox(img, Px, Py, border_only=True, color=color)
        
        # 绘制BFOV中心点
        center_px = int((center_x_rad + np.pi) / (2 * np.pi) * erp_w)
        center_py = int((np.pi/2 - center_y_rad) / np.pi * erp_h)
        cv2.circle(img, (center_px, center_py), 3, color, -1)
        
        # 绘制中心点标签
        label = f"{i+1}"
        cv2.putText(img, label, (center_px + 5, center_py), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    # 保存结果图像
    cv2.imwrite(output_path, img)
    print(f"\nBFOV可视化结果已保存到: {output_path}")
    
    return img, bfov_list


if __name__ == '__main__':
    # 生成并可视化5个随机BFOV
    visualize_random_bfovs(num_bfov=10, erp_w=1024, erp_h=512)
    
    print("\n脚本执行完成！")