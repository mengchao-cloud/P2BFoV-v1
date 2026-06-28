import numpy as np
import cv2
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入RBFoV相关的ImageRecorder和旋转函数
from PANDORA.RoIoU.libs.ImageRecorder import ImageRecorder
from PANDORA.RoIoU.libs.tools import ro_Shpbbox


def generate_random_rbfov(num_rbfov=10):
    """
    随机生成RBFoV参数
    
    参数:
    num_rbfov: 生成的RBFoV数量
    
    返回:
    rbfov_list: RBFoV参数列表，每个元素为 [center_x, center_y, fov_x, fov_y, angle]（角度制）
    """
    rbfov_list = []
    
    for _ in range(num_rbfov):
        # 随机生成RBFoV参数
        # center_x: 经度，范围 [-180, 180] 度
        center_x = np.random.uniform(-180, 180)
        # center_y: 纬度，范围 [-90, 90] 度
        center_y = np.random.uniform(-90, 90)
        # fov_x: 水平视场角，范围 [5, 120] 度
        fov_x = np.random.uniform(5, 100)
        # fov_y: 垂直视场角，范围 [5, 120] 度
        fov_y = np.random.uniform(5, 80)
        # angle: 旋转角度，范围 [-160, 160] 度
        angle = np.random.uniform(-100, 80)
        
        rbfov_list.append([center_x, center_y, fov_x, fov_y, angle])
    
    return rbfov_list


def rbfov_to_bfov(rbfov_params, erp_w=1024, erp_h=512):
    """
    将RBFoV转换为对应的BFoV
    
    参数:
    rbfov_params: RBFoV参数，格式为 [center_x, center_y, fov_x, fov_y, angle]（角度制）
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    
    返回:
    bfov_params: BFoV参数，格式为 [center_x, center_y, fov_x, fov_y]（角度制）
    """
    center_x, center_y, fov_x, fov_y, angle = rbfov_params
    
    # 创建ImageRecorder实例
    BFoV = ImageRecorder(erp_w, erp_h, view_angle_w=fov_x, view_angle_h=fov_y, long_side=erp_w)
    
    # 将中心点角度转换为弧度制
    center_x_rad = center_x / 180 * np.pi
    center_y_rad = center_y / 180 * np.pi
    
    # 获取RBFoV的边界点
    Px, Py = BFoV._sample_points(center_x_rad, center_y_rad, border_only=True)
    
    # 使用ro_Shpbbox对边界点进行旋转变换
    rbfov_np = np.array(rbfov_params)
    Px_rot, Py_rot = ro_Shpbbox(rbfov_np, Px, Py, erp_w=erp_w, erp_h=erp_h)
    
    # 找到旋转后边界点的最小和最大坐标
    Px_min, Px_max = Px_rot.min(), Px_rot.max()
    Py_min, Py_max = Py_rot.min(), Py_rot.max()
    
    # 计算BFoV的中心点（像素坐标）
    bfov_center_px = (Px_min + Px_max) / 2
    bfov_center_py = (Py_min + Py_max) / 2
    
    # 转换为经纬度（角度制）
    bfov_center_x = (bfov_center_px / erp_w) * 360 - 180
    bfov_center_y = 90 - (bfov_center_py / erp_h) * 180
    
    # 计算BFoV的视场角
    # 水平视场角 = 宽度 / erp_w * 360度
    # 垂直视场角 = 高度 / erp_h * 180度
    bfov_fov_x = ((Px_max - Px_min) / erp_w) * 360
    bfov_fov_y = ((Py_max - Py_min) / erp_h) * 180
    
    # 确保视场角在有效范围内（小于160度，避免边界问题）
    bfov_fov_x = min(max(bfov_fov_x, 5), 160)
    bfov_fov_y = min(max(bfov_fov_y, 5), 160)
    
    return [bfov_center_x, bfov_center_y, bfov_fov_x, bfov_fov_y]


def visualize_rbfov_and_bfov(num_rbfov=3, erp_w=1024, erp_h=512, output_path='./bfov/rbfov_to_bfov_visualization.jpg'):
    """
    可视化RBFoV及其对应的BFoV
    
    参数:
    num_rbfov: 生成的RBFoV数量
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    output_path: 输出图像路径
    """
    # 创建空白图像（白色背景）
    img = np.zeros((erp_h, erp_w, 3), dtype=np.uint8)
    img[:] = (255, 255, 255)  # 白色背景
    
    # 生成随机RBFoV参数
    rbfov_list = generate_random_rbfov(num_rbfov)
    
    # 预定义颜色列表（用于区分不同RBFoV-BFoV对）
    colors = [
        (0, 0, 255),     # 红色 - RBFoV
        (0, 255, 0),     # 绿色 - BFoV
        (255, 0, 0),     # 蓝色 - RBFoV
        (0, 255, 255),   # 青色 - BFoV
        (255, 0, 255),   # 品红色 - RBFoV
        (255, 255, 0),   # 黄色 - BFoV
    ]
    
    print(f"生成了 {num_rbfov} 个随机RBFoV及其对应的BFoV:")
    
    # 遍历每个RBFoV并绘制
    for i, rbfov_params in enumerate(rbfov_list):
        center_x, center_y, fov_x, fov_y, angle = rbfov_params
        
        # 将RBFoV转换为BFoV
        bfov_params = rbfov_to_bfov(rbfov_params, erp_w, erp_h)
        bfov_center_x, bfov_center_y, bfov_fov_x, bfov_fov_y = bfov_params
        
        # 打印参数信息
        print(f"\nRBFoV[{i}]:")
        print(f"  中心点: ({center_x:.2f}°, {center_y:.2f}°)")
        print(f"  视场角: {fov_x:.2f}° × {fov_y:.2f}°")
        print(f"  旋转角度: {angle:.2f}°")
        print(f"对应的BFoV[{i}]:")
        print(f"  中心点: ({bfov_center_x:.2f}°, {bfov_center_y:.2f}°)")
        print(f"  视场角: {bfov_fov_x:.2f}° × {bfov_fov_y:.2f}°")
        
        # ================ 绘制RBFoV（使用颜色列表的奇数索引）===============
        rbfov_color = colors[i % len(colors)]
        BFoV_rbfov = ImageRecorder(erp_w, erp_h, view_angle_w=fov_x, view_angle_h=fov_y, long_side=erp_w)
        
        center_x_rad = center_x / 180 * np.pi
        center_y_rad = center_y / 180 * np.pi
        
        Px, Py = BFoV_rbfov._sample_points(center_x_rad, center_y_rad, border_only=True)
        
        # 应用旋转
        rbfov_np = np.array(rbfov_params)
        Px_rot, Py_rot = ro_Shpbbox(rbfov_np, Px, Py, erp_w=erp_w, erp_h=erp_h)
        
        # 绘制RBFoV边界
        BFoV_rbfov.draw_Sphbbox(img, Px_rot, Py_rot, border_only=True, color=rbfov_color)
        
        # 绘制RBFoV中心点
        center_px = int((center_x_rad + np.pi) / (2 * np.pi) * erp_w)
        center_py = int((np.pi/2 - center_y_rad) / np.pi * erp_h)
        cv2.circle(img, (center_px, center_py), 4, rbfov_color, -1)
        
        # ================ 绘制BFoV（使用对应的互补色）===============
        # BFoV使用比RBFoV更浅的颜色
        r, g, b = rbfov_color
        bfov_color = (min(r + 100, 255), min(g + 100, 255), min(b + 100, 255))
        BFoV_bfov = ImageRecorder(erp_w, erp_h, view_angle_w=bfov_fov_x, view_angle_h=bfov_fov_y, long_side=erp_w)
        
        bfov_center_x_rad = bfov_center_x / 180 * np.pi
        bfov_center_y_rad = bfov_center_y / 180 * np.pi
        
        Px_bfov, Py_bfov = BFoV_bfov._sample_points(bfov_center_x_rad, bfov_center_y_rad, border_only=True)
        
        # 绘制BFoV边界（不旋转，因为BFoV没有旋转角度）
        BFoV_bfov.draw_Sphbbox(img, Px_bfov, Py_bfov, border_only=True, color=bfov_color)
        
        # 绘制BFoV中心点
        bfov_center_px = int((bfov_center_x_rad + np.pi) / (2 * np.pi) * erp_w)
        bfov_center_py = int((np.pi/2 - bfov_center_y_rad) / np.pi * erp_h)
        cv2.circle(img, (bfov_center_px, bfov_center_py), 4, bfov_color, -1)
        
        # ================ 绘制标签 ================
        # RBFoV标签
        cv2.putText(img, f"R{i+1}", (center_px + 8, center_py + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, rbfov_color, 1, cv2.LINE_AA)
        # BFoV标签
        cv2.putText(img, f"B{i+1}", (bfov_center_px + 8, bfov_center_py + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bfov_color, 1, cv2.LINE_AA)
    
    # 添加图例
    legend_y = 20
    cv2.putText(img, "RBFoV (旋转球视野)", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "BFoV (球视野，RBFoV的外接矩形)", (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    # 保存结果图像
    cv2.imwrite(output_path, img)
    print(f"\nRBFoV和BFoV可视化结果已保存到: {output_path}")
    
    return img, rbfov_list


if __name__ == '__main__':
    # 生成并可视化3个随机RBFoV及其对应的BFoV
    visualize_rbfov_and_bfov(num_rbfov=5, erp_w=1024, erp_h=512)
    
    print("\n脚本执行完成！")
