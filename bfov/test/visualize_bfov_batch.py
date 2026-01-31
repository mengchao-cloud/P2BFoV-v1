import numpy as np
import cv2
import sys
import os

from PANDORA.PRDA.lib.ImageRecorder import ImageRecorder#文件lib当前的位置



def read_bfov_list(file_path):
    """
    读取bfovlist.txt文件中的数据
    
    参数:
    file_path: 文件路径
    
    返回:
    bfov_list: BFOV参数列表，每个元素是[longitude, latitude, fov_x, fov_y]（弧度制）
    """
    bfov_list = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 直接读取所有行，每行都是一个BFOV参数
    # 只读取前42组参数
    for i, line in enumerate(lines[:42]):
        params = list(map(float, line.strip().split()))
        if len(params) == 4:
            bfov_list.append(params)
    
    print(f"共读取 {len(bfov_list)} 个BFOV参数")
    return bfov_list

def visualize_bfovs_from_list(bfov_list, erp_w=1920, erp_h=960, output_dir='./visualization_results'):
    """
    可视化bfov列表中的所有BFOV
    
    参数:
    bfov_list: BFOV参数列表，每个元素是[longitude, latitude, fov_x, fov_y]（弧度制）
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建空白图像作为背景
    background = np.zeros((erp_h, erp_w, 3), dtype=np.uint8)
    background[:] = (255, 255, 255)  # 设置为白色背景
    
    # 为每个BFOV创建一个可视化
    for i, bfov_params in enumerate(bfov_list):
        longitude, latitude, fov_x, fov_y = bfov_params
        
        # 将弧度制转换为角度制
        center_x_deg = longitude * 180 / np.pi
        center_y_deg = latitude * 180 / np.pi
        fov_x_deg = fov_x * 180 / np.pi
        fov_y_deg = fov_y * 180 / np.pi
        
        print(f"BFOV {i+1}: center=({center_x_deg:.2f}, {center_y_deg:.2f}), fov=({fov_x_deg:.2f}, {fov_y_deg:.2f})")
        
        # 创建ImageRecorder实例
        BFoV = ImageRecorder(erp_w, erp_h, view_angle_w=fov_x_deg, view_angle_h=fov_y_deg, long_side=erp_w)
        
        # 获取BFOV的边界点
        Px, Py = BFoV._sample_points(longitude, latitude, border_only=True)
        
        # 绘制BFOV
        img = background.copy()
        # 绘制边界框
        img = BFoV.draw_BFoV(img, Px, Py, border_only=True, color=(0, 255, 0))
        
        # 绘制BFOV中心点
        center_px = int((longitude + np.pi) / (2 * np.pi) * erp_w)
        center_py = int((np.pi/2 - latitude) / np.pi * erp_h)
        cv2.circle(img, (center_px, center_py), 2, (0, 0, 255), -1)
        
        # 保存结果图像
        output_path = os.path.join(output_dir, f'bfov_{i+1}.jpg')
        cv2.imwrite(output_path, img)
        print(f"BFOV {i+1} 可视化结果已保存到: {output_path}")
    
    # 创建一个包含所有BFOV的综合图像
    all_bfovs_img = background.copy()
    for i, bfov_params in enumerate(bfov_list):
        longitude, latitude, fov_x, fov_y = bfov_params
        
        # 将弧度制转换为角度制
        fov_x_deg = fov_x * 180 / np.pi
        fov_y_deg = fov_y * 180 / np.pi
        
        # 创建ImageRecorder实例
        BFoV = ImageRecorder(erp_w, erp_h, view_angle_w=fov_x_deg, view_angle_h=fov_y_deg, long_side=erp_w)
        
        # 获取BFOV的边界点
        Px, Py = BFoV._sample_points(longitude, latitude, border_only=True)
        
        # 绘制BFOV
        all_bfovs_img = BFoV.draw_BFoV(all_bfovs_img, Px, Py, border_only=True, color=(0, 255, 0))
        
        # 绘制BFOV中心点
        center_px = int((longitude + np.pi) / (2 * np.pi) * erp_w)
        center_py = int((np.pi/2 - latitude) / np.pi * erp_h)
        cv2.circle(all_bfovs_img, (center_px, center_py), 2, (0, 0, 255), -1)
    
    # 保存综合图像
    all_output_path = os.path.join(output_dir, 'all_bfovs.jpg')
    cv2.imwrite(all_output_path, all_bfovs_img)
    print(f"所有BFOV的综合可视化结果已保存到: {all_output_path}")

if __name__ == '__main__':
    # 读取bfovlist.txt文件
    file_path = './bfovlist.txt'
    bfov_list = read_bfov_list(file_path)
    
    # 可视化所有BFOV
    visualize_bfovs_from_list(bfov_list)
