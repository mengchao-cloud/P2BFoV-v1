#!/usr/bin/env python3
"""
BFoV掩码生成与外接矩形提取工具
简单生成BFoV掩码并提取其外接矩形

功能:
- 生成BFoV掩码
- 提取掩码的外接矩形
- 可视化结果
"""

import torch
import numpy as np
import cv2
import os
import sys

# 添加父目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PANDORA.PRDA.lib.ReuseGPUImageRecorder import ReuseGPUImageRecorder

def generate_bfov_mask(lon, lat, fov_x, fov_y, erp_w=1920, erp_h=960, device='cuda'):
    """
    生成单个BFoV掩码
    
    参数:
    lon: 经度（弧度）
    lat: 纬度（弧度）
    fov_x: 水平视场（弧度）
    fov_y: 垂直视场（弧度）
    erp_w: ERP图像宽度（默认1920）
    erp_h: ERP图像高度（默认960）
    device: 设备（'cuda'或'cpu'）
    
    返回:
    mask: [erp_h, erp_w] 的掩码张量
    """
    # 确保张量在GPU上
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # 初始化掩码张量
    mask = torch.zeros((erp_h, erp_w), dtype=torch.uint8, device=device)
    
    # 创建ReuseGPUImageRecorder实例
    gpu_recorder = ReuseGPUImageRecorder(erp_w, erp_h, device=device)
    
    # 将视场角转换为角度制
    fov_x_deg = np.degrees(fov_x)
    fov_y_deg = np.degrees(fov_y)
    
    # 设置参数
    gpu_recorder.view_angle_w = fov_x_deg
    gpu_recorder.view_angle_h = fov_y_deg
    gpu_recorder.long_side = erp_w
    
    # 生成采样点
    Px, Py = gpu_recorder._sample_points(lon, lat, border_only=False)
    
    # 将采样点坐标转换为整数
    Px = Px.to(torch.int32)
    Py = Py.to(torch.int32)
    
    # 确保坐标在有效范围内
    valid_mask = (Px >= 0) & (Px < erp_w) & (Py >= 0) & (Py < erp_h)
    valid_Px = Px[valid_mask].to(torch.long)
    valid_Py = Py[valid_mask].to(torch.long)
    
    # 将有效采样点标记为1
    mask[valid_Py, valid_Px] = 1
    
    # 填充掩码区域
    current_mask = mask.cpu().numpy().astype(np.uint8)
    contours, hierarchy = cv2.findContours(current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(current_mask, contours, -1, 1, -1)
    
    # 转换回张量
    mask = torch.from_numpy(current_mask).to(device=device, dtype=torch.uint8)
    
    return mask

def get_bounding_box(mask):
    """
    提取掩码的外接矩形（直接找掩码的四个顶点）
    
    参数:
    mask: [H, W] 的掩码张量
    
    返回:
    bbox: 边界框，格式为 [x_min, y_min, x_max, y_max]
    """
    # 转换为numpy
    mask_np = mask.cpu().numpy().astype(np.uint8)
    
    # 找到所有非零像素的坐标
    non_zero_coords = np.where(mask_np == 1)
    
    if len(non_zero_coords[0]) > 0:
        # 计算最小和最大坐标
        y_min = np.min(non_zero_coords[0])
        y_max = np.max(non_zero_coords[0])
        x_min = np.min(non_zero_coords[1])
        x_max = np.max(non_zero_coords[1])
        
        # 构建边界框
        bbox = [x_min, y_min, x_max, y_max]
    else:
        # 如果没有非零像素，返回空边界框
        bbox = [0, 0, 0, 0]
    
    return bbox

def visualize_bfov_mask_with_bbox(mask, bbox, output_path, lon, lat, fov_x, fov_y):
    """
    可视化BFoV掩码和外接矩形
    
    参数:
    mask: 掩码张量
    bbox: 边界框
    output_path: 输出文件路径
    lon: 经度（弧度）
    lat: 纬度（弧度）
    fov_x: 水平视场（弧度）
    fov_y: 垂直视场（弧度）
    """
    erp_h, erp_w = mask.shape
    
    # 创建空白图像（白色背景）
    img = np.ones((erp_h, erp_w, 3), dtype=np.uint8) * 255
    
    # 绘制掩码区域
    mask_np = mask.cpu().numpy()
    mask_indices = np.where(mask_np == 1)
    img[mask_indices[0], mask_indices[1]] = (0, 255, 0)  # 绿色
    
    # 计算中心点像素坐标
    center_px = int((lon + np.pi) / (2 * np.pi) * erp_w)
    center_py = int((np.pi/2 - lat) / np.pi * erp_h)
    
    # 绘制BFoV中心点
    cv2.circle(img, (center_px, center_py), 8, (0, 0, 255), -1)  # 红色
    
    # 绘制外接矩形框
    if bbox != [0, 0, 0, 0]:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 0), 3)  # 黑色
    
    # 添加信息
    lon_deg = np.degrees(lon)
    lat_deg = np.degrees(lat)
    fov_x_deg = np.degrees(fov_x)
    fov_y_deg = np.degrees(fov_y)
    
    text = f'BFoV: Lon={lon_deg:7.2f}°, Lat={lat_deg:7.2f}°'
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    text = f'FOV: {fov_x_deg:5.1f}°x{fov_y_deg:5.1f}°'
    cv2.putText(img, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    if bbox != [0, 0, 0, 0]:
        text = f'BBox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]'
        cv2.putText(img, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # 保存图像
    cv2.imwrite(output_path, img)

def generate_random_bfov_params(num_bfovs=10):
    """
    生成随机的BFoV参数
    
    参数:
    num_bfovs: 生成的BFoV数量
    
    返回:
    bfov_params: BFoV参数列表，每个元素为 [经度, 纬度, 水平视场, 垂直视场]（弧度）
    """
    bfov_params = []
    
    for i in range(num_bfovs):
        # 随机生成参数
        # 经度：-π到π
        lon = np.random.uniform(-np.pi, np.pi)
        # 纬度：-π/2到π/2
        lat = np.random.uniform(-np.pi/2, np.pi/2)
        # 水平视场：30°到120°（转换为弧度）
        fov_x = np.random.uniform(np.pi/6, 2*np.pi/3)
        # 垂直视场：30°到120°（转换为弧度）
        fov_y = np.random.uniform(np.pi/6, 2*np.pi/3)
        
        bfov_params.append([lon, lat, fov_x, fov_y])
    
    return bfov_params

def main():
    """
    主函数：生成BFoV掩码并提取外接矩形
    """
    print("=== BFoV掩码生成与外接矩形提取工具 ===")
    
    # 检查GPU可用性
    if torch.cuda.is_available():
        device = 'cuda'
        print("✓ GPU可用，使用CUDA加速")
    else:
        device = 'cpu'
        print("⚠ GPU不可用，使用CPU模式")
    
    # 配置参数
    erp_w = 1920    # 分辨率宽度
    erp_h = 960     # 分辨率高度
    output_dir = './bfov/result/bfov_bbox/'
    num_bfovs = 20  # 生成的BFoV数量
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成随机BFoV参数
    print(f"生成 {num_bfovs} 个随机BFoV参数...")
    bfov_params = generate_random_bfov_params(num_bfovs)
    
    print(f"\n处理 {len(bfov_params)} 个BFoV...")
    
    for i, (lon, lat, fov_x, fov_y) in enumerate(bfov_params):
        print(f"\n处理BFoV {i+1}/{len(bfov_params)}...")
        
        # 1. 生成BFoV掩码
        print("  生成BFoV掩码...")
        mask = generate_bfov_mask(lon, lat, fov_x, fov_y, erp_w=erp_w, erp_h=erp_h, device=device)
        
        # 2. 提取外接矩形
        print("  提取外接矩形...")
        bbox = get_bounding_box(mask)
        print(f"  外接矩形: {bbox}")
        
        # 3. 可视化结果
        print("  可视化结果...")
        output_path = os.path.join(output_dir, f'bfov_{i+1:02d}_mask_bbox.jpg')
        visualize_bfov_mask_with_bbox(mask, bbox, output_path, lon, lat, fov_x, fov_y)
        print(f"  结果保存至: {output_path}")
    
    print("\n🎉 处理完成！")
    print(f"   输出目录: {output_dir}")


if __name__ == '__main__':
    main()
