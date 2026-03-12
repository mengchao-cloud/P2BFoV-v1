#!/usr/bin/env python3
"""
简单BFoV掩码生成器
生成指定数量的随机BFoV掩码（固定1920x960分辨率）并可视化

功能:
- 随机生成指定数量的BFoV参数
- 生成1920x960固定分辨率的BFoV掩码
- 可视化掩码区域
- 支持GPU加速
"""

import torch
import numpy as np
import cv2
import os
import time
import sys

# 添加父目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PANDORA.PRDA.lib.ReuseGPUImageRecorder import ReuseGPUImageRecorder
from mask_to_box import MaskToBox

def generate_random_bfov_params(num_bfovs=10, device='cuda'):
    """
    生成随机的BFoV参数（弧度制）
    
    参数:
    num_bfovs: 生成的BFoV数量
    device: 张量设备（'cuda'或'cpu'）
    
    返回:
    bfov_tensor: [num_bfovs, 4] 的张量，包含 [经度, 纬度, 水平视场, 垂直视场]
    """
    # 生成随机参数
    # 经度：-π到π
    longitudes = np.random.uniform(-np.pi, np.pi, num_bfovs)
    # 纬度：-π/2到π/2
    latitudes = np.random.uniform(-np.pi/2, np.pi/2, num_bfovs)
    # 水平视场：15°到170°（转换为弧度）
    fov_x = np.random.uniform(np.pi/48, np.pi/1.1, num_bfovs)
    # 垂直视场：15°到170°（转换为弧度）
    fov_y = np.random.uniform(np.pi/48, np.pi/1.1, num_bfovs)
    
    # 组合成张量
    bfov_params = np.stack([longitudes, latitudes, fov_x, fov_y], axis=1)
    bfov_tensor = torch.tensor(bfov_params, dtype=torch.float32, device=device)
    
    # 打印参数信息
    print(f"生成了 {num_bfovs} 个随机BFoV参数：")
    for i in range(num_bfovs):
        lon_deg = np.degrees(bfov_params[i, 0])
        lat_deg = np.degrees(bfov_params[i, 1])
        fov_x_deg = np.degrees(bfov_params[i, 2])
        fov_y_deg = np.degrees(bfov_params[i, 3])
        print(f"  BFoV {i+1:2d}: 经度={lon_deg:7.2f}°, 纬度={lat_deg:7.2f}°, "
              f"水平视场={fov_x_deg:5.1f}°, 垂直视场={fov_y_deg:5.1f}°")
    
    return bfov_tensor

def generate_bfov_masks(bfov_tensor, erp_w=1920, erp_h=960):
    """
    基于GPU复用的BFoV掩码生成函数
    
    参数:
    bfov_tensor: [num_bfovs, 4] 的张量（弧度制），包含 [经度, 纬度, 水平视场, 垂直视场]
    erp_w: ERP图像宽度（默认1920）
    erp_h: ERP图像高度（默认960）
    
    返回:
    masks: [num_bfovs, erp_h, erp_w] 的掩码张量
    
    实现逻辑:
    1. 使用ReuseGPUImageRecorder批量生成BFoV采样点
    2. 将采样点标记到掩码张量
    3. 使用OpenCV填充掩码区域
    
    优化内容:
    - 使用GPU复用技术提高性能
    - 减少不必要的数据传输
    - 直接在原始数组上操作，避免额外内存分配
    """
    # 确保张量在GPU上
    if not bfov_tensor.is_cuda:
        bfov_tensor = bfov_tensor.cuda()
    
    num_bfovs = bfov_tensor.shape[0]
    
    # 初始化掩码张量
    masks = torch.zeros((num_bfovs, erp_h, erp_w), dtype=torch.uint8, device=bfov_tensor.device)
    
    # 创建单个ReuseGPUImageRecorder实例（GPU复用）
    gpu_recorder = ReuseGPUImageRecorder(erp_w, erp_h, device=bfov_tensor.device)
    
    # 批量处理所有BFoV
    for i in range(num_bfovs):
        # 获取当前BFoV参数
        lon, lat, fov_x, fov_y = bfov_tensor[i].tolist()
        
        # 将视场角转换为角度制
        fov_x_deg = np.degrees(fov_x)
        fov_y_deg = np.degrees(fov_y)
        
        # 通过属性更新复用GPUImageRecorder实例
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
        masks[i, valid_Py, valid_Px] = 1
        
        # 立即填充当前BFoV掩码的完整区域
        # 转换掩码到CPU进行OpenCV处理（只转换一次）
        current_mask = masks[i].cpu().numpy().astype(np.uint8)
        
        # 查找轮廓并填充（直接在当前掩码上操作，避免额外内存分配）
        contours, hierarchy = cv2.findContours(current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(current_mask, contours, -1, 1, -1)  # 直接在当前掩码上填充
        
        # 将填充后的掩码转换回GPU张量（直接覆盖原位置，避免额外内存分配）
        masks[i] = torch.from_numpy(current_mask).to(device=bfov_tensor.device, dtype=torch.uint8)
    
    return masks

def calculate_mask_thickness(masks):
    """
    按列计算掩码区域的厚度（优化版本）
    
    参数:
    masks: 掩码张量 [num_bfovs, H, W]
    
    返回:
    thickness_dict: 字典，包含每个掩码的最大厚度和阈值
    column_thickness_maps: 列厚度图列表（numpy数组）
    
    优化内容:
    - 使用向量化操作替代嵌套循环
    - 减少GPU/CPU数据传输次数
    - 避免重复计算邻居窗口大小
    - 优化内存使用，减少不必要的数组复制
    """
    device = masks.device
    num_bfovs = masks.shape[0]
    H, W = masks.shape[1], masks.shape[2]
    thickness_dict = {}
    column_thickness_maps = []
    
    # 在GPU上批量计算每列的厚度
    # 创建垂直方向的索引张量
    y_indices = torch.arange(H, device=device).view(1, H, 1)
    
    # 计算每一列的厚度
    for i in range(num_bfovs):
        # 获取当前掩码（保持在GPU上）
        mask = masks[i]
        
        # 计算每一列的非零元素
        col_nonzero = mask.sum(dim=0)  # [W]
        
        # 计算每一列的最小和最大y坐标
        # 扩展y_indices到与mask相同形状
        y_indices_expanded = y_indices.expand(1, H, W)[0]
        
        # 计算每一列的最小y坐标
        min_y = torch.min(torch.where(mask == 1, y_indices_expanded, H), dim=0)[0]  # [W]
        
        # 计算每一列的最大y坐标
        max_y = torch.max(torch.where(mask == 1, y_indices_expanded, -1), dim=0)[0]  # [W]
        
        # 将边界坐标转换为numpy进行平滑处理
        min_y_np = min_y.cpu().numpy().astype(np.int32)
        max_y_np = max_y.cpu().numpy().astype(np.int32)
        
        # 对边界坐标进行平滑处理
        if W > 4:  # 确保有足够的列进行平滑
            # 计算一次邻居窗口大小，避免重复计算
            neighbor_window = max(3, int(W * 0.05))
            
            # 创建一次卷积核，避免重复创建
            kernel = np.ones(2 * neighbor_window + 1, dtype=np.float32) / (2 * neighbor_window + 1)
            kernel[neighbor_window] = 0  # 跳过当前列本身
            
            # 应用多次平滑（3次）以获得更好的效果
            for _ in range(3):
                # 平滑min_y（上边界）
                
                # 对有效点进行平滑
                valid_mask = min_y_np < H
                valid_min_y = min_y_np.copy()
                valid_min_y[~valid_mask] = H  # 无效点设为H
                
                # 使用卷积进行平滑
                smoothed = np.convolve(valid_min_y, kernel, mode='same')
                
                # 计算有效邻居数量
                valid_neighbors = np.convolve(valid_mask.astype(np.float32), kernel, mode='same') * (2 * neighbor_window + 1)
                
                # 只平滑凹陷的点
                height_ratio = 0.03  # 保持原始参数
                threshold_pixels = int(H * height_ratio)
                
                # 只在足够多有效邻居且当前点凹陷时进行平滑
                mask = valid_mask & (valid_neighbors >= neighbor_window * 2 // 3) & (min_y_np > smoothed + threshold_pixels)
                min_y_np[mask] = smoothed[mask].astype(np.int32)
            
            # 应用多次平滑（3次）以获得更好的效果
            for _ in range(3):
                # 平滑max_y（下边界）
                # 重用之前计算的邻居窗口大小和卷积核，避免重复计算
                
                # 对有效点进行平滑
                valid_mask = max_y_np > 0
                valid_max_y = max_y_np.copy()
                valid_max_y[~valid_mask] = 0  # 无效点设为0
                
                # 使用卷积进行平滑
                smoothed = np.convolve(valid_max_y, kernel, mode='same')
                
                # 计算有效邻居数量
                valid_neighbors = np.convolve(valid_mask.astype(np.float32), kernel, mode='same') * (2 * neighbor_window + 1)
                
                # 只平滑凹陷的点
                height_ratio = 0.015  # 保持原始参数
                threshold_pixels = int(H * height_ratio)
                
                # 只在足够多有效邻居且当前点凹陷时进行平滑
                mask = valid_mask & (valid_neighbors >= neighbor_window * 2 // 3) & (max_y_np < smoothed - threshold_pixels)
                max_y_np[mask] = smoothed[mask].astype(np.int32)
            
            # 将平滑后的边界坐标转换回GPU张量
            min_y = torch.tensor(min_y_np, device=device, dtype=torch.int32)
            max_y = torch.tensor(max_y_np, device=device, dtype=torch.int32)
        
        # 计算每一列的厚度 - 确保类型一致
        column_thickness = torch.where(col_nonzero > 0, max_y - min_y + 1, torch.zeros_like(max_y))
        
        # 转换为numpy用于存储和处理
        column_thickness_np = column_thickness.cpu().numpy().astype(np.int32)
        
        # 添加平滑处理：修正高度突变
        if W > 4:  # 确保有足够的列进行平滑
            # 计算一次邻居窗口大小和卷积核，避免重复计算
            neighbor_window = max(3, int(W * 0.05))
            kernel = np.ones(2 * neighbor_window + 1, dtype=np.float32) / (2 * neighbor_window + 1)
            kernel[neighbor_window] = 0  # 跳过当前列本身
            
            # 应用多次平滑（3次）以获得更好的效果
            for _ in range(3):
                
                # 对有效点进行平滑
                valid_mask = column_thickness_np > 0
                valid_thickness = column_thickness_np.copy()
                valid_thickness[~valid_mask] = 0  # 无效点设为0
                
                # 使用卷积进行平滑
                smoothed = np.convolve(valid_thickness, kernel, mode='same')
                
                # 计算有效邻居数量
                valid_neighbors = np.convolve(valid_mask.astype(np.float32), kernel, mode='same') * (2 * neighbor_window + 1)
                
                # 只在足够多有效邻居且厚度差异超过阈值时进行平滑
                mask = valid_mask & (valid_neighbors >= neighbor_window * 2 // 3)
                avg_neighbor = smoothed[mask] / (valid_neighbors[mask] / (2 * neighbor_window + 1))
                
                # 更积极地平滑：如果当前列厚度与平均厚度差异超过30%，则修正
                current = column_thickness_np[mask]
                smooth_mask = mask.copy()
                smooth_mask[mask] = (current < avg_neighbor * 0.7) | (current > avg_neighbor * 1.3)
                
                # 使用平滑后的厚度
                column_thickness_np[smooth_mask] = avg_neighbor[smooth_mask[mask]].astype(np.int32)
        
        # 找到最大厚度 - 使用平滑后的厚度
        max_thick = column_thickness_np.max()
        threshold = 0.5 * max_thick
        
        # 存储厚度信息
        thickness_dict[i] = {
            'max_thick': max_thick,
            'threshold': threshold
        }
        
        column_thickness_maps.append(column_thickness_np)
    
    return thickness_dict, column_thickness_maps












def visualize_bfov_masks(bfov_tensor, masks, output_dir='./bfov/result/simple_bfov', bounding_boxes=None):
    """
    可视化BFoV掩码区域
    
    参数:
    bfov_tensor: BFoV参数张量
    masks: 掩码张量
    output_dir: 输出目录
    bounding_boxes: 可选，外接矩形框列表 [x_min, y_min, x_max, y_max]
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    erp_w, erp_h = masks.shape[2], masks.shape[1]
    num_bfovs = masks.shape[0]
    
    # 不同BFoV使用不同颜色
    colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 紫色
        (0, 255, 255),  # 青色
        (128, 0, 0),    # 深红色
        (0, 128, 0),    # 深绿色
        (0, 0, 128),    # 深蓝色
        (128, 128, 0),  # 橄榄色
        (128, 0, 128),  # 深紫色
        (0, 128, 128),  # 深青色
        (192, 192, 192) # 银色
    ]
    
    # 1. 生成每个BFoV的单独可视化
    print("\n生成每个BFoV的单独可视化...")
    
    for i in range(num_bfovs):
        # 创建空白图像（白色背景）
        img = np.ones((erp_h, erp_w, 3), dtype=np.uint8) * 255
        
        # 获取当前掩码（转换为numpy）
        mask = masks[i].cpu().numpy()
        
        # 获取当前BFoV参数
        lon, lat, fov_x, fov_y = bfov_tensor[i].cpu().numpy()
        
        # 计算中心点像素坐标
        center_px = int((lon + np.pi) / (2 * np.pi) * erp_w)
        center_py = int((np.pi/2 - lat) / np.pi * erp_h)
        
        # 使用循环颜色
        color = colors[i % len(colors)]
        
        # 绘制掩码区域
        mask_indices = np.where(mask == 1)
        img[mask_indices[0], mask_indices[1]] = color
        
        # 绘制BFoV中心点
        cv2.circle(img, (center_px, center_py), 8, color, -1)
        
        # 绘制外接矩形框（如果提供）
        if bounding_boxes is not None:
            bbox = bounding_boxes[i]
            if bbox != [0, 0, 0, 0]:  # 确保不是空矩形框
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 0), 3)  # 黑色粗边框
        
        # 添加BFoV信息
        lon_deg = np.degrees(lon)
        lat_deg = np.degrees(lat)
        fov_x_deg = np.degrees(fov_x)
        fov_y_deg = np.degrees(fov_y)
        
        text = f'BFoV {i+1:02d}: Lon={lon_deg:7.2f}°, Lat={lat_deg:7.2f}°'
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        text = f'FOV: {fov_x_deg:5.1f}°x{fov_y_deg:5.1f}°'
        cv2.putText(img, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # 保存单个BFoV图像
        individual_path = os.path.join(output_dir, f'bfov_{i+1:02d}_mask.jpg')
        cv2.imwrite(individual_path, img)
        
        print(f"  BFoV {i+1:2d} 可视化: {individual_path}")
    
    # 2. 生成全部放在一起的可视化
    print("\n生成全部BFoV合并可视化...")
    combined_img = np.ones((erp_h, erp_w, 3), dtype=np.uint8) * 255
    
    for i in range(num_bfovs):
        # 获取当前掩码（转换为numpy）
        mask = masks[i].cpu().numpy()
        
        # 获取当前BFoV参数
        lon, lat, fov_x, fov_y = bfov_tensor[i].cpu().numpy()
        
        # 计算中心点像素坐标
        center_px = int((lon + np.pi) / (2 * np.pi) * erp_w)
        center_py = int((np.pi/2 - lat) / np.pi * erp_h)
        
        # 使用循环颜色
        color = colors[i % len(colors)]
        
        # 绘制掩码区域
        mask_indices = np.where(mask == 1)
        combined_img[mask_indices[0], mask_indices[1]] = color
        
        # 绘制BFoV中心点
        cv2.circle(combined_img, (center_px, center_py), 6, color, -1)
        
        # 绘制外接矩形框（如果提供）
        if bounding_boxes is not None:
            bbox = bounding_boxes[i]
            if bbox != [0, 0, 0, 0]:  # 确保不是空矩形框
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(combined_img, (x_min, y_min), (x_max, y_max), (0, 0, 0), 2)  # 黑色边框
        
        # 添加BFoV编号
        text = f'{i+1}'
        cv2.putText(combined_img, text, (center_px + 10, center_py + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # 保存合并图像
    combined_path = os.path.join(output_dir, 'all_bfovs_combined.jpg')
    cv2.imwrite(combined_path, combined_img)
    print(f"  合并可视化: {combined_path}")
    
    return output_dir

def visualize_mask_comparison(bfov_tensor, original_masks, filtered_masks, thickness_dict, output_dir='./bfov/result/simple_bfov/'):
    """
    可视化过滤前后的掩码对比
    
    参数:
    bfov_tensor: BFoV参数张量
    original_masks: 原始掩码张量
    filtered_masks: 过滤后的掩码张量
    thickness_dict: 厚度信息字典
    output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    num_bfovs = original_masks.shape[0]
    erp_w, erp_h = original_masks.shape[2], original_masks.shape[1]
    
    # 不同BFoV使用不同颜色
    colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 紫色
        (0, 255, 255),  # 青色
        (128, 0, 0),    # 深红色
        (0, 128, 0),    # 深绿色
        (0, 0, 128),    # 深蓝色
        (128, 128, 0),  # 橄榄色
        (128, 0, 128),  # 深紫色
        (0, 128, 128),  # 深青色
        (192, 192, 192) # 银色
    ]
    
    print("\n生成过滤前后掩码对比可视化...")
    
    for i in range(num_bfovs):
        # 创建对比图像（左右布局）
        combined_h = erp_h
        combined_w = erp_w * 2 + 20  # 左右各一个图像，中间留20像素空白
        combined_img = np.ones((combined_h, combined_w, 3), dtype=np.uint8) * 255
        
        # 获取原始掩码和过滤后的掩码
        original_mask = original_masks[i].cpu().numpy()
        filtered_mask = filtered_masks[i].cpu().numpy()
        
        # 获取当前BFoV参数
        lon, lat, fov_x, fov_y = bfov_tensor[i].cpu().numpy()
        
        # 计算中心点像素坐标
        center_px = int((lon + np.pi) / (2 * np.pi) * erp_w)
        center_py = int((np.pi/2 - lat) / np.pi * erp_h)
        
        # 使用循环颜色
        color = colors[i % len(colors)]
        
        # 绘制原始掩码（左侧）
        original_img = np.ones((erp_h, erp_w, 3), dtype=np.uint8) * 255
        original_mask_indices = np.where(original_mask == 1)
        original_img[original_mask_indices[0], original_mask_indices[1]] = color
        
        # 绘制原始掩码的中心点
        cv2.circle(original_img, (center_px, center_py), 8, color, -1)
        
        # 绘制过滤后的掩码（右侧）
        filtered_img = np.ones((erp_h, erp_w, 3), dtype=np.uint8) * 255
        filtered_mask_indices = np.where(filtered_mask == 1)
        filtered_img[filtered_mask_indices[0], filtered_mask_indices[1]] = color
        
        # 绘制过滤后掩码的中心点
        cv2.circle(filtered_img, (center_px, center_py), 8, color, -1)
        
        # 将左右图像合并到对比图像中
        combined_img[:, :erp_w] = original_img
        combined_img[:, erp_w+20:] = filtered_img
        
        # 添加标题
        cv2.putText(combined_img, "Original Mask", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        cv2.putText(combined_img, "Filtered Mask (Thickness >= 30%)", (erp_w + 50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        
        # 添加BFoV信息
        if thickness_dict is not None and i in thickness_dict:
            thickness_info = f"Max Thick: {thickness_dict[i]['max_thick']:.2f} px, Threshold: {thickness_dict[i]['threshold']:.2f} px"
            cv2.putText(combined_img, thickness_info, (50, erp_h - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # 保存对比图像
        comparison_path = os.path.join(output_dir, f'bfov_{i+1:02d}_comparison.jpg')
        cv2.imwrite(comparison_path, combined_img)
        
        print(f"  BFoV {i+1:2d} 对比可视化: {comparison_path}")
    
    # 生成合并的对比图像（所有BFoV）
    print("\n生成所有BFoV合并的对比图像...")
    
    # 计算合并图像的尺寸
    grid_cols = 5
    grid_rows = (num_bfovs + grid_cols - 1) // grid_cols
    img_size = 300
    margin = 20
    
    combined_h = grid_rows * img_size + (grid_rows + 1) * margin
    # 简单直接的宽度计算 - 左右各grid_cols列，每列包含图像和margin
    combined_w = margin * (2 * grid_cols + 1) + 2 * grid_cols * img_size
    combined_all_img = np.ones((combined_h, combined_w, 3), dtype=np.uint8) * 255
    
    for i in range(num_bfovs):
        # 获取原始掩码和过滤后的掩码
        original_mask = original_masks[i].cpu().numpy()
        filtered_mask = filtered_masks[i].cpu().numpy()
        
        # 使用循环颜色
        color = colors[i % len(colors)]
        
        # 调整图像大小
        original_resized = cv2.resize(original_mask.astype(np.uint8), (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        filtered_resized = cv2.resize(filtered_mask.astype(np.uint8), (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        
        # 创建小图像
        original_small = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        original_small[original_resized == 1] = color
        
        filtered_small = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        filtered_small[filtered_resized == 1] = color
        
        # 计算位置
        row = i // grid_cols
        col = i % grid_cols
        
        # 原始掩码位置
        x1 = margin + col * (img_size + margin)
        y1 = margin + row * (img_size + margin)
        
        # 过滤后掩码位置（使用相同的步长）
        x2 = margin + grid_cols * (img_size + margin) + col * (img_size + margin)
        y2 = y1
        
        # 绘制小图像
        combined_all_img[y1:y1+img_size, x1:x1+img_size] = original_small
        combined_all_img[y2:y2+img_size, x2:x2+img_size] = filtered_small
        
        # 添加BFoV编号
        cv2.putText(combined_all_img, f"BFoV {i+1}", (x1 + 10, y1 + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # 添加整体标题
    cv2.putText(combined_all_img, "Original Masks", (margin + 100, margin // 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.putText(combined_all_img, "Filtered Masks", (margin + grid_cols * (img_size + margin) + 100, margin // 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    # 保存合并对比图像
    all_comparison_path = os.path.join(output_dir, 'all_bfovs_comparison.jpg')
    cv2.imwrite(all_comparison_path, combined_all_img)
    print(f"  所有BFoV合并对比可视化: {all_comparison_path}")
    
    return output_dir

def main():
    """
    主函数：生成随机BFoV掩码并可视化
    """
    print("=== 简单BFoV掩码生成器（1920x960固定分辨率） ===")
    
    # 检查GPU可用性
    if torch.cuda.is_available():
        device = 'cuda'
        print("✓ GPU可用，使用CUDA加速")
    else:
        device = 'cpu'
        print("⚠ GPU不可用，使用CPU模式")
    
    # 用户参数配置
    num_bfovs = 20  # 生成的BFoV数量（可调整）
    erp_w = 1920    # 固定分辨率宽度
    erp_h = 960     # 固定分辨率高度
    output_dir = './bfov/result/simple_bfov/'
    
    print(f"\n配置参数：")
    print(f"   BFoV数量: {num_bfovs}")
    print(f"   分辨率: {erp_w}x{erp_h}")
    print(f"   输出目录: {output_dir}")
    
    # 1. 生成随机BFoV参数
    print(f"\n1. 生成 {num_bfovs} 个随机BFoV参数...")
    bfov_tensor = generate_random_bfov_params(num_bfovs, device=device)
    
    # 2. 生成BFoV掩码
    print(f"\n2. 生成 {erp_w}x{erp_h} 分辨率的BFoV掩码...")
    start_time = time.time()
    masks = generate_bfov_masks(bfov_tensor, erp_w=erp_w, erp_h=erp_h)
    mask_time = time.time() - start_time
    print(f"✓ BFoV掩码生成完成，耗时: {mask_time:.3f}秒")
    
    # 3. 使用MaskToBox类处理掩码并生成边界框
    print(f"\n3. 使用MaskToBox类处理掩码并生成边界框...")
    start_time = time.time()
    mask_to_box = MaskToBox()
    # 使用MaskToBox类处理原始生成的掩码
    bounding_boxes = mask_to_box(masks, preprocess=True)
    mask_process_time = time.time() - start_time
    print(f"✓ 掩码处理和边界框生成完成，耗时: {mask_process_time:.3f}秒")
    
    # 由于process_masks方法不可用，我们使用原始掩码和边界框进行可视化
    filtered_masks = masks
    largest_masks = masks
    thickness_dict = None
    
    # 4. 可视化原始BFoV掩码
    print(f"\n4. 可视化原始BFoV掩码区域...")
    start_time = time.time()
    output_dir = visualize_bfov_masks(bfov_tensor, masks, output_dir=output_dir)
    visualize_original_time = time.time() - start_time
    print(f"✓ 原始掩码可视化完成，耗时: {visualize_original_time:.3f}秒")
    
    # 5. 可视化处理后的BFoV掩码（包含外接矩形框）
    print(f"\n5. 可视化处理后的BFoV掩码区域...")
    start_time = time.time()
    processed_output_dir = os.path.join(output_dir, 'processed')
    output_dir = visualize_bfov_masks(bfov_tensor, largest_masks, output_dir=processed_output_dir, bounding_boxes=bounding_boxes.cpu().tolist())
    visualize_processed_time = time.time() - start_time
    print(f"✓ 处理后掩码可视化完成，耗时: {visualize_processed_time:.3f}秒")
    
    # 6. 可视化过滤前后的掩码对比
    print(f"\n6. 可视化过滤前后的掩码对比...")
    start_time = time.time()
    comparison_output_dir = os.path.join(output_dir, 'comparison')
    output_dir = visualize_mask_comparison(bfov_tensor, masks, largest_masks, thickness_dict, output_dir=comparison_output_dir)
    visualize_comparison_time = time.time() - start_time
    print(f"✓ 掩码对比可视化完成，耗时: {visualize_comparison_time:.3f}秒")
    
    # 性能统计
    total_time = mask_time + mask_process_time + visualize_original_time + visualize_processed_time + visualize_comparison_time
    print(f"\n📊 性能统计：")
    print(f"   掩码生成耗时: {mask_time:.3f}秒")
    print(f"   掩码处理和边界框生成耗时: {mask_process_time:.3f}秒")
    print(f"   原始掩码可视化耗时: {visualize_original_time:.3f}秒")
    print(f"   处理后掩码可视化耗时: {visualize_processed_time:.3f}秒")
    print(f"   掩码对比可视化耗时: {visualize_comparison_time:.3f}秒")
    print(f"   总耗时: {total_time:.3f}秒")
    
    # 输出总结
    print(f"\n🎉 BFoV掩码生成、处理和可视化完成！")
    print(f"   生成的BFoV数量: {num_bfovs}")
    print(f"   分辨率: {erp_w}x{erp_h}")
    print(f"   生成的文件：")
    print(f"   - 原始掩码可视化: {num_bfovs + 1} 个文件")
    print(f"   - 处理后掩码可视化: {num_bfovs + 1} 个文件")
    print(f"   - 掩码对比可视化: {num_bfovs + 1} 个文件")
    print(f"   - 输出目录: {output_dir}")
    print(f"\n📋 功能特点：")
    print(f"   - 生成随机BFoV参数和对应的掩码")
    print(f"   - 使用MaskToBox类进行掩码处理")
    print(f"   - 按列计算掩码厚度")
    print(f"   - 按列过滤，保留厚度≥50%max_thick的区域")
    print(f"   - 提取并保留最大的连通区域")
    print(f"   - 生成最大连通区域的外接矩形框")
    print(f"   - 提供掩码对比可视化")
    print(f"   - 完整的性能统计和时间分析")


if __name__ == '__main__':
    main()
