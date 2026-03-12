#!/usr/bin/env python3
"""
GPU复用BFoV掩码生成器
基于GPU复用的高效掩码生成，支持指定BFoV参数和两种分辨率（最小和最大）

新增功能：
- 支持将低分辨率采样至高分辨率并绘制边界框
- 支持将高分辨率采样至低分辨率并绘制边界框
"""

import torch
import numpy as np
import cv2
import os
import time
from PANDORA.PRDA.lib.ReuseGPUImageRecorder import ReuseGPUImageRecorder


def generate_bfov_masks_gpu_reuse(bfov_tensor, erp_w=1920, erp_h=960):
    """
    基于GPU复用的BFoV掩码生成函数
    
    参数:
    bfov_tensor: [num_bfovs, 4] 的张量（弧度制）
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    
    返回:
    masks: [num_bfovs, erp_h, erp_w] 的掩码张量
    """
    # 确保张量在GPU上
    if not bfov_tensor.is_cuda:
        bfov_tensor = bfov_tensor.cuda()
    
    num_bfovs = bfov_tensor.shape[0]
    
    # 初始化掩码张量
    masks = torch.zeros((num_bfovs, erp_h, erp_w), dtype=torch.uint8, device=bfov_tensor.device)
    
    # 🔑 关键优化：创建单个ReuseGPUImageRecorder实例
    gpu_recorder = ReuseGPUImageRecorder(erp_w, erp_h, device=bfov_tensor.device)
    
    # 批量处理所有BFoV
    for i in range(num_bfovs):
        # 获取当前BFoV参数
        lon, lat, fov_x, fov_y = bfov_tensor[i].tolist()
        
        # 将视场角转换为角度制
        fov_x_deg = np.degrees(fov_x)
        fov_y_deg = np.degrees(fov_y)
        
        # 🔑 关键优化：通过属性更新复用GPUImageRecorder实例
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
    
    return masks


def generate_bfov_masks_multiple_resolutions(bfov_tensor, resolutions):
    """
    在多种分辨率下生成BFoV掩码
    
    参数:
    bfov_tensor: [num_bfovs, 4] 的张量（弧度制）
    resolutions: 分辨率列表，格式为 [(width, height), ...]
    
    返回:
    masks_dict: 字典，键为分辨率，值为对应的掩码张量
    """
    masks_dict = {}
    
    for resolution in resolutions:
        erp_w, erp_h = resolution
        print(f"   生成 {erp_w}x{erp_h} 分辨率的掩码...")
        
        # 使用GPU复用生成当前分辨率的掩码
        masks = generate_bfov_masks_gpu_reuse(bfov_tensor, erp_w, erp_h)
        masks_dict[resolution] = masks
        
        # 统计信息
        total_pixels = masks.shape[1] * masks.shape[2]
        print(f"     {erp_w}x{erp_h} 分辨率掩码生成完成")
    
    return masks_dict


def masks_to_bboxes(mask_list, device='cuda'):
    """
    更高效的GPU批量掩码到边界框转换（使用非零坐标）
    
    参数:
    mask_list: 掩码列表，每个元素是形状为 [M, H, W] 的PyTorch张量
    device: 目标设备，默认为'cuda'
    
    返回:
    bboxes_list: 边界框列表，每个元素是形状为 [M, 4] 的张量 [x_min, y_min, x_max, y_max]
    """
    bboxes_list = []
    
    for masks in mask_list:
        masks = masks.to(device)
        M, H, W = masks.shape
        
        # 批量处理所有掩码
        bboxes = torch.zeros((M, 4), device=device)
        
        for i in range(M):
            mask = masks[i]
            # 找到非零像素的坐标
            nonzero_coords = torch.nonzero(mask > 0)
            
            if len(nonzero_coords) > 0:
                # 计算边界框
                y_coords = nonzero_coords[:, 0]
                x_coords = nonzero_coords[:, 1]
                
                y_min = y_coords.min()
                y_max = y_coords.max()
                x_min = x_coords.min()
                x_max = x_coords.max()
                
                bboxes[i] = torch.tensor([x_min, y_min, x_max, y_max], device=device)
            else:
                bboxes[i] = torch.tensor([0, 0, 0, 0], device=device)
        
        bboxes_list.append(bboxes)
    
    return bboxes_list


def generate_bboxes_for_masks_dict(masks_dict, device='cuda'):
    """
    为多种分辨率的掩码字典生成边界框
    
    参数:
    masks_dict: 掩码字典，键为分辨率，值为掩码张量
    device: 目标设备
    
    返回:
    bboxes_dict: 边界框字典，键为分辨率，值为边界框列表
    """
    bboxes_dict = {}
    
    for resolution, masks in masks_dict.items():
        # 将掩码张量包装成列表格式（符合masks_to_bboxes输入要求）
        mask_list = [masks]
        
        # 生成边界框
        bboxes_list = masks_to_bboxes(mask_list, device=device)
        bboxes_dict[resolution] = bboxes_list[0]  # 提取第一个元素
        
        print(f"    分辨率 {resolution[0]}x{resolution[1]} 边界框生成完成")
    
    return bboxes_dict


def resize_masks_and_resample_bboxes(masks, bboxes, source_res, target_res, device='cuda'):
    """
    调整掩码分辨率并重新采样边界框
    
    参数:
    masks: 原始掩码张量 [num_bfovs, source_h, source_w]
    bboxes: 原始边界框张量 [num_bfovs, 4]，格式为 [x_min, y_min, x_max, y_max]
    source_res: 源分辨率 (source_w, source_h)
    target_res: 目标分辨率 (target_w, target_h)
    device: 目标设备
    
    返回:
    resized_masks: 调整后的掩码张量 [num_bfovs, target_h, target_w]
    resampled_bboxes: 重新采样后的边界框张量 [num_bfovs, 4]
    """
    num_bfovs, source_h, source_w = masks.shape
    target_w, target_h = target_res
    
    # 计算缩放比例
    scale_w = target_w / source_w
    scale_h = target_h / source_h
    
    # 调整掩码分辨率
    # 首先将掩码转换为float32以进行插值
    masks_float = masks.float()
    
    # 使用双线性插值调整掩码大小
    # PyTorch的interpolate要求输入为4D张量：[batch, channels, height, width]
    resized_masks = torch.nn.functional.interpolate(
        masks_float.unsqueeze(1),  # 添加通道维度
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False
    ).squeeze(1)  # 移除通道维度
    
    # 将掩码转换回uint8
    resized_masks = (resized_masks > 0.5).to(torch.uint8)
    
    # 重新采样边界框
    resampled_bboxes = bboxes.clone()
    
    # 调整边界框坐标
    resampled_bboxes[:, 0] = bboxes[:, 0] * scale_w  # x_min
    resampled_bboxes[:, 1] = bboxes[:, 1] * scale_h  # y_min
    resampled_bboxes[:, 2] = bboxes[:, 2] * scale_w  # x_max
    resampled_bboxes[:, 3] = bboxes[:, 3] * scale_h  # y_max
    
    # 确保边界框在有效范围内
    resampled_bboxes[:, 0] = torch.clamp(resampled_bboxes[:, 0], 0, target_w - 1)
    resampled_bboxes[:, 1] = torch.clamp(resampled_bboxes[:, 1], 0, target_h - 1)
    resampled_bboxes[:, 2] = torch.clamp(resampled_bboxes[:, 2], 0, target_w - 1)
    resampled_bboxes[:, 3] = torch.clamp(resampled_bboxes[:, 3], 0, target_h - 1)
    
    return resized_masks, resampled_bboxes


def visualize_bfov_masks_on_blank(bfov_tensor, masks, bboxes=None, erp_w=1920, erp_h=960, 
                                 output_dir='./bfov/result', resolution_label=''):
    """
    在空白图像上可视化BFoV掩码和边界框
    生成每个BFoV的单独可视化和全部放在一起的可视化
    
    参数:
    bfov_tensor: BFoV参数张量
    masks: 掩码张量
    bboxes: 边界框张量 [num_bfovs, 4]，格式为 [x_min, y_min, x_max, y_max]
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    output_dir: 输出目录
    resolution_label: 分辨率标签，用于文件名
    
    返回:
    individual_paths: 单个BFoV可视化图像路径列表
    combined_path: 合并可视化图像路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
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
        (192, 192, 192),# 银色
        (128, 0, 0),    # 栗色
        (0, 0, 128)     # 海军蓝
    ]
    
    num_bfovs = masks.shape[0]
    individual_paths = []
    
    # 1. 生成每个BFoV的单独可视化
    print("   生成每个BFoV的单独可视化...")
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
        cv2.circle(img, (center_px, center_py), 10, color, -1)
        
        # 如果提供了边界框，绘制边界框
        if bboxes is not None:
            bbox = bboxes[i].cpu().numpy()
            x_min, y_min, x_max, y_max = bbox.astype(int)
            
            # 检查边界框是否有效（非零）
            if x_max > x_min and y_max > y_min:
                # 绘制边界框（使用对比色）
                bbox_color = (0, 0, 0)  # 黑色边界框
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), bbox_color, 3)
                
                # 添加边界框尺寸信息
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
                bbox_text = f'Box: {bbox_width}x{bbox_height}'
                cv2.putText(img, bbox_text, (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 1)
        
        # 添加BFoV编号和参数信息
        lon_deg = np.degrees(lon)
        lat_deg = np.degrees(lat)
        fov_x_deg = np.degrees(fov_x)
        fov_y_deg = np.degrees(fov_y)
        
        text1 = f'BFoV {i+1}'
        text2 = f'({lon_deg:.1f}°, {lat_deg:.1f}°)'
        text3 = f'FOV: {fov_x_deg:.1f}°x{fov_y_deg:.1f}°'
        
        cv2.putText(img, text1, (center_px + 15, center_py - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(img, text2, (center_px + 15, center_py), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(img, text3, (center_px + 15, center_py + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 保存单个BFoV图像
        filename = f'bfov_{i+1:02d}_individual'
        if resolution_label:
            filename += f'_{resolution_label}'
        individual_path = os.path.join(output_dir, f'{filename}.jpg')
        cv2.imwrite(individual_path, img)
        individual_paths.append(individual_path)
        
        print(f"      BFoV {i+1:2d} 单独可视化: {individual_path}")
    
    # 2. 生成全部放在一起的可视化
    print("   生成全部BFoV合并可视化...")
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
        cv2.circle(combined_img, (center_px, center_py), 8, color, -1)
        
        # 如果提供了边界框，绘制边界框
        if bboxes is not None:
            bbox = bboxes[i].cpu().numpy()
            x_min, y_min, x_max, y_max = bbox.astype(int)
            
            # 检查边界框是否有效（非零）
            if x_max > x_min and y_max > y_min:
                # 绘制边界框（使用对比色）
                bbox_color = (0, 0, 0)  # 黑色边界框
                cv2.rectangle(combined_img, (x_min, y_min), (x_max, y_max), bbox_color, 2)
        
        # 添加BFoV编号
        text = f'{i+1}'
        cv2.putText(combined_img, text, (center_px + 10, center_py), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # 保存合并图像
    filename = 'all_bfovs_combined'
    if resolution_label:
        filename += f'_{resolution_label}'
    combined_path = os.path.join(output_dir, f'{filename}.jpg')
    cv2.imwrite(combined_path, combined_img)
    print(f"     全部BFoV合并可视化: {combined_path}")
    
    return individual_paths, combined_path


def main():
    """主函数：测试GPU复用BFoV掩码生成（仅指定BFoV和两种分辨率）"""
    print("=== GPU复用BFoV掩码生成器（指定BFoV和两种分辨率） ===")
    
    # 检查GPU可用性
    if torch.cuda.is_available():
        device = 'cuda'
        print("✓ GPU可用，使用CUDA加速")
    else:
        device = 'cpu'
        print("⚠ GPU不可用，使用CPU模式")
    
    # 定义两种分辨率：最小和最大
    resolutions = [
        (960, 480),   # 最小分辨率
        (1920, 960)    # 最大分辨率
    ]
    
    print(f"\n支持的分辨率: {resolutions}")
    
    # 定义指定的BFoV参数（角度制）
    specified_bfovs = [
        # 格式：[经度(°), 纬度(°), 水平视场(°), 垂直视场(°)]
        [0, 60, 10, 60],     # 指定BFoV 1
        [0, 60, 10, 59.9],     # 指定BFoV 2
        [0, 60, 10, 60.1]      # 指定BFoV 3
    ]
    
    num_bfovs = len(specified_bfovs)
    
    # 将指定的BFoV参数转换为弧度制
    print(f"\n1. 处理 {num_bfovs} 个指定BFoV参数（角度制转换为弧度制）...")
    specified_bfovs_rad = []
    
    for i, bfov in enumerate(specified_bfovs):
        lon_deg, lat_deg, fov_x_deg, fov_y_deg = bfov
        # 转换为弧度制
        lon = np.radians(lon_deg)
        lat = np.radians(lat_deg)
        fov_x = np.radians(fov_x_deg)
        fov_y = np.radians(fov_y_deg)
        
        specified_bfovs_rad.append([lon, lat, fov_x, fov_y])
        
        print(f"  指定BFoV {i+1:2d}: 经度={lon_deg:7.2f}°, 纬度={lat_deg:7.2f}°, "
              f"水平视场={fov_x_deg:5.1f}°, 垂直视场={fov_y_deg:5.1f}°")
    
    # 将指定的BFoV参数转换为张量
    bfov_tensor = torch.tensor(specified_bfovs_rad, dtype=torch.float32, device=device)
    
    # 2. 在两种分辨率下生成掩码
    print(f"\n2. 在两种分辨率下生成BFoV掩码...")
    start_time = time.time()
    masks_dict = generate_bfov_masks_multiple_resolutions(bfov_tensor, resolutions)
    mask_time = time.time() - start_time
    print(f"✓ 所有分辨率掩码生成完成，耗时: {mask_time:.3f}秒")
    
    # 3. 为所有分辨率生成边界框
    print(f"\n3. 为所有分辨率生成边界框...")
    start_time = time.time()
    bboxes_dict = generate_bboxes_for_masks_dict(masks_dict, device=device)
    bbox_time = time.time() - start_time
    print(f"✓ 所有分辨率边界框生成完成，耗时: {bbox_time:.3f}秒")
    
    # 4. 在空白图像上可视化（两种分辨率）
    print(f"\n4. 在两种分辨率的空白图像上可视化BFoV掩码和边界框...")
    output_dir = './bfov/result/mask_to_box_two_resolutions/'
    
    all_individual_paths = []
    all_combined_paths = []
    
    for resolution in resolutions:
        erp_w, erp_h = resolution
        resolution_label = f"{erp_w}x{erp_h}"
        
        print(f"\n   分辨率 {resolution_label}:")
        masks = masks_dict[resolution]
        
        # 为当前分辨率创建子目录
        resolution_output_dir = os.path.join(output_dir, resolution_label)
        
        # 可视化当前分辨率的掩码和边界框
        individual_paths, combined_path = visualize_bfov_masks_on_blank(
            bfov_tensor, masks, bboxes_dict[resolution], erp_w, erp_h, resolution_output_dir, resolution_label
        )
        
        all_individual_paths.extend(individual_paths)
        all_combined_paths.append(combined_path)
        
        # 统计信息
        total_pixels = masks.shape[1] * masks.shape[2]
        print(f"    统计信息:")
        for i in range(num_bfovs):
            mask_pixels = torch.sum(masks[i]).item()
            coverage = mask_pixels / total_pixels * 100
            
            # 边界框统计
            bbox = bboxes_dict[resolution][i].cpu().numpy()
            x_min, y_min, x_max, y_max = bbox.astype(int)
            bbox_width = x_max - x_min if x_max > x_min else 0
            bbox_height = y_max - y_min if y_max > y_min else 0
            bbox_area = bbox_width * bbox_height
            
            print(f"      BFoV {i+1:2d}: 掩码像素数={mask_pixels:6d}, 覆盖率={coverage:5.2f}%, "
                  f"边界框={bbox_width}x{bbox_height}, 面积={bbox_area}")
    
    # 5. 分辨率转换和重采样可视化
    print(f"\n5. 分辨率转换和重采样可视化...")
    
    # 使用resolutions列表中定义的实际分辨率值
    # 假设resolutions列表第一个元素是低分辨率，第二个元素是高分辨率
    low_res = resolutions[0]
    high_res = resolutions[1]
    
    # 获取原始掩码和边界框
    low_res_masks = masks_dict[low_res]
    low_res_bboxes = bboxes_dict[low_res]
    high_res_masks = masks_dict[high_res]
    high_res_bboxes = bboxes_dict[high_res]
    
    # 5.1 低分辨率到高分辨率转换
    print(f"\n   低分辨率({low_res[0]}x{low_res[1]})到高分辨率({high_res[0]}x{high_res[1]})转换...")
    low_to_high_masks, low_to_high_bboxes = resize_masks_and_resample_bboxes(
        low_res_masks, low_res_bboxes, low_res, high_res, device=device
    )
    
    # 可视化低到高转换结果
    low_to_high_output_dir = os.path.join(output_dir, 'low_to_high')
    low_to_high_label = f"low_to_high_{low_res[0]}x{low_res[1]}_to_{high_res[0]}x{high_res[1]}"
    low_to_high_individual, low_to_high_combined = visualize_bfov_masks_on_blank(
        bfov_tensor, low_to_high_masks, low_to_high_bboxes, high_res[0], high_res[1], 
        low_to_high_output_dir, low_to_high_label
    )
    
    # 统计信息
    print(f"    低到高转换统计信息:")
    total_pixels = high_res[0] * high_res[1]
    for i in range(num_bfovs):
        mask_pixels = torch.sum(low_to_high_masks[i]).item()
        coverage = mask_pixels / total_pixels * 100
        
        # 边界框统计
        bbox = low_to_high_bboxes[i].cpu().numpy()
        x_min, y_min, x_max, y_max = bbox.astype(int)
        bbox_width = x_max - x_min if x_max > x_min else 0
        bbox_height = y_max - y_min if y_max > y_min else 0
        bbox_area = bbox_width * bbox_height
        
        print(f"      BFoV {i+1:2d}: 掩码像素数={mask_pixels:6d}, 覆盖率={coverage:5.2f}%, "
              f"边界框={bbox_width}x{bbox_height}, 面积={bbox_area}")
    
    # 5.2 高分辨率到低分辨率转换
    print(f"\n   高分辨率({high_res[0]}x{high_res[1]})到低分辨率({low_res[0]}x{low_res[1]})转换...")
    high_to_low_masks, high_to_low_bboxes = resize_masks_and_resample_bboxes(
        high_res_masks, high_res_bboxes, high_res, low_res, device=device
    )
    
    # 可视化高到低转换结果
    high_to_low_output_dir = os.path.join(output_dir, 'high_to_low')
    high_to_low_label = f"high_to_low_{high_res[0]}x{high_res[1]}_to_{low_res[0]}x{low_res[1]}"
    high_to_low_individual, high_to_low_combined = visualize_bfov_masks_on_blank(
        bfov_tensor, high_to_low_masks, high_to_low_bboxes, low_res[0], low_res[1], 
        high_to_low_output_dir, high_to_low_label
    )
    
    # 统计信息
    print(f"    高到低转换统计信息:")
    total_pixels = low_res[0] * low_res[1]
    for i in range(num_bfovs):
        mask_pixels = torch.sum(high_to_low_masks[i]).item()
        coverage = mask_pixels / total_pixels * 100
        
        # 边界框统计
        bbox = high_to_low_bboxes[i].cpu().numpy()
        x_min, y_min, x_max, y_max = bbox.astype(int)
        bbox_width = x_max - x_min if x_max > x_min else 0
        bbox_height = y_max - y_min if y_max > y_min else 0
        bbox_area = bbox_width * bbox_height
        
        print(f"      BFoV {i+1:2d}: 掩码像素数={mask_pixels:6d}, 覆盖率={coverage:5.2f}%, "
              f"边界框={bbox_width}x{bbox_height}, 面积={bbox_area}")
    
    # 5. 性能统计
    print(f"\n5. 性能统计：")
    total_time = mask_time + bbox_time
    print(f"   掩码生成耗时: {mask_time:.3f}秒")
    print(f"   边界框生成耗时: {bbox_time:.3f}秒")
    print(f"   总生成耗时: {total_time:.3f}秒")
    
    for resolution in resolutions:
        erp_w, erp_h = resolution
        total_pixels = erp_w * erp_h
        print(f"   分辨率 {erp_w}x{erp_h}: 总像素={total_pixels:,}")
    
    print(f"\n🎉 GPU复用BFoV掩码和边界框生成完成！")
    print(f"   生成的BFoV数量: {num_bfovs}")
    print(f"   支持的分辨率: {len(resolutions)} 种")
    print(f"   总生成耗时: {total_time:.3f}秒")
    print(f"   生成的文件：")
    
    for resolution in resolutions:
        erp_w, erp_h = resolution
        resolution_dir = f"{output_dir}/{erp_w}x{erp_h}/"
        print(f"   - 分辨率 {erp_w}x{erp_h}:")
        print(f"       单个BFoV可视化（含边界框）: {num_bfovs} 个文件")
        print(f"       全部BFoV合并可视化（含边界框）: 1 个文件")
        print(f"       输出目录: {resolution_dir}")
    
    print(f"\n📊 边界框生成特点：")
    print(f"   - 使用GPU加速的非零坐标检测")
    print(f"   - 自动处理空掩码（返回零边界框）")
    print(f"   - 支持批量处理，高效生成")
    print(f"   - 边界框格式: [x_min, y_min, x_max, y_max]")
    print(f"   - 可视化中显示边界框尺寸和面积")


if __name__ == '__main__':
    main()