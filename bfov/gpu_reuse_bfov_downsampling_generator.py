#!/usr/bin/env python3
"""
GPU复用BFoV掩码生成器 - 下采样版
基于GPU复用的高效掩码生成，支持指定BFoV参数和多次下采样

功能:
- 生成指定BFoV掩码
- 对掩码进行四次下采样（使用采样率: 4, 8, 16, 32）
- 生成每次下采样后的边界框
- 可视化所有掩码和边界框
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


def downsample_mask(mask, sampling_rate):
    """
    对单个掩码进行下采样
    
    参数:
    mask: 原始掩码张量 [H, W]
    sampling_rate: 采样率（原始尺寸与下采样后尺寸的比值）
                  例如：sampling_rate=4 表示下采样后尺寸是原始尺寸的1/4
    
    返回:
    downsampled_mask: 下采样后的掩码张量
    """
    # 转换为float32以进行插值
    mask_float = mask.float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # 计算新尺寸
    new_h = int(mask.shape[0] / sampling_rate)
    new_w = int(mask.shape[1] / sampling_rate)
    
    # 使用双线性插值下采样
    downsampled = torch.nn.functional.interpolate(
        mask_float,
        size=(new_h, new_w),
        mode='bilinear',
        align_corners=False
    )
    
    # 应用阈值：采样值大于0.5的区域置1
    downsampled_mask = (downsampled.squeeze() > 0.5).to(torch.uint8)
    
    return downsampled_mask


def downsample_masks_with_rates(masks, sampling_rates):
    """
    对批量掩码进行指定采样率的下采样
    
    参数:
    masks: 原始掩码张量 [num_bfovs, H, W]
    sampling_rates: 采样率列表，每个元素表示原始尺寸与下采样后尺寸的比值
                    例如：[4, 8, 16, 32] 表示分别下采样到1/4, 1/8, 1/16, 1/32尺寸
    
    返回:
    downsampled_masks_list: 包含原始掩码和所有下采样掩码的列表
    """
    num_bfovs, original_h, original_w = masks.shape
    downsampled_masks_list = [masks]  # 包含原始掩码
    
    for i, sampling_rate in enumerate(sampling_rates):
        print(f"   第 {i+1} 次下采样 (采样率: {sampling_rate})...")
        
        # 对每个BFoV掩码进行下采样
        downsampled_batch = []
        for bfov_idx in range(num_bfovs):
            # 获取单个BFoV掩码
            bfov_mask = masks[bfov_idx]  # 始终使用原始掩码进行下采样
            
            # 下采样
            downsampled_bfov_mask = downsample_mask(bfov_mask, sampling_rate)
            downsampled_batch.append(downsampled_bfov_mask)
        
        # 转换为张量并添加到列表
        downsampled_tensor = torch.stack(downsampled_batch)
        downsampled_masks_list.append(downsampled_tensor)
        
        # 打印信息
        new_h, new_w = downsampled_tensor.shape[1], downsampled_tensor.shape[2]
        print(f"     下采样后尺寸: {new_w}x{new_h}")
    
    return downsampled_masks_list


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


def visualize_bfov_masks_with_downsampling(bfov_tensor, masks_list, bboxes_list, original_res, 
                                          output_dir='./bfov/result/downsampling_result/'):
    """
    可视化原始掩码和所有下采样后的掩码及边界框
    
    参数:
    bfov_tensor: BFoV参数张量
    masks_list: 包含原始掩码和下采样掩码的列表
    bboxes_list: 包含原始边界框和下采样边界框的列表
    original_res: 原始分辨率 (original_w, original_h)
    output_dir: 输出目录
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
        (0, 255, 255)   # 青色
    ]
    
    num_bfovs = bfov_tensor.shape[0]
    num_levels = len(masks_list)
    
    for level in range(num_levels):
        print(f"\n   可视化第 {level} 级掩码和边界框...")
        print(f"     (级别0: 原始掩码, 级别1-3: 下采样掩码)")
        
        masks = masks_list[level]
        bboxes = bboxes_list[level]
        
        # 获取当前级别的分辨率
        current_h, current_w = masks.shape[1], masks.shape[2]
        level_label = f"level_{level}_{current_w}x{current_h}"
        
        # 创建当前级别的输出目录
        level_output_dir = os.path.join(output_dir, level_label)
        os.makedirs(level_output_dir, exist_ok=True)
        
        # 1. 生成每个BFoV的单独可视化
        print("     生成每个BFoV的单独可视化...")
        individual_paths = []
        
        for i in range(num_bfovs):
            # 创建空白图像（白色背景）
            img = np.ones((current_h, current_w, 3), dtype=np.uint8) * 255
            
            # 获取当前掩码（转换为numpy）
            mask = masks[i].cpu().numpy()
            
            # 获取当前BFoV参数
            lon, lat, fov_x, fov_y = bfov_tensor[i].cpu().numpy()
            
            # 计算中心点像素坐标（转换到当前分辨率）
            center_px = int((lon + np.pi) / (2 * np.pi) * current_w)
            center_py = int((np.pi/2 - lat) / np.pi * current_h)
            
            # 使用循环颜色
            color = colors[i % len(colors)]
            
            # 绘制掩码区域
            mask_indices = np.where(mask == 1)
            img[mask_indices[0], mask_indices[1]] = color
            
            # 绘制BFoV中心点
            cv2.circle(img, (center_px, center_py), 5 if level == 0 else 3, color, -1)
            
            # 绘制边界框
            bbox = bboxes[i].cpu().numpy()
            x_min, y_min, x_max, y_max = bbox.astype(int)
            
            # 检查边界框是否有效（非零）
            if x_max > x_min and y_max > y_min:
                # 绘制边界框
                bbox_color = (0, 0, 0)  # 黑色边界框
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), bbox_color, 2)
                
                # 添加边界框尺寸信息
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
                bbox_text = f'Box: {bbox_width}x{bbox_height}'
                cv2.putText(img, bbox_text, (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 1)
            
            # 添加BFoV编号和参数信息（仅在原始分辨率显示详细信息）
            if level == 0:
                text1 = f'BFoV {i+1}'
                lon_deg = np.degrees(lon)
                lat_deg = np.degrees(lat)
                fov_x_deg = np.degrees(fov_x)
                fov_y_deg = np.degrees(fov_y)
                text2 = f'({lon_deg:.1f}°, {lat_deg:.1f}°)'
                text3 = f'FOV: {fov_x_deg:.1f}°x{fov_y_deg:.1f}°'
                
                cv2.putText(img, text1, (center_px + 10, center_py - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(img, text2, (center_px + 10, center_py), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(img, text3, (center_px + 10, center_py + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                text = f'BFoV {i+1}'
                cv2.putText(img, text, (center_px + 8, center_py), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 保存单个BFoV图像
            filename = f'bfov_{i+1:02d}_individual_{level_label}.jpg'
            individual_path = os.path.join(level_output_dir, filename)
            cv2.imwrite(individual_path, img)
            individual_paths.append(individual_path)
            
            print(f"       BFoV {i+1:2d} 可视化: {individual_path}")
        
        # 2. 生成全部放在一起的可视化
        print("     生成全部BFoV合并可视化...")
        combined_img = np.ones((current_h, current_w, 3), dtype=np.uint8) * 255
        
        for i in range(num_bfovs):
            # 获取当前掩码（转换为numpy）
            mask = masks[i].cpu().numpy()
            
            # 获取当前BFoV参数
            lon, lat, fov_x, fov_y = bfov_tensor[i].cpu().numpy()
            
            # 计算中心点像素坐标
            center_px = int((lon + np.pi) / (2 * np.pi) * current_w)
            center_py = int((np.pi/2 - lat) / np.pi * current_h)
            
            # 使用循环颜色
            color = colors[i % len(colors)]
            
            # 绘制掩码区域
            mask_indices = np.where(mask == 1)
            combined_img[mask_indices[0], mask_indices[1]] = color
            
            # 绘制BFoV中心点
            cv2.circle(combined_img, (center_px, center_py), 4 if level == 0 else 2, color, -1)
            
            # 绘制边界框
            bbox = bboxes[i].cpu().numpy()
            x_min, y_min, x_max, y_max = bbox.astype(int)
            
            # 检查边界框是否有效（非零）
            if x_max > x_min and y_max > y_min:
                # 绘制边界框
                bbox_color = (0, 0, 0)  # 黑色边界框
                cv2.rectangle(combined_img, (x_min, y_min), (x_max, y_max), bbox_color, 1)
            
            # 添加BFoV编号
            text = f'{i+1}'
            cv2.putText(combined_img, text, (center_px + 5, center_py), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 保存合并图像
        filename = f'all_bfovs_combined_{level_label}.jpg'
        combined_path = os.path.join(level_output_dir, filename)
        cv2.imwrite(combined_path, combined_img)
        print(f"       全部BFoV合并可视化: {combined_path}")


def main():
    """主函数：测试GPU复用BFoV掩码生成和下采样"""
    print("=== GPU复用BFoV掩码生成器 - 下采样版 ===")
    
    # 检查GPU可用性
    if torch.cuda.is_available():
        device = 'cuda'
        print("✓ GPU可用，使用CUDA加速")
    else:
        device = 'cpu'
        print("⚠ GPU不可用，使用CPU模式")
    
    # 参数设置
    # 定义输入图像尺度（原始分辨率）
    original_res = (2048, 1024)  # (width, height)
    erp_w, erp_h = original_res
    
    # 定义指定的BFoV参数（角度制）
    specified_bfovs = [
        # 格式：[经度(°), 纬度(°), 水平视场(°), 垂直视场(°)]
        [0, 60, 10, 60],     # 指定BFoV 1
        [0, 60, 10, 59.9],     # 指定BFoV 2
        [0, 60, 10, 60.1]      # 指定BFoV 3
    ]
    
    num_bfovs = len(specified_bfovs)
    
    print(f"\n输入图像尺度: {erp_w}x{erp_h}")
    print(f"BFoV数量: {num_bfovs}")
    
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
    
    # 2. 生成原始分辨率的BFoV掩码
    print(f"\n2. 生成原始分辨率({erp_w}x{erp_h})的BFoV掩码...")
    start_time = time.time()
    original_masks = generate_bfov_masks_gpu_reuse(bfov_tensor, erp_w, erp_h)
    mask_time = time.time() - start_time
    print(f"✓ 原始掩码生成完成，耗时: {mask_time:.3f}秒")
    
    # 3. 对掩码进行四次下采样（使用指定采样率）
    print(f"\n3. 对掩码进行四次下采样（采样率: 4, 8, 16, 32）...")
    downsample_start_time = time.time()
    sampling_rates = [4, 8, 16, 32]  # 指定的采样率
    masks_list = downsample_masks_with_rates(original_masks, sampling_rates)
    downsample_time = time.time() - downsample_start_time
    print(f"✓ 四次下采样完成，耗时: {downsample_time:.3f}秒")
    
    # 4. 为所有级别生成边界框
    print(f"\n4. 为所有级别生成边界框...")
    bbox_start_time = time.time()
    bboxes_list = masks_to_bboxes(masks_list, device=device)
    bbox_time = time.time() - bbox_start_time
    print(f"✓ 所有级别边界框生成完成，耗时: {bbox_time:.3f}秒")
    
    # 5. 可视化所有结果
    print(f"\n5. 可视化原始掩码和所有下采样掩码及边界框...")
    output_dir = './bfov/result/bfov_downsampling_result/'
    visualize_bfov_masks_with_downsampling(bfov_tensor, masks_list, bboxes_list, original_res, output_dir)
    print(f"✓ 可视化完成")
    
    # 6. 性能统计
    print(f"\n6. 性能统计：")
    total_time = mask_time + downsample_time + bbox_time
    print(f"   原始掩码生成耗时: {mask_time:.3f}秒")
    print(f"   四次下采样耗时: {downsample_time:.3f}秒")
    print(f"   边界框生成耗时: {bbox_time:.3f}秒")
    print(f"   总耗时: {total_time:.3f}秒")
    
    # 输出信息
    print(f"\n🎉 BFoV掩码生成和下采样完成！")
    print(f"   原始分辨率: {erp_w}x{erp_h}")
    print(f"   采样率: {', '.join(map(str, sampling_rates))}")
    print(f"   下采样级别: 4级")
    print(f"   每个BFoV生成 {len(masks_list)} 个不同分辨率的掩码和边界框")
    print(f"   输出目录: {output_dir}")
    print(f"\n📊 下采样特点：")
    print(f"   - 使用指定采样率: {', '.join(map(str, sampling_rates))}")
    print(f"   - 采样值大于0.5的区域在新掩码中置1")
    print(f"   - 始终基于原始掩码进行下采样")
    print(f"   - 生成每个分辨率的边界框")
    print(f"   - 支持批量处理，高效生成")


if __name__ == '__main__':
    main()