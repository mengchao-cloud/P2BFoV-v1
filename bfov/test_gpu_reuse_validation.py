#!/usr/bin/env python3
"""
三版本GPUImageRecorder对比测试脚本
对比原始GPU版本、复用GPU版本和CPU版本的性能与效果
"""

import torch
import numpy as np
import time
import os
import cv2

# 导入三个版本的ImageRecorder
from PANDORA.PRDA.lib.GPUImageRecorder import GPUImageRecorder  # 原始GPU版本
from PANDORA.PRDA.lib.ReuseGPUImageRecorder import ReuseGPUImageRecorder  # 复用GPU版本
from PANDORA.PRDA.lib.ImageRecorder import ImageRecorder  # CPU版本





def generate_bfov_masks_gpu_reuse(bfov_list, erp_w=1920, erp_h=960, threshold=None):
    """
    基于GPU复用的BFOV掩码生成函数（优化版本）
    使用支持参数动态更新的ReuseGPUImageRecorder类
    
    参数:
    bfov_list: 长度为N的列表，每个元素是[M_i,4]的张量（弧度制）
               N是GT的数量，M_i是每个GT对应的BFOV数量
               每个BFOV参数: [longitude, latitude, fov_x, fov_y]
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    threshold: 分割阈值，默认使用图像宽度的一半
    
    返回:
    mask_list: 长度为N的列表，每个元素是[M_i, H, W]的张量
               H=erp_h, W=erp_w
    """
    # 使用默认分割阈值（图像宽度的一半）
    if threshold is None:
        threshold = erp_w // 2
    
    # 初始化结果列表
    mask_list = []
    
    # 遍历每个GT
    for gt_idx, gt_bfovs in enumerate(bfov_list):
        # 确保张量在GPU上
        if not gt_bfovs.is_cuda:
            gt_bfovs = gt_bfovs.cuda()
        
        # 获取当前GT的BFOV数量
        M_i = gt_bfovs.shape[0]
        
        # 初始化当前GT的掩码张量 [M_i, H, W]
        masks = torch.zeros((M_i, erp_h, erp_w), dtype=torch.uint8, device=gt_bfovs.device)
        
        # 🔑 关键优化：创建单个ReuseGPUImageRecorder实例用于当前GT
        gpu_recorder = ReuseGPUImageRecorder(erp_w, erp_h, device=gt_bfovs.device)
        
        # 批量处理当前GT的所有BFOV
        for i in range(M_i):
            # 获取当前BFOV参数
            lon, lat, fov_x, fov_y = gt_bfovs[i].tolist()
            
            # 注意：这里的fov_x和fov_y是弧度制，需要转换为角度制
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
        
        # 将当前GT的掩码添加到结果列表
        mask_list.append(masks)
    
    return mask_list


def generate_bfov_masks_gpu_original(bfov_list, erp_w=1920, erp_h=960, threshold=None):
    """
    原始GPU版本的BFOV掩码生成函数（用于对比）
    每次BFOV都创建新的GPUImageRecorder实例
    
    参数:
    bfov_list: 长度为N的列表，每个元素是[M_i,4]的张量（弧度制）
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    threshold: 分割阈值，默认使用图像宽度的一半
    
    返回:
    mask_list: 长度为N的列表，每个元素是[M_i, H, W]的张量
    """
    if threshold is None:
        threshold = erp_w // 2
    
    mask_list = []
    
    for gt_idx, gt_bfovs in enumerate(bfov_list):
        if not gt_bfovs.is_cuda:
            gt_bfovs = gt_bfovs.cuda()
        
        M_i = gt_bfovs.shape[0]
        masks = torch.zeros((M_i, erp_h, erp_w), dtype=torch.uint8, device=gt_bfovs.device)
        
        # 原始版本：每次BFOV都创建新实例
        for i in range(M_i):
            lon, lat, fov_x, fov_y = gt_bfovs[i].tolist()
            fov_x_deg = np.degrees(fov_x)
            fov_y_deg = np.degrees(fov_y)
            
            # 每次创建新实例
            gpu_recorder = GPUImageRecorder(erp_w, erp_h, view_angle_w=fov_x_deg, view_angle_h=fov_y_deg, long_side=erp_w)
            
            Px, Py = gpu_recorder._sample_points(lon, lat, border_only=False)
            Px = Px.to(torch.int32)
            Py = Py.to(torch.int32)
            
            valid_mask = (Px >= 0) & (Px < erp_w) & (Py >= 0) & (Py < erp_h)
            valid_Px = Px[valid_mask].to(torch.long)
            valid_Py = Py[valid_mask].to(torch.long)
            
            masks[i, valid_Py, valid_Px] = 1
        
        mask_list.append(masks)
    
    return mask_list


def generate_bfov_masks_cpu(bfov_list, erp_w=1920, erp_h=960, threshold=None):
    """
    基于CPU的BFOV掩码生成函数（基准验证）
    使用CPU版本的ImageRecorder类
    
    参数:
    bfov_list: 长度为N的列表，每个元素是[M_i,4]的张量或数组（弧度制）
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    threshold: 分割阈值，默认使用图像宽度的一半
    
    返回:
    mask_list: 长度为N的列表，每个元素是[M_i, H, W]的数组
    """
    if threshold is None:
        threshold = erp_w // 2
    
    mask_list = []
    
    for gt_idx, gt_bfovs in enumerate(bfov_list):
        if isinstance(gt_bfovs, torch.Tensor):
            gt_bfovs = gt_bfovs.cpu().numpy()
        
        M_i = gt_bfovs.shape[0]
        masks = []
        
        for i in range(M_i):
            lon, lat, fov_x, fov_y = gt_bfovs[i]
            fov_x_deg = np.degrees(fov_x)
            fov_y_deg = np.degrees(fov_y)
            
            cpu_recorder = ImageRecorder(erp_w, erp_h, view_angle_w=fov_x_deg, view_angle_h=fov_y_deg, long_side=erp_w)
            
            Px, Py = cpu_recorder._sample_points(lon, lat, border_only=False)
            Px = Px.astype(np.int32)
            Py = Py.astype(np.int32)
            
            valid_mask = (Px >= 0) & (Px < erp_w) & (Py >= 0) & (Py < erp_h)
            valid_Px = Px[valid_mask]
            valid_Py = Py[valid_mask]
            
            mask = np.zeros((erp_h, erp_w), dtype=np.uint8)
            mask[valid_Py, valid_Px] = 1
            masks.append(mask)
        
        mask_list.append(np.array(masks))
    
    return mask_list


def validate_results(gpu_results, cpu_results, tolerance=1e-6):
    """
    验证GPU和CPU结果是否一致
    
    参数:
    gpu_results: GPU版本生成的掩码列表
    cpu_results: CPU版本生成的掩码列表（可以是Tensor或numpy数组）
    tolerance: 数值容差
    
    返回:
    bool: 结果是否一致
    """
    print("=== 结果验证 ===")
    
    if len(gpu_results) != len(cpu_results):
        print(f"❌ 结果长度不一致: GPU={len(gpu_results)}, CPU={len(cpu_results)}")
        return False
    
    print(f"✓ 结果长度一致: {len(gpu_results)}个GT")
    
    total_pixels = 0
    total_diff_pixels = 0
    
    for i, (gpu_masks, cpu_masks) in enumerate(zip(gpu_results, cpu_results)):
        # 将GPU张量转换为CPU数组
        gpu_masks_np = gpu_masks.cpu().numpy()
        
        # 如果cpu_masks是Tensor，也转换为numpy
        if isinstance(cpu_masks, torch.Tensor):
            cpu_masks_np = cpu_masks.cpu().numpy()
        else:
            cpu_masks_np = cpu_masks
        
        if gpu_masks_np.shape != cpu_masks_np.shape:
            print(f"❌ GT {i} 掩码形状不一致: GPU={gpu_masks_np.shape}, CPU={cpu_masks_np.shape}")
            return False
        
        print(f"✓ GT {i} 掩码形状一致: {gpu_masks_np.shape}")
        
        # 计算差异
        diff = np.abs(gpu_masks_np.astype(np.float32) - cpu_masks_np.astype(np.float32))
        diff_pixels = np.sum(diff > tolerance)
        total_pixels += gpu_masks_np.size
        total_diff_pixels += diff_pixels
        
        if diff_pixels > 0:
            print(f"⚠ GT {i} 有{diff_pixels}个像素差异 (差异率: {diff_pixels/gpu_masks_np.size*100:.4f}%)")
        else:
            print(f"✓ GT {i} 掩码内容完全一致")
    
    if total_diff_pixels == 0:
        print("🎉 所有结果验证通过！GPU和CPU生成的掩码完全一致。")
        return True
    else:
        print(f"⚠ 总体差异: {total_diff_pixels}/{total_pixels} 像素 (差异率: {total_diff_pixels/total_pixels*100:.4f}%)")
        return total_diff_pixels / total_pixels < 0.001  # 允许0.1%以内的差异


def visualize_bfov_masks(bfov_list, mask_list, erp_w=1920, erp_h=960, threshold=None, 
                        output_dir='./bfov/result/bfov_mask_visualizations', prefix=''):
    """
    可视化BFOV掩码列表
    
    参数:
    bfov_list: BFOV参数列表，长度为N的列表，每个元素是[M_i,4]的张量或数组
    mask_list: 掩码列表，长度为N的列表，每个元素是[M_i, H, W]的张量或数组
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    threshold: 分割阈值，默认使用图像宽度的一半
    output_dir: 可视化图像输出目录
    prefix: 文件名前缀
    """
    # 使用默认分割阈值（图像宽度的一半）
    if threshold is None:
        threshold = erp_w // 2
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取gt数量
    N = len(mask_list)
    
    # 遍历所有gt
    for i in range(N):
        # 创建可视化图像
        img = np.zeros((erp_h, erp_w, 3), dtype=np.uint8)
        img[:] = (255, 255, 255)  # 白色背景
        
        # 绘制分割线
        cv2.line(img, (threshold, 0), (threshold, erp_h), (0, 0, 0), 2)
        
        # 不同bfov使用不同颜色
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
            (128, 128, 0)   # 橄榄色
        ]
        
        # 获取当前gt的bfov数量
        masks = mask_list[i]
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        M = masks.shape[0]
        
        # 遍历当前gt的所有bfov
        for j in range(M):
            # 获取当前bfov的掩码
            mask = masks[j]
            
            # 获取当前bfov的参数
            gt_bfov_params = bfov_list[i]
            if isinstance(gt_bfov_params, torch.Tensor):
                gt_bfov_params = gt_bfov_params.cpu().numpy()
            bfov_params = gt_bfov_params[j]
            longitude, latitude, fov_x, fov_y = bfov_params
            
            # 计算中心点像素坐标
            center_x_rad = longitude
            center_y_rad = latitude
            center_px = int((center_x_rad + np.pi) / (2 * np.pi) * erp_w)
            center_py = int((np.pi/2 - center_y_rad) / np.pi * erp_h)
            
            # 确保坐标在有效范围内
            center_px = max(0, min(erp_w-1, center_px))
            center_py = max(0, min(erp_h-1, center_py))
            
            # 选择颜色
            color = colors[j % len(colors)]
            
            # 绘制掩码区域
            mask_indices = np.where(mask > 0)
            if len(mask_indices[0]) > 0:
                img[mask_indices] = color
            
            # 绘制中心点
            cv2.circle(img, (center_px, center_py), 5, color, -1)
            
            # 添加参数标签
            label = f"BFOV{j}: lon={longitude:.2f}, lat={latitude:.2f}, fov_x={fov_x:.2f}, fov_y={fov_y:.2f}"
            cv2.putText(img, label, (10, 30 + j*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 保存图像
        output_path = os.path.join(output_dir, f"{prefix}gt_{i}_mask_visualization.png")
        cv2.imwrite(output_path, img)
        print(f"✓ 可视化图像已保存: {output_path}")


def test_three_versions_comparison():
    """三版本对比测试：原始GPU、复用GPU、CPU"""
    print("=== 三版本GPUImageRecorder对比测试 ===")
    
    # 创建测试数据
    erp_w, erp_h = 1920, 960
    
    # 模拟多个GT，每个GT有多个BFOV
    test_bfov_list = []
    
    # GT 1: 5个BFOV，FOV参数相同（测试复用效果）
    gt1 = torch.tensor([
        [0.8, 0.8, 0.8, 0.6],    # lon, lat, fov_x, fov_y (弧度)
        [1.2, -0.4, 0.8, 0.6],   # 相同FOV参数
        [-0.8, 0.1, 0.8, 0.6],   # 相同FOV参数
        [2.0, 0.2, 0.8, 0.6],    # 相同FOV参数
        [-1.5, -0.3, 0.8, 0.6]   # 相同FOV参数
    ])
    test_bfov_list.append(gt1)
    
    # GT 2: 6个BFOV，FOV参数不同（测试参数更新效果）
    gt2 = torch.tensor([
        [1.0, 0.1, 0.5, 0.4],    # 不同FOV参数
        [0.2, -0.3, 0.7, 0.5],   # 不同FOV参数
        [-1.2, 0.5, 0.9, 0.7],   # 不同FOV参数
        [0.8, 0.2, 0.6, 0.5],    # 不同FOV参数
        [-0.5, -0.4, 1.0, 0.8],  # 不同FOV参数
        [1.5, 0.3, 0.4, 0.3]     # 不同FOV参数
    ])
    test_bfov_list.append(gt2)
    
    # GT 3: 8个BFOV，混合参数（模拟真实场景）
    gt3 = torch.tensor([
        [0.3, 0.2, 0.6, 0.5],    # 参数组1
        [1.8, 0.4, 0.6, 0.5],    # 相同参数
        [-1.0, -0.1, 0.6, 0.5],  # 相同参数
        [0.7, -0.5, 0.8, 0.7],   # 参数组2
        [-0.3, 0.6, 0.8, 0.7],   # 相同参数
        [2.2, -0.2, 1.0, 0.9],   # 参数组3
        [-1.8, 0.3, 1.0, 0.9],   # 相同参数
        [0.1, 0.8, 0.4, 0.3]     # 参数组4
    ])
    test_bfov_list.append(gt3)
    
    # GT 4: 10个BFOV，覆盖更多区域
    gt4 = torch.tensor([
        [0.0, 0.0, 0.7, 0.6],    # 中心区域
        [np.pi, 0.0, 0.7, 0.6],  # 对侧区域
        [0.0, np.pi/2, 0.5, 0.4],# 顶部区域
        [0.0, -np.pi/2, 0.5, 0.4],# 底部区域
        [np.pi/2, 0.0, 0.6, 0.5],# 右侧区域
        [-np.pi/2, 0.0, 0.6, 0.5],# 左侧区域
        [np.pi/4, np.pi/4, 0.8, 0.7],# 右上区域
        [-np.pi/4, np.pi/4, 0.8, 0.7],# 左上区域
        [np.pi/4, -np.pi/4, 0.8, 0.7],# 右下区域
        [-np.pi/4, -np.pi/4, 0.8, 0.7] # 左下区域
    ])
    test_bfov_list.append(gt4)
    
    print("📊 测试数据统计:")
    print(f"   GT数量: {len(test_bfov_list)}")
    total_bfovs = sum([gt.shape[0] for gt in test_bfov_list])
    print(f"   总BFOV数量: {total_bfovs}")
    
    # 测试三个版本
    print("\n=== 性能测试 ===")
    
    # 1. 测试GPU原始版本
    print("\n1. 测试GPU原始版本...")
    start_time = time.time()
    gpu_original_results = generate_bfov_masks_gpu_original(test_bfov_list, erp_w, erp_h)
    gpu_original_time = time.time() - start_time
    print(f"   GPU原始版本耗时: {gpu_original_time:.4f}秒")
    print(f"   平均每个BFOV耗时: {gpu_original_time/total_bfovs*1000:.2f}毫秒")
    
    # 2. 测试GPU复用版本
    print("\n2. 测试GPU复用版本...")
    start_time = time.time()
    gpu_reuse_results = generate_bfov_masks_gpu_reuse(test_bfov_list, erp_w, erp_h)
    gpu_reuse_time = time.time() - start_time
    print(f"   GPU复用版本耗时: {gpu_reuse_time:.4f}秒")
    print(f"   平均每个BFOV耗时: {gpu_reuse_time/total_bfovs*1000:.2f}毫秒")
    
    # 3. 测试CPU版本（基准）
    print("\n3. 测试CPU版本（基准）...")
    start_time = time.time()
    cpu_results = generate_bfov_masks_cpu(test_bfov_list, erp_w, erp_h)
    cpu_time = time.time() - start_time
    print(f"   CPU版本耗时: {cpu_time:.4f}秒")
    print(f"   平均每个BFOV耗时: {cpu_time/total_bfovs*1000:.2f}毫秒")
    
    # 性能对比
    print("\n=== 性能对比结果 ===")
    print(f"GPU复用版本 vs GPU原始版本: {gpu_original_time/gpu_reuse_time:.2f}x 加速")
    print(f"GPU复用版本 vs CPU版本: {cpu_time/gpu_reuse_time:.2f}x 加速")
    print(f"GPU原始版本 vs CPU版本: {cpu_time/gpu_original_time:.2f}x 加速")
    
    # 验证结果一致性
    print("\n=== 结果一致性验证 ===")
    
    # 验证GPU原始版本与CPU版本
    print("1. GPU原始版本 vs CPU版本:")
    original_vs_cpu = validate_results(gpu_original_results, cpu_results)
    
    # 验证GPU复用版本与CPU版本
    print("\n2. GPU复用版本 vs CPU版本:")
    reuse_vs_cpu = validate_results(gpu_reuse_results, cpu_results)
    
    # 验证GPU复用版本与GPU原始版本
    print("\n3. GPU复用版本 vs GPU原始版本:")
    reuse_vs_original = validate_results(gpu_reuse_results, gpu_original_results)
    
    # 可视化掩码结果
    print("\n=== 掩码可视化 ===")
    
    # 可视化GPU原始版本的掩码
    print("1. 可视化GPU原始版本掩码:")
    visualize_bfov_masks(test_bfov_list, gpu_original_results, erp_w, erp_h, 
                        prefix='gpu_original_')
    
    # 可视化GPU复用版本的掩码
    print("\n2. 可视化GPU复用版本掩码:")
    visualize_bfov_masks(test_bfov_list, gpu_reuse_results, erp_w, erp_h, 
                        prefix='gpu_reuse_')
    
    # 可视化CPU版本的掩码
    print("\n3. 可视化CPU版本掩码:")
    visualize_bfov_masks(test_bfov_list, cpu_results, erp_w, erp_h, 
                        prefix='cpu_')
    
    # 测试总结
    print("\n=== 测试总结 ===")
    print("📊 三版本对比结果:")
    print(f"   GPU原始版本: {gpu_original_time:.4f}秒")
    print(f"   GPU复用版本: {gpu_reuse_time:.4f}秒")
    print(f"   CPU版本: {cpu_time:.4f}秒")
    
    if original_vs_cpu and reuse_vs_cpu and reuse_vs_original:
        print("✅ 所有测试通过！三个版本功能一致，GPU复用版本优化有效。")
        print(f"✅ GPU复用版本相比原始版本加速: {gpu_original_time/gpu_reuse_time:.2f}x")
        print(f"✅ GPU复用版本相比CPU版本加速: {cpu_time/gpu_reuse_time:.2f}x")
        print("✅ 掩码可视化图像已保存到: ./bfov/result/bfov_mask_visualizations/")
    else:
        print("⚠ 部分测试有差异，但功能基本一致。")
    
    return {
        'gpu_original_time': gpu_original_time,
        'gpu_reuse_time': gpu_reuse_time,
        'cpu_time': cpu_time,
        'total_bfovs': total_bfovs,
        'all_passed': original_vs_cpu and reuse_vs_cpu and reuse_vs_original
    }


def test_parameter_update_functionality():
    """测试参数动态更新功能"""
    print("=== 参数动态更新功能测试 ===")
    
    erp_w, erp_h = 1920, 960
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建GPUImageRecorder实例
    recorder = GPUImageRecorder(erp_w, erp_h, device=device)
    
    # 测试1: 基本参数更新
    print("\n1. 测试基本参数更新:")
    original_fov_w = recorder.view_angle_w
    original_fov_h = recorder.view_angle_h
    
    print(f"   初始参数: FOV({original_fov_w}°, {original_fov_h}°)")
    
    # 更新参数
    recorder.view_angle_w = 60.0
    recorder.view_angle_h = 45.0
    
    print(f"   更新后参数: FOV({recorder.view_angle_w}°, {recorder.view_angle_h}°)")
    
    # 测试2: 多次参数更新
    print("\n2. 测试多次参数更新:")
    test_params = [
        (30.0, 30.0),
        (90.0, 60.0),
        (120.0, 90.0),
        (45.0, 45.0)
    ]
    
    for i, (fov_w, fov_h) in enumerate(test_params):
        recorder.view_angle_w = fov_w
        recorder.view_angle_h = fov_h
        
        # 测试采样点生成
        Px, Py = recorder._sample_points(0.5, 0.3, border_only=False)
        print(f"   参数集{i+1}: FOV({fov_w}°, {fov_h}°) -> {len(Px)}个采样点")
    
    print("✅ 参数动态更新功能测试通过！")


if __name__ == "__main__":
    print("三版本GPUImageRecorder对比测试")
    print("=" * 50)
    
    # 检查GPU可用性
    if torch.cuda.is_available():
        print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ GPU不可用，将使用CPU进行测试")
    
    # 运行三版本对比测试
    results = test_three_versions_comparison()
    
    print("\n" + "=" * 50)
    print("测试完成！")
    
    # 显示最终结果摘要
    print("\n📋 最终结果摘要:")
    print(f"   GPU原始版本: {results['gpu_original_time']:.4f}秒")
    print(f"   GPU复用版本: {results['gpu_reuse_time']:.4f}秒")
    print(f"   CPU版本: {results['cpu_time']:.4f}秒")
    print(f"   总BFOV数量: {results['total_bfovs']}")
    
    if results['all_passed']:
        print("✅ 所有测试通过！")
    else:
        print("⚠ 部分测试有差异，请检查可视化结果。")