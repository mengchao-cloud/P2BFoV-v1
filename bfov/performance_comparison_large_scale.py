#!/usr/bin/env python3
"""
大规模BFOV掩码生成性能对比测试
对比GPU原始版本和GPU复用版本的性能差异
生成大量随机BFOV数据进行严格测试
"""

import torch
import numpy as np
import time
import random
import sys
import os

# 添加路径以导入必要的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from PANDORA.PRDA.lib.GPUImageRecorder import GPUImageRecorder
except ImportError as e:
    print(f"导入GPUImageRecorder失败: {e}")
    sys.exit(1)

try:
    from PANDORA.PRDA.lib.ReuseGPUImageRecorder import ReuseGPUImageRecorder
except ImportError as e:
    print(f"导入ReuseGPUImageRecorder失败: {e}")
    sys.exit(1)


def generate_random_bfov_data(num_gts=10, min_bfovs_per_gt=5, max_bfovs_per_gt=20, 
                              erp_w=1920, erp_h=960):
    """
    生成大量随机BFOV测试数据
    严格限制FOV参数范围：大于0小于π
    
    参数:
    num_gts: GT数量
    min_bfovs_per_gt: 每个GT最少BFOV数量
    max_bfovs_per_gt: 每个GT最多BFOV数量
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    
    返回:
    bfov_list: 随机BFOV参数列表
    """
    bfov_list = []
    
    print(f"生成随机BFOV数据...")
    print(f"GT数量: {num_gts}")
    
    total_bfovs = 0
    
    for gt_idx in range(num_gts):
        # 随机确定当前GT的BFOV数量
        num_bfovs = random.randint(min_bfovs_per_gt, max_bfovs_per_gt)
        total_bfovs += num_bfovs
        
        gt_bfovs = []
        
        for bfov_idx in range(num_bfovs):
            # 生成随机参数，严格限制范围
            # longitude: [-π, π]
            lon = random.uniform(-np.pi, np.pi)
            
            # latitude: [-π/2, π/2]
            lat = random.uniform(-np.pi/2, np.pi/2)
            
            # fov_x, fov_y: (0, π) 严格大于0小于π
            fov_x = random.uniform(0.1, np.pi - 0.1)  # 避免边界值
            fov_y = random.uniform(0.1, np.pi - 0.1)
            
            gt_bfovs.append([lon, lat, fov_x, fov_y])
        
        # 转换为Tensor并添加到列表
        gt_tensor = torch.tensor(gt_bfovs, dtype=torch.float32)
        bfov_list.append(gt_tensor)
    
    print(f"总BFOV数量: {total_bfovs}")
    print(f"参数范围验证:")
    print(f"  longitude: [-π, π] = [{-np.pi:.2f}, {np.pi:.2f}]")
    print(f"  latitude: [-π/2, π/2] = [{-np.pi/2:.2f}, {np.pi/2:.2f}]")
    print(f"  fov_x, fov_y: (0, π) = (0.00, {np.pi:.2f})")
    
    return bfov_list, total_bfovs


def generate_bfov_masks_gpu_original(bfov_list, erp_w=1920, erp_h=960):
    """
    原始GPU版本的BFOV掩码生成函数
    每次BFOV都创建新的GPUImageRecorder实例
    """
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
            gpu_recorder = GPUImageRecorder(erp_w, erp_h, view_angle_w=fov_x_deg, 
                                          view_angle_h=fov_y_deg, long_side=erp_w)
            
            Px, Py = gpu_recorder._sample_points(lon, lat, border_only=False)
            Px = Px.to(torch.int32)
            Py = Py.to(torch.int32)
            
            valid_mask = (Px >= 0) & (Px < erp_w) & (Py >= 0) & (Py < erp_h)
            valid_Px = Px[valid_mask].to(torch.long)
            valid_Py = Py[valid_mask].to(torch.long)
            
            masks[i, valid_Py, valid_Px] = 1
        
        mask_list.append(masks)
    
    return mask_list


def generate_bfov_masks_gpu_reuse(bfov_list, erp_w=1920, erp_h=960):
    """
    GPU复用版本的BFOV掩码生成函数
    使用单个ReuseGPUImageRecorder实例，通过属性更新实现复用
    """
    mask_list = []
    
    for gt_idx, gt_bfovs in enumerate(bfov_list):
        if not gt_bfovs.is_cuda:
            gt_bfovs = gt_bfovs.cuda()
        
        M_i = gt_bfovs.shape[0]
        masks = torch.zeros((M_i, erp_h, erp_w), dtype=torch.uint8, device=gt_bfovs.device)
        
        # 复用版本：创建单个实例用于当前GT
        gpu_recorder = ReuseGPUImageRecorder(erp_w, erp_h, device='cuda')
        
        for i in range(M_i):
            lon, lat, fov_x, fov_y = gt_bfovs[i].tolist()
            fov_x_deg = np.degrees(fov_x)
            fov_y_deg = np.degrees(fov_y)
            
            # 通过属性更新复用ReuseGPUImageRecorder实例
            gpu_recorder.view_angle_w = fov_x_deg
            gpu_recorder.view_angle_h = fov_y_deg
            gpu_recorder.long_side = erp_w
            
            Px, Py = gpu_recorder._sample_points(lon, lat, border_only=False)
            Px = Px.to(torch.int32)
            Py = Py.to(torch.int32)
            
            valid_mask = (Px >= 0) & (Px < erp_w) & (Py >= 0) & (Py < erp_h)
            valid_Px = Px[valid_mask].to(torch.long)
            valid_Py = Py[valid_mask].to(torch.long)
            
            masks[i, valid_Py, valid_Px] = 1
        
        mask_list.append(masks)
    
    return mask_list


def validate_results_consistency(gpu_original_results, gpu_reuse_results):
    """
    验证两个GPU版本的结果一致性
    """
    print("\n=== 结果一致性验证 ===")
    
    if len(gpu_original_results) != len(gpu_reuse_results):
        print(f"❌ 结果长度不一致: 原始版本={len(gpu_original_results)}, 复用版本={len(gpu_reuse_results)}")
        return False
    
    print(f"✓ 结果长度一致: {len(gpu_original_results)}个GT")
    
    total_pixels = 0
    total_diff_pixels = 0
    tolerance = 1e-6
    
    for i, (original_masks, reuse_masks) in enumerate(zip(gpu_original_results, gpu_reuse_results)):
        # 转换为numpy数组进行比较
        original_np = original_masks.cpu().numpy()
        reuse_np = reuse_masks.cpu().numpy()
        
        if original_np.shape != reuse_np.shape:
            print(f"❌ GT {i} 掩码形状不一致: 原始版本={original_np.shape}, 复用版本={reuse_np.shape}")
            return False
        
        # 计算差异
        diff = np.abs(original_np.astype(np.float32) - reuse_np.astype(np.float32))
        diff_pixels = np.sum(diff > tolerance)
        total_pixels += original_np.size
        total_diff_pixels += diff_pixels
        
        if diff_pixels > 0:
            print(f"⚠ GT {i} 有{diff_pixels}个像素差异 (差异率: {diff_pixels/original_np.size*100:.4f}%)")
        else:
            print(f"✓ GT {i} 掩码内容完全一致")
    
    if total_diff_pixels == 0:
        print("🎉 所有结果验证通过！两个GPU版本生成的掩码完全一致。")
        return True
    else:
        print(f"⚠ 总体差异: {total_diff_pixels}/{total_pixels} 像素 (差异率: {total_diff_pixels/total_pixels*100:.4f}%)")
        return total_diff_pixels / total_pixels < 0.001  # 允许0.1%以内的差异


def run_performance_test(bfov_list, total_bfovs, erp_w=1920, erp_h=960, num_runs=3):
    """
    运行性能测试，多次运行取平均值
    """
    print("\n=== 性能测试开始 ===")
    print(f"测试配置:")
    print(f"  ERP尺寸: {erp_w}x{erp_h}")
    print(f"  总BFOV数量: {total_bfovs}")
    print(f"  运行次数: {num_runs}")
    
    # 预热GPU
    print("\n预热GPU...")
    dummy_tensor = torch.randn(1000, 1000, device='cuda')
    torch.cuda.synchronize()
    
    # 测试GPU原始版本
    print("\n1. 测试GPU原始版本...")
    original_times = []
    
    for run in range(num_runs):
        torch.cuda.empty_cache()  # 清理GPU缓存
        torch.cuda.synchronize()
        
        start_time = time.time()
        gpu_original_results = generate_bfov_masks_gpu_original(bfov_list, erp_w, erp_h)
        torch.cuda.synchronize()
        end_time = time.time()
        
        run_time = end_time - start_time
        original_times.append(run_time)
        print(f"   第{run+1}次运行: {run_time:.4f}秒")
    
    original_avg_time = np.mean(original_times)
    original_std_time = np.std(original_times)
    print(f"   平均耗时: {original_avg_time:.4f}秒 (±{original_std_time:.4f}秒)")
    print(f"   平均每个BFOV耗时: {original_avg_time/total_bfovs*1000:.2f}毫秒")
    
    # 测试GPU复用版本
    print("\n2. 测试GPU复用版本...")
    reuse_times = []
    
    for run in range(num_runs):
        torch.cuda.empty_cache()  # 清理GPU缓存
        torch.cuda.synchronize()
        
        start_time = time.time()
        gpu_reuse_results = generate_bfov_masks_gpu_reuse(bfov_list, erp_w, erp_h)
        torch.cuda.synchronize()
        end_time = time.time()
        
        run_time = end_time - start_time
        reuse_times.append(run_time)
        print(f"   第{run+1}次运行: {run_time:.4f}秒")
    
    reuse_avg_time = np.mean(reuse_times)
    reuse_std_time = np.std(reuse_times)
    print(f"   平均耗时: {reuse_avg_time:.4f}秒 (±{reuse_std_time:.4f}秒)")
    print(f"   平均每个BFOV耗时: {reuse_avg_time/total_bfovs*1000:.2f}毫秒")
    
    # 验证结果一致性
    consistency_passed = validate_results_consistency(gpu_original_results, gpu_reuse_results)
    
    # 性能对比
    print("\n=== 性能对比结果 ===")
    speedup = original_avg_time / reuse_avg_time
    print(f"GPU复用版本 vs GPU原始版本: {speedup:.2f}x 加速")
    print(f"性能提升: {(speedup - 1) * 100:.1f}%")
    
    # 内存使用统计
    print("\n=== 内存使用统计 ===")
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.max_memory_reserved() / 1024**3  # GB
        print(f"峰值显存分配: {memory_allocated:.2f} GB")
        print(f"峰值显存保留: {memory_reserved:.2f} GB")
    
    return {
        'original_avg_time': original_avg_time,
        'original_std_time': original_std_time,
        'reuse_avg_time': reuse_avg_time,
        'reuse_std_time': reuse_std_time,
        'speedup': speedup,
        'total_bfovs': total_bfovs,
        'consistency_passed': consistency_passed,
        'memory_allocated': memory_allocated if torch.cuda.is_available() else 0,
        'memory_reserved': memory_reserved if torch.cuda.is_available() else 0
    }


def main():
    """主函数"""
    print("大规模BFOV掩码生成性能对比测试")
    print("=" * 60)
    
    # 检查GPU可用性
    if torch.cuda.is_available():
        print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("❌ GPU不可用，测试无法进行")
        return
    
    # 设置随机种子以确保结果可复现
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 生成大规模随机BFOV数据
    print("\n生成大规模随机BFOV数据...")
    bfov_list, total_bfovs = generate_random_bfov_data(
        num_gts=20,           # 20个GT
        min_bfovs_per_gt=10,  # 每个GT最少10个BFOV
        max_bfovs_per_gt=30,  # 每个GT最多30个BFOV
        erp_w=1920,
        erp_h=960
    )
    
    # 运行性能测试
    results = run_performance_test(bfov_list, total_bfovs, num_runs=5)
    
    # 输出最终结果
    print("\n" + "=" * 60)
    print("📊 最终测试结果摘要")
    print("=" * 60)
    print(f"测试规模: {total_bfovs} 个BFOV")
    print(f"GPU原始版本: {results['original_avg_time']:.4f}秒 (±{results['original_std_time']:.4f})")
    print(f"GPU复用版本: {results['reuse_avg_time']:.4f}秒 (±{results['reuse_std_time']:.4f})")
    print(f"性能加速: {results['speedup']:.2f}x (提升{(results['speedup'] - 1) * 100:.1f}%)")
    print(f"峰值显存使用: {results['memory_allocated']:.2f} GB")
    
    if results['consistency_passed']:
        print("✅ 结果一致性: 通过")
    else:
        print("⚠ 结果一致性: 有微小差异")
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()