#!/usr/bin/env python3
"""
测试优化后的BFoV掩码生成和处理代码
"""

import torch
import numpy as np
import time
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入优化后的代码
from bfov_mask_filiter_to_box import (
    generate_random_bfov_params,
    generate_bfov_masks,
    calculate_mask_thickness
)
from mask_to_box import MaskToBox

def test_bfov_pipeline():
    """
    测试完整的BFoV处理流水线
    """
    print("=== 测试BFoV处理流水线 ===")
    
    # 设置参数
    num_bfovs = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    try:
        # 1. 生成随机BFoV参数
        print("\n1. 生成随机BFoV参数...")
        bfov_params = generate_random_bfov_params(num_bfovs=num_bfovs, device=device)
        print(f"   ✓ 成功生成 {num_bfovs} 个BFoV参数")
        
        # 2. 生成BFoV掩码
        print("\n2. 生成BFoV掩码...")
        start_time = time.time()
        masks = generate_bfov_masks(bfov_params)
        elapsed_time = time.time() - start_time
        print(f"   ✓ 成功生成掩码，形状: {masks.shape}")
        print(f"   ✓ 耗时: {elapsed_time:.4f} 秒")
        
        # 3. 计算掩码厚度
        print("\n3. 计算掩码厚度...")
        start_time = time.time()
        thickness_dict, column_thickness_maps = calculate_mask_thickness(masks)
        elapsed_time = time.time() - start_time
        print(f"   ✓ 成功计算厚度")
        print(f"   ✓ 耗时: {elapsed_time:.4f} 秒")
        
        # 4. 根据厚度过滤掩码
        print("\n4. 根据厚度过滤掩码...")
        start_time = time.time()
        filtered_masks = filter_masks_by_thickness(masks, column_thickness_maps, thickness_dict)
        elapsed_time = time.time() - start_time
        print(f"   ✓ 成功过滤掩码，形状: {filtered_masks.shape}")
        print(f"   ✓ 耗时: {elapsed_time:.4f} 秒")
        
        # 5. 填充掩码连通区域
        print("\n5. 填充掩码连通区域...")
        start_time = time.time()
        filled_masks = fill_mask_connected_regions(filtered_masks)
        elapsed_time = time.time() - start_time
        print(f"   ✓ 成功填充掩码，形状: {filled_masks.shape}")
        print(f"   ✓ 耗时: {elapsed_time:.4f} 秒")
        
        # 6. 保留最大连通区域
        print("\n6. 保留最大连通区域...")
        start_time = time.time()
        largest_masks = keep_largest_connected_region(filled_masks)
        elapsed_time = time.time() - start_time
        print(f"   ✓ 成功保留最大连通区域，形状: {largest_masks.shape}")
        print(f"   ✓ 耗时: {elapsed_time:.4f} 秒")
        
        # 7. 生成边界框
        print("\n7. 生成边界框...")
        start_time = time.time()
        bounding_boxes = generate_bounding_boxes(largest_masks)
        elapsed_time = time.time() - start_time
        print(f"   ✓ 成功生成 {len(bounding_boxes)} 个边界框")
        print(f"   ✓ 耗时: {elapsed_time:.4f} 秒")
        
        # 8. 验证结果
        print("\n8. 验证结果...")
        for i in range(num_bfovs):
            mask_sum = masks[i].sum().item()
            filtered_sum = filtered_masks[i].sum().item()
            filled_sum = filled_masks[i].sum().item()
            largest_sum = largest_masks[i].sum().item()
            
            print(f"   BFoV {i+1}: 原始={mask_sum} → 过滤={filtered_sum} → 填充={filled_sum} → 最大连通={largest_sum}")
            
            # 确保处理过程合理
            assert mask_sum >= 0
            assert filtered_sum <= mask_sum
            assert filled_sum >= filtered_sum
            assert largest_sum <= filled_sum
        
        print("\n=== 测试完成，所有功能正常！ ===")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_bfov_pipeline()
