#!/usr/bin/env python3
"""
简单掩码处理与边界框生成工具

功能:
1. 计算掩码区域的列厚度
2. 过滤掉厚度小于0.5倍最大厚度的列
3. 将过滤后的掩码从中间分开
4. 分别生成左右半图的最小外接矩形框
5. 根据左右框是否相邻生成最终边界框
"""

import torch
import numpy as np

def calculate_column_thickness(mask):
    """
    计算掩码的列厚度
    
    参数:
    mask: 2D掩码张量或numpy数组 [H, W]
    
    返回:
    column_thickness: 每列的厚度 [W]
    max_thickness: 最大厚度值
    """
    # 确保输入是numpy数组
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # 计算每列的厚度（非零像素的数量）
    column_thickness = mask.sum(axis=0)
    max_thickness = column_thickness.max()
    
    return column_thickness, max_thickness

def filter_mask_by_thickness(mask, column_thickness, max_thickness, threshold_ratio=0.5):
    """
    根据列厚度过滤掩码
    
    参数:
    mask: 2D掩码张量或numpy数组 [H, W]
    column_thickness: 每列的厚度 [W]
    max_thickness: 最大厚度值
    threshold_ratio: 厚度阈值比例（默认0.5）
    
    返回:
    filtered_mask: 过滤后的掩码
    """
    # 确保输入是numpy数组
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # 计算厚度阈值
    threshold = threshold_ratio * max_thickness
    
    # 创建过滤掩码
    filter_mask = column_thickness >= threshold
    
    # 应用过滤
    filtered_mask = mask.copy()
    filtered_mask[:, ~filter_mask] = 0
    
    return filtered_mask

def generate_bounding_box(mask):
    """
    生成掩码的最小外接矩形框
    
    参数:
    mask: 2D掩码张量或numpy数组 [H, W]
    
    返回:
    box: [x_min, y_min, x_max, y_max] 格式的边界框，如果掩码为空则返回None
    """
    # 确保输入是numpy数组
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # 找到所有非零像素的坐标
    non_zero_indices = np.where(mask == 1)
    
    if len(non_zero_indices[0]) == 0:
        return None
    
    # 计算边界框
    y_min = non_zero_indices[0].min()
    y_max = non_zero_indices[0].max()
    x_min = non_zero_indices[1].min()
    x_max = non_zero_indices[1].max()
    
    return [x_min, y_min, x_max, y_max]

def are_boxes_adjacent(box1, box2, adjacency_threshold=10):
    """
    检查两个边界框是否相邻
    
    参数:
    box1: 第一个边界框 [x_min, y_min, x_max, y_max]
    box2: 第二个边界框 [x_min, y_min, x_max, y_max]
    adjacency_threshold: 相邻阈值（像素数）
    
    返回:
    is_adjacent: 布尔值，表示两个框是否相邻
    """
    if box1 is None or box2 is None:
        return False
    
    # 检查是否在水平方向相邻
    left_box, right_box = (box1, box2) if box1[0] < box2[0] else (box2, box1)
    
    # 检查水平距离是否小于阈值
    horizontal_distance = right_box[0] - left_box[2]
    if horizontal_distance > adjacency_threshold:
        return False
    
    # 检查垂直方向是否有重叠
    vertical_overlap = not (left_box[3] < right_box[1] - adjacency_threshold or left_box[1] > right_box[3] + adjacency_threshold)
    
    return vertical_overlap

def get_larger_box(box1, box2):
    """
    返回两个边界框中面积较大的那个
    
    参数:
    box1: 第一个边界框 [x_min, y_min, x_max, y_max]
    box2: 第二个边界框 [x_min, y_min, x_max, y_max]
    
    返回:
    larger_box: 面积较大的边界框
    """
    if box1 is None:
        return box2
    if box2 is None:
        return box1
    
    # 计算两个框的面积
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    return box1 if area1 >= area2 else box2

def generate_enclosing_box(box1, box2):
    """
    生成包围两个边界框的最小边界框
    
    参数:
    box1: 第一个边界框 [x_min, y_min, x_max, y_max]
    box2: 第二个边界框 [x_min, y_min, x_max, y_max]
    
    返回:
    enclosing_box: 包围两个框的最小边界框
    """
    if box1 is None:
        return box2
    if box2 is None:
        return box1
    
    x_min = min(box1[0], box2[0])
    y_min = min(box1[1], box2[1])
    x_max = max(box1[2], box2[2])
    y_max = max(box1[3], box2[3])
    
    return [x_min, y_min, x_max, y_max]

def process_mask(mask):
    """
    处理掩码并生成最终边界框
    
    参数:
    mask: 2D掩码张量或numpy数组 [H, W]
    
    返回:
    final_box: 最终边界框 [x_min, y_min, x_max, y_max]
    filtered_mask: 过滤后的掩码
    left_box: 左半图边界框
    right_box: 右半图边界框
    """
    # 1. 计算列厚度
    column_thickness, max_thickness = calculate_column_thickness(mask)
    
    # 2. 过滤掉厚度小于0.5倍最大厚度的列
    filtered_mask = filter_mask_by_thickness(mask, column_thickness, max_thickness, threshold_ratio=0.5)
    
    # 3. 将过滤后的掩码从中间分开
    H, W = filtered_mask.shape
    mid_col = W // 2
    
    left_mask = filtered_mask[:, :mid_col]
    right_mask = filtered_mask[:, mid_col:]
    
    # 4. 分别生成左右半图的最小外接矩形框
    left_box = generate_bounding_box(left_mask)
    right_box = generate_bounding_box(right_mask)
    
    # 调整右半图的边界框坐标（因为右半图的x坐标是从0开始的）
    if right_box is not None:
        right_box = [right_box[0] + mid_col, right_box[1], right_box[2] + mid_col, right_box[3]]
    
    # 5. 根据左右框是否相邻生成最终边界框
    if are_boxes_adjacent(left_box, right_box):
        final_box = generate_enclosing_box(left_box, right_box)
    else:
        final_box = get_larger_box(left_box, right_box)
    
    return final_box, filtered_mask, left_box, right_box


# 使用示例
if __name__ == '__main__':
    # 创建一个测试掩码
    H, W = 100, 200
    test_mask = np.zeros((H, W), dtype=np.uint8)
    
    # 绘制一个沙漏形状的掩码
    for x in range(W):
        # 左侧部分
        if x < W // 2:
            y_min = int(H * 0.3 - (H * 0.1 * x / (W // 2)))
            y_max = int(H * 0.7 + (H * 0.1 * x / (W // 2)))
        # 右侧部分
        else:
            y_min = int(H * 0.2 + (H * 0.1 * (x - W // 2) / (W // 2)))
            y_max = int(H * 0.8 - (H * 0.1 * (x - W // 2) / (W // 2)))
        
        test_mask[y_min:y_max+1, x] = 1
    
    print("原始掩码尺寸:", test_mask.shape)
    print("原始掩码非零像素数:", test_mask.sum())
    
    # 处理掩码
    final_box, filtered_mask, left_box, right_box = process_mask(test_mask)
    
    # 输出结果
    print("\n处理结果:")
    print(f"左半图边界框: {left_box}")
    print(f"右半图边界框: {right_box}")
    print(f"最终边界框: {final_box}")
    
    # 如果有左边界框，计算其面积
    if left_box is not None:
        left_area = (left_box[2] - left_box[0] + 1) * (left_box[3] - left_box[1] + 1)
        print(f"左半图边界框面积: {left_area}")
    
    # 如果有右边界框，计算其面积
    if right_box is not None:
        right_area = (right_box[2] - right_box[0] + 1) * (right_box[3] - right_box[1] + 1)
        print(f"右半图边界框面积: {right_area}")
    
    # 如果有最终边界框，计算其面积
    if final_box is not None:
        final_area = (final_box[2] - final_box[0] + 1) * (final_box[3] - final_box[1] + 1)
        print(f"最终边界框面积: {final_area}")
    
    print("\n处理完成!")
