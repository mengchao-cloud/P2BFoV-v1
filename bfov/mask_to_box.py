#!/usr/bin/env python3
"""
BFoV掩码转边界框接口

提供从原始掩码生成边界框的核心功能

功能:
- 从原始掩码直接转换为边界框
- 提供可选的掩码预处理功能

示例用法:
    from bfov.mask_to_box import MaskToBox
    
    # 初始化接口
    mask_to_box = MaskToBox()
    
    # 从原始掩码生成边界框
    bboxes = mask_to_box(masks)
    
    # 从原始掩码生成边界框并进行预处理
    bboxes = mask_to_box(masks, preprocess=True)
"""

import torch
import numpy as np
import cv2


class MaskToBox:
    """
    掩码转边界框接口类
    
    提供从原始掩码生成边界框的核心功能
    """
    
    def __init__(self):
        """
        初始化接口
        """
        pass
    
    def calculate_mask_thickness(self, masks):
        """
        按列计算掩码区域的厚度（可选预处理）
        
        参数:
        masks: 掩码张量 [num_bfovs, H, W]
        
        返回:
        thickness_dict: 字典，包含每个掩码的最大厚度和阈值
        column_thickness_maps: 列厚度图列表（numpy数组）
        """
        device = masks.device
        num_bfovs = masks.shape[0]
        H, W = masks.shape[1], masks.shape[2]
        thickness_dict = {}
        column_thickness_maps = []
        
        # 在GPU上批量计算每列的厚度
        y_indices = torch.arange(H, device=device).view(1, H, 1)
        
        for i in range(num_bfovs):
            mask = masks[i]
            col_nonzero = mask.sum(dim=0)  # [W]
            
            # 计算每一列的最小和最大y坐标
            y_indices_expanded = y_indices.expand(1, H, W)[0]
            min_y = torch.min(torch.where(mask == 1, y_indices_expanded, H), dim=0)[0]  # [W]
            max_y = torch.max(torch.where(mask == 1, y_indices_expanded, -1), dim=0)[0]  # [W]
            
            # 将边界坐标转换为numpy进行平滑处理
            min_y_np = min_y.cpu().numpy().astype(np.int32)
            max_y_np = max_y.cpu().numpy().astype(np.int32)
            
            # 对边界坐标进行平滑处理
            if W > 4:
                neighbor_window = max(3, int(W * 0.05))
                kernel = np.ones(2 * neighbor_window + 1, dtype=np.float32) / (2 * neighbor_window + 1)
                kernel[neighbor_window] = 0  # 跳过当前列本身
                
                # 应用多次平滑
                for _ in range(3):
                    # 平滑min_y（上边界）
                    valid_mask = min_y_np < H
                    valid_min_y = min_y_np.copy()
                    valid_min_y[~valid_mask] = H  # 无效点设为H
                    
                    smoothed = np.convolve(valid_min_y, kernel, mode='same')
                    valid_neighbors = np.convolve(valid_mask.astype(np.float32), kernel, mode='same') * (2 * neighbor_window + 1)
                    
                    height_ratio = 0.03
                    threshold_pixels = int(H * height_ratio)
                    
                    mask_smooth = valid_mask & (valid_neighbors >= neighbor_window * 2 // 3) & (min_y_np > smoothed + threshold_pixels)
                    min_y_np[mask_smooth] = smoothed[mask_smooth].astype(np.int32)
                
                # 平滑max_y（下边界）
                for _ in range(3):
                    valid_mask = max_y_np > 0
                    valid_max_y = max_y_np.copy()
                    valid_max_y[~valid_mask] = 0  # 无效点设为0
                    
                    smoothed = np.convolve(valid_max_y, kernel, mode='same')
                    valid_neighbors = np.convolve(valid_mask.astype(np.float32), kernel, mode='same') * (2 * neighbor_window + 1)
                    
                    height_ratio = 0.015
                    threshold_pixels = int(H * height_ratio)
                    
                    mask_smooth = valid_mask & (valid_neighbors >= neighbor_window * 2 // 3) & (max_y_np < smoothed - threshold_pixels)
                    max_y_np[mask_smooth] = smoothed[mask_smooth].astype(np.int32)
                
                # 将平滑后的边界坐标转换回GPU张量
                min_y = torch.tensor(min_y_np, device=device, dtype=torch.int32)
                max_y = torch.tensor(max_y_np, device=device, dtype=torch.int32)
            
            # 计算每一列的厚度
            column_thickness = torch.where(col_nonzero > 0, max_y - min_y + 1, torch.zeros_like(max_y))
            
            # 转换为numpy用于存储和处理
            column_thickness_np = column_thickness.cpu().numpy().astype(np.int32)
            
            # 添加平滑处理：修正高度突变
            if W > 4:
                neighbor_window = max(3, int(W * 0.05))
                kernel = np.ones(2 * neighbor_window + 1, dtype=np.float32) / (2 * neighbor_window + 1)
                kernel[neighbor_window] = 0  # 跳过当前列本身
                
                for _ in range(3):
                    valid_mask = column_thickness_np > 0
                    valid_thickness = column_thickness_np.copy()
                    valid_thickness[~valid_mask] = 0  # 无效点设为0
                    
                    smoothed = np.convolve(valid_thickness, kernel, mode='same')
                    valid_neighbors = np.convolve(valid_mask.astype(np.float32), kernel, mode='same') * (2 * neighbor_window + 1)
                    
                    mask_valid = valid_mask & (valid_neighbors >= neighbor_window * 2 // 3)
                    avg_neighbor = smoothed[mask_valid] / (valid_neighbors[mask_valid] / (2 * neighbor_window + 1))
                    
                    current = column_thickness_np[mask_valid]
                    smooth_mask = mask_valid.copy()
                    smooth_mask[mask_valid] = (current < avg_neighbor * 0.7) | (current > avg_neighbor * 1.3)
                    
                    column_thickness_np[smooth_mask] = avg_neighbor[smooth_mask[mask_valid]].astype(np.int32)
            
            # 找到最大厚度 - 使用平滑后的厚度
            max_thick = column_thickness_np.max()
            threshold = 0.3 * max_thick
            
            # 存储厚度信息
            thickness_dict[i] = {
                'max_thick': max_thick,
                'threshold': threshold
            }
            
            column_thickness_maps.append(column_thickness_np)
        
        return thickness_dict, column_thickness_maps
    
    def filter_masks_by_thickness(self, masks, column_thickness_maps, thickness_dict):
        """
        根据列厚度过滤掩码（可选预处理）
        
        参数:
        masks: 原始掩码张量
        column_thickness_maps: 列厚度图列表
        thickness_dict: 厚度信息字典
        
        返回:
        filtered_masks: 过滤后的掩码张量
        """
        device = masks.device
        num_bfovs = masks.shape[0]
        H, W = masks.shape[1], masks.shape[2]
        filtered_masks = torch.zeros_like(masks)
        
        for i in range(num_bfovs):
            # 获取当前掩码
            mask = masks[i]
            
            # 获取阈值
            threshold = thickness_dict[i]['threshold']
            
            # 将列厚度图转换为GPU张量
            column_thickness = torch.tensor(column_thickness_maps[i], device=device, dtype=torch.float32)
            
            # 创建列过滤掩码（形状为[W]）
            col_filter = (column_thickness >= threshold).float()
            
            # 扩展列过滤掩码到与mask相同形状[H, W]
            col_filter_expanded = col_filter.view(1, W).expand(H, W)
            
            # 在GPU上进行过滤
            filtered_mask = mask * col_filter_expanded
            
            # 存储过滤后的掩码
            filtered_masks[i] = filtered_mask
        
        return filtered_masks
    
    def fill_mask_connected_regions(self, masks):
        """
        基于列边界分析的掩码填充函数（可选预处理）
        
        参数:
        masks: 掩码张量 [num_bfovs, H, W]
        
        返回:
        filled_masks: 填充后的掩码张量
        """
        device = masks.device
        num_bfovs = masks.shape[0]
        H, W = masks.shape[1], masks.shape[2]
        filled_masks = torch.zeros_like(masks)
        
        for i in range(num_bfovs):
            # 获取当前掩码
            mask = masks[i]
            
            # 创建一个y坐标索引矩阵 [H, W]
            y_indices = torch.arange(H, device=device).view(H, 1).expand(H, W)
            
            # 对每一列找到最上面的非零像素（最小y值）
            min_y_per_col = torch.where(mask == 1, y_indices, H)
            min_y_per_col = torch.min(min_y_per_col, dim=0)[0]  # 每列的最小y值
            
            # 对每一列找到最下面的非零像素（最大y值）
            max_y_per_col = torch.where(mask == 1, y_indices, -1)
            max_y_per_col = torch.max(max_y_per_col, dim=0)[0]  # 每列的最大y值
            
            # 平滑处理边界，避免突变
            if W > 5:
                # 动态窗口大小（5%的图像宽度，至少5）
                window_size = max(5, int(W * 0.05))
                
                # 转换为numpy进行平滑处理
                min_y_np = min_y_per_col.cpu().numpy().astype(np.int32)
                max_y_np = max_y_per_col.cpu().numpy().astype(np.int32)
                
                # 1. 平滑上边界（min_y）
                smoothed_min = min_y_np.copy()
                center_col = W // 2
                
                # 创建距离加权核
                weights = np.array([1.0 / (abs(offset) + 1) for offset in range(-window_size, window_size + 1)], dtype=np.float32)
                weights[window_size] = 0  # 跳过当前列本身
                
                # 创建有效掩码
                edge_mask = (np.arange(W) >= 5) & (np.arange(W) <= W - 6)
                center_mask = abs(np.arange(W) - center_col) > 1
                valid_mask = edge_mask & center_mask & (min_y_np < H)
                
                # 应用加权平滑
                for col in np.where(valid_mask)[0]:
                    # 计算邻居窗口范围
                    start = max(5, col - window_size)
                    end = min(W - 5, col + window_size + 1)
                    
                    # 获取有效邻居
                    neighbor_cols = np.arange(start, end)
                    neighbor_valid = (min_y_np[neighbor_cols] < H)
                    
                    if np.any(neighbor_valid):
                        # 计算偏移量
                        offsets = neighbor_cols - col
                        
                        # 获取权重
                        weight_indices = offsets + window_size
                        col_weights = weights[weight_indices]
                        
                        # 计算加权平均
                        y_values = min_y_np[neighbor_cols][neighbor_valid]
                        col_weights = col_weights[neighbor_valid]
                        
                        total_weight = np.sum(col_weights)
                        if total_weight > 0:
                            weighted_avg = np.sum(y_values * col_weights) / total_weight
                            smoothed_min[col] = int(weighted_avg)
                
                # 2. 平滑下边界（max_y）
                smoothed_max = max_y_np.copy()
                
                # 重用之前创建的距离加权核和掩码
                valid_mask = edge_mask & center_mask & (max_y_np >= 0)
                
                # 应用加权平滑
                for col in np.where(valid_mask)[0]:
                    # 计算邻居窗口范围
                    start = max(5, col - window_size)
                    end = min(W - 5, col + window_size + 1)
                    
                    # 获取有效邻居
                    neighbor_cols = np.arange(start, end)
                    neighbor_valid = (max_y_np[neighbor_cols] >= 0)
                    
                    if np.any(neighbor_valid):
                        # 计算偏移量
                        offsets = neighbor_cols - col
                        
                        # 获取权重
                        weight_indices = offsets + window_size
                        col_weights = weights[weight_indices]
                        
                        # 计算加权平均
                        y_values = max_y_np[neighbor_cols][neighbor_valid]
                        col_weights = col_weights[neighbor_valid]
                        
                        total_weight = np.sum(col_weights)
                        if total_weight > 0:
                            weighted_avg = np.sum(y_values * col_weights) / total_weight
                            smoothed_max[col] = int(weighted_avg)
                
                # 将平滑后的边界转换回张量
                min_y_smooth = torch.tensor(smoothed_min, device=device, dtype=torch.int32)
                max_y_smooth = torch.tensor(smoothed_max, device=device, dtype=torch.int32)
            else:
                # 小图像直接使用原始边界
                min_y_smooth = min_y_per_col
                max_y_smooth = max_y_per_col
            
            # 创建填充掩码
            filled_mask = torch.zeros_like(mask)
            
            # 对每一列填充最上面和最下面非零像素之间的区域
            for x in range(W):
                min_y = min_y_smooth[x]
                max_y = max_y_smooth[x]
                
                # 避免填充图像边缘（距离边缘至少3个像素）
                if x >= 3 and x <= W - 4 and min_y < H - 3 and max_y >= 3 and min_y <= max_y:
                    # 填充min_y到max_y之间的所有行
                    filled_mask[min_y:max_y+1, x] = 1
            
            # 存储填充后的掩码
            filled_masks[i] = filled_mask
        
        return filled_masks
    
    def keep_largest_connected_region(self, masks):
        """
        保留掩码中最大的连通区域（可选预处理）
        
        参数:
        masks: 掩码张量 [num_bfovs, H, W]
        
        返回:
        largest_masks: 只包含最大连通区域的掩码张量
        """
        device = masks.device
        num_bfovs = masks.shape[0]
        largest_masks = torch.zeros_like(masks)
        
        for i in range(num_bfovs):
            # 获取当前掩码
            mask = masks[i].cpu().numpy().astype(np.uint8)
            
            if mask.sum() == 0:
                # 如果掩码为空，直接跳过
                continue
            
            # 找到所有连通区域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            if num_labels <= 1:
                # 如果只有背景或只有一个连通区域，直接使用
                largest_masks[i] = masks[i]
                continue
            
            # 找到面积最大的连通区域（跳过背景标签0）
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_area_idx = np.argmax(areas) + 1  # 加1是因为stats[0]是背景
            
            # 创建只包含最大连通区域的掩码
            largest_component = (labels == max_area_idx).astype(np.uint8)
            
            # 转换回张量
            largest_masks[i] = torch.tensor(largest_component, dtype=torch.uint8, device=device)
        
        return largest_masks
    
    def generate_bounding_boxes(self, masks):
        """
        为掩码区域生成外接矩形框
        
        参数:
        masks: 掩码张量 [num_bfovs, H, W]
        
        返回:
        bounding_boxes: 边界框张量 [num_bfovs, 4]，格式为 [x_min, y_min, x_max, y_max]
        """
        num_bfovs = masks.shape[0]
        H, W = masks.shape[1], masks.shape[2]
        bounding_boxes = torch.zeros((num_bfovs, 4), dtype=torch.int32, device=masks.device)
        
        for i in range(num_bfovs):
            # 获取当前掩码
            mask = masks[i]
            
            # 找到所有非零像素的坐标
            non_zero_indices = torch.nonzero(mask, as_tuple=True)
            
            if len(non_zero_indices[0]) > 0:
                # 计算外接矩形框
                y_min = non_zero_indices[0].min()
                y_max = non_zero_indices[0].max()
                x_min = non_zero_indices[1].min()
                x_max = non_zero_indices[1].max()
                
                bounding_boxes[i] = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.int32, device=masks.device)
            else:
                # 如果没有非零像素，返回空矩形框
                bounding_boxes[i] = torch.tensor([0, 0, 0, 0], dtype=torch.int32, device=masks.device)
        
        return bounding_boxes
    
    def __call__(self, masks, preprocess=False):
        """
        从原始掩码生成边界框的主接口
        
        参数:
        masks: 原始掩码张量 [num_bfovs, H, W]
        preprocess: 是否进行掩码预处理（默认False）
            - True: 进行厚度过滤、连通区域填充和最大连通区域保留
            - False: 直接从原始掩码生成边界框
        
        返回:
        bounding_boxes: 边界框张量 [num_bfovs, 4]，格式为 [x_min, y_min, x_max, y_max]
        """
        # 验证输入
        if not isinstance(masks, torch.Tensor):
            raise TypeError("masks必须是torch.Tensor类型")
            
        if masks.dim() != 3:
            raise ValueError("masks必须是3维张量 [num_bfovs, H, W]")
        
        # 进行预处理（如果需要）
        if preprocess:
            # 1. 计算厚度
            thickness_dict, column_thickness_maps = self.calculate_mask_thickness(masks)
            
            # 2. 过滤掩码
            filtered_masks = self.filter_masks_by_thickness(masks, column_thickness_maps, thickness_dict)
            
            # 3. 填充连通区域
            filled_masks = self.fill_mask_connected_regions(filtered_masks)
            
            # 4. 保留最大连通区域
            processed_masks = self.keep_largest_connected_region(filled_masks)
        else:
            # 不进行预处理，直接使用原始掩码
            processed_masks = masks
        
        # 5. 生成边界框
        bounding_boxes = self.generate_bounding_boxes(processed_masks)
        
        return bounding_boxes


# 便捷函数
def mask_to_box(masks, preprocess=False):
    """
    从原始掩码生成边界框的便捷函数
    
    参数:
    masks: 原始掩码张量 [num_bfovs, H, W]
    preprocess: 是否进行掩码预处理（默认False）
        
    返回:
    bounding_boxes: 边界框张量 [num_bfovs, 4]，格式为 [x_min, y_min, x_max, y_max]
    """
    converter = MaskToBox()
    return converter(masks, preprocess)


if __name__ == "__main__":
    # 示例：从随机掩码生成边界框
    print("测试掩码转边界框接口...")
    
    # 创建随机掩码（示例）
    num_bfovs = 3
    H, W = 960, 1920
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建随机掩码
    masks = torch.zeros((num_bfovs, H, W), dtype=torch.uint8, device=device)
    
    # 为每个掩码添加一个随机矩形区域
    for i in range(num_bfovs):
        x1 = torch.randint(0, W//2, (1,), device=device).item()
        y1 = torch.randint(0, H//2, (1,), device=device).item()
        x2 = torch.randint(x1+1, W, (1,), device=device).item()
        y2 = torch.randint(y1+1, H, (1,), device=device).item()
        masks[i, y1:y2, x1:x2] = 1
    
    # 初始化接口
    converter = MaskToBox()
    
    # 方法1: 使用__call__方法
    print("\n1. 直接从原始掩码生成边界框:")
    bboxes1 = converter(masks)
    print(f"   边界框张量形状: {bboxes1.shape}")
    for i, bbox in enumerate(bboxes1):
        print(f"   BFoV {i+1}: {bbox.tolist()}")
    
    # 方法2: 使用便捷函数
    print("\n2. 使用便捷函数生成边界框:")
    bboxes2 = mask_to_box(masks)
    for i, bbox in enumerate(bboxes2):
        print(f"   BFoV {i+1}: {bbox.tolist()}")
    
    print("\n接口测试成功！")
