#!/usr/bin/env python3
"""
BFoV掩码转边界框接口

提供从原始掩码生成边界框的核心功能，使用高效的GPU加速实现

功能:
- 从原始掩码直接转换为边界框
- 提供全面的掩码预处理功能：
  - 计算掩码厚度
  - 根据厚度过滤掩码
  - 保留最大连通区域

示例用法:
    from bfov.mask_to_box import MaskToBox
    
    # 初始化接口
    mask_to_box = MaskToBox()
    
    # 从原始掩码生成边界框
    bboxes = mask_to_box(masks)
    
    # 从原始掩码生成边界框并进行全面预处理
    bboxes = mask_to_box(masks, preprocess=True)
"""

import torch
import numpy as np
import cv2


class MaskToBox:
    """
    掩码转边界框接口类
    
    提供从原始掩码生成边界框的核心功能，使用高效的GPU加速实现
    """
    
    def __init__(self):
        """
        初始化接口
        """
        pass
    
    def calculate_column_thickness(self, masks):
        """
        计算掩码的列厚度
        
        参数:
        masks: 3D掩码张量 [num_bfovs, H, W]
        
        返回:
        thickness_dict: 字典，包含每个掩码的列厚度和最大厚度
        """
        device = masks.device
        num_bfovs = masks.shape[0]
        thickness_dict = {}
        
        # 在GPU上批量计算每列的厚度
        for i in range(num_bfovs):
            mask = masks[i]
            
            # 计算每一列的非零元素
            column_thickness = mask.sum(dim=0)  # [W]
            max_thickness = column_thickness.max().item() if column_thickness.numel() > 0 else 0
            
            thickness_dict[i] = {
                'column_thickness': column_thickness.cpu().numpy(),
                'max_thickness': max_thickness
            }
        
        return thickness_dict
    
    def filter_masks_by_thickness(self, masks, thickness_dict, threshold_ratio=0.5):
        """
        根据列厚度过滤掩码
        
        参数:
        masks: 3D掩码张量 [num_bfovs, H, W]
        thickness_dict: 包含列厚度信息的字典
        threshold_ratio: 厚度阈值比例（默认0.5）
        
        返回:
        filtered_masks: 过滤后的掩码张量
        """
        device = masks.device
        num_bfovs = masks.shape[0]
        H, W = masks.shape[1], masks.shape[2]
        filtered_masks = torch.zeros_like(masks)
        
        for i in range(num_bfovs):
            mask = masks[i]
            column_thickness = thickness_dict[i]['column_thickness']
            max_thickness = thickness_dict[i]['max_thickness']
            
            if max_thickness == 0:
                continue
                
            # 计算厚度阈值
            threshold = threshold_ratio * max_thickness
            
            # 创建过滤掩码
            filter_mask = torch.tensor(column_thickness >= threshold, device=device, dtype=torch.uint8)
            
            # 应用过滤
            filtered_mask = mask * filter_mask.view(1, W).expand(H, W)
            filtered_masks[i] = filtered_mask
        
        return filtered_masks
    
    def generate_bounding_box(self, mask):
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
    
    def find_largest_connected_component_box(self, mask):
        """
        查找掩码中最大的连通区域并生成其边界框
        
        参数:
        mask: 2D掩码张量或numpy数组 [H, W]
        
        返回:
        largest_box: 最大连通区域的边界框 [x_min, y_min, x_max, y_max]，如果掩码为空则返回None
        """
        # 确保输入是numpy数组
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy().astype(np.uint8)
        
        if mask.sum() == 0:
            return None
        
        # 找到所有连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        if num_labels <= 1:
            # 如果只有背景或只有一个连通区域
            return self.generate_bounding_box(mask)
        
        # 找到面积最大的连通区域（跳过背景标签0）
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_area_idx = np.argmax(areas) + 1  # 加1是因为stats[0]是背景
        
        # 创建只包含最大连通区域的掩码
        largest_component = (labels == max_area_idx).astype(np.uint8)
        
        # 生成最大连通区域的边界框
        return self.generate_bounding_box(largest_component)
    
    def generate_bounding_boxes(self, masks):
        """
        为掩码区域生成外接矩形框
        
        参数:
        masks: 掩码张量 [num_bfovs, H, W]
        
        返回:
        bounding_boxes: 边界框张量 [num_bfovs, 4]，格式为 [x_min, y_min, x_max, y_max]
        """
        device = masks.device
        num_bfovs = masks.shape[0]
        H, W = masks.shape[1], masks.shape[2]
        
        # 初始化边界框张量 [num_bfovs, 4]
        bounding_boxes = torch.zeros((num_bfovs, 4), dtype=torch.int32, device=device)
        
        # 遍历每个掩码
        for i in range(num_bfovs):
            mask = masks[i]
            
            # 找到所有非零像素的坐标
            coords = torch.nonzero(mask)
            
            if coords.numel() == 0:
                continue
            
            # 计算边界框
            y_min, x_min = coords.min(dim=0)[0]
            y_max, x_max = coords.max(dim=0)[0]
            
            # 存储边界框
            bounding_boxes[i] = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.int32, device=device)
        
        return bounding_boxes
    
    def keep_largest_connected_region(self, masks):
        """
        保留掩码中最大的连通区域
        
        参数:
        masks: 掩码张量 [num_bfovs, H, W]
        
        返回:
        largest_masks: 只包含最大连通区域的掩码张量 [num_bfovs, H, W]
        """
        device = masks.device
        num_bfovs = masks.shape[0]
        largest_masks = torch.zeros_like(masks)
        
        for i in range(num_bfovs):
            # 获取当前掩码
            mask = masks[i].cpu().numpy().astype(np.uint8)
            
            if mask.sum() == 0:
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
            largest_masks[i] = torch.tensor(largest_component, dtype=torch.uint8, device=device)
        
        return largest_masks
    
    def process_masks(self, masks):
        """
        处理掩码并生成最终边界框
        
        参数:
        masks: 3D掩码张量 [num_bfovs, H, W]
        
        返回:
        result_dict: 包含处理结果的字典
        """
        device = masks.device
        num_bfovs = masks.shape[0]
        H, W = masks.shape[1], masks.shape[2]
        
        # 计算列厚度
        thickness_dict = self.calculate_column_thickness(masks)
        
        # 过滤掩码
        filtered_masks = self.filter_masks_by_thickness(masks, thickness_dict, threshold_ratio=0.5)
        
        # 保留最大连通区域
        largest_masks = self.keep_largest_connected_region(filtered_masks)
        
        # 生成最终边界框
        bounding_boxes = self.generate_bounding_boxes(largest_masks)
        
        return {
            'filtered_masks': filtered_masks,
            'largest_masks': largest_masks,
            'bounding_boxes': bounding_boxes,
            'thickness_dict': thickness_dict
        }
    
    def __call__(self, masks, preprocess=False):
        """
        从原始掩码生成边界框的主接口
        
        参数:
        masks: 原始掩码张量 [num_bfovs, H, W]
        preprocess: 是否进行掩码预处理（默认False）
            - True: 进行完整的预处理流程（计算厚度、过滤、保留最大连通区域）
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
            # 预处理流程：厚度计算 -> 过滤 -> 保留最大连通区域
            result = self.process_masks(masks)
            bounding_boxes = result['bounding_boxes']
        else:
            # 不进行预处理，直接使用原始掩码生成边界框
            bounding_boxes = self.generate_bounding_boxes(masks)
        
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
    
    # 方法3: 进行预处理后生成边界框
    print("\n3. 预处理后生成边界框:")
    bboxes3 = converter(masks, preprocess=True)
    for i, bbox in enumerate(bboxes3):
        print(f"   BFoV {i+1}: {bbox.tolist()}")
    
    print("\n接口测试成功！")
