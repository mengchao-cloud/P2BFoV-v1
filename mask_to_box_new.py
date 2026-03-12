#!/usr/bin/env python3
"""
BFoV掩码处理与边界框生成工具

功能:
1. 生成随机BFoV参数和对应的掩码
2. 计算掩码区域的列厚度
3. 过滤掉厚度小于0.5倍最大厚度的列
4. 将过滤后的掩码从中间分开
5. 分别生成左右半图的最小外接矩形框
6. 根据左右框是否相邻生成最终边界框
7. 使用与原脚本一致的格式进行可视化
"""

import torch
import numpy as np
import cv2
import os
import time
from PANDORA.PRDA.lib.ReuseGPUImageRecorder import ReuseGPUImageRecorder

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
    longitudes = np.random.uniform(-np.pi, np.pi, num_bfovs)
    latitudes = np.random.uniform(-np.pi/2, np.pi/2, num_bfovs)
    fov_x = np.random.uniform(np.pi/48, np.pi/1.1, num_bfovs)
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
        current_mask = masks[i].cpu().numpy().astype(np.uint8)
        contours, hierarchy = cv2.findContours(current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(current_mask, contours, -1, 1, -1)
        masks[i] = torch.from_numpy(current_mask).to(device=bfov_tensor.device, dtype=torch.uint8)
    
    return masks

def calculate_column_thickness(masks):
    """
    计算掩码的列厚度
    
    参数:
    masks: 3D掩码张量 [num_bfovs, H, W]
    
    返回:
    thickness_dict: 字典，包含每个掩码的列厚度和最大厚度
    """
    device = masks.device
    num_bfovs = masks.shape[0]
    H, W = masks.shape[1], masks.shape[2]
    thickness_dict = {}
    
    # 在GPU上批量计算每列的厚度
    y_indices = torch.arange(H, device=device).view(1, H, 1)
    
    for i in range(num_bfovs):
        mask = masks[i]
        
        # 计算每一列的非零元素
        column_thickness = mask.sum(dim=0)  # [W]
        max_thickness = column_thickness.max().item()
        
        thickness_dict[i] = {
            'column_thickness': column_thickness.cpu().numpy(),
            'max_thickness': max_thickness
        }
    
    return thickness_dict

def filter_masks_by_thickness(masks, thickness_dict, threshold_ratio=0.5):
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
        
        # 计算厚度阈值
        threshold = threshold_ratio * max_thickness
        
        # 创建过滤掩码
        filter_mask = torch.tensor(column_thickness >= threshold, device=device, dtype=torch.uint8)
        
        # 应用过滤
        filtered_mask = mask * filter_mask.view(1, W).expand(H, W)
        filtered_masks[i] = filtered_mask
    
    return filtered_masks

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

def find_largest_connected_component_box(mask):
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
        return generate_bounding_box(mask)
    
    # 找到面积最大的连通区域（跳过背景标签0）
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area_idx = np.argmax(areas) + 1  # 加1是因为stats[0]是背景
    
    # 创建只包含最大连通区域的掩码
    largest_component = (labels == max_area_idx).astype(np.uint8)
    
    # 生成最大连通区域的边界框
    return generate_bounding_box(largest_component)

def process_masks(masks):
    """
    处理掩码并生成最终边界框
    
    参数:
    masks: 3D掩码张量 [num_bfovs, H, W]
    
    返回:
    result_dict: 包含处理结果的字典
    """
    num_bfovs = masks.shape[0]
    
    # 计算列厚度
    thickness_dict = calculate_column_thickness(masks)
    
    # 过滤掩码
    filtered_masks = filter_masks_by_thickness(masks, thickness_dict, threshold_ratio=0.5)
    
    # 处理每个掩码：找到最大连通区域并生成边界框
    final_boxes = []
    
    for i in range(num_bfovs):
        filtered_mask = filtered_masks[i]
        
        # 在过滤后的掩码中找到最大连通区域的边界框
        largest_box = find_largest_connected_component_box(filtered_mask)
        
        final_boxes.append(largest_box if largest_box is not None else [0, 0, 0, 0])
    
    return {
        'filtered_masks': filtered_masks,
        'final_boxes': final_boxes,
        'thickness_dict': thickness_dict
    }

def visualize_bfov_masks(bfov_tensor, masks, output_dir='./bfov/result/simple_bfov', bounding_boxes=None):
    """
    可视化BFoV掩码区域（与原脚本一致的格式）
    
    参数:
    bfov_tensor: BFoV参数张量
    masks: 掩码张量
    output_dir: 输出目录
    bounding_boxes: 可选，外接矩形框列表
    """
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
    
    print("\n生成每个BFoV的单独可视化...")
    
    for i in range(num_bfovs):
        # 创建空白图像
        img = np.ones((erp_h, erp_w, 3), dtype=np.uint8) * 255
        
        # 获取当前掩码
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
        
        # 绘制外接矩形框
        if bounding_boxes is not None:
            bbox = bounding_boxes[i]
            if bbox != [0, 0, 0, 0]:
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 0), 3)
        
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
    
    # 生成全部放在一起的可视化
    print("\n生成全部BFoV合并可视化...")
    combined_img = np.ones((erp_h, erp_w, 3), dtype=np.uint8) * 255
    
    for i in range(num_bfovs):
        mask = masks[i].cpu().numpy()
        lon, lat, fov_x, fov_y = bfov_tensor[i].cpu().numpy()
        
        center_px = int((lon + np.pi) / (2 * np.pi) * erp_w)
        center_py = int((np.pi/2 - lat) / np.pi * erp_h)
        
        color = colors[i % len(colors)]
        
        # 绘制掩码区域
        mask_indices = np.where(mask == 1)
        combined_img[mask_indices[0], mask_indices[1]] = color
        
        # 绘制BFoV中心点
        cv2.circle(combined_img, (center_px, center_py), 6, color, -1)
        
        # 绘制外接矩形框
        if bounding_boxes is not None:
            bbox = bounding_boxes[i]
            if bbox != [0, 0, 0, 0]:
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(combined_img, (x_min, y_min), (x_max, y_max), (0, 0, 0), 2)
        
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
    可视化原始掩码和过滤后的掩码对比
    
    参数:
    bfov_tensor: BFoV参数张量
    original_masks: 原始掩码张量
    filtered_masks: 过滤后的掩码张量
    thickness_dict: 厚度信息字典
    output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_bfovs = original_masks.shape[0]
    erp_w, erp_h = original_masks.shape[2], original_masks.shape[1]
    
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
        # 创建对比图像
        combined_h = erp_h
        combined_w = erp_w * 2 + 20
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
        cv2.putText(combined_img, "Filtered Mask", (erp_w + 50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        
        # 添加BFoV信息
        thickness_info = f"Max Thick: {thickness_dict[i]['max_thickness']:.2f} px"
        cv2.putText(combined_img, thickness_info, (50, erp_h - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # 保存对比图像
        comparison_path = os.path.join(output_dir, f'bfov_{i+1:02d}_comparison.jpg')
        cv2.imwrite(comparison_path, combined_img)
        
        print(f"  BFoV {i+1:2d} 对比可视化: {comparison_path}")
    
    return output_dir

def main():
    """
    主函数：生成BFoV掩码并处理
    """
    print("=== BFoV掩码处理与边界框生成工具 ===")
    
    # 检查GPU可用性
    if torch.cuda.is_available():
        device = 'cuda'
        print("✓ GPU可用，使用CUDA加速")
    else:
        device = 'cpu'
        print("⚠ GPU不可用，使用CPU模式")
    
    # 用户参数配置
    num_bfovs = 10
    erp_w = 1920
    erp_h = 960
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
    
    # 3. 处理掩码并生成边界框
    print(f"\n3. 处理掩码并生成边界框...")
    start_time = time.time()
    result = process_masks(masks)
    process_time = time.time() - start_time
    print(f"✓ 掩码处理完成，耗时: {process_time:.3f}秒")
    
    # 4. 可视化结果
    print(f"\n4. 可视化原始BFoV掩码区域...")
    start_time = time.time()
    output_dir = visualize_bfov_masks(bfov_tensor, masks, output_dir=output_dir)
    visualize_original_time = time.time() - start_time
    print(f"✓ 原始掩码可视化完成，耗时: {visualize_original_time:.3f}秒")
    
    # 5. 可视化过滤后的掩码和边界框
    print(f"\n5. 可视化过滤后的BFoV掩码区域...")
    start_time = time.time()
    filtered_output_dir = os.path.join(output_dir, 'filtered')
    output_dir = visualize_bfov_masks(bfov_tensor, result['filtered_masks'], 
                                      output_dir=filtered_output_dir, 
                                      bounding_boxes=result['final_boxes'])
    visualize_filtered_time = time.time() - start_time
    print(f"✓ 过滤后掩码可视化完成，耗时: {visualize_filtered_time:.3f}秒")
    
    # 6. 可视化过滤前后的掩码对比
    print(f"\n6. 可视化过滤前后的掩码对比...")
    start_time = time.time()
    comparison_output_dir = os.path.join(output_dir, '..', 'comparison')
    output_dir = visualize_mask_comparison(bfov_tensor, masks, result['filtered_masks'], 
                                          result['thickness_dict'], 
                                          output_dir=comparison_output_dir)
    visualize_comparison_time = time.time() - start_time
    print(f"✓ 掩码对比可视化完成，耗时: {visualize_comparison_time:.3f}秒")
    
    # 性能统计
    total_time = mask_time + process_time + visualize_original_time + visualize_filtered_time + visualize_comparison_time
    print(f"\n📊 性能统计：")
    print(f"   掩码生成耗时: {mask_time:.3f}秒")
    print(f"   掩码处理耗时: {process_time:.3f}秒")
    print(f"   原始掩码可视化耗时: {visualize_original_time:.3f}秒")
    print(f"   过滤后掩码可视化耗时: {visualize_filtered_time:.3f}秒")
    print(f"   掩码对比可视化耗时: {visualize_comparison_time:.3f}秒")
    print(f"   总耗时: {total_time:.3f}秒")
    
    # 输出总结
    print(f"\n🎉 BFoV掩码处理与边界框生成完成！")
    print(f"   生成的BFoV数量: {num_bfovs}")
    print(f"   分辨率: {erp_w}x{erp_h}")
    print(f"   厚度过滤阈值: 每个BFoV的0.5倍最大厚度")
    print(f"   输出目录: {output_dir}")


if __name__ == '__main__':
    main()
