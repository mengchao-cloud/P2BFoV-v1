#!/usr/bin/env python3
"""
GPU复用版本的BFOV转bbox脚本
从COCO JSON文件中提取bfov参数，生成掩码，计算最小外接矩形，替换bbox参数
"""

import json
import numpy as np
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PANDORA.PRDA.lib.ReuseGPUImageRecorder import ReuseGPUImageRecorder


def generate_bfov_mask_gpu_reuse(bfov_params, erp_w=1920, erp_h=960):
    """
    基于GPU复用的单个BFOV掩码生成函数
    
    参数:
    bfov_params: BFOV参数 [longitude, latitude, fov_x, fov_y]（弧度制）
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    
    返回:
    mask: [H, W]的numpy数组，H=erp_h, W=erp_w
    """
    lon, lat, fov_x, fov_y = bfov_params
    
    if fov_x > np.pi or fov_y > np.pi:
        return None
    
    fov_x_deg = np.degrees(fov_x)
    fov_y_deg = np.degrees(fov_y)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    gpu_recorder = ReuseGPUImageRecorder(erp_w, erp_h, device=device)
    gpu_recorder.view_angle_w = fov_x_deg
    gpu_recorder.view_angle_h = fov_y_deg
    gpu_recorder.long_side = erp_w
    
    Px, Py = gpu_recorder._sample_points(lon, lat, border_only=False)
    
    Px = Px.to(torch.int32)
    Py = Py.to(torch.int32)
    
    valid_mask = (Px >= 0) & (Px < erp_w) & (Py >= 0) & (Py < erp_h)
    valid_Px = Px[valid_mask].to(torch.long)
    valid_Py = Py[valid_mask].to(torch.long)
    
    if len(valid_Px) == 0:
        return None
    
    mask = torch.zeros((erp_h, erp_w), dtype=torch.uint8, device=device)
    mask[valid_Py, valid_Px] = 1
    
    return mask.cpu().numpy()


def compute_bounding_box(mask):
    """
    计算掩码的外接矩形包围框
    
    参数:
    mask: [H, W]的numpy数组
    
    返回:
    bbox: [x, y, width, height] 包围框参数
          x: 左上角x坐标
          y: 左上角y坐标
          width: 宽度
          height: 高度
    """
    if mask is None or mask.size == 0 or len(mask.shape) == 0:
        return None
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None
    
    y_min = np.where(rows)[0][0]
    y_max = np.where(rows)[0][-1]
    
    x_indices = np.where(cols)[0]
    x_min = x_indices[0]
    x_max = x_indices[-1]
    
    erp_w = mask.shape[1]
    x_diff = x_max - x_min
    
    if x_diff > erp_w * 0.6:
        left_mask = mask.copy()
        left_mask[:, erp_w//2:] = 0
        
        right_mask = mask.copy()
        right_mask[:, :erp_w//2] = 0
        
        left_rows = np.any(left_mask, axis=1)
        left_cols = np.any(left_mask, axis=0)
        
        right_rows = np.any(right_mask, axis=1)
        right_cols = np.any(right_mask, axis=0)
        
        left_area = 0
        right_area = 0
        
        if np.any(left_rows) and np.any(left_cols):
            left_x_min = np.where(left_cols)[0][0]
            left_x_max = np.where(left_cols)[0][-1]
            left_y_min = np.where(left_rows)[0][0]
            left_y_max = np.where(left_rows)[0][-1]
            left_area = (left_x_max - left_x_min + 1) * (left_y_max - left_y_min + 1)
        
        if np.any(right_rows) and np.any(right_cols):
            right_x_min = np.where(right_cols)[0][0]
            right_x_max = np.where(right_cols)[0][-1]
            right_y_min = np.where(right_rows)[0][0]
            right_y_max = np.where(right_rows)[0][-1]
            right_area = (right_x_max - right_x_min + 1) * (right_y_max - right_y_min + 1)
        
        if left_area > right_area:
            x_min = left_x_min
            x_max = left_x_max
        else:
            x_min = right_x_min
            x_max = right_x_max
    
    x = x_min
    y = y_min
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    
    return [x, y, width, height]


def bfov_to_bbox_gpu(bfov_params, erp_w=1920, erp_h=960):
    """
    将BFOV参数转换为bbox参数（GPU复用版本）
    
    参数:
    bfov_params: BFOV参数 [longitude, latitude, fov_x, fov_y]（弧度制）
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    
    返回:
    bbox: [x, y, width, height] 包围框参数
    """
    mask = generate_bfov_mask_gpu_reuse(bfov_params, erp_w, erp_h)
    
    if mask is None:
        return None
    
    bbox = compute_bounding_box(mask)
    
    return bbox


def convert_bfov_to_bbox_in_coco_gpu(input_json_path, output_json_path):
    """
    将COCO格式JSON文件中的bfov参数转换为bbox参数（GPU复用版本）
    
    参数:
    input_json_path: 输入的COCO格式JSON文件路径
    output_json_path: 输出的COCO格式JSON文件路径
    """
    print(f"读取JSON文件: {input_json_path}")
    
    with open(input_json_path, 'r') as f:
        coco_data = json.load(f)
    
    print(f"原始数据包含 {len(coco_data['images'])} 张图像")
    print(f"原始数据包含 {len(coco_data['annotations'])} 个标注")
    
    erp_w = 1920
    erp_h = 960
    
    converted_count = 0
    failed_count = 0
    fov_too_large_count = 0
    mask_empty_count = 0
    
    for ann in coco_data['annotations']:
        if 'bfov' in ann:
            bfov_params = ann['bfov']
            
            lon, lat, fov_x, fov_y = bfov_params
            
            if fov_x > np.pi or fov_y > np.pi:
                failed_count += 1
                fov_too_large_count += 1
                img_info = next((img for img in coco_data['images'] if img['id'] == ann['image_id']), None)
                
                original_bbox = ann['bbox']
                new_bbox = [original_bbox[0], original_bbox[1], 16, 16]
                
                print(f"\n警告: 标注 {ann['id']} 的 bfov 视场角过大")
                print(f"  图像: {img_info['file_name'] if img_info else 'unknown'} (id={ann['image_id']})")
                print(f"  bfov 参数: [lon={lon:.4f}, lat={lat:.4f}, fov_x={np.degrees(fov_x):.1f}°, fov_y={np.degrees(fov_y):.1f}°]")
                print(f"  原始 bbox: {original_bbox}")
                print(f"  新 bbox: {new_bbox} (保留原始位置，宽高改为16x16)")
                
                ann['bbox'] = new_bbox
                ann['area'] = 256
                continue
            
            mask = generate_bfov_mask_gpu_reuse(bfov_params, erp_w, erp_h)
            
            if mask is None:
                failed_count += 1
                mask_empty_count += 1
                img_info = next((img for img in coco_data['images'] if img['id'] == ann['image_id']), None)
                print(f"\n警告: 标注 {ann['id']} 的 bfov 掩码为空（所有采样点超出图像边界）")
                print(f"  图像: {img_info['file_name'] if img_info else 'unknown'} (id={ann['image_id']})")
                print(f"  bfov 参数: [lon={lon:.4f}, lat={lat:.4f}, fov_x={np.degrees(fov_x):.1f}°, fov_y={np.degrees(fov_y):.1f}°]")
                print(f"  使用默认 bbox: [952, 472, 16, 16]")
                ann['bbox'] = [952, 472, 16, 16]
                ann['area'] = 256
                continue
            
            new_bbox = compute_bounding_box(mask)
            
            if new_bbox is not None:
                ann['bbox'] = [int(x) for x in new_bbox]
                width, height = ann['bbox'][2], ann['bbox'][3]
                ann['area'] = int(width * height)
                converted_count += 1
            else:
                failed_count += 1
                img_info = next((img for img in coco_data['images'] if img['id'] == ann['image_id']), None)
                print(f"\n警告: 标注 {ann['id']} 的 bfov 转换失败")
                print(f"  图像: {img_info['file_name'] if img_info else 'unknown'} (id={ann['image_id']})")
                print(f"  bfov 参数: [lon={lon:.4f}, lat={lat:.4f}, fov_x={np.degrees(fov_x):.1f}°, fov_y={np.degrees(fov_y):.1f}°]")
                print(f"  使用默认 bbox: [952, 472, 16, 16]")
                ann['bbox'] = [952, 472, 16, 16]
                ann['area'] = 256
    
    print(f"\n转换完成!")
    print(f"成功转换: {converted_count} 个标注")
    print(f"转换失败: {failed_count} 个标注")
    print(f"  - 视场角过大: {fov_too_large_count} 个")
    print(f"  - 掩码为空: {mask_empty_count} 个")
    
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"输出文件已保存: {output_json_path}")


def main():
    """主函数"""
    # input_json = "360indoor-short/ann/test_coco_short.json"
    # output_json = "360indoor-short/ann/test_coco_short_gpu_bbox.json"
    input_json = "../360indoor-short/ann/train_coco_short.json"
    output_json = "../360indoor-short/ann/train_coco_short_gpu_bbox.json"   
    
    convert_bfov_to_bbox_in_coco_gpu(input_json, output_json)


if __name__ == "__main__":
    main()
