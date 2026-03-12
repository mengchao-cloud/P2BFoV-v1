import json
import os
import numpy as np
# 设置文件路径
detection_file = './bfov/temp_json/result_json/head7-1/_512_latest_result.json'
coco_file = './bfov/temp_json/test_coco.json'
output_file = './bfov/temp_json/merge_result/merged_coco_results.json'

def merge_detection_results(detection_file, coco_file, output_file):
    """
    将检测结果与COCO标注文件合并，生成包含图像信息的完整COCO格式文件
    
    Args:
        detection_file: 检测结果文件路径 (JSON数组格式)
        coco_file: COCO标注文件路径 (完整COCO格式)
        output_file: 输出文件路径
    """
    # 读取检测结果
    with open(detection_file, 'r') as f:
        detection_results = json.load(f)
    
    # 读取COCO标注文件
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # 创建新的COCO格式数据结构
    merged_data = {
        'images': coco_data['images'],
        'categories': coco_data['categories'],
        'annotations': []
    }
    
    # 转换检测结果为COCO标注格式
    # 确保每个检测结果有唯一的ID
    max_ann_id = max(ann['id'] for ann in coco_data['annotations']) if 'annotations' in coco_data else 0
    
    for i, detection in enumerate(detection_results):
        # 创建COCO格式的标注
        coco_annotation = {
            'id': max_ann_id + i + 1,  # 确保ID唯一
            'image_id': detection['image_id'],
            'category_id': detection['category_id'],
            'bbox': detection['bbox'],
            'bfov': detection['bbox'],
            'score': detection['score'],  # 保留检测分数
            # 'area': detection['bbox'][2] * detection['bbox'][3],  # 计算面积
            'area': 4 * np.arccos(-np.sin(detection['bbox'][2] / 2) * np.sin(detection['bbox'][3] / 2)) - 2 * np.pi,  # 计算面积
            'iscrowd': 0,  # 检测结果通常不是人群
            'segmentation': [],  # 检测结果通常没有分割信息
            'detection_id': detection['ann_id']  # 保留原始检测ID
        }
        
        merged_data['annotations'].append(coco_annotation)
    
    # 保存合并后的文件
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"合并完成！")
    print(f"- 原始COCO文件: {len(coco_data['images'])}张图像, {len(coco_data['annotations'])}个标注")
    print(f"- 检测结果: {len(detection_results)}个检测框")
    print(f"- 合并后文件: {len(merged_data['images'])}张图像, {len(merged_data['annotations'])}个标注")
    print(f"- 输出文件: {output_file}")

if __name__ == '__main__':
    merge_detection_results(detection_file, coco_file, output_file)
