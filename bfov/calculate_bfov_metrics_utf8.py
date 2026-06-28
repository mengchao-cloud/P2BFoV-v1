import json
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

# 添加bfov/lib目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

from sphdet.iou.sph_iou_api import sph2pob_efficient_iou

def load_json_file(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_bfov_by_image(json_data):
    """按image_id提取BFoV信息"""
    bfov_by_image = {}
    if 'annotations' in json_data:
        total_annotations = len(json_data['annotations'])
        print(f"Total annotations in file: {total_annotations}")
        for ann in tqdm(json_data['annotations'], desc="Extracting annotations"):
            image_id = ann['image_id']
            if 'bfov' in ann:
                if image_id not in bfov_by_image:
                    bfov_by_image[image_id] = []
                bfov_by_image[image_id].append({
                    'id': ann['id'],
                    'bfov': ann['bfov'],
                    'score': ann.get('score', 1.0),  # 对于gt，score默认为1.0
                    'category_id': ann.get('category_id', None)
                })
    return bfov_by_image

def calculate_sphere_iou(gt_bfov, pred_bfov):
    """计算单个BFoV对的sphere_iou，使用sph2pob_efficient_iou方法"""
    # 转换为numpy数组
    gt_array = np.array([gt_bfov])
    pred_array = np.array([pred_bfov])
    
    # 转换为torch张量
    gt_tensor = torch.tensor(gt_array, dtype=torch.float32)
    pred_tensor = torch.tensor(pred_array, dtype=torch.float32)
    
    # 将弧度转换为度
    gt_deg = torch.rad2deg(gt_tensor)
    pred_deg = torch.rad2deg(pred_tensor)
    
    # 使用sph2pob_efficient_iou计算IoU（对齐计算）
    try:
        iou_tensor = sph2pob_efficient_iou(pred_deg, gt_deg, is_aligned=True)
        sphere_iou = iou_tensor.cpu().numpy()[0]
        return float(sphere_iou)
    except Exception as e:
        print(f"Error calculating sphere_iou: {e}")
        return 0.0

def get_latitude_band(lat_rad):
    """根据纬度（弧度）返回纬度带"""
    lat_deg = np.degrees(lat_rad)
    abs_lat = abs(lat_deg)
    
    if abs_lat <= 30:
        return "低纬度"
    elif abs_lat <= 60:
        return "中纬度"
    else:
        return "高纬度"

def lon_lat_to_3d(lon, lat):
    """将经纬度（弧度）转换为三维坐标"""
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.array([x, y, z])

def calculate_angle(vec1, vec2):
    """计算两个向量的夹角（弧度）"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cos_theta = dot_product / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1, 1)  # 确保在有效范围内
    return np.arccos(cos_theta)

def calculate_offset_metric(gt_bfov, pred_bfov):
    """计算偏移度指标"""
    # 提取经纬度
    gt_lon, gt_lat = gt_bfov[0], gt_bfov[1]
    pred_lon, pred_lat = pred_bfov[0], pred_bfov[1]
    
    # 转换为三维坐标
    gt_vec = lon_lat_to_3d(gt_lon, gt_lat)
    pred_vec = lon_lat_to_3d(pred_lon, pred_lat)
    
    # 计算夹角
    angle = calculate_angle(gt_vec, pred_vec)
    
    # 计算市场尺度
    fov_x, fov_y = gt_bfov[2], gt_bfov[3]
    market_scale = np.sqrt(fov_x * fov_y)
    
    # 计算偏移度指标
    if market_scale == 0:
        return 0.0
    offset_metric = angle / market_scale
    return float(offset_metric)

def main():
    # 文件路径
    gt_file = '/home/mengchao/workspace/P2BFoV/P2BNet-main/TOV_mmdetection/bfov/temp_json/test_coco.json'
    pred_file = '/home/mengchao/workspace/P2BFoV/P2BNet-main/TOV_mmdetection/bfov/temp_json/merge_result/xiaorong/merged_qie_infer_coco_results.json'
    
    # 加载文件
    print("Loading JSON files...")
    gt_data = load_json_file(gt_file)
    pred_data = load_json_file(pred_file)
    
    # 提取BFoV信息（按image_id）
    print("\nExtracting BFoV information from GT file...")
    gt_bfov_by_image = extract_bfov_by_image(gt_data)
    
    print("\nExtracting BFoV information from pred file...")
    pred_bfov_by_image = extract_bfov_by_image(pred_data)
    
    # 计算指标
    print("\nCalculating metrics...")
    all_metrics = []
    latitude_metrics = {
        "低纬度": {"ious": [], "scores": [], "offsets": []},
        "中纬度": {"ious": [], "scores": [], "offsets": []},
        "高纬度": {"ious": [], "scores": [], "offsets": []}
    }
    
    # 按image_id匹配，同一image_id内按顺序匹配
    for image_id in tqdm(gt_bfov_by_image.keys(), desc="Processing images"):
        if image_id in pred_bfov_by_image:
            gt_bfovs = gt_bfov_by_image[image_id]
            pred_bfovs = pred_bfov_by_image[image_id]
            
            # 按顺序匹配，取最小长度
            min_len = min(len(gt_bfovs), len(pred_bfovs))
            
            for i in range(min_len):
                gt = gt_bfovs[i]
                pred = pred_bfovs[i]
                
                # 计算sphere_iou
                sphere_iou = calculate_sphere_iou(gt['bfov'], pred['bfov'])
                
                # 计算偏移度指标
                offset_metric = calculate_offset_metric(gt['bfov'], pred['bfov'])
                
                # 获取纬度带
                gt_lat = gt['bfov'][1]  # BFoV格式：[经度, 纬度, 水平视场, 竖直视场]
                lat_band = get_latitude_band(gt_lat)
                
                # 记录指标
                metric = {
                    'image_id': image_id,
                    'gt_id': gt['id'],
                    'pred_id': pred['id'],
                    'gt_category_id': gt['category_id'],
                    'pred_category_id': pred['category_id'],
                    'gt_bfov': gt['bfov'],
                    'pred_bfov': pred['bfov'],
                    'sphere_iou': sphere_iou,
                    'pred_score': pred['score'],
                    'offset_metric': offset_metric,
                    'latitude_band': lat_band
                }
                all_metrics.append(metric)
                latitude_metrics[lat_band]["ious"].append(sphere_iou)
                latitude_metrics[lat_band]["scores"].append(pred['score'])
                latitude_metrics[lat_band]["offsets"].append(offset_metric)
    
    # 计算平均指标
    if all_metrics:
        avg_sphere_iou = sum(m['sphere_iou'] for m in all_metrics) / len(all_metrics)
        avg_score = sum(m['pred_score'] for m in all_metrics) / len(all_metrics)
        avg_offset = sum(m['offset_metric'] for m in all_metrics) / len(all_metrics)
        
        print(f"\nAverage metrics:")
        print(f"Sphere IoU: {avg_sphere_iou:.4f}")
        print(f"Average score: {avg_score:.4f}")
        print(f"Average offset metric: {avg_offset:.4f}")
        print(f"Total pairs: {len(all_metrics)}")
        
        # 计算各纬度带的平均IoU、平均得分和平均偏移度
        print("\nLatitude band metrics:")
        for band, data in latitude_metrics.items():
            ious = data["ious"]
            scores = data["scores"]
            offsets = data["offsets"]
            if ious:
                avg_iou = sum(ious) / len(ious)
                avg_score_band = sum(scores) / len(scores)
                avg_offset_band = sum(offsets) / len(offsets)
                print(f"{band}: Sphere IoU = {avg_iou:.4f}, Avg Score = {avg_score_band:.4f}, Avg Offset = {avg_offset_band:.4f} ({len(ious)} pairs)")
            else:
                print(f"{band}: N/A (0 pairs)")
    else:
        print("No matching pairs found.")
    
    # 保存详细结果
    output_file = '/home/mengchao/workspace/P2BFoV/P2BNet-main/TOV_mmdetection/bfov/bfov_metrics_results_qie.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
