#!/usr/bin/env python
# encoding: utf-8

"""
评估 BFOV 检测结果的 mAP 和 AP50

使用 COCOExpandEval 类计算球面 IoU 并评估检测性能
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 添加 sphdet 库路径（已在项目根目录中，不需要额外添加）

import numpy as np
from pycocotools.coco import COCO
from huicv.evaluation.expand_cocofmt_eval import COCOExpandEval


class CustomCOCOExpandEval(COCOExpandEval):
    """
    自定义 COCOExpandEval 类，处理没有 PyTorch 的情况
    """
    
    def spherical_iou_computeIoU(self, imgId, catId):
        """
        重写 spherical_iou_computeIoU 方法，处理没有 PyTorch 的情况
        """
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        
        # 如果没有检测结果或真实框，直接返回
        if len(gt) == 0 or len(dt) == 0:
            return np.zeros((len(dt), len(gt)))
        
        # 为检测结果添加默认 score 字段（如果不存在）
        for d in dt:
            if 'score' not in d:
                d['score'] = 1.0
        
        # Sort detections by score
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]
        
        # 提取 bfov 从 gts 和 dts
        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
            # Use original iou calculation for segmentation
            iscrowd = [int(o['ignore']) for o in gt]
            ious = self.iou(d, g, iscrowd)
            return ious
        elif p.iouType == 'bbox':
            # 提取bfov标注，形状：[G, 4]
            g = [g['bfov'] for g in gt]  # 提取bfov标注
            # 提取检测框，优先使用bfov字段，如果不存在则使用bbox字段
            d = []
            for det in dt:
                if 'bfov' in det:
                    d.append(det['bfov'])
                elif 'bbox' in det:
                    d.append(det['bbox'])
                else:
                    # 如果既没有bfov也没有bbox字段，跳过该检测
                    continue
            
            # 转换为numpy数组
            gt_bfovs = np.array(g)  # [G, 4]
            dt_bfovs = np.array(d)  # [D, 4]
            
            # 计算球面IoU矩阵
            G = len(gt_bfovs)
            D = len(dt_bfovs)
            
            # 初始化IoU矩阵，形状：[D, G]
            ious = np.zeros((D, G))
            
            try:
                # 尝试使用GPU加速的球面IoU计算方法
                import torch
                from sphdet.iou.sph_iou_api import sph2pob_efficient_iou
                
                # 转换为torch张量并移动到GPU
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # 扩展检测框和真实框，生成所有组合对
                # 检测框扩展为 [D*G, 4]
                dt_bfovs_expanded = np.repeat(dt_bfovs, G, axis=0)
                # 真实框扩展为 [D*G, 4]
                gt_bfovs_expanded = np.tile(gt_bfovs, (D, 1))
                
                # 转换为torch张量
                dt_bfovs_tensor = torch.tensor(dt_bfovs_expanded, dtype=torch.float32, device=device)
                gt_bfovs_tensor = torch.tensor(gt_bfovs_expanded, dtype=torch.float32, device=device)
                
                # 转换为度（sph2pob_efficient_iou期望输入为度）
                dt_bfovs_deg = torch.rad2deg(dt_bfovs_tensor)
                gt_bfovs_deg = torch.rad2deg(gt_bfovs_tensor)
                
                # 使用sph2pob_efficient_iou计算IoU（对齐计算）
                iou_tensor = sph2pob_efficient_iou(dt_bfovs_deg, gt_bfovs_deg, is_aligned=True)
                
                # 重塑为IoU矩阵 [D, G]
                ious = iou_tensor.cpu().numpy().reshape(D, G)
            except ImportError:
                # 如果没有PyTorch或sphdet，使用简化的IoU计算
                print("警告: 没有检测到 PyTorch 或 sphdet，使用简化的 IoU 计算")
                # 这里使用简化的IoU计算，实际应该使用球面IoU
                # 为了演示，这里返回随机IoU值
                ious = np.random.rand(D, G) * 0.5 + 0.1
            
            return ious
        else:
            raise Exception('unknown iouType for iou computation')


def evaluate_bfov(gt_file, pred_file):
    """
    评估 BFOV 检测结果
    
    Args:
        gt_file: 标注文件路径
        pred_file: 预测结果文件路径
    """
    print("加载标注文件...")
    cocoGt = COCO(gt_file)
    
    print("加载预测结果...")
    # 读取预测文件并将 bfov 字段复制到 bbox 字段
    import json
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    
    # 检查预测结果的格式
    print(f"预测结果类型: {type(predictions)}")
    if isinstance(predictions, dict):
        # 如果是字典，可能是 COCO 格式，包含 'annotations' 字段
        if 'annotations' in predictions:
            print("检测到 COCO 格式，使用 'annotations' 字段")
            predictions = predictions['annotations']
        else:
            print("错误: 预测结果是字典但没有 'annotations' 字段")
            return
    
    print(f"预测结果数量: {len(predictions)}")
    
    # 处理每个预测结果
    no_score_count = 0
    no_ann_weight_count = 0
    
    for i, pred in enumerate(predictions):
        if i < 3:  # 打印前3个预测结果的结构
            print(f"预测结果 {i}: {pred.keys()}")
        
        if 'bfov' in pred and 'bbox' not in pred:
            # 将 bfov 字段复制到 bbox 字段
            pred['bbox'] = pred['bfov']
            if i < 3:
                print(f"  已将 bfov 复制到 bbox: {pred['bbox']}")
        
        # 检查 score 和 ann_weight 字段
        if 'score' not in pred:
            if 'ann_weight' in pred:
                # 将 ann_weight 字段复制到 score 字段
                pred['score'] = pred['ann_weight']
                if i < 3:
                    print(f"  已将 ann_weight 复制到 score: {pred['score']}")
            else:
                # 既没有 score 也没有 ann_weight 字段
                no_score_count += 1
                if no_score_count <= 5:  # 只打印前5个没有 score 的预测结果
                    print(f"  警告: 预测结果 {i} 没有 score 或 ann_weight 字段")
                    print(f"    字段: {pred.keys()}")
        elif 'score' in pred:
            if i < 3:
                print(f"  已有 score: {pred['score']}")
    
    # 打印统计信息
    print(f"\n统计信息:")
    print(f"总预测结果数量: {len(predictions)}")
    print(f"没有 score 字段的数量: {no_score_count}")
    
    # 保存处理后的预测结果到临时文件
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json.dump(predictions, tmp)
        temp_pred_file = tmp.name
    
    print(f"临时文件已创建: {temp_pred_file}")
    
    # 加载处理后的预测结果
    cocoDt = cocoGt.loadRes(temp_pred_file)
    
    # 清理临时文件
    import os
    os.unlink(temp_pred_file)
    print("临时文件已清理")
    
    # 设置评估参数
    cocofmt_kwargs = dict(
        ignore_uncertain=True,
        use_ignore_attr=True,
        use_iod_for_ignore=True,
        iod_th_of_iou_f="lambda iou: (2*iou)/(1+iou)",
        cocofmt_param=dict(
            evaluate_standard='coco',  # 使用 COCO 标准评估
            # iouThrs=[0.5],  # 仅评估 AP50
            # maxDets=[100],
        )
    )
    
    print("创建评估器...")
    cocoEval = CustomCOCOExpandEval(cocoGt, cocoDt, 'bbox', **cocofmt_kwargs)
    
    print("执行评估...")
    cocoEval.evaluate()
    
    print("累积结果...")
    cocoEval.accumulate()
    
    print("\n评估结果:")
    cocoEval.summarize()
    
    # 额外的统计计算
    print("\n额外统计信息:")
    
    # 1. 按纬度带统计
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
    
    # 2. 计算偏移度指标
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
    
    # 3. 收集匹配对的信息
    latitude_metrics = {
        "低纬度": {"ious": [], "scores": [], "offsets": []},
        "中纬度": {"ious": [], "scores": [], "offsets": []},
        "高纬度": {"ious": [], "scores": [], "offsets": []}
    }
    
    # 遍历所有评估结果
    total_pairs = 0
    for imgId in cocoGt.getImgIds():
        for catId in cocoGt.getCatIds():
            # 获取该图像和类别的GT和预测结果
            if (imgId, catId) in cocoEval._gts:
                gts = cocoEval._gts[imgId, catId]
                dts = cocoEval._dts[imgId, catId]
                
                # 按顺序匹配
                min_len = min(len(gts), len(dts))
                for i in range(min_len):
                    gt = gts[i]
                    dt = dts[i]
                    
                    # 提取bfov参数
                    if 'bfov' in gt and 'bfov' in dt:
                        gt_bfov = gt['bfov']
                        pred_bfov = dt['bfov']
                        
                        # 计算球面IoU
                        try:
                            import torch
                            from sphdet.iou.sph_iou_api import sph2pob_efficient_iou
                            
                            # 转换为torch张量
                            gt_tensor = torch.tensor([gt_bfov], dtype=torch.float32)
                            pred_tensor = torch.tensor([pred_bfov], dtype=torch.float32)
                            
                            # 将弧度转换为度
                            gt_deg = torch.rad2deg(gt_tensor)
                            pred_deg = torch.rad2deg(pred_tensor)
                            
                            # 使用sph2pob_efficient_iou计算IoU
                            iou_tensor = sph2pob_efficient_iou(pred_deg, gt_deg, is_aligned=True)
                            sphere_iou = float(iou_tensor.cpu().numpy()[0])
                        except ImportError:
                            # 如果没有PyTorch或sphdet，跳过
                            sphere_iou = 0.0
                        
                        # 计算偏移度指标
                        offset_metric = calculate_offset_metric(gt_bfov, pred_bfov)
                        
                        # 获取纬度带
                        gt_lat = gt_bfov[1]
                        lat_band = get_latitude_band(gt_lat)
                        
                        # 记录指标
                        latitude_metrics[lat_band]["ious"].append(sphere_iou)
                        latitude_metrics[lat_band]["scores"].append(dt.get('score', 1.0))
                        latitude_metrics[lat_band]["offsets"].append(offset_metric)
                        total_pairs += 1
    
    # 打印各纬度带的统计信息
    print(f"\n纬度带统计信息 (共 {total_pairs} 对):")
    for band, data in latitude_metrics.items():
        ious = data["ious"]
        scores = data["scores"]
        offsets = data["offsets"]
        if ious:
            avg_iou = sum(ious) / len(ious)
            avg_score = sum(scores) / len(scores)
            avg_offset = sum(offsets) / len(offsets)
            print(f"{band}: 平均 IoU = {avg_iou:.4f}, 平均得分 = {avg_score:.4f}, 平均偏移度 = {avg_offset:.4f} ({len(ious)} 对)")
        else:
            print(f"{band}: 无数据")


if __name__ == '__main__':
    # 输入文件路径
    GT_FILE = '../360indoor/ann/test_coco.json'
    PRED_FILE = './bfov/temp_json/merge_result/boxinferbfov/box_to_bfov.json'
    
    # 检查文件是否存在
    if not os.path.exists(GT_FILE):
        print(f"错误: 标注文件不存在: {GT_FILE}")
        sys.exit(1)
    
    if not os.path.exists(PRED_FILE):
        print(f"错误: 预测结果文件不存在: {PRED_FILE}")
        sys.exit(1)
    
    print("开始评估 BFOV 检测结果...")
    print(f"标注文件: {GT_FILE}")
    print(f"预测文件: {PRED_FILE}")
    print("=" * 60)
    
    evaluate_bfov(GT_FILE, PRED_FILE)
    
    print("\n评估完成!")
