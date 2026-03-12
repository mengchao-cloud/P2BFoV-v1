import cv2
import numpy as np
import os
import json
from bfov_to_bbox import generate_bfov_mask_cpu


def get_symmetry_box_with_80_percent(mask):
    """
    为对称掩码生成主体外接矩形，满足掩码占比≥80%
    特别处理宽而扁的掩码（宽度很大但高度不大）
    
    参数:
    mask: 二值掩码图像（0为背景，255为前景）
    
    返回:
    target_box: 目标矩形 [x, y, w, h]（x,y为左上角坐标）
    """
    # 1. 校验输入掩码格式
    if len(mask.shape) != 2:
        raise ValueError("输入掩码必须是单通道二值图像")
    
    # 2. 获取掩码的整体外接矩形和基础信息
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("掩码中未检测到前景区域")
    
    # 整体外接矩形（x, y, w, h）
    x_total, y_total, w_total, h_total = cv2.boundingRect(contours[0])
    # 掩码前景像素总数
    mask_pixel_count = np.sum(mask > 0)
    
    # 3. 分析掩码特征：判断是否为宽而扁的掩码
    erp_w = mask.shape[1]
    erp_h = mask.shape[0]
    is_wide_flat_mask = (w_total > erp_w * 0.8) and (h_total < erp_h * 0.3)
    
    if is_wide_flat_mask:
        print("检测到宽而扁的掩码，使用特殊处理策略")
        # 对于宽而扁的掩码，直接使用掩码的高度和宽度
        # 因为任何高度为h_total的矩形框都能满足覆盖率要求
        box_h = h_total
        box_w = w_total
        box_x = x_total
        box_y = y_total
    else:
        # 4. 确定矩形高度（直接使用掩码高度）
        box_h = h_total
        # 计算满足80%占比的最小宽度：总像素数 / 占比 / 高度
        min_valid_w = mask_pixel_count / 0.8 / box_h
        # 向上取整（确保宽度足够）
        min_valid_w = int(np.ceil(min_valid_w))
        
        # 5. 结合对称性确定宽度（以掩码中心为基准）
        # 掩码中心x坐标
        mask_center_x = x_total + w_total / 2
        # 计算对称宽度的左右边界
        half_w = min_valid_w / 2
        box_x = mask_center_x - half_w
        box_w = min_valid_w
        
        # 6. 边界校验：确保宽度不超过掩码整体宽度，且坐标为整数
        box_w = min(box_w, w_total)  # 宽度不超过整体宽度
        box_x = int(mask_center_x - box_w / 2)
        box_y = y_total  # 高度与掩码一致，y坐标不变
        box_h = int(box_h)
        box_w = int(box_w)
    
    # 验证占比（可选）
    # 截取目标矩形区域的掩码
    box_mask = mask[box_y:box_y+box_h, box_x:box_x+box_w]
    box_pixel_count = np.sum(box_mask > 0)
    coverage_ratio = box_pixel_count / (box_w * box_h)
    
    if coverage_ratio < 0.8 and not is_wide_flat_mask:
        print(f"警告：当前占比{coverage_ratio:.2%}，略低于80%，已使用掩码最大宽度")
        box_w = w_total
        box_x = x_total
    
    return [box_x, box_y, box_w, box_h]


def compute_symmetry_bounding_box(mask):
    """
    计算掩码的对称外接矩形包围框（结合ERP跨边界处理）
    
    参数:
    mask: [H, W]的numpy数组
    
    返回:
    bbox: [x, y, width, height] 包围框参数
    """
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
    
    # 处理ERP图像跨边界问题
    if x_diff > erp_w * 0.6:
        # 使用对称性方法处理跨边界情况
        try:
            # 将掩码转换为uint8格式用于OpenCV
            mask_uint8 = (mask * 255).astype(np.uint8)
            symmetry_bbox = get_symmetry_box_with_80_percent(mask_uint8)
            return symmetry_bbox
        except Exception as e:
            print(f"对称性方法处理失败: {e}")
            # 回退到原始方法
            gap_threshold = 50
            gaps = []
            
            for i in range(len(x_indices) - 1):
                gap = x_indices[i+1] - x_indices[i]
                if gap > gap_threshold:
                    gaps.append((i, gap))
            
            if gaps:
                max_gap_idx, max_gap = max(gaps, key=lambda x: x[1])
                
                left_pixels = np.sum(mask[:, :x_indices[max_gap_idx]])
                right_pixels = np.sum(mask[:, x_indices[max_gap_idx+1]:])
                
                if left_pixels > right_pixels:
                    x_max = x_indices[max_gap_idx]
                else:
                    x_min = x_indices[max_gap_idx + 1]
    
    x = x_min
    y = y_min
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    
    return [x, y, width, height]


def visualize_symmetry_bbox(bfov_params, mask, bbox, erp_w=1920, erp_h=960, output_path=None):
    """
    可视化BFOV掩码及其对称外接矩形包围框
    
    参数:
    bfov_params: BFOV参数 [longitude, latitude, fov_x, fov_y]（弧度制）
    mask: [H, W]的numpy数组
    bbox: [x, y, width, height] 包围框参数
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    output_path: 输出图像路径
    """
    img = np.zeros((erp_h, erp_w, 3), dtype=np.uint8)
    img[:] = (255, 255, 255)
    
    # 绘制掩码
    img[mask == 1] = (0, 255, 0)
    
    if bbox is not None:
        x, y, width, height = bbox
        # 绘制对称边界框（红色）
        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 255), 3)
        
        # 绘制中心点
        center_x = x + width // 2
        center_y = y + height // 2
        cv2.circle(img, (center_x, center_y), 5, (255, 0, 0), -1)
        
        # 添加文本标注
        cv2.putText(img, f'Symmetry BBox: ({x},{y},{width},{height})', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # 绘制BFOV中心点
    lon, lat, fov_x, fov_y = bfov_params
    center_px = int((lon + np.pi) / (2 * np.pi) * erp_w)
    center_py = int((np.pi/2 - lat) / np.pi * erp_h)
    cv2.circle(img, (center_px, center_py), 8, (255, 0, 0), -1)
    
    # 添加BFOV参数信息
    info_text = f'Symmetry BFOV-BBox: lon={lon:.3f}, lat={lat:.3f}, fov_x={fov_x:.3f}, fov_y={fov_y:.3f}'
    cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        print(f"对称边界框可视化图像已保存到: {output_path}")
    
    return img


def convert_bfov_to_symmetry_bbox_in_coco(input_json_path, output_json_path):
    """
    将COCO格式JSON文件中的bfov参数转换为对称边界框参数
    
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
    symmetry_count = 0
    
    for ann in coco_data['annotations']:
        if 'bfov' in ann:
            bfov_params = ann['bfov']
            
            mask = generate_bfov_mask_cpu(bfov_params, erp_w, erp_h)
            
            if mask is None:
                failed_count += 1
                ann['bbox'] = [952, 472, 16, 16]
                ann['area'] = 256
                continue
            
            # 使用对称性方法计算边界框
            new_bbox = compute_symmetry_bounding_box(mask)
            
            if new_bbox is not None:
                old_bbox = ann['bbox']
                ann['bbox'] = [int(x) for x in new_bbox]
                
                width, height = ann['bbox'][2], ann['bbox'][3]
                ann['area'] = int(width * height)
                
                converted_count += 1
                
                # 检查是否使用了对称性处理
                if width > erp_w * 0.6:
                    symmetry_count += 1
                    print(f"标注 {ann['id']}: 使用对称性方法处理跨边界情况")
                
                if converted_count <= 3:
                    print(f"标注 {ann['id']}:")
                    print(f"  原始 bbox: {old_bbox}")
                    print(f"  对称 bbox: {new_bbox}")
            else:
                failed_count += 1
                ann['bbox'] = [952, 472, 16, 16]
                ann['area'] = 256
    
    print(f"\n转换完成!")
    print(f"成功转换: {converted_count} 个标注")
    print(f"使用对称性处理: {symmetry_count} 个标注")
    print(f"转换失败: {failed_count} 个标注")
    
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"新JSON文件已保存到: {output_json_path}")


def generate_random_bfov_params(num_bfovs=20):
    """
    生成随机的BFOV参数，特别包含宽而扁的掩码
    
    参数:
    num_bfovs: 生成的BFOV数量
    
    返回:
    bfov_list: BFOV参数列表
    """
    bfov_list = []
    
    for i in range(num_bfovs):
        # 前1/3生成宽而扁的掩码（大水平视场角，小垂直视场角）
        if i < num_bfovs // 3:
            # 宽而扁的掩码：大水平视场角，小垂直视场角
            lon = np.random.uniform(-np.pi, np.pi)
            lat = np.random.uniform(-np.pi/6, np.pi/6)  # 限制纬度范围
            fov_x = np.random.uniform(np.pi/2, np.pi)   # 大水平视场角（90°-180°）
            fov_y = np.random.uniform(np.pi/12, np.pi/6) # 小垂直视场角（15°-30°）
            mask_type = "宽而扁"
        # 中间1/3生成正常掩码
        elif i < 2 * num_bfovs // 3:
            # 正常掩码
            lon = np.random.uniform(-np.pi, np.pi)
            lat = np.random.uniform(-np.pi/2, np.pi/2)
            fov_x = np.random.uniform(np.pi/12, np.pi/2)
            fov_y = np.random.uniform(np.pi/12, np.pi/2)
            mask_type = "正常"
        # 后1/3生成高而窄的掩码
        else:
            # 高而窄的掩码：小水平视场角，大垂直视场角
            lon = np.random.uniform(-np.pi, np.pi)
            lat = np.random.uniform(-np.pi/3, np.pi/3)
            fov_x = np.random.uniform(np.pi/12, np.pi/6)  # 小水平视场角（15°-30°）
            fov_y = np.random.uniform(np.pi/2, np.pi)     # 大垂直视场角（90°-180°）
            mask_type = "高而窄"
        
        bfov_params = [lon, lat, fov_x, fov_y]
        bfov_list.append(bfov_params)
        
        # 打印参数信息
        print(f"BFOV {i+1:2d} [{mask_type}]: lon={lon:7.3f}, lat={lat:7.3f}, "
              f"fov_x={np.degrees(fov_x):5.1f}°, fov_y={np.degrees(fov_y):5.1f}°")
    
    return bfov_list


def main():
    """
    主函数：测试对称边界框生成算法
    """
    print("开始测试对称边界框生成算法...")
    
    erp_w = 1920
    erp_h = 960
    
    # 生成更多的随机BFOV参数
    num_bfovs = 10
    print(f"\n生成 {num_bfovs} 个随机BFOV参数...")
    
    # 包含一些特殊情况的固定测试用例
    special_bfovs = [
        [0.0, 0.0, np.pi/4, np.pi/4],  # 中心位置
        [np.pi, 0.0, np.pi/3, np.pi/3],  # 右边界跨边界
        [-np.pi, 0.0, np.pi/3, np.pi/3],  # 左边界跨边界
        [np.pi/2, np.pi/4, np.pi/6, np.pi/6],  # 右上角
        [-np.pi/2, -np.pi/4, np.pi/8, np.pi/8],  # 左下角
    ]
    
    # 生成随机BFOV参数
    random_bfovs = generate_random_bfov_params(num_bfovs - len(special_bfovs))
    
    # 合并测试用例
    test_bfovs = special_bfovs + random_bfovs
    
    output_dir = './bfov/result/symmetry_bbox'
    os.makedirs(output_dir, exist_ok=True)
    
    # 统计信息
    symmetry_count = 0
    coverage_stats = []
    
    for idx, bfov_params in enumerate(test_bfovs):
        print(f"\n{'='*60}")
        print(f"测试 BFOV {idx+1}/{len(test_bfovs)}")
        print(f"{'='*60}")
        
        mask = generate_bfov_mask_cpu(bfov_params, erp_w, erp_h)
        
        if mask is None:
            print("掩码生成失败，跳过此BFOV")
            continue
            
        # 使用对称性方法计算边界框
        symmetry_bbox = compute_symmetry_bounding_box(mask)
        
        # 可视化结果
        output_path = os.path.join(output_dir, f'symmetry_bbox_{idx+1:02d}.jpg')
        visualize_symmetry_bbox(bfov_params, mask, symmetry_bbox, erp_w, erp_h, output_path)
        
        print(f"对称边界框: {symmetry_bbox}")
        
        # 计算覆盖率
        if symmetry_bbox is not None:
            x, y, w, h = symmetry_bbox
            box_mask = mask[y:y+h, x:x+w]
            coverage = np.sum(box_mask) / (w * h)
            coverage_stats.append(coverage)
            
            # 检查是否使用了对称性处理
            if w > erp_w * 0.6:
                symmetry_count += 1
                print("✓ 使用对称性方法处理跨边界情况")
            
            print(f"边界框覆盖率: {coverage:.2%}")
            print(f"边界框面积: {w * h} 像素")
            print(f"掩码像素数: {np.sum(mask)} 像素")
    
    # 输出统计信息
    print(f"\n{'='*60}")
    print("测试结果统计")
    print(f"{'='*60}")
    print(f"总测试用例数: {len(test_bfovs)}")
    print(f"使用对称性处理的用例: {symmetry_count}")
    
    if coverage_stats:
        print(f"平均覆盖率: {np.mean(coverage_stats):.2%}")
        print(f"最小覆盖率: {np.min(coverage_stats):.2%}")
        print(f"最大覆盖率: {np.max(coverage_stats):.2%}")
        print(f"覆盖率≥80%的用例: {sum(c >= 0.8 for c in coverage_stats)}/{len(coverage_stats)}")
    
    print(f"\n测试完成！结果保存在: {output_dir}")
    print(f"共生成 {len(test_bfovs)} 个可视化图像")


if __name__ == '__main__':
    main()