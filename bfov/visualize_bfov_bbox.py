import torch
import numpy as np
import cv2
import os

from PANDORA.PRDA.lib.ImageRecorder import ImageRecorder


def generate_bfov_mask_cpu(bfov_params, erp_w=1920, erp_h=960):
    """
    生成单个BFOV的掩码（CPU版本）
    
    参数:
    bfov_params: BFOV参数 [longitude, latitude, fov_x, fov_y]（弧度制）
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    
    返回:
    mask: [H, W]的numpy数组，H=erp_h, W=erp_w
    """
    lon, lat, fov_x, fov_y = bfov_params
    
    fov_x_deg = np.degrees(fov_x)
    fov_y_deg = np.degrees(fov_y)
    
    cpu_recorder = ImageRecorder(erp_w, erp_h, view_angle_w=fov_x_deg, view_angle_h=fov_y_deg, long_side=erp_w)
    
    Px, Py = cpu_recorder._sample_points(lon, lat, border_only=False)
    
    Px = Px.astype(np.int32)
    Py = Py.astype(np.int32)
    
    valid_mask = (Px >= 0) & (Px < erp_w) & (Py >= 0) & (Py < erp_h)
    valid_Px = Px[valid_mask]
    valid_Py = Py[valid_mask]
    
    mask = np.zeros((erp_h, erp_w), dtype=np.uint8)
    mask[valid_Py, valid_Px] = 1
    
    return mask


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


def visualize_bfov_with_bbox(bfov_params, mask, bbox, erp_w=1920, erp_h=960, output_path=None):
    """
    可视化BFOV掩码及其外接矩形包围框
    
    参数:
    bfov_params: BFOV参数 [longitude, latitude, fov_x, fov_y]（弧度制）
    mask: [H, W]的numpy数组
    bbox: [x, y, width, height] 包围框参数
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    output_path: 输出图像路径，如果为None则不保存
    """
    img = np.zeros((erp_h, erp_w, 3), dtype=np.uint8)
    img[:] = (255, 255, 255)
    
    img[mask == 1] = (0, 255, 0)
    
    if bbox is not None:
        x, y, width, height = bbox
        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 255), 3)
        
        center_x = x + width // 2
        center_y = y + height // 2
        cv2.circle(img, (center_x, center_y), 5, (255, 0, 0), -1)
        
        cv2.putText(img, f'BBox: ({x},{y},{width},{height})', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    lon, lat, fov_x, fov_y = bfov_params
    center_px = int((lon + np.pi) / (2 * np.pi) * erp_w)
    center_py = int((np.pi/2 - lat) / np.pi * erp_h)
    
    cv2.circle(img, (center_px, center_py), 8, (255, 0, 0), -1)
    cv2.putText(img, f'Center: ({center_px},{center_py})', (center_px + 10, center_py - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    info_text = f'BFOV: lon={lon:.3f}, lat={lat:.3f}, fov_x={fov_x:.3f}, fov_y={fov_y:.3f}'
    cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        print(f"可视化图像已保存到: {output_path}")
    
    return img


def print_bfov_info(bfov_params, mask, bbox):
    """
    打印BFOV和包围框的详细信息
    
    参数:
    bfov_params: BFOV参数 [longitude, latitude, fov_x, fov_y]（弧度制）
    mask: [H, W]的numpy数组
    bbox: [x, y, width, height] 包围框参数
    """
    print("=" * 60)
    print("BFOV参数（弧度制）:")
    print("=" * 60)
    lon, lat, fov_x, fov_y = bfov_params
    print(f"经度: {lon:.6f} rad ({np.degrees(lon):.2f}°)")
    print(f"纬度: {lat:.6f} rad ({np.degrees(lat):.2f}°)")
    print(f"水平视场角: {fov_x:.6f} rad ({np.degrees(fov_x):.2f}°)")
    print(f"垂直视场角: {fov_y:.6f} rad ({np.degrees(fov_y):.2f}°)")
    
    print("\n" + "=" * 60)
    print("掩码统计信息:")
    print("=" * 60)
    print(f"掩码形状: {mask.shape}")
    print(f"掩码中像素总数: {mask.size}")
    print(f"掩码中有效像素数: {np.sum(mask)}")
    print(f"掩码覆盖率: {np.sum(mask) / mask.size * 100:.2f}%")
    
    print("\n" + "=" * 60)
    print("外接矩形包围框参数:")
    print("=" * 60)
    if bbox is not None:
        x, y, width, height = bbox
        print(f"左上角x坐标: {x}")
        print(f"左上角y坐标: {y}")
        print(f"宽度: {width}")
        print(f"高度: {height}")
        print(f"右下角坐标: ({x + width}, {y + height})")
        print(f"中心点坐标: ({x + width // 2}, {y + height // 2})")
        print(f"包围框面积: {width * height}")
        print(f"包围框参数格式: [x, y, width, height] = [{x}, {y}, {width}, {height}]")
    else:
        print("无法计算包围框（掩码为空）")
    print("=" * 60)


def main():
    """
    主函数：生成BFOV掩码，计算并可视化包围框
    """
    print("开始生成BFOV掩码并计算外接矩形包围框...")
    
    erp_w = 1920
    erp_h = 960
    
    num_bfovs = 5
    print(f"\n生成 {num_bfovs} 个随机BFOV参数...")
    
    bfov_list = []
    for i in range(num_bfovs):
        lon = np.random.uniform(-np.pi, np.pi)
        lat = np.random.uniform(-np.pi/2, np.pi/2)
        fov_x = np.random.uniform(np.pi/6, np.pi/2)
        fov_y = np.random.uniform(np.pi/6, np.pi/2)
        bfov_params = [lon, lat, fov_x, fov_y]
        bfov_list.append(bfov_params)
        print(f"BFOV {i+1}: lon={lon:.3f}, lat={lat:.3f}, fov_x={fov_x:.3f}, fov_y={fov_y:.3f}")
    
    output_dir = './bfov/result/bfov_bbox_visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    colors = [
        (255, 0, 0),    # 蓝色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 红色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 紫色
        (0, 255, 255),  # 青色
        (128, 0, 128),  # 紫色
        (128, 128, 0),  # 橄榄色
        (0, 128, 128),  # 青色
        (128, 128, 128)  # 灰色
    ]
    
    for idx, bfov_params in enumerate(bfov_list):
        print(f"\n{'='*60}")
        print(f"处理 BFOV {idx+1}/{num_bfovs}")
        print(f"{'='*60}")
        
        mask = generate_bfov_mask_cpu(bfov_params, erp_w, erp_h)
        print(f"掩码生成完成，形状: {mask.shape}")
        
        bbox = compute_bounding_box(mask)
        print(f"包围框计算完成")
        
        print_bfov_info(bfov_params, mask, bbox)
        
        output_path = os.path.join(output_dir, f'bfov_{idx+1}_with_bbox.jpg')
        img = visualize_bfov_with_bbox(bfov_params, mask, bbox, erp_w, erp_h, output_path)
        
        print("\n" + "=" * 60)
        print("包围框参数（可直接用于检测任务）:")
        print("=" * 60)
        if bbox is not None:
            print(f"bbox = [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}")
            print(f"bbox = [x, y, width, height]")
        print("=" * 60)
    
    summary_img = np.zeros((erp_h, erp_w, 3), dtype=np.uint8)
    summary_img[:] = (255, 255, 255)
    
    for idx, bfov_params in enumerate(bfov_list):
        mask = generate_bfov_mask_cpu(bfov_params, erp_w, erp_h)
        bbox = compute_bounding_box(mask)
        color = colors[idx % len(colors)]
        
        summary_img[mask == 1] = color
        
        if bbox is not None:
            x, y, width, height = bbox
            cv2.rectangle(summary_img, (x, y), (x + width, y + height), color, 2)
            
            lon, lat, fov_x, fov_y = bfov_params
            center_px = int((lon + np.pi) / (2 * np.pi) * erp_w)
            center_py = int((np.pi/2 - lat) / np.pi * erp_h)
            cv2.circle(summary_img, (center_px, center_py), 5, color, -1)
            cv2.putText(summary_img, f'BFOV{idx+1}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    summary_output_path = os.path.join(output_dir, 'all_bfovs_summary.jpg')
    cv2.imwrite(summary_output_path, summary_img)
    print(f"\n所有BFOV的汇总可视化已保存到: {summary_output_path}")


if __name__ == '__main__':
    main()
