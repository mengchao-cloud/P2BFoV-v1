import json
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bfov_to_bbox import bfov_to_bbox, generate_bfov_mask_cpu, compute_bounding_box


def convert_bfov_to_bbox_in_coco(input_json_path, output_json_path):
    """
    将COCO格式JSON文件中的bfov参数转换为bbox参数
    
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
                print(f"\n警告: 标注 {ann['id']} 的 bfov 视场角过大")
                print(f"  图像: {img_info['file_name'] if img_info else 'unknown'} (id={ann['image_id']})")
                print(f"  bfov 参数: [lon={lon:.4f}, lat={lat:.4f}, fov_x={np.degrees(fov_x):.1f}°, fov_y={np.degrees(fov_y):.1f}°]")
                print(f"  使用默认 bbox: [952, 472, 16, 16]")
                ann['bbox'] = [952, 472, 16, 16]
                ann['area'] = 256
                continue
            
            mask = generate_bfov_mask_cpu(bfov_params, erp_w, erp_h)
            
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
                old_bbox = ann['bbox']
                ann['bbox'] = [int(x) for x in new_bbox]
                
                width, height = ann['bbox'][2], ann['bbox'][3]
                ann['area'] = int(width * height)
                
                converted_count += 1
                
                if converted_count <= 3:
                    print(f"\n标注 {ann['id']}:")
                    print(f"  原始 bbox: {old_bbox}")
                    print(f"  bfov 参数: {bfov_params}")
                    print(f"  新 bbox: {new_bbox}")
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
    
    print(f"新JSON文件已保存到: {output_json_path}")


def main():
    # input_json = '360indoor-short/ann/train_coco_short.json'
    # output_json = '360indoor-short/ann/train_coco_short_bfov_bbox.json'
    input_json = 'bfov/temp_json/test_coco.json'
    output_json = 'bfov/temp_json/test_coco_bfov_bbox.json'
    
    convert_bfov_to_bbox_in_coco(input_json, output_json)


if __name__ == '__main__':
    main()
