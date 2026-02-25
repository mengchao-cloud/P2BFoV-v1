import json
import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bfov_to_bbox import generate_bfov_mask_cpu


def visualize_bfov_and_bbox(json_path, images_dir, output_dir):
    """
    将 bfov 掩码和 bbox 绘制到图像上
    
    参数:
    json_path: COCO格式JSON文件路径
    images_dir: 图像目录路径
    output_dir: 输出目录路径
    """
    print(f"读取JSON文件: {json_path}")
    
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    print(f"包含 {len(coco_data['images'])} 张图像")
    print(f"包含 {len(coco_data['annotations'])} 个标注")
    
    os.makedirs(output_dir, exist_ok=True)
    
    erp_w = 1920
    erp_h = 960
    
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
    
    max_images = 6
    img_count = 0
    
    for img_info in coco_data['images']:
        if img_count >= max_images:
            print(f"\n已处理 {max_images} 张图像，停止处理")
            break
        
        img_count += 1
        img_id = img_info['id']
        img_filename = img_info['file_name']
        img_path = os.path.join(images_dir, img_filename)
        
        print(f"\n处理图像 {img_id}: {img_filename}")
        
        if not os.path.exists(img_path):
            print(f"  警告: 图像文件不存在: {img_path}")
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"  警告: 无法读取图像: {img_path}")
            continue
        
        overlay = img.copy()
        
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
        print(f"  找到 {len(annotations)} 个标注")
        
        for idx, ann in enumerate(annotations):
            color = colors[idx % len(colors)]
            
            bfov_params = ann['bfov']
            bbox = ann['bbox']
            
            mask = generate_bfov_mask_cpu(bfov_params, erp_w, erp_h)
            
            mask_colored = np.zeros_like(overlay)
            mask_colored[mask == 1] = color
            
            overlay = cv2.addWeighted(overlay, 1, mask_colored, 0.3, 0)
            
            x, y, width, height = bbox
            cv2.rectangle(overlay, (x, y), (x + width, y + height), color, 3)
            
            center_x = x + width // 2
            center_y = y + height // 2
            cv2.circle(overlay, (center_x, center_y), 5, color, -1)
            
            lon, lat, fov_x, fov_y = bfov_params
            center_px = int((lon + np.pi) / (2 * np.pi) * erp_w)
            center_py = int((np.pi/2 - lat) / np.pi * erp_h)
            cv2.circle(overlay, (center_px, center_py), 8, (255, 255, 255), -1)
            cv2.circle(overlay, (center_px, center_py), 5, color, -1)
            
            label = f"ID{ann['id']}"
            cv2.putText(overlay, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            print(f"    标注 {ann['id']}: bbox={bbox}, bfov=[{lon:.3f}, {lat:.3f}, {fov_x:.3f}, {fov_y:.3f}]")
        
        result_img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        
        output_filename = f"visualized_{img_filename}"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, result_img)
        print(f"  保存到: {output_path}")
    
    print(f"\n完成! 所有图像已保存到: {output_dir}")


def main():
    # json_path = '360indoor-short/ann/test_coco_short_gpu_bbox.json'
    json_path = "../360indoor/ann/test_coco_gpu_bbox.json" 
    images_dir = '../360indoor/images'
    output_dir = 'bfov/result/visualized_bfov_bbox_images/test'
    
    visualize_bfov_and_bbox(json_path, images_dir, output_dir)


if __name__ == '__main__':
    main()
