import json
import os
import math

# 读取指定的list.txt文件，获取需要转换的文件名列表
def get_file_list(list_type='train'):
    # 根据list_type选择使用哪个list文件
    list_file = f'data_list/{list_type}_list_short.txt'
    with open(list_file, 'r') as f:
        lines = f.readlines()
    # 去掉文件扩展名，因为我们需要的是JSON文件名
    file_list = [line.strip().split('.')[0] for line in lines]
    return file_list

# 转换单个JSON文件到COCO格式
def convert_single_file(file_name, coco_format, image_id, annotation_id):
    # 构建JSON文件路径
    json_path = f'annotations/{file_name}.json'
    
    # 读取原始JSON文件
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 添加图像信息
    image = {
        "id": image_id,
        "file_name": f"{file_name}.jpg",
        "width": 1920,  # 假设图像宽度，需要根据实际情况调整
        "height": 960   # 假设图像高度，需要根据实际情况调整
    }
    coco_format["images"].append(image)
    
    # 转换标注信息
    for box in data["boxes"]:
        # box格式：[x, y, 3D位置1, 3D位置2, 宽度, 高度, 类别名称]
        x, y, pos1, pos2, width, height, category_name = box
        
        # 根据类别名称获取category_id
        category_id = None
        for category in coco_format["categories"]:
            if category["name"] == category_name:
                category_id = category["id"]
                break
        
        if category_id is None:
            print(f"警告：未找到类别 {category_name} 的映射，跳过该标注")
            continue
        
        # 将bfov中的width和height从角度转换为弧度
        bfov_width_rad = math.radians(width)
        bfov_height_rad = math.radians(height)
        
        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x, y, width, height],  # [x, y, width, height] 像素坐标，保持角度
            "bfov": [pos1, pos2, bfov_width_rad, bfov_height_rad],  # [pos1, pos2, width, height] 3D位置，转换为弧度
            "point": [pos1, pos2],  # [pos1, pos2] 3D中心点
            "area": width * height,  # 面积，使用原始角度计算
            "iscrowd": 0  # 是否为拥挤区域
        }
        
        coco_format["annotations"].append(annotation)
        annotation_id += 1
    
    return image_id + 1, annotation_id

# 主函数
def main():
    # 获取文件列表
    file_list = get_file_list('train')
    print(f"共找到 {len(file_list)} 个文件需要转换")
    
    # 使用用户提供的完整categories列表
    categories = [ 
        { "id": 0, "name": "toilet", "supercategory": "indoor" }, 
        { "id": 1, "name": "board", "supercategory": "indoor" }, 
        { "id": 2, "name": "mirror", "supercategory": "indoor" }, 
        { "id": 3, "name": "bed", "supercategory": "indoor" }, 
        { "id": 4, "name": "potted plant", "supercategory": "indoor" }, 
        { "id": 5, "name": "book", "supercategory": "indoor" }, 
        { "id": 6, "name": "clock", "supercategory": "indoor" }, 
        { "id": 7, "name": "phone", "supercategory": "indoor" }, 
        { "id": 8, "name": "keyboard", "supercategory": "indoor" }, 
        { "id": 9, "name": "tv", "supercategory": "indoor" }, 
        { "id": 10, "name": "fan", "supercategory": "indoor" }, 
        { "id": 11, "name": "backpack", "supercategory": "indoor" }, 
        { "id": 12, "name": "light", "supercategory": "indoor" }, 
        { "id": 13, "name": "refrigerator", "supercategory": "indoor" }, 
        { "id": 14, "name": "bathtub", "supercategory": "indoor" }, 
        { "id": 15, "name": "wine glass", "supercategory": "indoor" }, 
        { "id": 16, "name": "airconditioner", "supercategory": "indoor" }, 
        { "id": 17, "name": "cabinet", "supercategory": "indoor" }, 
        { "id": 18, "name": "sofa", "supercategory": "indoor" }, 
        { "id": 19, "name": "bowl", "supercategory": "indoor" }, 
        { "id": 20, "name": "sink", "supercategory": "indoor" }, 
        { "id": 21, "name": "computer", "supercategory": "indoor" }, 
        { "id": 22, "name": "cup", "supercategory": "indoor" }, 
        { "id": 23, "name": "bottle", "supercategory": "indoor" }, 
        { "id": 24, "name": "washer", "supercategory": "indoor" }, 
        { "id": 25, "name": "chair", "supercategory": "indoor" }, 
        { "id": 26, "name": "picture", "supercategory": "indoor" }, 
        { "id": 27, "name": "window", "supercategory": "indoor" }, 
        { "id": 28, "name": "door", "supercategory": "indoor" }, 
        { "id": 29, "name": "heater", "supercategory": "indoor" }, 
        { "id": 30, "name": "fireplace", "supercategory": "indoor" }, 
        { "id": 31, "name": "mouse", "supercategory": "indoor" }, 
        { "id": 32, "name": "oven", "supercategory": "indoor" }, 
        { "id": 33, "name": "microwave", "supercategory": "indoor" }, 
        { "id": 34, "name": "person", "supercategory": "indoor" }, 
        { "id": 35, "name": "vase", "supercategory": "indoor" }, 
        { "id": 36, "name": "table", "supercategory": "indoor" }
    ]
    
    # 初始化COCO格式
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    # 遍历所有文件进行转换
    image_id = 1
    annotation_id = 1
    success_count = 0
    fail_count = 0
    
    for file_name in file_list:
        try:
            image_id, annotation_id = convert_single_file(file_name, coco_format, image_id, annotation_id)
            success_count += 1
            print(f"成功转换文件：{file_name}")
        except Exception as e:
            fail_count += 1
            print(f"转换文件 {file_name} 失败：{str(e)}")
    
    # 保存为COCO格式JSON文件
    output_path = 'train_coco_short.json'
    with open(output_path, 'w') as f:
        json.dump(coco_format, f, indent=2)
    
    print(f"\n批量转换完成！")
    print(f"成功转换：{success_count} 个文件")
    print(f"转换失败：{fail_count} 个文件")
    print(f"生成的COCO格式文件：{output_path}")
    print(f"\nCOCO文件内容结构：")
    print(f"- images: {len(coco_format['images'])} 张图像")
    print(f"- annotations: {len(coco_format['annotations'])} 个标注")
    print(f"- categories: {len(coco_format['categories'])} 个类别")

if __name__ == "__main__":
    main()