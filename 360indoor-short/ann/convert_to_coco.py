import json

# 读取原始JSON文件
with open('annotations/7fB0x.json', 'r') as f:
    data = json.load(f)

# COCO格式基本结构
coco_format = {
    "images": [],
    "annotations": [],
    "categories": []
}

# 添加图像信息（假设只有一张图像）
coco_format["images"].append({
    "id": 1,
    "file_name": "7fB0x.jpg",  # 假设图像文件名与标注文件前缀相同
    "width": 1920,  # 假设图像宽度，需要根据实际情况调整
    "height": 960  # 假设图像高度，需要根据实际情况调整
})

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
coco_format["categories"] = categories

# 类别映射（从类别名到ID）
category_map = {}
for category in categories:
    category_map[category["name"]] = category["id"]

# 转换标注信息
annotation_id = 1
for i, box in enumerate(data["boxes"]):
    # box格式：[x, y, 3D位置1, 3D位置2, 宽度, 高度, 类别名称]
    # 按照要求去掉x和y坐标，只保留3D位置、宽度、高度和类别
    _, _, pos1, pos2, width, height, category_name = box

    
    # 在COCO格式中，bbox通常是[x, y, width, height]，但这里我们要去掉x和y
    # 由于COCO格式要求bbox字段，我们可以将其设置为[0, 0, width, height]或者其他合适的值
    # 按照要求，将pos1, pos2与width, height调换位置，使用[pos1, pos2, width, height]作为bbox
    
    annotation = {
        "id": annotation_id,
        "image_id": 1,
        "category_id": category_map[category_name],
        "bbox": [pos1, pos2, width, height],  # 去掉x和y坐标，使用[pos1, pos2, width, height]作为bbox
        "area": width * height,  # 面积
        "iscrowd": 0  # 是否为拥挤区域
    }
    
    coco_format["annotations"].append(annotation)
    annotation_id += 1

# 保存为COCO格式JSON文件
with open('7fB0x_coco.json', 'w') as f:
    json.dump(coco_format, f, indent=2)

print("转换完成！COCO格式文件已保存为 7fB0x_coco.json")