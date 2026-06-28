#!/usr/bin/env python3
"""
绘制图像 7PhJB.jpg 上的 BFoV 标注，只绘制3-6这五个BFoV，且中心有适当偏离

使用方法：
conda run -n pointobb python draw_bfov_7PhJB_5dev.py
"""

import os
import sys
import numpy as np
import cv2

# 添加sphdet路径到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../sphdet/visualizers'))
from ImageRecorder import ImageRecorder as BFoV

# 添加PANDORA路径用于三维旋转
sys.path.append(os.path.join(os.path.dirname(__file__), '../../PANDORA/RoIoU/libs'))
try:
    from tools import rotate_image, tools
    print("成功导入 rotate_image 和 tools")
except ImportError as e:
    print(f"导入失败: {e}")
    raise

# 图像 7PhJB.jpg 的原始 BFoV 标注（保留）
BFoV_ANNOTATIONS = [
    [1, 0.14562590946327697, -0.011453723216212833, 0.3490658503988659, 0.4188790204786391],
    [8, 0.07363107781851078, -0.2503456645829367, 0.13962634015954636, 0.06981317007977318],
    [21, 0.0867210472084682, -0.15871587885323438, 0.20943951023931956, 0.2792526803190927],
    [25, 0.8492117641734908, -1.0750137361502572, 0.3490658503988659, 0.2792526803190927],
    [25, -0.14562590946327697, -1.0651962591077895, 0.3490658503988659, 0.2792526803190927],
    [25, -0.5743224069843842, -0.9179341034707679, 0.20943951023931956, 0.2792526803190927],
    [25, 0.8197593330460868, -0.7968518866136611, 0.20943951023931956, 0.2792526803190927],
    [25, 0.7870344095711929, -0.5710499146368947, 0.20943951023931956, 0.2792526803190927],
    [25, 0.6103198228067673, -0.502327575339618, 0.13962634015954636, 0.20943951023931956],
    [25, 0.5612324375944268, -0.36815538909255385, 0.13962634015954636, 0.06981317007977318],
    [25, -0.3027055421427664, -0.3779728661350221, 0.13962634015954636, 0.20943951023931956],
    [28, -2.541090307825494, -0.2372556951929793, 0.2792526803190927, 2.0943951023931953],
    [31, 0.1390809247682979, -0.2601631416254046, 0.06981317007977318, 0.06981317007977318],
    [34, -0.0801760625134895, -0.16853335589570229, 0.2792526803190927, 0.4886921905584123],
    [34, 0.3550654197025965, -0.1947132946756175, 0.2792526803190927, 0.5585053606381855],
    [36, 0.0703585854710216, -0.6594072080191075, 1.117010721276371, 0.8377580409572782],
]

# 固定的FOV参数组合（9种）
FIXED_FOV_PARAMS = [
    (0.2612193871, 0.1306096936),  # 0
    (0.1847100000, 0.1847100000),  # 1
    (0.1306096936, 0.2612193871),  # 2 (3号)
    (0.6791704065, 0.3395852032),  # 3 (4号)
    (0.4802460000, 0.4802460000),  # 4 (5号)
    (0.3395852032, 0.6791704065),  # 5 (6号)
    (1.2016091807, 0.6008045903),  # 6 (7号)
    (0.8496660000, 0.8496660000),  # 7 (8号)
    (0.6008045903, 1.2016091807),  # 8 (9号)
]

# person的中心点
PERSON_THETA = -0.0801760625134895
PERSON_PHI = -0.16853335589570229

# chair的中心点
CHAIR_THETA = 0.8492117641734908
CHAIR_PHI = -1.0750137361502572

# 选择3-6这五个FOV参数用于有偏离的BFoV（索引2-6）
SELECTED_FOV_PARAMS_OFFSET = FIXED_FOV_PARAMS[2:7]

# 选择5-9这五个FOV参数用于没有偏离的BFoV（索引4-8）
SELECTED_FOV_PARAMS_NO_OFFSET = FIXED_FOV_PARAMS[4:9]

# 为person和chair生成带有中心偏离的标注
def generate_annotations_with_offset(base_theta, base_phi, fov_params, category_id):
    """
    生成带有中心偏离的BFoV标注
    
    参数:
    base_theta: 基础中心的theta（弧度）
    base_phi: 基础中心的phi（弧度）
    fov_params: FOV参数列表
    category_id: 类别ID
    
    返回:
    标注列表
    """
    annotations = []
    
    # 为每个FOV生成不同的中心偏离
    # 偏离模式：围绕基础中心在不同方向上偏移（更大幅度）
    offsets = [
        (0.15, 0.10),     # 右上
        (-0.12, 0.08),    # 左上
        (0.0, -0.15),     # 下
        (0.10, -0.10),    # 右下
        (-0.15, -0.08),   # 左下
    ]
    
    for i, (fov_x, fov_y) in enumerate(fov_params):
        theta_offset, phi_offset = offsets[i]
        new_theta = base_theta + theta_offset
        new_phi = base_phi + phi_offset
        annotations.append([category_id, new_theta, new_phi, fov_x, fov_y])
    
    return annotations

# 为person和chair生成没有中心偏离的标注
def generate_annotations_no_offset(base_theta, base_phi, fov_params, category_id):
    """
    生成没有中心偏离的BFoV标注（使用原始中心）
    
    参数:
    base_theta: 基础中心的theta（弧度）
    base_phi: 基础中心的phi（弧度）
    fov_params: FOV参数列表
    category_id: 类别ID
    
    返回:
    标注列表
    """
    annotations = []
    
    for fov_x, fov_y in fov_params:
        annotations.append([category_id, base_theta, base_phi, fov_x, fov_y])
    
    return annotations

# 生成person和chair的标注（同时包含有偏离和无偏离）
# 有偏离的使用3-6号FOV参数，没有偏离的使用5-9号FOV参数
person_annotations_offset = generate_annotations_with_offset(PERSON_THETA, PERSON_PHI, SELECTED_FOV_PARAMS_OFFSET, 34)
person_annotations_no_offset = generate_annotations_no_offset(PERSON_THETA, PERSON_PHI, SELECTED_FOV_PARAMS_NO_OFFSET, 34)
chair_annotations_offset = generate_annotations_with_offset(CHAIR_THETA, CHAIR_PHI, SELECTED_FOV_PARAMS_OFFSET, 25)
chair_annotations_no_offset = generate_annotations_no_offset(CHAIR_THETA, CHAIR_PHI, SELECTED_FOV_PARAMS_NO_OFFSET, 25)

# 合并所有标注（有偏离 + 无偏离）
SELECTED_BFOV_ANNOTATIONS = person_annotations_offset + person_annotations_no_offset + chair_annotations_offset + chair_annotations_no_offset

# 类别名称映射
CATEGORY_NAMES = {
    0: 'toilet',
    1: 'board',
    2: 'mirror',
    3: 'bed',
    4: 'potted plant',
    5: 'book',
    6: 'clock',
    7: 'phone',
    8: 'keyboard',
    9: 'tv',
    10: 'fan',
    11: 'backpack',
    12: 'light',
    13: 'refrigerator',
    14: 'bathtub',
    15: 'wine glass',
    16: 'airconditioner',
    17: 'cabinet',
    18: 'sofa',
    19: 'bowl',
    20: 'sink',
    21: 'computer',
    22: 'cup',
    23: 'bottle',
    24: 'washer',
    25: 'chair',
    26: 'picture',
    27: 'window',
    28: 'door',
    29: 'heater',
    30: 'fireplace',
    31: 'mouse',
    32: 'oven',
    33: 'microwave',
    34: 'person',
    35: 'vase',
    36: 'table',
}

def draw_bfov_on_image(img, bfov_annotations, img_w, img_h):
    """
    在图像上绘制多个BFoV
    
    参数:
    img: 输入图像
    bfov_annotations: BFoV标注列表
    img_w: 图像宽度
    img_h: 图像高度
    
    返回:
    img: 绘制后的图像
    """
    category_colors = {
        25: (0, 255, 255), # chair - 黄色（与原脚本一致）
        34: (128, 0, 0),   # person - 深蓝色（与原脚本一致）
    }
    
    for i, ann in enumerate(bfov_annotations):
        category_id, theta, phi, fov_x, fov_y = ann
        
        theta_deg = theta * 180 / np.pi
        phi_deg = phi * 180 / np.pi
        fov_x_deg = fov_x * 180 / np.pi
        fov_y_deg = fov_y * 180 / np.pi
        
        category_name = CATEGORY_NAMES.get(category_id, f'category_{category_id}')
        
        print(f"  绘制BFoV {i+1}: {category_name}")
        print(f"    中心: ({theta_deg:.2f}°, {phi_deg:.2f}°)")
        print(f"    FOV:  ({fov_x_deg:.2f}° x {fov_y_deg:.2f}°)")
        
        color = category_colors.get(category_id, (128, 128, 128))
        
        try:
            bfov = BFoV(img_w, img_h, 
                       view_angle_w=fov_x_deg, 
                       view_angle_h=fov_y_deg, 
                       long_side=img_w)
            
            px, py = bfov._sample_points(theta, phi, border_only=True)
            
            for j in range(px.shape[0]):
                x, y = int(px[j]), int(py[j])
                if 0 <= y < img_h and 0 <= x < img_w:
                    cv2.circle(img, (x, y), 1, color, -1, lineType=cv2.LINE_AA)
            
        except Exception as e:
            print(f"  绘制BFoV {i+1}失败: {e}")
            import traceback
            traceback.print_exc()
    
    return img


def main():
    img_path = './7PhJB.jpg'
    output_path_offset = './7PhJB_bfov_5dev_offset.jpg'    # 有偏离的BFoV
    output_path_no_offset = './7PhJB_bfov_5dev_nooffset.jpg'  # 没有偏离的BFoV
    
    if not os.path.exists(img_path):
        print(f"错误: 图像文件不存在: {img_path}")
        return
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"错误: 无法读取图像: {img_path}")
        return
    
    img_h, img_w = img.shape[:2]
    print(f"图像尺寸: {img_w} x {img_h}")
    print(f"有偏离的FOV参数数量: {len(SELECTED_FOV_PARAMS_OFFSET)}")
    print(f"没有偏离的FOV参数数量: {len(SELECTED_FOV_PARAMS_NO_OFFSET)}")
    
    # 生成有偏离的标注（使用3-6号FOV）
    annotations_offset = person_annotations_offset + chair_annotations_offset
    print(f"有偏离的标注数量: {len(annotations_offset)}")
    
    # 生成没有偏离的标注（使用5-9号FOV）
    annotations_no_offset = person_annotations_no_offset + chair_annotations_no_offset
    print(f"没有偏离的标注数量: {len(annotations_no_offset)}")
    
    # ========== 绘制第一幅图：有偏离的BFoV ==========
    print("\n=== 绘制有偏离的BFoV ===")
    img_with_offset = draw_bfov_on_image(img.copy(), annotations_offset, img_w, img_h)
    cv2.imwrite(output_path_offset, img_with_offset)
    print(f"有偏离的BFoV图像保存到: {os.path.abspath(output_path_offset)}")
    
    # ========== 绘制第二幅图：没有偏离的BFoV ==========
    print("\n=== 绘制没有偏离的BFoV ===")
    img_without_offset = draw_bfov_on_image(img.copy(), annotations_no_offset, img_w, img_h)
    cv2.imwrite(output_path_no_offset, img_without_offset)
    print(f"没有偏离的BFoV图像保存到: {os.path.abspath(output_path_no_offset)}")

if __name__ == '__main__':
    main()
