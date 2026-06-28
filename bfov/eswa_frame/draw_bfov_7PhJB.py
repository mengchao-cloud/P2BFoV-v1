#!/usr/bin/env python3
"""
绘制图像 7PhJB.jpg 上的 BFoV 标注，并进行随机旋转

使用方法：
conda run -n pointobb python draw_bfov_7PhJB.py
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

# 图像 7PhJB.jpg 的 BFoV 标注（硬编码）
# 格式: [category_id, theta, phi, fov_x, fov_y]
# theta: 经度 (-π ~ π) - 弧度
# phi: 纬度 (-π/2 ~ π/2) - 弧度
# fov_x: 水平视场角（弧度）
# fov_y: 垂直视场角（弧度）
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
    (0.2612193871, 0.1306096936),
    (0.1847100000, 0.1847100000),
    (0.1306096936, 0.2612193871),
    (0.6791704065, 0.3395852032),
    (0.4802460000, 0.4802460000),
    (0.3395852032, 0.6791704065),
    (1.2016091807, 0.6008045903),
    (0.8496660000, 0.8496660000),
    (0.6008045903, 1.2016091807),
]

# person的中心点
PERSON_THETA = -0.0801760625134895
PERSON_PHI = -0.16853335589570229

# chair的中心点
CHAIR_THETA = 0.8492117641734908
CHAIR_PHI = -1.0750137361502572

# 生成person和chair的固定FOV标注
FIXED_BFOV_ANNOTATIONS = []
# person: category_id=34
for fov_x, fov_y in FIXED_FOV_PARAMS:
    FIXED_BFOV_ANNOTATIONS.append([34, PERSON_THETA, PERSON_PHI, fov_x, fov_y])
# chair: category_id=25
for fov_x, fov_y in FIXED_FOV_PARAMS:
    FIXED_BFOV_ANNOTATIONS.append([25, CHAIR_THETA, CHAIR_PHI, fov_x, fov_y])

# 类别名称映射（根据360indoor数据集标注文件）
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

def box_transform(box):
    """
    将BFoV参数从弧度制转换为角度制，并调整坐标格式
    
    参数:
    box: [theta_rad, phi_rad, fov_x_rad, fov_y_rad]
    - theta: -π ~ π（弧度）→ -180° ~ 180°（角度）
    - phi: -π/2 ~ π/2（弧度）→ -90° ~ 90°（角度）
    
    返回:
    转换后的box: [lon_deg, lat_deg, fov_x_deg, fov_y_deg]
    - lon_deg: 0° ~ 360°（经度）
    - lat_deg: 0° ~ 180°（纬度）
    """
    eps = 1
    # 弧度转角度
    lon_deg = box[0] * 180 / np.pi  # -180 ~ 180
    lat_deg = box[1] * 180 / np.pi  # -90 ~ 90
    fov_x_deg = box[2] * 180 / np.pi
    fov_y_deg = box[3] * 180 / np.pi
    
    # 调整坐标范围
    lon_deg = lon_deg + 180  # -180~180 转换为 0~360
    lat_deg = lat_deg + 90   # -90~90 转换为 0~180
    fov_x_deg = max(min(fov_x_deg, 180-eps), eps)
    fov_y_deg = max(min(fov_y_deg, 180-eps), eps)
    
    return [lon_deg, lat_deg, fov_x_deg, fov_y_deg]

def random_rotate_image_3d(img, max_angles=[30, 30, 30], rotate_ratio=1.0):
    """
    对ERP图像进行随机三维旋转
    
    参数:
    img: 输入图像
    max_angles: 三个旋转角度的最大值 [roll_max, pitch_max, yaw_max]，单位为度
    rotate_ratio: 比例系数，随机值范围为 [-max*ratio, max*ratio]
    
    返回:
    rotated_img: 旋转后的图像
    rotation_params: 使用的旋转参数 [roll, pitch, yaw]，单位为度
    """
    # 生成随机旋转角度
    roll = np.random.uniform(-max_angles[0], max_angles[0]) * rotate_ratio
    pitch = np.random.uniform(-max_angles[1], max_angles[1]) * rotate_ratio
    yaw = np.random.uniform(-max_angles[2], max_angles[2]) * rotate_ratio
    
    print(f"随机旋转参数: Roll={roll:.2f}°, Pitch={pitch:.2f}°, Yaw={yaw:.2f}°")
    
    # 应用三维旋转（参考 rotate_erp_3d.py，直接使用角度，不需要转换为弧度）
    rotated_img = img.copy()
    
    if abs(roll) > 1e-6:
        rotated_img = rotate_image(rotated_img, roll, np.array([1, 0, 0], dtype=np.float64))
    if abs(pitch) > 1e-6:
        rotated_img = rotate_image(rotated_img, pitch, np.array([0, 1, 0], dtype=np.float64))
    if abs(yaw) > 1e-6:
        rotated_img = rotate_image(rotated_img, yaw, np.array([0, 0, 1], dtype=np.float64))
    
    return rotated_img, [roll, pitch, yaw]

def transform_bfov_params(bfov_params, rotation_angles, erp_w, erp_h):
    """
    参考 rotate_erp_3d.py 中的 transform_rbfov_params 函数
    计算BFoV在三维旋转后的新参数（应用 Case 5 逻辑: Roll+, Pitch-, Yaw+）
    
    参数:
    bfov_params: [category_id, theta_rad, phi_rad, fov_x_rad, fov_y_rad]
                 theta: -π~π (弧度), phi: -π/2~π/2 (弧度)
    rotation_angles: [roll, pitch, yaw] 单位为度
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    
    返回:
    transformed_bfov: [category_id, new_theta_rad, new_phi_rad, fov_x_rad, fov_y_rad]
    """
    category_id, theta, phi, fov_x, fov_y = bfov_params
    roll, pitch, yaw = rotation_angles
    
    t = tools(erp_w, erp_h)
    
    # 将弧度转换为角度（与参考脚本一致）
    center_x = theta * 180 / np.pi  # -180 ~ 180
    center_y = phi * 180 / np.pi     # -90 ~ 90
    fov_x_deg = fov_x * 180 / np.pi
    fov_y_deg = fov_y * 180 / np.pi
    
    # ========== 1. 确定变换角度（Case 5: Pitch 取反）==========
    roll_t = roll
    pitch_t = -pitch  # 关键修正
    yaw_t = yaw
    
    # ========== 2. 变换中心点坐标 ==========
    # 计算原始中心点的像素坐标
    center_px = (center_x + 180) / 360 * erp_w
    center_py = (90 - center_y) / 180 * erp_h
    
    # 转为三维单位向量
    center_xyz = np.array(t.pxpy2xyz([center_px + 0.5, center_py + 0.5]))
    center_xyz = center_xyz / np.linalg.norm(center_xyz)
    
    # 应用旋转矩阵
    if abs(roll_t) > 1e-6:
        center_xyz = np.array(t.roll_T(np.array([1, 0, 0]), center_xyz, roll_t))
    if abs(pitch_t) > 1e-6:
        center_xyz = np.array(t.roll_T(np.array([0, 1, 0]), center_xyz, pitch_t))
    if abs(yaw_t) > 1e-6:
        center_xyz = np.array(t.roll_T(np.array([0, 0, 1]), center_xyz, yaw_t))
        
    center_xyz = center_xyz / np.linalg.norm(center_xyz)
    
    # 转回经纬度（角度）
    px_new, py_new = t.xyz2pxpy(center_xyz)
    new_center_x = (px_new / erp_w) * 360 - 180
    new_center_y = 90 - (py_new / erp_h) * 180
    
    # 转换回弧度
    new_theta = new_center_x * np.pi / 180
    new_phi = new_center_y * np.pi / 180
    
    return [category_id, new_theta, new_phi, fov_x, fov_y]

def draw_bfov_on_image(img, bfov_annotations, img_w, img_h, use_filter=False):
    """
    在图像上绘制多个BFoV
    
    参数:
    img: 输入图像
    bfov_annotations: BFoV标注列表，每个元素为 [category_id, theta, phi, fov_x, fov_y]
    img_w: 图像宽度
    img_h: 图像高度
    use_filter: 是否启用过滤逻辑（True: 只绘制person和chair，每个类别只绘制一个）
    
    返回:
    img: 绘制后的图像
    """
    # 为每个类别分配不同的颜色（BGR格式）
    category_colors = {
        1: (0, 255, 0),    # person - 绿色
        8: (0, 0, 255),    # chair - 红色
        21: (255, 0, 0),   # monitor - 蓝色
        25: (0, 255, 255), # bottle - 黄色
        28: (255, 0, 255), # wall - 品红色
        31: (255, 255, 0), # lamp - 青色
        34: (128, 0, 0),   # table - 深蓝色
        36: (128, 128, 0), # floor - 橄榄色
    }
    
    # 跟踪已绘制的类别（每个类别只绘制第一个）
    drawn_categories = set()
    
    # 绘制每个BFoV
    for i, ann in enumerate(bfov_annotations):
        category_id, theta, phi, fov_x, fov_y = ann
        
        # 转换为角度制用于显示
        theta_deg = theta * 180 / np.pi
        phi_deg = phi * 180 / np.pi
        fov_x_deg = fov_x * 180 / np.pi
        fov_y_deg = fov_y * 180 / np.pi
        
        # 检查FOV参数是否有效
        if fov_x_deg <= 0 or fov_y_deg <= 0:
            print(f"  跳过无效BFoV {i+1}: FOV参数为负")
            continue
        
        # 获取类别名称
        category_name = CATEGORY_NAMES.get(category_id, f'category_{category_id}')
        
        # 如果启用过滤，只绘制 person 和 chair 这两个类别
        if use_filter:
            if category_name not in ['person', 'chair']:
                print(f"  跳过BFoV {i+1}: {category_name}（该类别不需要绘制）")
                continue
            
            # 每个类别只绘制第一个标注
            if category_id in drawn_categories:
                print(f"  跳过BFoV {i+1}: {category_name}（该类别已绘制）")
                continue
            drawn_categories.add(category_id)
        
        print(f"  绘制BFoV {i+1}: {category_name}")
        print(f"    中心: ({theta_deg:.2f}°, {phi_deg:.2f}°)")
        print(f"    FOV:  ({fov_x_deg:.2f}° x {fov_y_deg:.2f}°)")
        
        # 选择颜色
        color = category_colors.get(category_id, (128, 128, 128))  # 默认灰色
        
        # 使用BFoV绘制
        try:
            # 创建BFoV实例（使用角度制FOV参数）
            bfov = BFoV(img_w, img_h, 
                       view_angle_w=fov_x_deg, 
                       view_angle_h=fov_y_deg, 
                       long_side=img_w)
            
            # 获取BFoV的边界点（直接使用弧度制坐标，theta: -π~π, phi: -π/2~π/2）
            px, py = bfov._sample_points(theta, phi, border_only=True)
            
            # 逐点绘制BFoV边界（不连接顶点，使用半径1的小圆点）
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
    # 配置参数
    img_path = './7PhJB.jpg'  # 图像路径
    output_path = './7PhJB_bfov.jpg'  # 输出路径（绘制BFoV后）
    rotated_output_path = './7PhJB_bfov_rotated.jpg'  # 旋转后的输出路径
    
    # 检查图像是否存在
    if not os.path.exists(img_path):
        print(f"错误: 图像文件不存在: {img_path}")
        return
    
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"错误: 无法读取图像: {img_path}")
        return
    
    img_h, img_w = img.shape[:2]
    print(f"图像尺寸: {img_w} x {img_h}")
    print(f"固定FOV标注数量: {len(FIXED_BFOV_ANNOTATIONS)}")
    
    # 在原始图像上绘制固定FOV的BFoV（不启用过滤，绘制所有BFoV）
    img_with_bfov = draw_bfov_on_image(img.copy(), FIXED_BFOV_ANNOTATIONS, img_w, img_h, use_filter=False)
    
    # 保存绘制BFoV后的图像
    cv2.imwrite(output_path, img_with_bfov)
    print(f"\n绘制BFoV后的图像保存到: {os.path.abspath(output_path)}")
    
    # ========== 新逻辑：先旋转图像，再变换BFoV参数，最后绘制到旋转后的图像上 ==========
    print("\n=== 开始随机三维旋转 ===")
    max_angles = [60, 60, 60]  # Roll, Pitch, Yaw 的最大角度（度）
    rotate_ratio = 1.0  # 比例系数
    rotated_img, rotation_params = random_rotate_image_3d(img.copy(), max_angles, rotate_ratio)
    
    # 变换BFoV参数（应用同样的旋转）
    print("\n=== 变换BFoV参数 ===")
    transformed_annotations = []
    for ann in FIXED_BFOV_ANNOTATIONS:
        transformed_ann = transform_bfov_params(ann, rotation_params, img_w, img_h)
        transformed_annotations.append(transformed_ann)
    
    # 在旋转后的图像上绘制变换后的BFoV（不启用过滤，绘制所有BFoV）
    rotated_img_with_bfov = draw_bfov_on_image(rotated_img.copy(), transformed_annotations, img_w, img_h, use_filter=False)
    
    # 保存旋转后并绘制BFoV的图像
    cv2.imwrite(rotated_output_path, rotated_img_with_bfov)
    print(f"旋转后并绘制BFoV的图像保存到: {os.path.abspath(rotated_output_path)}")
    
    # 显示统计信息
    print("\n=== 标注统计 ===")
    category_counts = {}
    for ann in BFoV_ANNOTATIONS:
        category_id = ann[0]
        category_counts[category_id] = category_counts.get(category_id, 0) + 1
    
    for category_id, count in sorted(category_counts.items()):
        category_name = CATEGORY_NAMES.get(category_id, f'category_{category_id}')
        print(f"  {category_name}: {count} 个标注")

if __name__ == '__main__':
    main()
