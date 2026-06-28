#!/usr/bin/env python3
"""
绘制图像 7PhJB.jpg 上的 person 和 chair 的原始 BFoV 标注及抖动效果图
参考 P2BFoV.py 中的提案抖动逻辑

使用方法：
conda run -n pointobb python draw_bfov_7PhJB_orig.py
"""

import os
import sys
import numpy as np
import cv2
import math

# 添加sphdet路径到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../sphdet/visualizers'))
from ImageRecorder import ImageRecorder as BFoV

# 图像 7PhJB.jpg 的原始 BFoV 标注（硬编码）
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

# 抖动参数（参考P2BFoV.py）
SHAKE_RATIO = 1.4  # 抖动比例（固定值）

# 扩展提案参数
SCALE_FACTORS = [2.0, 4.0]  # 面积尺度因子
ASPECT_RATIOS = [0.5, 1.0, 2.0]  # fovx/fovy比例

def constrain_spherical_coords(coords):
    """
    约束球面坐标在有效范围内
    参考 P2BFoV.py 中的实现
    
    参数:
    coords: 坐标张量，形状 [..., 2]，其中[...,0]是经度(lambda), [...,1]是纬度(phi)
    
    返回:
    约束后的坐标
    """
    # coords[..., 0] 是经度 lambda: [-π, π]
    # coords[..., 1] 是纬度 phi: [-π/2, π/2]
    
    # 获取经度和纬度
    lambda_coords = coords[..., 0]
    phi_coords = coords[..., 1]
    
    # 约束纬度 phi: [-π/2, π/2]
    # 使用取余运算处理纬度跨越
    phi_coords = np.mod(phi_coords + np.pi/2, np.pi) - np.pi/2
    
    # 约束经度 lambda: [-π, π]
    # 使用取余运算处理经度跨越
    lambda_coords = np.mod(lambda_coords + np.pi, 2*np.pi) - np.pi
    
    # 复制coords并更新
    constrained_coords = coords.copy()
    constrained_coords[..., 0] = lambda_coords
    constrained_coords[..., 1] = phi_coords
    
    return constrained_coords


def expand_proposals(proposals, scale_factors, aspect_ratios):
    """
    对BFoV提案进行尺度扩展
    
    参数:
    proposals: 提案列表，每个元素为 [category_id, theta, phi, fov_x, fov_y]
    scale_factors: 面积尺度因子列表（相对于当前面积）
    aspect_ratios: fovx/fovy比例列表
    
    返回:
    expanded_proposals: 扩展后的提案列表
    """
    expanded_proposals = []
    
    for proposal in proposals:
        category_id, theta, phi, fov_x, fov_y = proposal
        
        # 当前面积
        current_area = fov_x * fov_y
        
        # 生成所有尺度和比例的组合
        for scale in scale_factors:
            for aspect_ratio in aspect_ratios:
                # 新面积 = scale * 当前面积
                new_area = scale * current_area
                
                # 根据比例计算新的fov_x和fov_y
                # fov_x / fov_y = aspect_ratio
                # fov_x * fov_y = new_area
                # => fov_x = sqrt(aspect_ratio * new_area)
                # => fov_y = fov_x / aspect_ratio
                new_fov_x = np.sqrt(aspect_ratio * new_area)
                new_fov_y = new_fov_x / aspect_ratio
                
                # 确保FOV不超过pi
                new_fov_x = min(new_fov_x, np.pi - 1e-6)
                new_fov_y = min(new_fov_y, np.pi - 1e-6)
                
                expanded_proposals.append([category_id, theta, phi, new_fov_x, new_fov_y])
    
    return expanded_proposals


def shake_bfov_proposals_horizontal(proposals, ratio):
    """
    对BFoV提案进行水平抖动（左/右）（参考P2BFoV.py的fine_proposals_from_cfg函数）
    
    参数:
    proposals: 提案列表，每个元素为 [category_id, theta, phi, fov_x, fov_y]
    ratio: 抖动比例（单个值）
    
    返回:
    shaken_proposals: 抖动后的提案列表，只包含左右抖动提案（不包含原始提案）
    """
    shaken_proposals = []
    
    for proposal in proposals:
        category_id, lambda0, phi0, fov_x, fov_y = proposal
        
        # ========== 水平抖动（左/右）：沿着纬线方向 ==========
        alpha = ratio * fov_x  # 水平抖动角度
        
        # 计算新纬度 phi2
        sin_phi2 = np.sin(phi0) * np.cos(alpha)
        # 限制范围避免浮点误差
        sin_phi2 = np.clip(sin_phi2, -1.0 + 1e-6, 1.0 - 1e-6)
        phi2 = np.arcsin(sin_phi2)
        
        # 计算经度变化量 delta_lambda
        cos_phi0 = np.cos(phi0) + 1e-8  # 避免除零
        delta_lambda = np.arctan2(np.sin(alpha), cos_phi0 * np.cos(alpha))
        
        # 左抖动和右抖动
        lambda2_l = lambda0 + delta_lambda  # 向左
        lambda2_r = lambda0 - delta_lambda  # 向右
        
        # 创建左右抖动后的坐标
        shaken_centers = [
            [lambda2_l, phi2],   # 左
            [lambda2_r, phi2],   # 右
        ]
        
        # 约束坐标到有效范围
        for shaken_center in shaken_centers:
            shaken_center = constrain_spherical_coords(np.array([shaken_center]))[0]
            shaken_proposals.append([category_id, shaken_center[0], shaken_center[1], fov_x, fov_y])
    
    return shaken_proposals


def shake_bfov_proposals_vertical(proposals, ratio):
    """
    对BFoV提案进行垂直抖动（上/下）（参考P2BFoV.py的fine_proposals_from_cfg函数）
    
    参数:
    proposals: 提案列表，每个元素为 [category_id, theta, phi, fov_x, fov_y]
    ratio: 抖动比例（单个值）
    
    返回:
    shaken_proposals: 抖动后的提案列表，只包含上下抖动提案（不包含原始提案）
    """
    shaken_proposals = []
    
    for proposal in proposals:
        category_id, lambda0, phi0, fov_x, fov_y = proposal
        
        # ========== 垂直抖动（上/下）：沿着经线方向 ==========
        phi_t = phi0 + ratio * fov_y  # 向上
        phi_d = phi0 - ratio * fov_y  # 向下
        
        # 创建上下抖动后的坐标
        shaken_centers = [
            [lambda0, phi_t],    # 上
            [lambda0, phi_d],    # 下
        ]
        
        # 约束坐标到有效范围
        for shaken_center in shaken_centers:
            shaken_center = constrain_spherical_coords(np.array([shaken_center]))[0]
            shaken_proposals.append([category_id, shaken_center[0], shaken_center[1], fov_x, fov_y])
    
    return shaken_proposals


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
        25: (0, 255, 255), # chair - 黄色
        34: (128, 0, 0),   # person - 深蓝色
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
    
    # 创建输出文件夹
    output_dir = './bfov_shaken_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出文件夹: {os.path.abspath(output_dir)}")
    
    output_path_orig = os.path.join(output_dir, '7PhJB_bfov_orig.jpg')
    output_path_horizontal = os.path.join(output_dir, '7PhJB_bfov_shaken_horizontal.jpg')
    output_path_vertical = os.path.join(output_dir, '7PhJB_bfov_shaken_vertical.jpg')
    
    if not os.path.exists(img_path):
        print(f"错误: 图像文件不存在: {img_path}")
        return
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"错误: 无法读取图像: {img_path}")
        return
    
    img_h, img_w = img.shape[:2]
    print(f"图像尺寸: {img_w} x {img_h}")
    
    # 只绘制一个 person (34) 和一个 chair (25) 的原始标注
    # 将fov_x和fov_y都乘以0.5（缩小一半）
    person_ann = BFoV_ANNOTATIONS[13].copy()
    chair_ann = BFoV_ANNOTATIONS[3].copy()
    
    # person的FOV缩小一半
    person_ann[3] *= 0.5  # fov_x
    person_ann[4] *= 0.5  # fov_y
    
    # chair的FOV缩小一半
    chair_ann[3] *= 0.5  # fov_x
    chair_ann[4] *= 0.5  # fov_y
    
    selected_annotations = [
        person_ann,   # person (FOV缩小一半)
        chair_ann,    # chair (FOV缩小一半)
    ]
    
    print(f"原始标注总数: {len(BFoV_ANNOTATIONS)}")
    print(f"选择的标注数量: {len(selected_annotations)} (1 person + 1 chair)")
    print(f"抖动比例: {SHAKE_RATIO}")
    print(f"尺度因子: {SCALE_FACTORS}")
    print(f"宽高比: {ASPECT_RATIOS}")
    
    # ========== 绘制原始BFoV（带扩展提案）==========
    print("\n=== 绘制原始BFoV（带扩展提案）===")
    # 对原始提案进行尺度扩展
    expanded_orig_proposals = expand_proposals(selected_annotations, SCALE_FACTORS, ASPECT_RATIOS)
    print(f"扩展后提案数: {len(expanded_orig_proposals)} (每个原始提案 × {len(SCALE_FACTORS)}尺度 × {len(ASPECT_RATIOS)}比例)")
    
    img_with_orig = draw_bfov_on_image(img.copy(), expanded_orig_proposals, img_w, img_h)
    cv2.imwrite(output_path_orig, img_with_orig)
    print(f"原始BFoV扩展图像保存到: {os.path.abspath(output_path_orig)}")
    
    # ========== 对每个BFoV进行水平抖动（左/右）==========
    print("\n=== 对每个BFoV进行水平抖动（左/右）===")
    all_horizontal_shaken = []
    
    for i, proposal in enumerate(selected_annotations):
        category_id = proposal[0]
        category_name = CATEGORY_NAMES.get(category_id, f'category_{category_id}')
        print(f"\n原始提案 {i+1}: {category_name}")
        print(f"  中心: ({proposal[1]*180/np.pi:.2f}°, {proposal[2]*180/np.pi:.2f}°)")
        print(f"  FOV:  ({proposal[3]*180/np.pi:.2f}° x {proposal[4]*180/np.pi:.2f}°)")
        
        # 使用原始FOV进行抖动（得到抖动后的中心位置）
        shaken_centers = shake_bfov_proposals_horizontal([proposal], SHAKE_RATIO)
        
        # 对每个抖动后的中心，应用所有尺度扩展
        for shaken_center in shaken_centers:
            shaken_proposal = shaken_center.copy()
            # 扩展提案：保持抖动后的中心，改变FOV大小
            expanded_proposals = expand_proposals([shaken_proposal], SCALE_FACTORS, ASPECT_RATIOS)
            all_horizontal_shaken.extend(expanded_proposals)
        
        print(f"  左右抖动提案数: {len(shaken_centers) * len(SCALE_FACTORS) * len(ASPECT_RATIOS)} (2方向 × {len(SCALE_FACTORS)}尺度 × {len(ASPECT_RATIOS)}比例)")
    
    # ========== 绘制水平抖动后的BFoV ==========
    print(f"\n=== 绘制水平抖动后的BFoV ===")
    print(f"总提案数: {len(all_horizontal_shaken)}")
    
    img_with_horizontal = draw_bfov_on_image(img.copy(), all_horizontal_shaken, img_w, img_h)
    cv2.imwrite(output_path_horizontal, img_with_horizontal)
    print(f"水平抖动后BFoV图像保存到: {os.path.abspath(output_path_horizontal)}")
    
    # ========== 对每个BFoV进行垂直抖动（上/下）==========
    print("\n=== 对每个BFoV进行垂直抖动（上/下）===")
    all_vertical_shaken = []
    
    for i, proposal in enumerate(selected_annotations):
        category_id = proposal[0]
        category_name = CATEGORY_NAMES.get(category_id, f'category_{category_id}')
        print(f"\n原始提案 {i+1}: {category_name}")
        print(f"  中心: ({proposal[1]*180/np.pi:.2f}°, {proposal[2]*180/np.pi:.2f}°)")
        print(f"  FOV:  ({proposal[3]*180/np.pi:.2f}° x {proposal[4]*180/np.pi:.2f}°)")
        
        # 使用原始FOV进行抖动（得到抖动后的中心位置）
        shaken_centers = shake_bfov_proposals_vertical([proposal], SHAKE_RATIO)
        
        # 对每个抖动后的中心，应用所有尺度扩展
        for shaken_center in shaken_centers:
            shaken_proposal = shaken_center.copy()
            # 扩展提案：保持抖动后的中心，改变FOV大小
            expanded_proposals = expand_proposals([shaken_proposal], SCALE_FACTORS, ASPECT_RATIOS)
            all_vertical_shaken.extend(expanded_proposals)
        
        print(f"  上下抖动提案数: {len(shaken_centers) * len(SCALE_FACTORS) * len(ASPECT_RATIOS)} (2方向 × {len(SCALE_FACTORS)}尺度 × {len(ASPECT_RATIOS)}比例)")
    
    # ========== 绘制垂直抖动后的BFoV ==========
    print(f"\n=== 绘制垂直抖动后的BFoV ===")
    print(f"总提案数: {len(all_vertical_shaken)}")
    
    img_with_vertical = draw_bfov_on_image(img.copy(), all_vertical_shaken, img_w, img_h)
    cv2.imwrite(output_path_vertical, img_with_vertical)
    print(f"垂直抖动后BFoV图像保存到: {os.path.abspath(output_path_vertical)}")
    
    print("\n处理完成！")

if __name__ == '__main__':
    main()
