import numpy as np
import sys
import os

# 添加PRDA/lib目录到Python路径
# sys.path.append(os.path.join(os.path.dirname(__file__), 'PRDA', 'lib'))
# from ImageRecorder import ImageRecorder
from PANDORA.PRDA.lib.ImageRecorder import ImageRecorder#文件lib当前的位置


def generate_bfov_masks(bfov_list, erp_w=1920, erp_h=960, threshold=None):
    """
    批量生成BFOV掩码列表，并将每个掩码分为左右两部分
    
    参数:
    bfov_list: BFOV参数列表，形状为[N, M, 4] 或列表形式
               N表示gt数量，M表示每个gt对应可变数量的bfov，4表示每个bfov的参数(经度,纬度,水平视场,竖直视场)
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    threshold: 分割阈值，默认使用图像宽度的一半
    
    返回:
    mask_left_list: 左侧掩码列表，形状为列表形式，每个gt对应可变数量的掩码
    mask_right_list: 右侧掩码列表，形状为列表形式，每个gt对应可变数量的掩码
    """
    # 使用默认分割阈值（图像宽度的一半）
    if threshold is None:
        threshold = erp_w // 2
    
    # 确保bfov_list是numpy数组或列表
    if isinstance(bfov_list, list):
        # 如果是列表形式，直接使用
        pass
    elif hasattr(bfov_list, 'shape'):
        # 如果是numpy数组，转换为列表处理
        bfov_list = [bfov_list[i] for i in range(bfov_list.shape[0])]
    
    # 获取gt数量
    N = len(bfov_list)
    
    # 初始化掩码列表
    mask_left_list = []
    mask_right_list = []
    
    # 遍历所有gt
    for i in range(N):
        # 获取当前gt对应的bfov参数，数量可变
        gt_bfov_params = bfov_list[i]
        
        gt_left_masks = []
        gt_right_masks = []
        
        # 获取当前gt的bfov数量
        M = gt_bfov_params.shape[0] if hasattr(gt_bfov_params, 'shape') else len(gt_bfov_params)
        
        # 遍历当前gt的所有bfov
        for j in range(M):
            # 提取当前bfov的参数
            bfov_params = gt_bfov_params[j]
            longitude, latitude, fov_x, fov_y = bfov_params
            center_x = longitude
            center_y = latitude
            angle = 0  # 旋转角为0
            
            # 创建ImageRecorder实例
            BFoV = ImageRecorder(erp_w, erp_h, view_angle_w=fov_x, view_angle_h=fov_y, long_side=erp_w)
            
            # 将中心点角度转换为弧度制
            center_x_rad = center_x / 180 * np.pi
            center_y_rad = center_y / 180 * np.pi
            
            # 生成BFOV掩码
            mask = np.zeros((erp_h, erp_w), dtype=np.uint8)
            
            # 获取BFOV区域内的所有采样点
            Px, Py = BFoV._sample_points(center_x_rad, center_y_rad, border_only=False)
            
            # 将采样点坐标转换为整数
            Px = Px.astype(np.int32)
            Py = Py.astype(np.int32)
            
            # 确保坐标在有效范围内
            valid_mask = (Px >= 0) & (Px < erp_w) & (Py >= 0) & (Py < erp_h)
            valid_Px = Px[valid_mask]
            valid_Py = Py[valid_mask]
            
            # 将有效采样点标记为1
            mask[valid_Py, valid_Px] = 1
            
            # 将掩码分为左右两部分
            # 左侧掩码：x < threshold
            left_mask = mask.copy()
            left_mask[:, threshold:] = 0
            
            # 右侧掩码：x >= threshold
            right_mask = mask.copy()
            right_mask[:, :threshold] = 0
            
            # 添加到当前gt的掩码列表
            gt_left_masks.append(left_mask)
            gt_right_masks.append(right_mask)
        
        # 将当前gt的掩码列表添加到总列表
        mask_left_list.append(gt_left_masks)
        mask_right_list.append(gt_right_masks)
    
    return mask_left_list, mask_right_list


if __name__ == '__main__':
    # 示例用法
    # 创建一个示例bfov_list，使用列表形式，每个gt对应不同数量的bfov
    bfov_list = []
    
    # 第1个gt - 3个bfov
    gt1_bfovs = np.array([
        [0, 0, 80, 40],      # 经度0°，纬度0°，水平视场80°，竖直视场40°
        [90, 30, 90, 50],    # 经度90°，纬度30°，水平视场90°，竖直视场50°
        [180, 0, 100, 60]    # 经度180°，纬度0°，水平视场100°，竖直视场60°
    ])
    bfov_list.append(gt1_bfovs)
    
    # 第2个gt - 2个bfov
    gt2_bfovs = np.array([
        [15, 15, 85, 42],     # 经度15°，纬度15°，水平视场85°，竖直视场42°
        [105, 45, 95, 55]     # 经度105°，纬度45°，水平视场95°，竖直视场55°
    ])
    bfov_list.append(gt2_bfovs)
    
    # 调用掩码生成函数
    mask_left_list, mask_right_list = generate_bfov_masks(bfov_list, erp_w=1920, erp_h=960)
    
    # 打印结果信息
    N = len(bfov_list)
    print(f"gt数量: {N}")
    
    for i in range(N):
        M = len(mask_left_list[i])
        print(f"GT {i+1}对应的bfov数量: {M}")
        print(f"GT {i+1}左侧掩码列表长度: {len(mask_left_list[i])}")
        print(f"GT {i+1}右侧掩码列表长度: {len(mask_right_list[i])}")
        print(f"GT {i+1}单个掩码形状: {mask_left_list[i][0].shape}")
