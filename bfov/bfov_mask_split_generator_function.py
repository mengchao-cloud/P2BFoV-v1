import numpy as np
import cv2
import sys
import os

# 添加PRDA/lib目录到Python路径
# sys.path.append(os.path.join(os.path.dirname(__file__), 'PRDA', 'lib'))
# from ImageRecorder import ImageRecorder
from PANDORA.PRDA.lib.ImageRecorder import ImageRecorder#文件lib当前的位置


def generate_bfov_masks_split(bfov_list, erp_w=1920, erp_h=960, threshold=None):
    """
    批量生成BFOV掩码，并将每个掩码分为左右两部分
    
    参数:
    bfov_list: BFOV参数列表，形状为[N, 5, 4]，其中：
              N是GT的数量，
              5是每个GT对应的BFOV数量，
              4是每个BFOV的参数[longitude, latitude, fov_x, fov_y]，单位：弧度
    erp_w: ERP图像宽度，默认1920
    erp_h: ERP图像高度，默认960
    threshold: 分割阈值，默认使用图像宽度的一半
    
    返回:
    mask_left_list: 左侧掩码列表，形状为[N, 5, erp_h, erp_w]
    mask_right_list: 右侧掩码列表，形状为[N, 5, erp_h, erp_w]
    """
    # 使用默认分割阈值（图像宽度的一半）
    if threshold is None:
        threshold = erp_w // 2
    
    N = len(bfov_list)  # GT数量
    mask_left_list = []
    mask_right_list = []
    
    # 遍历每个GT
    for i in range(N):
        gt_bfovs = bfov_list[i]  # 当前GT的5个BFOV
        gt_left_masks = []
        gt_right_masks = []
        
        # 遍历当前GT的每个BFOV
        for j in range(5):
            bfov_params = gt_bfovs[j]  # 当前BFOV参数
            longitude, latitude, fov_x, fov_y = bfov_params
            angle = 0  # 旋转角为0
            
            # 将fov从弧度转换为角度
            fov_x_deg = np.degrees(fov_x)
            fov_y_deg = np.degrees(fov_y)
            
            # 创建ImageRecorder实例
            BFoV = ImageRecorder(erp_w, erp_h, view_angle_w=fov_x_deg, view_angle_h=fov_y_deg, long_side=erp_w)
            
            # 生成BFOV掩码
            mask = np.zeros((erp_h, erp_w), dtype=np.uint8)
            
            # 获取BFOV区域内的所有采样点
            Px, Py = BFoV._sample_points(longitude, latitude, border_only=False)
            
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
            
            # 将掩码添加到当前GT的列表中
            gt_left_masks.append(left_mask)
            gt_right_masks.append(right_mask)
        
        # 将当前GT的掩码列表添加到总列表中
        mask_left_list.append(gt_left_masks)
        mask_right_list.append(gt_right_masks)
    
    # 转换为numpy数组
    mask_left_list = np.array(mask_left_list)
    mask_right_list = np.array(mask_right_list)
    
    return mask_left_list, mask_right_list


# 测试代码
if __name__ == '__main__':
    # 创建测试数据，形状为[N, 5, 4]
    # 示例：2个GT，每个GT有5个BFOV
    N = 2
    test_bfov_list = []
    
    for i in range(N):
        gt_bfovs = []
        for j in range(5):
            # 生成随机BFOV参数：longitude, latitude, fov_x, fov_y
            longitude = np.random.uniform(-np.pi, np.pi)
            latitude = np.random.uniform(-np.pi/2, np.pi/2)
            fov_x = np.random.uniform(np.pi/6, np.pi/2)  # 30°-90°
            fov_y = np.random.uniform(np.pi/6, np.pi/2)  # 30°-90°
            gt_bfovs.append([longitude, latitude, fov_x, fov_y])
        test_bfov_list.append(gt_bfovs)
    
    test_bfov_list = np.array(test_bfov_list)
    print(f"测试数据形状: {test_bfov_list.shape}")
    
    # 调用函数生成掩码
    mask_left, mask_right = generate_bfov_masks_split(test_bfov_list)
    
    print(f"左侧掩码形状: {mask_left.shape}")
    print(f"右侧掩码形状: {mask_right.shape}")
    print("掩码生成完成！")