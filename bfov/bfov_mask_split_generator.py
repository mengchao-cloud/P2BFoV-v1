import numpy as np
import cv2
import sys
import os

# 添加PRDA/lib目录到Python路径
# sys.path.append(os.path.join(os.path.dirname(__file__), 'PRDA', 'lib'))#lib之前的位置

# from ImageRecorder import ImageRecorder
from PANDORA.PRDA.lib.ImageRecorder import ImageRecorder#文件lib当前的位置

def generate_split_bfov_mask(bfov_params, erp_w=1920, erp_h=960, threshold=None, output_image_path='./result/bfov_mask_split_result.jpg', output_left_tensor_path='./result/bfov_mask_left_tensor.txt', output_right_tensor_path='./result/bfov_mask_right_tensor.txt'):
    """
    生成跨越左右边界的BFOV掩码，并将其分为左右两部分保存
    
    参数:
    bfov_params: BFOV参数 [center_x, center_y, fov_x, fov_y, angle]，角度制
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    threshold: 分割阈值，默认使用图像宽度的一半
    output_image_path: 掩码图像输出路径
    output_left_tensor_path: 左侧掩码张量输出路径
    output_right_tensor_path: 右侧掩码张量输出路径
    """
    # 使用默认分割阈值（图像宽度的一半）
    if threshold is None:
        threshold = erp_w // 2
    
    # 提取BFOV参数
    center_x, center_y, fov_x, fov_y, angle = bfov_params
    print(f"BFOV参数: center_x={center_x}, center_y={center_y}, fov_x={fov_x}, fov_y={fov_y}, angle={angle}")
    print(f"分割阈值: {threshold}")
    
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
    
    print(f"采样点总数: {len(Px)}")
    print(f"有效采样点: {len(valid_Px)}")
    
    # 将有效采样点标记为1
    mask[valid_Py, valid_Px] = 1
    
    # 将掩码分为左右两部分
    # 左侧掩码：x < threshold
    left_mask = mask.copy()
    left_mask[:, threshold:] = 0
    
    # 右侧掩码：x >= threshold
    right_mask = mask.copy()
    right_mask[:, :threshold] = 0
    
    # 生成可视化图像
    img = np.zeros((erp_h, erp_w, 3), dtype=np.uint8)
    img[:] = (255, 255, 255)  # 白色背景
    
    # 绘制左侧掩码（蓝色）
    img[left_mask == 1] = (255, 0, 0)
    
    # 绘制右侧掩码（红色）
    img[right_mask == 1] = (0, 0, 255)
    
    # 绘制分割线
    cv2.line(img, (threshold, 0), (threshold, erp_h), (0, 0, 0), 2)
    
    # 绘制BFOV中心点
    center_px = int((center_x_rad + np.pi) / (2 * np.pi) * erp_w)
    center_py = int((np.pi/2 - center_y_rad) / np.pi * erp_h)
    cv2.circle(img, (center_px, center_py), 8, (0, 255, 0), -1)
    
    print(f"中心点坐标: ({center_px}, {center_py})")
    
    # 保存掩码图像
    cv2.imwrite(output_image_path, img)
    print(f"BFOV分割掩码图像已保存到: {output_image_path}")
    
    # 保存左侧掩码张量
    left_mask_tensor = left_mask[np.newaxis, :, :]
    save_tensor_to_txt(left_mask_tensor, output_left_tensor_path)
    print(f"左侧BFOV掩码张量已保存到: {output_left_tensor_path}")
    
    # 保存右侧掩码张量
    right_mask_tensor = right_mask[np.newaxis, :, :]
    save_tensor_to_txt(right_mask_tensor, output_right_tensor_path)
    print(f"右侧BFOV掩码张量已保存到: {output_right_tensor_path}")
    
    return left_mask, right_mask


def save_tensor_to_txt(tensor, output_path):
    """
    将张量保存为txt文件
    
    参数:
    tensor: 输入张量，形状为(1, H, W)
    output_path: 输出文件路径
    """
    with open(output_path, 'w') as f:
        # 写入形状信息
        f.write(f"Shape: {tensor.shape}\n")
        # 写入张量数据，按行展开
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                row_str = ' '.join(map(str, tensor[i, j, :]))
                f.write(f"{row_str}\n")


if __name__ == '__main__':
    # BFOV参数 [center_x, center_y, fov_x, fov_y, angle]，角度制
    # 这个BFOV跨越左右边界
    bfov_params = [0, 40, 80, 40, 0]
    
    # 调用掩码生成函数
    generate_split_bfov_mask(bfov_params)
