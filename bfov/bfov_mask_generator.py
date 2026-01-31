import numpy as np
import cv2
import sys
import os

# 添加PRDA/lib目录到Python路径
# sys.path.append(os.path.join(os.path.dirname(__file__), 'PRDA', 'lib'))
# from ImageRecorder import ImageRecorder
from PANDORA.PRDA.lib.ImageRecorder import ImageRecorder#文件lib当前的位置


def save_tensor_to_txt(tensor, output_path):
    """
    将张量保存为txt文件
    
    参数:
    tensor: 输入张量
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


def generate_bfov_mask(bfov_params, erp_w=1920, erp_h=960, threshold=None, return_center=False):
    """
    生成单个BFOV掩码，并将其分为左右两部分
    
    参数:
    bfov_params: BFOV参数 [longitude, latitude, fov_x, fov_y]，旋转角为0，角度制
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    threshold: 分割阈值，默认使用图像宽度的一半
    return_center: 是否返回中心点像素坐标
    
    返回:
    left_mask: 左侧掩码
    right_mask: 右侧掩码
    (可选) center_px, center_py: 中心点像素坐标
    """
    # 使用默认分割阈值（图像宽度的一半）
    if threshold is None:
        threshold = erp_w // 2
    
    # 提取BFOV参数，添加旋转角0
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
    
    # 计算中心点像素坐标
    center_px = int((center_x_rad + np.pi) / (2 * np.pi) * erp_w)
    center_py = int((np.pi/2 - center_y_rad) / np.pi * erp_h)
    
    if return_center:
        return left_mask, right_mask, center_px, center_py
    else:
        return left_mask, right_mask


def visualize_bfov_masks(bfov_list, mask_left_list, mask_right_list, erp_w=1920, erp_h=960, threshold=None, output_dir='./bfov_mask_visualizations'):
    """
    可视化BFOV掩码列表
    
    参数:
    bfov_list: BFOV参数列表，形状为[N, M, 4] 或列表形式
               N表示gt数量，M表示每个gt对应可变数量的bfov
    mask_left_list: 左侧掩码列表，列表形式，每个gt对应可变数量的掩码
    mask_right_list: 右侧掩码列表，列表形式，每个gt对应可变数量的掩码
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    threshold: 分割阈值，默认使用图像宽度的一半
    output_dir: 可视化图像输出目录
    """
    # 使用默认分割阈值（图像宽度的一半）
    if threshold is None:
        threshold = erp_w // 2
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取gt数量
    N = len(mask_left_list)
    
    # 遍历所有gt
    for i in range(N):
        # 创建可视化图像
        img = np.zeros((erp_h, erp_w, 3), dtype=np.uint8)
        img[:] = (255, 255, 255)  # 白色背景
        
        # 绘制分割线
        cv2.line(img, (threshold, 0), (threshold, erp_h), (0, 0, 0), 2)
        
        # 不同bfov使用不同颜色
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
        
        # 获取当前gt的bfov数量
        M = len(mask_left_list[i])
        
        # 遍历当前gt的所有bfov
        for j in range(M):
            # 获取当前bfov的左右掩码
            left_mask = mask_left_list[i][j]
            right_mask = mask_right_list[i][j]
            
            # 获取当前bfov的参数
            gt_bfov_params = bfov_list[i]
            bfov_params = gt_bfov_params[j] if hasattr(gt_bfov_params, '__getitem__') else gt_bfov_params[j]
            longitude, latitude, fov_x, fov_y = bfov_params
            
            # 计算中心点像素坐标
            center_x_rad = longitude / 180 * np.pi
            center_y_rad = latitude / 180 * np.pi
            center_px = int((center_x_rad + np.pi) / (2 * np.pi) * erp_w)
            center_py = int((np.pi/2 - center_y_rad) / np.pi * erp_h)
            
            # 使用循环颜色
            color = colors[j % len(colors)]
            
            # 绘制左侧掩码（使用对应颜色）
            img[left_mask == 1] = color
            
            # 绘制右侧掩码（使用对应颜色）
            img[right_mask == 1] = color
            
            # 绘制BFOV中心点
            cv2.circle(img, (center_px, center_py), 8, color, -1)
            
            # 添加bfov编号
            cv2.putText(img, f'BFOV {j+1}', (center_px + 10, center_py - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 保存可视化图像
        output_path = os.path.join(output_dir, f'gt_{i+1}_bfov_masks.jpg')
        cv2.imwrite(output_path, img)
        print(f"GT {i+1}的BFOV掩码可视化已保存到: {output_path}")
    
    # 生成所有gt的汇总图像
    summary_img = np.zeros((erp_h, erp_w, 3), dtype=np.uint8)
    summary_img[:] = (255, 255, 255)  # 白色背景
    
    # 绘制分割线
    cv2.line(summary_img, (threshold, 0), (threshold, erp_h), (0, 0, 0), 2)
    
    # 使用不同颜色区分不同gt
    gt_colors = [
        (255, 0, 0),    # 蓝色 - gt 0
        (0, 255, 0),    # 绿色 - gt 1
        (0, 0, 255),    # 红色 - gt 2
        (255, 255, 0),  # 黄色 - gt 3
        (255, 0, 255),  # 紫色 - gt 4
        (0, 255, 255),  # 青色 - gt 5
        (128, 0, 128),  # 紫色 - gt 6
        (128, 128, 0),  # 橄榄色 - gt 7
        (0, 128, 128),  # 青色 - gt 8
        (128, 128, 128)  # 灰色 - gt 9
    ]
    
    # 遍历所有gt和bfov，绘制汇总图
    for i in range(N):
        # 获取当前gt的bfov数量
        M = len(mask_left_list[i])
        
        for j in range(M):
            left_mask = mask_left_list[i][j]
            right_mask = mask_right_list[i][j]
            
            # 使用不同颜色区分不同gt
            gt_color = gt_colors[i % len(gt_colors)]
            
            # 绘制左侧掩码
            summary_img[left_mask == 1] = gt_color
            
            # 绘制右侧掩码
            summary_img[right_mask == 1] = gt_color
    
    # 保存汇总图像
    summary_output_path = os.path.join(output_dir, f'all_gt_bfov_masks.jpg')
    cv2.imwrite(summary_output_path, summary_img)
    print(f"所有GT的BFOV掩码汇总可视化已保存到: {summary_output_path}")


def generate_bfov_masks(bfov_list, erp_w=1920, erp_h=960, threshold=None, visualize=False, output_dir='./result/bfov_mask_visualizations', save_txt=False, txt_output_dir='./result/bfov_mask_tensors'):
    """
    生成BFOV掩码列表，并将每个掩码分为左右两部分
    
    参数:
    bfov_list: BFOV参数列表，形状为[N, M, 4] 或列表形式
               N表示gt数量，M表示每个gt对应可变数量的bfov，4表示每个bfov的参数(经度,纬度,水平视场,竖直视场)
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    threshold: 分割阈值，默认使用图像宽度的一半
    visualize: 是否生成可视化图像
    output_dir: 可视化图像输出目录
    save_txt: 是否将掩码张量保存到txt文件
    txt_output_dir: 掩码张量输出目录
    
    返回:
    mask_left_list: 左侧掩码列表，形状为列表形式，每个gt对应可变数量的掩码
    mask_right_list: 右侧掩码列表，形状为列表形式，每个gt对应可变数量的掩码
    """
    # 确保bfov_list是numpy数组
    if isinstance(bfov_list, list):
        bfov_list = np.array(bfov_list, dtype=object)
    
    # 获取gt数量
    N = bfov_list.shape[0]
    
    # 初始化掩码列表
    mask_left_list = []
    mask_right_list = []
    
    # 遍历所有gt
    for i in range(N):
        # 获取当前gt对应的bfov参数，数量可变
        gt_bfov_params = bfov_list[i]  # 形状为[M, 4]，M为当前gt的bfov数量
        
        gt_left_masks = []
        gt_right_masks = []
        
        # 获取当前gt的bfov数量
        M = gt_bfov_params.shape[0] if hasattr(gt_bfov_params, 'shape') else len(gt_bfov_params)
        
        # 遍历当前gt的所有bfov
        for j in range(M):
            bfov_params = gt_bfov_params[j]  # 形状为[4]
            
            # 生成当前bfov的左右掩码
            left_mask, right_mask = generate_bfov_mask(bfov_params, erp_w, erp_h, threshold)
            
            # 添加到当前gt的掩码列表
            gt_left_masks.append(left_mask)
            gt_right_masks.append(right_mask)
        
        # 将当前gt的掩码列表添加到总列表
        mask_left_list.append(gt_left_masks)
        mask_right_list.append(gt_right_masks)
    
    # 生成可视化图像
    if visualize:
        visualize_bfov_masks(bfov_list, mask_left_list, mask_right_list, erp_w, erp_h, threshold, output_dir)
    
    # 保存掩码张量到txt文件
    if save_txt:
        # 创建输出目录
        os.makedirs(txt_output_dir, exist_ok=True)
        
        # 遍历所有gt和bfov，保存每个掩码的张量
        for i in range(N):
            # 获取当前gt的bfov数量
            M = len(mask_left_list[i])
            
            for j in range(M):
                # 获取当前bfov的左右掩码
                left_mask = mask_left_list[i][j]
                right_mask = mask_right_list[i][j]
                
                # 扩展维度，使其形状为(1, H, W)
                left_mask_tensor = left_mask[np.newaxis, :, :]
                right_mask_tensor = right_mask[np.newaxis, :, :]
                
                # 保存左侧掩码张量
                left_output_path = os.path.join(txt_output_dir, f'gt_{i+1}_bfov_{j+1}_left_tensor.txt')
                save_tensor_to_txt(left_mask_tensor, left_output_path)
                print(f"GT {i+1} BFOV {j+1} 左侧掩码张量已保存到: {left_output_path}")
                
                # 保存右侧掩码张量
                right_output_path = os.path.join(txt_output_dir, f'gt_{i+1}_bfov_{j+1}_right_tensor.txt')
                save_tensor_to_txt(right_mask_tensor, right_output_path)
                print(f"GT {i+1} BFOV {j+1} 右侧掩码张量已保存到: {right_output_path}")
    
    return mask_left_list, mask_right_list


if __name__ == '__main__':
    # 示例用法 - 每个gt对应不同数量的bfov
    # 创建一个示例bfov_list，使用列表形式，每个gt对应不同数量的bfov
    bfov_list = []
    
    # 第1个gt - 3个bfov
    gt1_bfovs = []
    gt1_bfovs.append([0, 0, 80, 40])      # 经度0°，纬度0°，水平视场80°，竖直视场40°
    gt1_bfovs.append([90, 30, 90, 50])    # 经度90°，纬度30°，水平视场90°，竖直视场50°
    gt1_bfovs.append([180, 0, 100, 60])   # 经度180°，纬度0°，水平视场100°，竖直视场60°
    bfov_list.append(np.array(gt1_bfovs))
    
    # 第2个gt - 5个bfov
    gt2_bfovs = []
    gt2_bfovs.append([15, 15, 85, 42])     # 经度15°，纬度15°，水平视场85°，竖直视场42°
    gt2_bfovs.append([105, 45, 95, 55])    # 经度105°，纬度45°，水平视场95°，竖直视场55°
    gt2_bfovs.append([195, -15, 110, 65])  # 经度195°，纬度-15°，水平视场110°，竖直视场65°
    gt2_bfovs.append([285, -45, 75, 48])   # 经度285°，纬度-45°，水平视场75°，竖直视场48°
    gt2_bfovs.append([60, 75, 65, 33])     # 经度60°，纬度75°，水平视场65°，竖直视场33°
    bfov_list.append(np.array(gt2_bfovs))
    
    # 第3个gt - 2个bfov
    gt3_bfovs = []
    gt3_bfovs.append([-45, 20, 70, 35])    # 经度-45°，纬度20°，水平视场70°，竖直视场35°
    gt3_bfovs.append([135, -30, 85, 45])   # 经度135°，纬度-30°，水平视场85°，竖直视场45°
    bfov_list.append(np.array(gt3_bfovs))
    
    # 调用掩码生成函数，启用可视化和张量保存
    mask_left_list, mask_right_list = generate_bfov_masks(bfov_list, visualize=True, save_txt=True)
    
    # 打印结果信息
    N = len(bfov_list)
    print(f"gt数量: {N}")
    
    for i in range(N):
        M = len(mask_left_list[i])
        print(f"GT {i+1}对应的bfov数量: {M}")
        print(f"GT {i+1}左侧掩码列表长度: {len(mask_left_list[i])}")
        print(f"GT {i+1}右侧掩码列表长度: {len(mask_right_list[i])}")
        print(f"GT {i+1}单个掩码形状: {mask_left_list[i][0].shape}")