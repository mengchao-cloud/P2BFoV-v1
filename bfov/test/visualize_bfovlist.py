import torch
import numpy as np
import cv2
import os


# 导入GPUImageRecorder
from PANDORA.PRDA.lib.GPUImageRecorder import GPUImageRecorder
# 导入CPU版本的ImageRecorder用于验证
from PANDORA.PRDA.lib.ImageRecorder import ImageRecorder


def read_bfovlist(file_path, start_line=0, num_lines=42):
    """
    从指定行开始读取bfovlist.txt文件的num_lines行
    
    参数:
    file_path: 文件路径
    start_line: 开始读取的行号（从0开始），默认0
    num_lines: 读取的行数，默认42
    
    返回:
    bfov_list: 包含BFOV参数的列表
    """
    bfov_list = []
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            # 跳过前start_line行
            if i < start_line:
                continue
            # 读取指定数量的行后停止
            if i >= start_line + num_lines:
                break
            
            # 解析每行数据
            parts = line.strip().split()
            if len(parts) == 4:
                lon = float(parts[0])
                lat = float(parts[1])
                fov_x = float(parts[2])
                fov_y = float(parts[3])
                
                # 限制fov_x和fov_y在0到pi之间
                fov_x = max(0, min(np.pi, fov_x))
                fov_y = max(0, min(np.pi, fov_y))
                
                bfov_list.append([lon, lat, fov_x, fov_y])
    
    return bfov_list


def generate_bfov_masks_gpu(bfov_list, erp_w=1920, erp_h=960):
    """
    基于GPU加速的BFOV掩码生成函数
    
    参数:
    bfov_list: BFOV参数列表，每个元素是[lon, lat, fov_x, fov_y]
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    
    返回:
    mask_list: 掩码列表
    """
    # 准备GPU输入数据
    if bfov_list:
        # 将列表转换为张量
        bfov_tensor = torch.tensor(bfov_list, dtype=torch.float32).cuda()
        # 包装成列表格式（符合函数预期）
        gpu_input = [bfov_tensor]
    else:
        return []
    
    # 初始化结果列表
    mask_list = []
    
    # 遍历每个GT
    for gt_idx, gt_bfovs in enumerate(gpu_input):
        # 确保张量在GPU上
        if not gt_bfovs.is_cuda:
            gt_bfovs = gt_bfovs.cuda()
        
        # 获取当前GT的BFOV数量
        M_i = gt_bfovs.shape[0]
        
        # 初始化当前GT的掩码张量 [M_i, H, W]
        masks = torch.zeros((M_i, erp_h, erp_w), dtype=torch.uint8, device=gt_bfovs.device)
        
        # 批量处理当前GT的所有BFOV
        for i in range(M_i):
            # 获取当前BFOV参数
            lon, lat, fov_x, fov_y = gt_bfovs[i].tolist()
            
            # 创建GPUImageRecorder实例
            # 注意：这里的fov_x和fov_y是弧度制，需要转换为角度制
            fov_x_deg = np.degrees(fov_x)
            fov_y_deg = np.degrees(fov_y)
            
            # 限制fov值，避免过大导致内存问题
            max_fov = 180.0
            fov_x_deg = min(fov_x_deg, max_fov)
            fov_y_deg = min(fov_y_deg, max_fov)
            
            gpu_recorder = GPUImageRecorder(erp_w, erp_h, view_angle_w=fov_x_deg, view_angle_h=fov_y_deg, long_side=erp_w)
            
            # 生成采样点
            Px, Py = gpu_recorder._sample_points(lon, lat, border_only=False)
            
            # 将采样点坐标转换为整数
            Px = Px.to(torch.int32)
            Py = Py.to(torch.int32)
            
            # 确保坐标在有效范围内
            valid_mask = (Px >= 0) & (Px < erp_w) & (Py >= 0) & (Py < erp_h)
            valid_Px = Px[valid_mask].to(torch.long)
            valid_Py = Py[valid_mask].to(torch.long)
            
            # 将有效采样点标记为1
            masks[i, valid_Py, valid_Px] = 1
        
        # 将当前GT的掩码添加到结果列表
        mask_list.append(masks)
    
    return mask_list


def generate_bfov_masks_cpu(bfov_list, erp_w=1920, erp_h=960):
    """
    基于CPU的BFOV掩码生成函数（用于验证）
    
    参数:
    bfov_list: BFOV参数列表，每个元素是[lon, lat, fov_x, fov_y]
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    
    返回:
    mask_list: 掩码列表
    """
    # 准备CPU输入数据
    if bfov_list:
        # 将列表转换为数组
        bfov_array = np.array(bfov_list, dtype=np.float32)
        # 包装成列表格式（符合函数预期）
        cpu_input = [bfov_array]
    else:
        return []
    
    # 初始化结果列表
    mask_list = []
    
    # 遍历每个GT
    for gt_idx, gt_bfovs in enumerate(cpu_input):
        # 获取当前GT的BFOV数量
        M_i = gt_bfovs.shape[0]
        
        # 初始化当前GT的掩码列表
        masks = []
        
        # 处理当前GT的所有BFOV
        for i in range(M_i):
            # 获取当前BFOV参数
            lon, lat, fov_x, fov_y = gt_bfovs[i]
            
            # 创建CPU版本的ImageRecorder实例
            # 注意：这里的fov_x和fov_y是弧度制，需要转换为角度制
            fov_x_deg = np.degrees(fov_x)
            fov_y_deg = np.degrees(fov_y)
            
            # 限制fov值，避免过大导致内存问题
            max_fov = 180.0
            fov_x_deg = min(fov_x_deg, max_fov)
            fov_y_deg = min(fov_y_deg, max_fov)
            
            cpu_recorder = ImageRecorder(erp_w, erp_h, view_angle_w=fov_x_deg, view_angle_h=fov_y_deg, long_side=erp_w)
            
            # 生成采样点
            Px, Py = cpu_recorder._sample_points(lon, lat, border_only=False)
            
            # 将采样点坐标转换为整数
            Px = Px.astype(np.int32)
            Py = Py.astype(np.int32)
            
            # 确保坐标在有效范围内
            valid_mask = (Px >= 0) & (Px < erp_w) & (Py >= 0) & (Py < erp_h)
            valid_Px = Px[valid_mask]
            valid_Py = Py[valid_mask]
            
            # 创建掩码
            mask = np.zeros((erp_h, erp_w), dtype=np.uint8)
            mask[valid_Py, valid_Px] = 1
            masks.append(mask)
        
        # 将当前GT的掩码添加到结果列表
        mask_list.append(np.array(masks))
    
    return mask_list


def visualize_bfov_masks(bfov_list, mask_list, erp_w=1920, erp_h=960, output_dir='./visualization_results'):
    """
    可视化BFOV掩码列表
    
    参数:
    bfov_list: BFOV参数列表
    mask_list: 掩码列表
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    output_dir: 可视化图像输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取gt数量
    N = len(mask_list)
    
    # 遍历所有gt
    for i in range(N):
        # 创建可视化图像
        img = np.zeros((erp_h, erp_w, 3), dtype=np.uint8)
        img[:] = (255, 255, 255)  # 白色背景
        
        # 绘制分割线
        threshold = erp_w // 2
        cv2.line(img, (threshold, 0), (threshold, erp_h), (0, 0, 0), 2)
        
        # 不同bfov使用不同颜色
        colors = [
            (255, 0, 0),    # 红色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 蓝色
            (255, 255, 0),  # 黄色
            (255, 0, 255),  # 紫色
            (0, 255, 255),  # 青色
            (128, 0, 128),  # 紫色
            (128, 128, 0),  # 橄榄色
            (0, 128, 128),  # 青色
            (128, 128, 128)  # 灰色
        ]
        
        # 获取当前gt的掩码
        masks = mask_list[i]
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        M = masks.shape[0]
        
        # 遍历当前gt的所有bfov
        for j in range(M):
            # 获取当前bfov的掩码
            mask = masks[j]
            
            # 获取当前bfov的参数
            bfov_params = bfov_list[j]
            longitude, latitude, fov_x, fov_y = bfov_params
            
            # 计算中心点像素坐标
            center_x_rad = longitude
            center_y_rad = latitude
            center_px = int((center_x_rad + np.pi) / (2 * np.pi) * erp_w)
            center_py = int((np.pi/2 - center_y_rad) / np.pi * erp_h)
            
            # 使用循环颜色
            color = colors[j % len(colors)]
            
            # 绘制掩码
            img[mask == 1] = color
            
            # 绘制BFOV中心点
            cv2.circle(img, (center_px, center_py), 8, color, -1)
            
            # 添加bfov编号
            cv2.putText(img, f'BFOV {j+1}', (center_px + 10, center_py - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 保存可视化图像
        output_path = os.path.join(output_dir, f'bfovlist_visualization.jpg')
        cv2.imwrite(output_path, img)
        print(f"BFOV列表可视化已保存到: {output_path}")
    
    # 生成汇总图像（仅显示边界）
    summary_img = np.zeros((erp_h, erp_w, 3), dtype=np.uint8)
    summary_img[:] = (255, 255, 255)  # 白色背景
    
    # 绘制分割线
    threshold = erp_w // 2
    cv2.line(summary_img, (threshold, 0), (threshold, erp_h), (0, 0, 0), 2)
    
    # 遍历所有gt和bfov，绘制汇总图
    for i in range(N):
        # 获取当前gt的掩码
        masks = mask_list[i]
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        
        # 获取当前gt的bfov数量
        M = masks.shape[0]
        
        for j in range(M):
            mask = masks[j]
            
            # 使用循环颜色
            color = colors[j % len(colors)]
            
            # 绘制掩码边界
            # 计算掩码的边界
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # 绘制轮廓，使用细线
                cv2.drawContours(summary_img, contours, -1, color, 1)
    
    # 保存汇总图像
    summary_output_path = os.path.join(output_dir, f'bfovlist_boundaries.jpg')
    cv2.imwrite(summary_output_path, summary_img)
    print(f"BFOV列表边界可视化已保存到: {summary_output_path}")


def visualize_range(start_line, num_lines):
    """
    可视化bfovlist.txt的指定行范围数据
    
    参数:
    start_line: 开始读取的行号（从0开始）
    num_lines: 读取的行数
    """
    print(f"开始可视化bfovlist.txt的第{start_line+1}行到第{start_line+num_lines}行数据...")
    
    # 设置参数
    erp_w = 1920
    erp_h = 960
    bfovlist_path = './bfovlist.txt'
    
    # 读取bfovlist.txt文件
    print(f"读取{bfovlist_path}的第{start_line+1}行到第{start_line+num_lines}行数据...")
    bfov_list = read_bfovlist(bfovlist_path, start_line, num_lines)
    print(f"成功读取{len(bfov_list)}行数据")
    
    if not bfov_list:
        print("错误: 未读取到任何数据！")
        return
    
    # 检查是否有可用的GPU
    use_gpu = torch.cuda.is_available()
    print(f"是否使用GPU: {use_gpu}")
    
    # 生成掩码
    if use_gpu:
        print("使用GPU生成BFOV掩码...")
        masks = generate_bfov_masks_gpu(bfov_list, erp_w, erp_h)
    else:
        print("使用CPU生成BFOV掩码...")
        masks = generate_bfov_masks_cpu(bfov_list, erp_w, erp_h)
    
    if not masks:
        print("错误: 生成掩码失败！")
        return
    
    # 可视化掩码
    print("可视化BFOV掩码...")
    # 为每个范围创建单独的输出目录
    output_dir = f'./visualization_results_range_{start_line}_{start_line+num_lines}'
    visualize_bfov_masks(bfov_list, masks, erp_w, erp_h, output_dir)
    
    print(f"第{start_line+1}行到第{start_line+num_lines}行数据可视化完成！")


def main(ranges):
    """
    可视化bfovlist.txt的多个指定行范围数据
    
    参数:
    ranges: 包含多个(start_line, num_lines)元组的列表
    """
    print(f"开始可视化bfovlist.txt的{len(ranges)}个数据范围...")
    
    for i, (start_line, num_lines) in enumerate(ranges):
        print(f"\n处理第{i+1}个数据范围:")
        visualize_range(start_line, num_lines)
    
    print("\n所有数据范围可视化完成！")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化bfovlist.txt的多个指定行范围数据')
    parser.add_argument('--range', action='append', nargs=2, type=int, metavar=('START', 'COUNT'),
                       help='数据范围：开始行号（从0开始）和读取行数，可多次指定')
    parser.add_argument('--default-range', action='store_true', help='使用默认数据范围（0-42）')
    
    args = parser.parse_args()
    
    # 处理数据范围
    ranges = []
    if args.range:
        for start, count in args.range:
            ranges.append((start, count))
    elif args.default_range or not args.range:
        # 默认处理0-42行
        ranges.append((0, 42))
    
    if not ranges:
        print("错误: 未指定数据范围！")
        print("请使用 --range START COUNT 或 --default-range")
        exit(1)
    
    main(ranges)
