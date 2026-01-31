import torch
import numpy as np
import time
import cv2
import os

# 导入GPUImageRecorder
from PANDORA.PRDA.lib.GPUImageRecorder import GPUImageRecorder
# 导入CPU版本的ImageRecorder用于验证
from PANDORA.PRDA.lib.ImageRecorder import ImageRecorder


def generate_bfov_masks_gpu(bfov_list, erp_w=1920, erp_h=960, threshold=None):
    """
    基于GPU加速的BFOV掩码生成函数
    使用GPUImageRecorder类，与CPU版本的ImageRecorder行为一致
    
    参数:
    bfov_list: 长度为N的列表，每个元素是[M_i,4]的张量（弧度制）
               N是GT的数量，M_i是每个GT对应的BFOV数量
               每个BFOV参数: [longitude, latitude, fov_x, fov_y]
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    threshold: 分割阈值，默认使用图像宽度的一半
    
    返回:
    mask_list: 长度为N的列表，每个元素是[M_i, H, W]的张量
               H=erp_h, W=erp_w
    """
    # 使用默认分割阈值（图像宽度的一半）
    if threshold is None:
        threshold = erp_w // 2
    
    # 初始化结果列表
    mask_list = []
    
    # 遍历每个GT
    for gt_idx, gt_bfovs in enumerate(bfov_list):
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


def generate_bfov_masks_cpu(bfov_list, erp_w=1920, erp_h=960, threshold=None):
    """
    基于CPU的BFOV掩码生成函数（用于验证）
    使用CPU版本的ImageRecorder类，与GPU版本保持一致的逻辑
    
    参数:
    bfov_list: 长度为N的列表，每个元素是[M_i,4]的张量或数组（弧度制）
    erp_w: ERP图像宽度
    erp_h: ERP图像高度
    threshold: 分割阈值，默认使用图像宽度的一半
    
    返回:
    mask_list: 长度为N的列表，每个元素是[M_i, H, W]的数组
    """
    # 使用默认分割阈值（图像宽度的一半）
    if threshold is None:
        threshold = erp_w // 2
    
    # 初始化结果列表
    mask_list = []
    
    # 遍历每个GT
    for gt_idx, gt_bfovs in enumerate(bfov_list):
        # 确保数据在CPU上
        if isinstance(gt_bfovs, torch.Tensor):
            gt_bfovs = gt_bfovs.cpu().numpy()
        
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


def validate_results(gpu_results, cpu_results):
    """
    验证GPU和CPU结果是否一致
    
    参数:
    gpu_results: GPU版本生成的掩码列表
    cpu_results: CPU版本生成的掩码列表
    
    返回:
    bool: 结果是否一致
    """
    if len(gpu_results) != len(cpu_results):
        print(f"结果长度不一致: GPU={len(gpu_results)}, CPU={len(cpu_results)}")
        return False
    
    for i, (gpu_masks, cpu_masks) in enumerate(zip(gpu_results, cpu_results)):
        # 将GPU张量转换为CPU数组
        gpu_masks_np = gpu_masks.cpu().numpy()
        
        if gpu_masks_np.shape != cpu_masks.shape:
            print(f"GT {i} 掩码形状不一致: GPU={gpu_masks_np.shape}, CPU={cpu_masks.shape}")
            return False
        
        # 计算差异
        diff = np.abs(gpu_masks_np - cpu_masks)
        if np.any(diff > 0):
            print(f"GT {i} 掩码内容不一致，差异像素数: {np.sum(diff)}")
            return False
    
    print("所有结果验证通过！GPU和CPU生成的掩码完全一致。")
    return True


def visualize_bfov_masks_test(bfov_list, mask_list, erp_w=1920, erp_h=960, threshold=None, output_dir='./result/bfov_mask_visualizations'):
    """
    可视化BFOV掩码列表（测试版本）
    
    参数:
    bfov_list: BFOV参数列表，长度为N的列表，每个元素是[M_i,4]的张量或数组
    mask_list: 掩码列表，长度为N的列表，每个元素是[M_i, H, W]的张量或数组
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
    N = len(mask_list)
    
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
        masks = mask_list[i]
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        M = masks.shape[0]
        
        # 遍历当前gt的所有bfov
        for j in range(M):
            # 获取当前bfov的掩码
            mask = masks[j]
            
            # 获取当前bfov的参数
            gt_bfov_params = bfov_list[i]
            if isinstance(gt_bfov_params, torch.Tensor):
                gt_bfov_params = gt_bfov_params.cpu().numpy()
            bfov_params = gt_bfov_params[j]
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
        # 获取当前gt的掩码
        masks = mask_list[i]
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        
        # 获取当前gt的bfov数量
        M = masks.shape[0]
        
        for j in range(M):
            mask = masks[j]
            
            # 使用不同颜色区分不同gt
            gt_color = gt_colors[i % len(gt_colors)]
            
            # 绘制掩码
            summary_img[mask == 1] = gt_color
    
    # 保存汇总图像
    summary_output_path = os.path.join(output_dir, f'all_gt_bfov_masks.jpg')
    cv2.imwrite(summary_output_path, summary_img)
    print(f"所有GT的BFOV掩码汇总可视化已保存到: {summary_output_path}")


def main():
    """
    测试GPU版本的BFOV掩码生成
    """
    print("开始测试GPU版本的BFOV掩码生成...")
    
    # 检查是否有可用的GPU
    if not torch.cuda.is_available():
        print("错误: 没有可用的GPU设备！")
        return
    
    # 设置测试参数
    erp_w = 1920
    erp_h = 960
    
    # 准备测试数据
    print("准备测试数据...")
    
    # 创建测试数据
    # 测试场景1: 单个GT，多个BFOV
    test_case_1 = []
    gt1_bfovs = torch.tensor([
        [0, 0, np.pi/2, np.pi/3],      # 经度0，纬度0，水平视场90°，竖直视场60°
        [np.pi/2, np.pi/6, np.pi/2, np.pi/3],  # 经度90°，纬度30°
        [np.pi, 0, 2*np.pi/3, np.pi/2]  # 经度180°，纬度0，水平视场120°，竖直视场90°
    ], dtype=torch.float32).cuda()
    test_case_1.append(gt1_bfovs)
    
    # 测试场景2: 多个GT，每个GT有不同数量的BFOV
    test_case_2 = []
    # GT 1: 3个BFOV
    gt1_bfovs = torch.tensor([
        [0, 0, np.pi/2, np.pi/3],
        [np.pi/2, np.pi/6, np.pi/2, np.pi/3],
        [np.pi, 0, 2*np.pi/3, np.pi/2]
    ], dtype=torch.float32).cuda()
    test_case_2.append(gt1_bfovs)
    
    # GT 2: 2个BFOV
    gt2_bfovs = torch.tensor([
        [np.pi/12, np.pi/12, np.pi/2, np.pi/3],  # 经度15°，纬度15°
        [7*np.pi/12, np.pi/4, 2*np.pi/3, np.pi/3]  # 经度105°，纬度45°
    ], dtype=torch.float32).cuda()
    test_case_2.append(gt2_bfovs)
    
    # 测试场景3: 多个GT，每个GT有多个BFOV（复杂场景）
    test_case_3 = []
    # GT 1: 4个BFOV
    gt1_bfovs = torch.tensor([
        [0, 0, np.pi/3, np.pi/4],      # 经度0°，纬度0°，水平视场60°，竖直视场45°
        [np.pi/3, np.pi/4, np.pi/2, np.pi/3],  # 经度60°，纬度45°
        [2*np.pi/3, -np.pi/6, 2*np.pi/3, np.pi/2],  # 经度120°，纬度-30°
        [np.pi, np.pi/3, np.pi/4, np.pi/5]  # 经度180°，纬度60°
    ], dtype=torch.float32).cuda()
    test_case_3.append(gt1_bfovs)
    
    # GT 2: 5个BFOV
    gt2_bfovs = torch.tensor([
        [np.pi/6, np.pi/6, np.pi/2, np.pi/3],  # 经度30°，纬度30°
        [np.pi/2, -np.pi/3, 2*np.pi/3, np.pi/2],  # 经度90°，纬度-60°
        [5*np.pi/6, np.pi/4, np.pi/3, np.pi/4],  # 经度150°，纬度45°
        [7*np.pi/6, -np.pi/4, np.pi/2, np.pi/3],  # 经度210°，纬度-45°
        [3*np.pi/2, np.pi/6, 2*np.pi/3, np.pi/2]  # 经度270°，纬度30°
    ], dtype=torch.float32).cuda()
    test_case_3.append(gt2_bfovs)
    
    # GT 3: 3个BFOV
    gt3_bfovs = torch.tensor([
        [11*np.pi/6, -np.pi/6, np.pi/3, np.pi/4],  # 经度330°，纬度-30°
        [np.pi/4, np.pi/2, np.pi/2, np.pi/3],  # 经度45°，纬度90°（极点）
        [5*np.pi/4, -np.pi/2, 2*np.pi/3, np.pi/2]  # 经度225°，纬度-90°（极点）
    ], dtype=torch.float32).cuda()
    test_case_3.append(gt3_bfovs)
    
    # 测试场景4: 边缘情况测试
    test_case_4 = []
    # GT 1: 各种边缘情况
    gt1_bfovs = torch.tensor([
        [0, 0, np.pi/6, np.pi/6],      # 小视场角
        [0, 0, 5*np.pi/6, 5*np.pi/6],  # 大视场角
        [np.pi/2, 0, np.pi, np.pi/2],  # 水平180°视场
        [0, np.pi/2, np.pi/2, np.pi],  # 垂直180°视场（极点）
        [0, -np.pi/2, np.pi/2, np.pi]  # 垂直180°视场（南极点）
    ], dtype=torch.float32).cuda()
    test_case_4.append(gt1_bfovs)
    
    # 运行测试场景1
    print("\n测试场景1: 单个GT，多个BFOV")
    start_time = time.time()
    gpu_results_1 = generate_bfov_masks_gpu(test_case_1, erp_w, erp_h)
    gpu_time_1 = time.time() - start_time
    print(f"GPU版本运行时间: {gpu_time_1:.4f}秒")
    
    # 准备CPU测试数据
    cpu_test_case_1 = []
    for gt_bfovs in test_case_1:
        cpu_test_case_1.append(gt_bfovs.cpu().numpy())
    
    # 运行CPU版本进行验证
    start_time = time.time()
    cpu_results_1 = generate_bfov_masks_cpu(cpu_test_case_1, erp_w, erp_h)
    cpu_time_1 = time.time() - start_time
    print(f"CPU版本运行时间: {cpu_time_1:.4f}秒")
    print(f"GPU加速比: {cpu_time_1/gpu_time_1:.2f}x")
    
    # 验证结果
    validate_results(gpu_results_1, cpu_results_1)
    
    # 打印GPU结果信息
    print(f"\nGPU结果信息:")
    for i, masks in enumerate(gpu_results_1):
        print(f"GT {i} 掩码形状: {masks.shape}")
        print(f"GT {i} 掩码类型: {type(masks)}")
        print(f"GT {i} 掩码设备: {masks.device}")
        # 计算掩码中1的数量
        mask_sum = torch.sum(masks).item()
        print(f"GT {i} 掩码中1的数量: {mask_sum}")
    
    # 可视化GPU生成的掩码
    print("\n可视化GPU生成的掩码...")
    gpu_output_dir = './result/gpu_bfov_mask_visualizations'
    visualize_bfov_masks_test(test_case_1, gpu_results_1, erp_w, erp_h, threshold=None, output_dir=gpu_output_dir)
    
    # 可视化CPU生成的掩码
    print("\n可视化CPU生成的掩码...")
    cpu_output_dir = './result/cpu_bfov_mask_visualizations'
    visualize_bfov_masks_test(cpu_test_case_1, cpu_results_1, erp_w, erp_h, threshold=None, output_dir=cpu_output_dir)
    
    # 运行测试场景2
    print("\n测试场景2: 多个GT，每个GT有不同数量的BFOV")
    start_time = time.time()
    gpu_results_2 = generate_bfov_masks_gpu(test_case_2, erp_w, erp_h)
    gpu_time_2 = time.time() - start_time
    print(f"GPU版本运行时间: {gpu_time_2:.4f}秒")
    
    # 准备CPU测试数据
    cpu_test_case_2 = []
    for gt_bfovs in test_case_2:
        cpu_test_case_2.append(gt_bfovs.cpu().numpy())
    
    # 运行CPU版本进行验证
    start_time = time.time()
    cpu_results_2 = generate_bfov_masks_cpu(cpu_test_case_2, erp_w, erp_h)
    cpu_time_2 = time.time() - start_time
    print(f"CPU版本运行时间: {cpu_time_2:.4f}秒")
    print(f"GPU加速比: {cpu_time_2/gpu_time_2:.2f}x")
    
    # 验证结果
    validate_results(gpu_results_2, cpu_results_2)
    
    # 打印GPU结果信息
    print(f"\nGPU结果信息:")
    for i, masks in enumerate(gpu_results_2):
        print(f"GT {i} 掩码形状: {masks.shape}")
        print(f"GT {i} 掩码类型: {type(masks)}")
        print(f"GT {i} 掩码设备: {masks.device}")
        # 计算掩码中1的数量
        mask_sum = torch.sum(masks).item()
        print(f"GT {i} 掩码中1的数量: {mask_sum}")
    
    # 可视化GPU生成的掩码（场景2）
    print("\n可视化GPU生成的掩码（场景2）...")
    gpu_output_dir_2 = './result/gpu_bfov_mask_visualizations_scene2'
    visualize_bfov_masks_test(test_case_2, gpu_results_2, erp_w, erp_h, threshold=None, output_dir=gpu_output_dir_2)
    
    # 可视化CPU生成的掩码（场景2）
    print("\n可视化CPU生成的掩码（场景2）...")
    cpu_output_dir_2 = './result/cpu_bfov_mask_visualizations_scene2'
    visualize_bfov_masks_test(cpu_test_case_2, cpu_results_2, erp_w, erp_h, threshold=None, output_dir=cpu_output_dir_2)
    
    # 运行测试场景3: 多个GT，每个GT有多个BFOV（复杂场景）
    print("\n测试场景3: 多个GT，每个GT有多个BFOV（复杂场景）")
    start_time = time.time()
    gpu_results_3 = generate_bfov_masks_gpu(test_case_3, erp_w, erp_h)
    gpu_time_3 = time.time() - start_time
    print(f"GPU版本运行时间: {gpu_time_3:.4f}秒")
    
    # 准备CPU测试数据
    cpu_test_case_3 = []
    for gt_bfovs in test_case_3:
        cpu_test_case_3.append(gt_bfovs.cpu().numpy())
    
    # 运行CPU版本进行验证
    start_time = time.time()
    cpu_results_3 = generate_bfov_masks_cpu(cpu_test_case_3, erp_w, erp_h)
    cpu_time_3 = time.time() - start_time
    print(f"CPU版本运行时间: {cpu_time_3:.4f}秒")
    print(f"GPU加速比: {cpu_time_3/gpu_time_3:.2f}x")
    
    # 验证结果
    validate_results(gpu_results_3, cpu_results_3)
    
    # 打印GPU结果信息
    print(f"\nGPU结果信息:")
    for i, masks in enumerate(gpu_results_3):
        print(f"GT {i} 掩码形状: {masks.shape}")
        print(f"GT {i} 掩码类型: {type(masks)}")
        print(f"GT {i} 掩码设备: {masks.device}")
        # 计算掩码中1的数量
        mask_sum = torch.sum(masks).item()
        print(f"GT {i} 掩码中1的数量: {mask_sum}")
    
    # 可视化GPU生成的掩码（场景3）
    print("\n可视化GPU生成的掩码（场景3）...")
    gpu_output_dir_3 = './result/gpu_bfov_mask_visualizations_scene3'
    visualize_bfov_masks_test(test_case_3, gpu_results_3, erp_w, erp_h, threshold=None, output_dir=gpu_output_dir_3)
    
    # 可视化CPU生成的掩码（场景3）
    print("\n可视化CPU生成的掩码（场景3）...")
    cpu_output_dir_3 = './result/cpu_bfov_mask_visualizations_scene3'
    visualize_bfov_masks_test(cpu_test_case_3, cpu_results_3, erp_w, erp_h, threshold=None, output_dir=cpu_output_dir_3)
    
    # 运行测试场景4: 边缘情况测试
    print("\n测试场景4: 边缘情况测试")
    start_time = time.time()
    gpu_results_4 = generate_bfov_masks_gpu(test_case_4, erp_w, erp_h)
    gpu_time_4 = time.time() - start_time
    print(f"GPU版本运行时间: {gpu_time_4:.4f}秒")
    
    # 准备CPU测试数据
    cpu_test_case_4 = []
    for gt_bfovs in test_case_4:
        cpu_test_case_4.append(gt_bfovs.cpu().numpy())
    
    # 运行CPU版本进行验证
    start_time = time.time()
    cpu_results_4 = generate_bfov_masks_cpu(cpu_test_case_4, erp_w, erp_h)
    cpu_time_4 = time.time() - start_time
    print(f"CPU版本运行时间: {cpu_time_4:.4f}秒")
    print(f"GPU加速比: {cpu_time_4/gpu_time_4:.2f}x")
    
    # 验证结果
    validate_results(gpu_results_4, cpu_results_4)
    
    # 打印GPU结果信息
    print(f"\nGPU结果信息:")
    for i, masks in enumerate(gpu_results_4):
        print(f"GT {i} 掩码形状: {masks.shape}")
        print(f"GT {i} 掩码类型: {type(masks)}")
        print(f"GT {i} 掩码设备: {masks.device}")
        # 计算掩码中1的数量
        mask_sum = torch.sum(masks).item()
        print(f"GT {i} 掩码中1的数量: {mask_sum}")
    
    # 可视化GPU生成的掩码（场景4）
    print("\n可视化GPU生成的掩码（场景4）...")
    gpu_output_dir_4 = './result/gpu_bfov_mask_visualizations_scene4'
    visualize_bfov_masks_test(test_case_4, gpu_results_4, erp_w, erp_h, threshold=None, output_dir=gpu_output_dir_4)
    
    # 可视化CPU生成的掩码（场景4）
    print("\n可视化CPU生成的掩码（场景4）...")
    cpu_output_dir_4 = './result/cpu_bfov_mask_visualizations_scene4'
    visualize_bfov_masks_test(cpu_test_case_4, cpu_results_4, erp_w, erp_h, threshold=None, output_dir=cpu_output_dir_4)
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()