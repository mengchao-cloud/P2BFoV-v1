import os
import time
import numpy as np
import torch
import tqdm


# 导入两种IoU计算方法
from sphdet.iou.sph_iou_api import sph2pob_efficient_iou
from sphdet.iou.unbiased_iou_bfov import Sph as UnbiasedSph

def read_bfovlist(file_path, num_lines=100):
    """
    读取bfovlist.txt文件的前num_lines行
    
    参数:
    file_path: 文件路径
    num_lines: 读取的行数
    
    返回:
    bfov_list: BFOV参数列表，每个元素是 [longitude, latitude, fov_x, fov_y]
    """
    bfov_list = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_lines:
                break
            parts = line.strip().split()
            if len(parts) == 4:
                # 假设文件中的数据是弧度制，需要转换为角度制
                longitude_rad = float(parts[0])
                latitude_rad = float(parts[1])
                fov_x_rad = float(parts[2])
                fov_y_rad = float(parts[3])
                
                # 弧度转角度
                longitude = np.rad2deg(longitude_rad)
                latitude = np.rad2deg(latitude_rad)
                fov_x = np.rad2deg(fov_x_rad)
                fov_y = np.rad2deg(fov_y_rad)
                
                bfov_list.append([longitude, latitude, fov_x, fov_y])
    return bfov_list

def generate_test_data(num_boxes=100):
    """
    生成测试数据
    
    参数:
    num_boxes: 生成的边界框数量
    
    返回:
    dets: 检测框，形状为(num_boxes, 5)
    gts: 真实框，形状为(num_boxes, 5)
    """
    # 生成随机边界框
    # 格式: [longitude, latitude, fov_x, fov_y, angle]
    # longitude: -180~180度
    # latitude: -90~90度
    # fov_x: 0~180度
    # fov_y: 0~180度
    # angle: -180~180度
    np.random.seed(42)  # 设置随机种子，保证结果可复现
    
    dets = np.random.rand(num_boxes, 5)
    dets[:, 0] = dets[:, 0] * 360 - 180  # longitude
    dets[:, 1] = dets[:, 1] * 180 - 90   # latitude
    dets[:, 2] = dets[:, 2] * 180        # fov_x
    dets[:, 3] = dets[:, 3] * 180        # fov_y
    dets[:, 4] = dets[:, 4] * 360 - 180  # angle
    
    gts = np.random.rand(num_boxes, 5)
    gts[:, 0] = gts[:, 0] * 360 - 180  # longitude
    gts[:, 1] = gts[:, 1] * 180 - 90   # latitude
    gts[:, 2] = gts[:, 2] * 180        # fov_x
    gts[:, 3] = gts[:, 3] * 180        # fov_y
    gts[:, 4] = gts[:, 4] * 360 - 180  # angle
    
    return dets, gts

def compare_iou_methods():
    """
    比较两种IoU计算方法的速度和精度
    """
    print("开始比较两种IoU计算方法...")
    
    # 读取bfovlist.txt文件
    bfovlist_path = './bfovlist.txt'
    if os.path.exists(bfovlist_path):
        print(f"使用{bfovlist_path}中的数据进行测试")
        print("注意: 假设文件中的数据是弧度制，将自动转换为角度制")
        num_boxes = 100  # 读取100个数据进行测试
        bfov_list = read_bfovlist(bfovlist_path, num_boxes)
        print(f"成功读取{len(bfov_list)}个BFOV参数")
        print("数据已从弧度制转换为角度制")
        
        # 转换为测试数据
        dets = np.array(bfov_list)
        gts = np.array(bfov_list)
        
        # 添加角度维度（角度为0）
        dets_with_angle = np.concatenate([dets, np.zeros((dets.shape[0], 1))], axis=1)
        gts_with_angle = np.concatenate([gts, np.zeros((gts.shape[0], 1))], axis=1)
    else:
        print(f"{bfovlist_path} 文件不存在，使用随机生成的数据进行测试")
        # 生成测试数据
        num_boxes = 100
        dets_with_angle, gts_with_angle = generate_test_data(num_boxes)
    
    print(f"测试数据形状: dets={dets_with_angle.shape}, gts={gts_with_angle.shape}")
    print(f"需要计算{len(dets_with_angle) * len(gts_with_angle)}个IoU值")
    
    # 转换为PyTorch张量
    dets_tensor = torch.tensor(dets_with_angle, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
    gts_tensor = torch.tensor(gts_with_angle, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 测试 sph2pob_efficient_iou 方法
    print("\n测试 sph2pob_efficient_iou 方法...")
    start_time = time.time()
    
    # 使用批量处理以减少内存使用
    batch_size = 50
    total_dets = dets_tensor.shape[0]
    efficient_iou = torch.zeros((total_dets, len(gts_with_angle)), device=dets_tensor.device)
    
    with tqdm.tqdm(total=total_dets, desc="sph2pob_efficient_iou 计算", unit="框") as pbar:
        for i in range(0, total_dets, batch_size):
            end_idx = min(i + batch_size, total_dets)
            dets_batch = dets_tensor[i:end_idx]
            iou_batch = sph2pob_efficient_iou(dets_batch, gts_tensor, is_aligned=False)
            efficient_iou[i:end_idx] = iou_batch
            pbar.update(end_idx - i)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    efficient_time = time.time() - start_time
    print(f"sph2pob_efficient_iou 运行时间: {efficient_time:.4f}秒")
    
    # 2. 测试 unbiased_iou_bfov 方法
    print("\n测试 unbiased_iou_bfov 方法...")
    start_time = time.time()
    
    unbiased_calculator = UnbiasedSph()
    unbiased_iou = np.zeros((len(dets_with_angle), len(gts_with_angle)))
    
    # 由于unbiased方法计算较慢，只计算前50个检测框
    test_size = min(50, len(dets_with_angle))
    with tqdm.tqdm(total=test_size, desc="unbiased_iou_bfov 计算", unit="框") as pbar:
        for i in range(test_size):
            single_det = dets_with_angle[i:i+1]
            single_det_tensor = torch.tensor(single_det, dtype=torch.float32)
            iou_value = unbiased_calculator.sphIoU(single_det_tensor, torch.tensor(gts_with_angle, dtype=torch.float32))
            unbiased_iou[i] = iou_value.cpu().numpy()[0]
            pbar.update(1)
    
    unbiased_time = time.time() - start_time
    print(f"unbiased_iou_bfov 运行时间（前{test_size}个检测框）: {unbiased_time:.4f}秒")
    estimated_unbiased_time = unbiased_time * len(dets_with_angle) / test_size
    print(f"unbiased_iou_bfov 预计总运行时间: {estimated_unbiased_time:.4f}秒")
    
    # 3. 比较两种方法的精度差异
    print("\n比较两种方法的精度差异...")
    
    # 只比较前test_size个检测框的结果
    efficient_iou_np = efficient_iou.cpu().numpy()
    diff = np.abs(efficient_iou_np[:test_size] - unbiased_iou[:test_size])
    
    # 计算差异统计信息
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)
    min_diff = np.min(diff)
    std_diff = np.std(diff)
    
    print(f"差异统计信息:")
    print(f"  平均差异: {mean_diff:.8f}")
    print(f"  最大差异: {max_diff:.8f}")
    print(f"  最小差异: {min_diff:.8f}")
    print(f"  标准差: {std_diff:.8f}")
    
    # 计算不同阈值下的差异比例
    thresholds = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    print("\n不同阈值下的差异比例:")
    for threshold in thresholds:
        count = np.sum(diff > threshold)
        ratio = count / diff.size * 100
        print(f"  差异 > {threshold}: {count}个 ({ratio:.2f}%)")
    
    # 4. 性能对比
    print("\n性能对比:")
    print(f"sph2pob_efficient_iou 速度: {len(dets_with_angle) / efficient_time:.2f} 框/秒")
    print(f"unbiased_iou_bfov 速度: {test_size / unbiased_time:.2f} 框/秒")
    speed_up = (test_size / unbiased_time) / (len(dets_with_angle) / efficient_time)
    print(f"sph2pob_efficient_iou 比 unbiased_iou_bfov 快: {1/speed_up:.2f}倍")
    
    # 5. 保存比较结果
    output_dir = 'test/iou_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存两种方法的IoU值
    np.savetxt(os.path.join(output_dir, 'efficient_iou.txt'), efficient_iou_np[:test_size], fmt='%.8f')
    np.savetxt(os.path.join(output_dir, 'unbiased_iou.txt'), unbiased_iou[:test_size], fmt='%.8f')
    np.savetxt(os.path.join(output_dir, 'iou_diff.txt'), diff, fmt='%.8f')
    
    # 保存比较报告
    with open(os.path.join(output_dir, 'comparison_report.txt'), 'w') as f:
        f.write("# IoU计算方法比较报告\n\n")
        f.write(f"## 测试配置\n")
        f.write(f"- 测试数据数量: {len(dets_with_angle)}个检测框, {len(gts_with_angle)}个GT框\n")
        f.write(f"- 计算IoU数量: {len(dets_with_angle) * len(gts_with_angle)}\n")
        f.write(f"- 硬件环境: {'GPU' if torch.cuda.is_available() else 'CPU'}\n")
        if torch.cuda.is_available():
            f.write(f"- GPU型号: {torch.cuda.get_device_name(0)}\n")
        f.write(f"- PyTorch版本: {torch.__version__}\n\n")
        
        f.write("## 性能对比\n")
        f.write(f"- sph2pob_efficient_iou 运行时间: {efficient_time:.4f}秒\n")
        f.write(f"- unbiased_iou_bfov 运行时间（前{test_size}个检测框）: {unbiased_time:.4f}秒\n")
        f.write(f"- unbiased_iou_bfov 预计总运行时间: {estimated_unbiased_time:.4f}秒\n")
        f.write(f"- sph2pob_efficient_iou 速度: {len(dets_with_angle) / efficient_time:.2f} 框/秒\n")
        f.write(f"- unbiased_iou_bfov 速度: {test_size / unbiased_time:.2f} 框/秒\n")
        f.write(f"- 速度提升: {1/speed_up:.2f}倍\n\n")
        
        f.write("## 精度对比\n")
        f.write(f"- 平均差异: {mean_diff:.8f}\n")
        f.write(f"- 最大差异: {max_diff:.8f}\n")
        f.write(f"- 最小差异: {min_diff:.8f}\n")
        f.write(f"- 标准差: {std_diff:.8f}\n\n")
        
        f.write("## 差异分布\n")
        for threshold in thresholds:
            count = np.sum(diff > threshold)
            ratio = count / diff.size * 100
            f.write(f"- 差异 > {threshold}: {count}个 ({ratio:.2f}%)\n")
        
        f.write("\n## 结论\n")
        if mean_diff < 1e-4:
            f.write("sph2pob_efficient_iou 在速度上有显著优势，同时保持了较高的精度。\n")
        else:
            f.write("sph2pob_efficient_iou 在速度上有显著优势，但与unbiased方法存在一定精度差异。\n")
    
    print(f"\n比较结果已保存到: {output_dir}")
    print("\n比较完成！")

if __name__ == "__main__":
    # 打印环境信息
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"CUDA版本: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"当前GPU: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    
    compare_iou_methods()
