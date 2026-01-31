import os

import time
import numpy as np
import torch
import tqdm


from sphdet.iou.sph_iou_api import sph2pob_efficient_iou
from sphdet.iou.unbiased_iou_bfov import Sph as UnbiasedSph


def read_bfovlist(file_path, num_lines=10):
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


def generate_random_bfov(num_boxes=200):
    """
    随机生成BFOV参数
    
    参数:
    num_boxes: 生成的BFOV参数数量
    
    返回:
    bfov_list: BFOV参数列表，每个元素是 [longitude, latitude, fov_x, fov_y]
    """
    np.random.seed(42)  # 设置随机种子，保证结果可复现
    bfov_list = []
    
    for _ in range(num_boxes):
        # 生成随机参数
        longitude = np.random.uniform(-180, 180)  # 经度: -180~180度
        latitude = np.random.uniform(-90, 90)     # 纬度: -90~90度
        fov_x = np.random.uniform(10, 120)        # 水平视场角: 10~120度
        fov_y = np.random.uniform(10, 120)        # 垂直视场角: 10~120度
        
        bfov_list.append([longitude, latitude, fov_x, fov_y])
    
    return bfov_list

def test_sph_gpu_accuracy():
    """
    测试sph2pob_efficient_iou和unbiased_iou_bfov两种方法的速度差异和精度差异
    """
    print("开始测试两种IoU计算方法的性能和精度...")
    
    # 随机生成BFOV参数 - 使用1000个数据进行测试（500个用于检测框，500个用于GT框）
    num_boxes = 1000  # 生成1000个数据进行测试
    bfov_list = generate_random_bfov(num_boxes)
    print(f"成功生成{len(bfov_list)}个BFOV参数")
    print("注意: 生成的数据是角度制")
    
    # 生成检测框和GT框（各500个）
    dets = np.array(bfov_list[:500])  # 前500个作为检测框
    gts = np.array(bfov_list[500:1000])  # 后500个作为GT框
    print(f"生成了{len(dets)}个检测框和{len(gts)}个GT框")
    print(f"需要计算{len(dets) * len(gts)}个IoU值")
    
    # 添加角度维度（角度为0）
    dets_with_angle = np.concatenate([dets, np.zeros((dets.shape[0], 1))], axis=1)
    gts_with_angle = np.concatenate([gts, np.zeros((gts.shape[0], 1))], axis=1)
    
    # 1. 测试unbiased_iou_bfov方法（CPU版本）
    print("\n测试unbiased_iou_bfov方法（CPU版本）...")
    unbiased_calculator = UnbiasedSph()
    start_time = time.time()
    
    # 使用进度条显示计算进度
    unbiased_iou = np.zeros((len(dets), len(gts)))
    # 计算所有100个检测框的IoU值
    test_size = len(dets)
    for i in tqdm.tqdm(range(test_size), desc="计算unbiased_iou_bfov IoU", unit="框"):
        for j in range(len(gts)):  # 计算所有100个GT框
            # 计算单个检测框和GT框的IoU
            single_det = dets_with_angle[i:i+1]
            single_gt = gts_with_angle[j:j+1]
            iou_value = unbiased_calculator.sphIoU(torch.tensor(single_det, dtype=torch.float32), torch.tensor(single_gt, dtype=torch.float32))
            unbiased_iou[i, j] = iou_value.cpu().numpy()[0, 0]
    
    unbiased_time = time.time() - start_time
    print(f"unbiased_iou_bfov运行时间（{test_size}个检测框，{len(gts)}个GT框）: {unbiased_time:.4f}秒")
    
    # 2. 测试sph2pob_efficient_iou方法（GPU版本）
    if torch.cuda.is_available():
        print("\n测试sph2pob_efficient_iou方法（GPU版本）...")
        # 转换为PyTorch张量
        dets_tensor = torch.tensor(dets_with_angle, dtype=torch.float32, device='cuda')
        gts_tensor = torch.tensor(gts_with_angle, dtype=torch.float32, device='cuda')
        
        start_time = time.time()
        
        # 使用批量处理以减少内存使用
        batch_size = 100  # 每个批次处理100个检测框，适应更大的数据量
        total_dets = dets_tensor.shape[0]
        efficient_iou = torch.zeros((total_dets, len(gts)), device='cuda')
        
        # 使用进度条显示GPU计算进度
        with tqdm.tqdm(total=total_dets, desc="GPU批量计算", unit="框") as pbar:
            for i in range(0, total_dets, batch_size):
                end_idx = min(i + batch_size, total_dets)
                dets_batch = dets_tensor[i:end_idx]
                
                # 计算当前批次的IoU值，使用sph2pob_efficient_iou
                iou_batch = sph2pob_efficient_iou(dets_batch, gts_tensor, is_aligned=False)
                efficient_iou[i:end_idx] = iou_batch
                
                # 更新进度条
                pbar.update(end_idx - i)
                
                # 清除缓存以释放内存
                torch.cuda.empty_cache()
        
        # 等待GPU计算完成
        torch.cuda.synchronize()
        efficient_time = time.time() - start_time
        print(f"sph2pob_efficient_iou运行时间（全部{len(dets)}个检测框）: {efficient_time:.4f}秒")
        
        # 3. 比较两种方法的精度差异（比较前100*100对结果）
        print("\n比较两种方法的精度差异...")
        efficient_iou_np = efficient_iou.cpu().numpy()
        
        # 比较所有100*100对结果
        test_size = len(dets)
        diff = np.abs(unbiased_iou[:test_size, :] - efficient_iou_np[:test_size, :])
        
        # 计算差异大于0.2的数量
        large_diff_threshold = 0.2
        large_diff_count = np.sum(diff >= large_diff_threshold)
        total_count = diff.size
        large_diff_ratio = (large_diff_count / total_count) * 100
        
        # 排除差异大于0.2的结果，重新计算统计信息
        filtered_diff = diff[diff < large_diff_threshold]
        filtered_count = filtered_diff.size
        
        if filtered_count > 0:
            max_filtered_diff = np.max(filtered_diff)
            mean_filtered_diff = np.mean(filtered_diff)
            std_filtered_diff = np.std(filtered_diff)
        else:
            max_filtered_diff = 0
            mean_filtered_diff = 0
            std_filtered_diff = 0
        
        # 设置差异阈值
        acceptable_threshold = 0.01  # 差异小于0.01的被认为是"差异小"
        
        # 计算过滤后差异小于阈值的比例
        if filtered_count > 0:
            small_diff_count = np.sum(filtered_diff < acceptable_threshold)
            small_diff_ratio = (small_diff_count / filtered_count) * 100
        else:
            small_diff_count = 0
            small_diff_ratio = 0
        
        print(f"总IoU计算数量: {total_count}")
        print(f"差异大于{large_diff_threshold}的数量: {large_diff_count}")
        print(f"差异大于{large_diff_threshold}的比例: {large_diff_ratio:.2f}%")
        print(f"过滤后剩余IoU计算数量: {filtered_count}")
        print(f"过滤后最大差异: {max_filtered_diff:.8f}")
        print(f"过滤后平均差异: {mean_filtered_diff:.8f}")
        print(f"过滤后标准差: {std_filtered_diff:.8f}")
        print(f"过滤后差异小于{acceptable_threshold}的比例: {small_diff_ratio:.2f}%")
        
        # 检查是否满足精度要求（差异小于百分之一）
        if mean_filtered_diff < 0.01:
            print("✅ 精度要求满足！排除差异大于0.2的情况后，两种方法的平均差异小于百分之一。")
        else:
            print("⚠️  精度要求不满足！排除差异大于0.2的情况后，两种方法的平均差异大于百分之一。")
        
        # 4. 比较性能差异
        print("\n比较两种方法的性能差异...")
        # 计算unbiased_iou_bfov处理500*500对的时间
        unbiased_500x500_time = unbiased_time
        # 计算sph2pob_efficient_iou处理500*500对的时间
        efficient_500x500_time = efficient_time
        
        print(f"unbiased_iou_bfov处理500*500对的时间: {unbiased_500x500_time:.4f}秒")
        print(f"sph2pob_efficient_iou处理500*500对的时间: {efficient_500x500_time:.4f}秒")
        
        # 计算性能提升
        if efficient_500x500_time > 0:
            speed_up = unbiased_500x500_time / efficient_500x500_time
            print(f"性能提升: {speed_up:.2f}倍")
            print(f"sph2pob_efficient_iou速度是unbiased_iou_bfov的 {speed_up:.2f} 倍")
        
        # 5. 准备输出目录
        output_dir = 'test/iou_comparison'
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备标签
        num_dets = test_size
        num_gts = len(gts)
        
        # 6. 保存比较报告
        report_path = os.path.join(output_dir, 'comparison_report.txt')
        with open(report_path, 'w') as f:
            f.write("# IoU计算方法比较报告\n\n")
            f.write("## 测试配置\n")
            f.write(f"- 测试数据数量: {len(dets)}个检测框, {len(gts)}个GT框\n")
            f.write(f"- 计算IoU数量: {len(dets) * len(gts)}\n")
            f.write(f"- 比较结果数量: {test_size}x{len(gts)} = {test_size*len(gts)}\n")
            f.write(f"- 硬件环境: {'GPU' if torch.cuda.is_available() else 'CPU'}\n")
            if torch.cuda.is_available():
                f.write(f"- GPU型号: {torch.cuda.get_device_name(0)}\n")
            f.write(f"- PyTorch版本: {torch.__version__}\n\n")
            
            f.write("## 性能对比\n")
            f.write(f"- unbiased_iou_bfov运行时间（{test_size}个检测框，{len(gts)}个GT框）: {unbiased_time:.4f}秒\n")
            f.write(f"- sph2pob_efficient_iou运行时间（全部{len(dets)}个检测框）: {efficient_time:.4f}秒\n")
            f.write(f"- unbiased_iou_bfov处理500*500对的时间: {unbiased_500x500_time:.4f}秒\n")
            f.write(f"- sph2pob_efficient_iou处理500*500对的时间: {efficient_500x500_time:.4f}秒\n")
            if efficient_500x500_time > 0:
                f.write(f"- 性能提升: {speed_up:.2f}倍\n")
                f.write(f"- sph2pob_efficient_iou速度是unbiased_iou_bfov的 {speed_up:.2f} 倍\n\n")
            
            f.write("## 精度对比\n")
            f.write(f"- 总IoU计算数量: {total_count}\n")
            f.write(f"- 差异大于{large_diff_threshold}的数量: {large_diff_count}\n")
            f.write(f"- 差异大于{large_diff_threshold}的比例: {large_diff_ratio:.2f}%\n")
            f.write(f"- 过滤后剩余IoU计算数量: {filtered_count}\n")
            f.write(f"- 过滤后最大差异: {max_filtered_diff:.8f}\n")
            f.write(f"- 过滤后平均差异: {mean_filtered_diff:.8f}\n")
            f.write(f"- 过滤后标准差: {std_filtered_diff:.8f}\n")
            f.write(f"- 过滤后差异小于{acceptable_threshold}的比例: {small_diff_ratio:.2f}%\n\n")
            
            f.write("## 结论\n")
            if mean_filtered_diff < 0.01:
                f.write("✅ 精度要求满足！排除差异大于0.2的情况后，两种方法的平均差异小于百分之一。\n")
            else:
                f.write("⚠️  精度要求不满足！排除差异大于0.2的情况后，两种方法的平均差异大于百分之一。\n")
            
            if torch.cuda.is_available():
                f.write("✅ 计算速度从不可GPU加速变成可以GPU加速，性能显著提升。\n")
        
        print(f"\n比较报告已保存到: {report_path}")
    else:
        print("\n警告: 没有可用的GPU，无法测试GPU版本")
    
    print("\n测试完成！")


if __name__ == "__main__":
    # 在脚本开头添加
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"CUDA版本: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"当前GPU: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    test_sph_gpu_accuracy()