import torch
import time
import numpy as np

# 添加兼容层，处理不同版本的 PyTorch
if not hasattr(torch, 'concat'):
    torch.concat = torch.cat

# 添加对 torch.pi 的兼容处理
if not hasattr(torch, 'pi'):
    torch.pi = np.pi

from sphdet.iou.sph_iou_api import sph2pob_efficient_iou, unbiased_iou

# 设置随机种子以确保结果可重复
np.random.seed(42)
torch.manual_seed(42)

class IoUComparator:
    def __init__(self):
        pass
    
    def generate_test_data(self, num_samples=100):
        """
        生成符合要求的测试数据
        - 第一个参数 (theta): -pi 到 pi，但不能等于边界值
        - 第二个参数 (phi): -pi/2 到 pi/2，但不能等于边界值
        - 第三个参数 (fov_x): 0 到 pi，但不能等于边界值
        - 第四个参数 (fov_y): 0 到 pi，但不能等于边界值
        - 第五个参数 (angle): 0
        """
        # 添加一个小的 epsilon 值，确保生成的数据不接近边界
        eps = 1e-6
        
        # 生成随机数据
        theta = np.random.uniform(-np.pi + eps, np.pi - eps, num_samples)
        phi = np.random.uniform(-np.pi/2 + eps, np.pi/2 - eps, num_samples)
        fov_x = np.random.uniform(eps, np.pi - eps, num_samples)
        fov_y = np.random.uniform(eps, np.pi - eps, num_samples)
        angle = np.zeros(num_samples)
        
        # 组合数据
        data = np.stack([theta, phi, fov_x, fov_y, angle], axis=1)
        return torch.tensor(data, dtype=torch.float32)
    
    def convert_rad_to_deg(self, data):
        """
        将弧度转换为度
        """
        data_deg = data.clone()
        # 前四个参数转换为度
        data_deg[:, :4] = torch.rad2deg(data_deg[:, :4])
        return data_deg
    
    def calculate_iou_with_sph2pob(self, sph_gt, sph_pred):
        """
        使用 sph2pob_efficient_iou 计算 IoU
        """
        # 转换为度
        sph_gt_deg = self.convert_rad_to_deg(sph_gt)
        sph_pred_deg = self.convert_rad_to_deg(sph_pred)
        
        # 只使用前四个参数（因为 sph2pob_efficient_iou 不使用 angle 参数）
        sph_gt_deg = sph_gt_deg[:, :4]
        sph_pred_deg = sph_pred_deg[:, :4]
        
        # 计算 IoU
        iou = sph2pob_efficient_iou(sph_pred_deg, sph_gt_deg, is_aligned=True)
        return iou
    
    def calculate_iou_with_unbiased(self, sph_gt, sph_pred):
        """
        使用 unbiased_iou 计算 IoU
        """
        # 转换为度
        sph_gt_deg = self.convert_rad_to_deg(sph_gt)
        sph_pred_deg = self.convert_rad_to_deg(sph_pred)
        
        # 只使用前四个参数（因为 unbiased_iou 不使用 angle 参数）
        sph_gt_deg = sph_gt_deg[:, :4]
        sph_pred_deg = sph_pred_deg[:, :4]
        
        # 计算 IoU
        iou = unbiased_iou(sph_pred_deg, sph_gt_deg, is_aligned=True)
        return iou
    
    def compare_methods(self, num_samples=100):
        """
        比较两种方法的效率和精度
        """
        # 生成测试数据
        print(f"生成 {num_samples} 个测试样本...")
        sph_gt = self.generate_test_data(num_samples)
        sph_pred = self.generate_test_data(num_samples)
        
        # 使用 sph2pob_efficient_iou 计算 IoU
        print("\n使用 sph2pob_efficient_iou 计算 IoU...")
        start_time = time.time()
        iou_sph2pob = self.calculate_iou_with_sph2pob(sph_gt, sph_pred)
        time_sph2pob = time.time() - start_time
        print(f"sph2pob_efficient_iou 耗时: {time_sph2pob:.4f} 秒")
        
        # 使用 unbiased_iou 计算 IoU
        print("\n使用 unbiased_iou 计算 IoU...")
        start_time = time.time()
        iou_unbiased = self.calculate_iou_with_unbiased(sph_gt, sph_pred)
        time_unbiased = time.time() - start_time
        print(f"unbiased_iou 耗时: {time_unbiased:.4f} 秒")
        
        # 计算精度差异
        print("\n计算精度差异...")
        diff = torch.abs(iou_sph2pob - iou_unbiased)
        mean_diff = torch.mean(diff).item()
        max_diff = torch.max(diff).item()
        print(f"平均差异: {mean_diff:.6f}")
        print(f"最大差异: {max_diff:.6f}")
        
        # 计算效率比
        efficiency_ratio = time_unbiased / time_sph2pob
        print(f"\n效率比 (unbiased / sph2pob): {efficiency_ratio:.2f}")
        if efficiency_ratio > 1:
            print(f"sph2pob_efficient_iou 比 unbiased_iou 快 {efficiency_ratio:.2f} 倍")
        else:
            print(f"unbiased_iou 比 sph2pob_efficient_iou 快 {1/efficiency_ratio:.2f} 倍")
        
        # 输出一些样本结果进行比较
        print("\n部分样本结果比较:")
        print("样本索引 | sph2pob IoU | unbiased IoU | 差异")
        print("-" * 60)
        for i in range(min(10, num_samples)):
            print(f"{i:8d} | {iou_sph2pob[i].item():12.6f} | {iou_unbiased[i].item():12.6f} | {diff[i].item():8.6f}")
        
        return {
            'time_sph2pob': time_sph2pob,
            'time_unbiased': time_unbiased,
            'mean_diff': mean_diff,
            'max_diff': max_diff,
            'efficiency_ratio': efficiency_ratio
        }

if __name__ == "__main__":
    comparator = IoUComparator()
    # 增大计算数量到 10000 个样本
    results = comparator.compare_methods(num_samples=1000000)
    
    print("\n总结:")
    print(f"sph2pob_efficient_iou 耗时: {results['time_sph2pob']:.4f} 秒")
    print(f"unbiased_iou 耗时: {results['time_unbiased']:.4f} 秒")
    print(f"平均精度差异: {results['mean_diff']:.6f}")
    print(f"最大差异: {results['max_diff']:.6f}")
    print(f"效率比: {results['efficiency_ratio']:.2f}")
