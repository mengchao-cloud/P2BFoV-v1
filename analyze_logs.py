import json
import numpy as np


def parse_log_file(log_path):
    """
    解析日志文件，提取训练指标
    
    参数:
    log_path: 日志文件路径
    
    返回:
    data: 包含所有训练记录的列表
    """
    data = []
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line.startswith('{'):
                try:
                    record = json.loads(line)
                    if record.get('mode') == 'train':
                        data.append(record)
                except json.JSONDecodeError:
                    continue
    return data


def compare_logs(log_path_before, log_path_before_name, log_path_after, log_path_after_name):
    """
    对比优化前后的日志指标
    
    参数:
    log_path_before: 优化前的日志路径
    log_path_before_name: 优化前的名称
    log_path_after: 优化后的日志路径
    log_path_after_name: 优化后的名称
    """
    print("=" * 80)
    print(f"优化前后对比分析")
    print("=" * 80)
    print(f"优化前: {log_path_before_name}")
    print(f"优化后: {log_path_after_name}")
    print("=" * 80)
    print()
    
    # 解析日志文件
    data_before = parse_log_file(log_path_before)
    data_after = parse_log_file(log_path_after)
    
    # 提取每个epoch的第50次迭代的数据
    def extract_epoch_data(data):
        epoch_data = {}
        for record in data:
            epoch = record.get('epoch')
            iter_num = record.get('iter')
            if epoch is not None and iter_num == 50:
                epoch_data[epoch] = record
        return epoch_data
    
    epoch_data_before = extract_epoch_data(data_before)
    epoch_data_after = extract_epoch_data(data_after)
    
    # 获取共同的epoch
    common_epochs = sorted(set(epoch_data_before.keys()) & set(epoch_data_after.keys()))
    
    if not common_epochs:
        print("警告: 没有找到共同的epoch数据")
        return
    
    print(f"对比 {len(common_epochs)} 个epoch的数据 (epoch {min(common_epochs)} - {max(common_epochs)})")
    print()
    
    # 定义要对比的指标
    metrics = [
        ('loss', '总损失'),
        ('stage0_loss_instance_mil', 'Stage0 MIL损失'),
        ('stage0_bag_acc', 'Stage0 Bag准确率(%)'),
        ('stage0_mean_ious', 'Stage0 平均IoU'),
        ('stage1_loss_instance_mil', 'Stage1 MIL损失'),
        ('stage1_bag_acc', 'Stage1 Bag准确率(%)'),
        ('stage1_neg_loss', 'Stage1 负样本损失'),
        ('stage1_mean_ious', 'Stage1 平均IoU'),
        ('time', '训练时间(s)'),
        ('memory', '内存使用(MB)'),
    ]
    
    # 打印详细对比表格
    print("=" * 120)
    print(f"{'Epoch':<6} {'指标':<25} {'优化前':<12} {'优化后':<12} {'变化':<12} {'变化率':<10}")
    print("=" * 120)
    
    for epoch in common_epochs:
        print(f"\nEpoch {epoch}:")
        print("-" * 120)
        
        record_before = epoch_data_before[epoch]
        record_after = epoch_data_after[epoch]
        
        for metric_key, metric_name in metrics:
            value_before = record_before.get(metric_key, 0)
            value_after = record_after.get(metric_key, 0)
            
            # 计算变化
            change = value_after - value_before
            change_rate = (change / value_before * 100) if value_before != 0 else 0
            
            # 格式化输出
            value_before_str = f"{value_before:.4f}"
            value_after_str = f"{value_after:.4f}"
            change_str = f"{change:+.4f}"
            change_rate_str = f"{change_rate:+.2f}%"
            
            # 对于准确率，变化率符号相反（准确率提高是好的）
            if 'acc' in metric_key:
                if change > 0:
                    change_rate_str = f"↑{change_rate:.2f}%"
                else:
                    change_rate_str = f"↓{abs(change_rate):.2f}%"
            else:
                # 对于损失，降低是好的
                if change < 0:
                    change_rate_str = f"↓{abs(change_rate):.2f}%"
                else:
                    change_rate_str = f"↑{change_rate:.2f}%"
            
            print(f"{'':<6} {metric_name:<25} {value_before_str:<12} {value_after_str:<12} {change_str:<12} {change_rate_str:<10}")
    
    # 计算平均变化
    print("\n" + "=" * 120)
    print("平均变化汇总:")
    print("=" * 120)
    print(f"{'指标':<25} {'平均变化':<15} {'平均变化率':<15}")
    print("-" * 120)
    
    for metric_key, metric_name in metrics:
        changes = []
        change_rates = []
        
        for epoch in common_epochs:
            record_before = epoch_data_before[epoch]
            record_after = epoch_data_after[epoch]
            
            value_before = record_before.get(metric_key, 0)
            value_after = record_after.get(metric_key, 0)
            
            if value_before != 0:
                change = value_after - value_before
                change_rate = (change / value_before * 100)
                changes.append(change)
                change_rates.append(change_rate)
        
        if changes:
            avg_change = np.mean(changes)
            avg_change_rate = np.mean(change_rates)
            
            change_str = f"{avg_change:+.4f}"
            change_rate_str = f"{avg_change_rate:+.2f}%"
            
            # 对于准确率，变化率符号相反
            if 'acc' in metric_key:
                if avg_change > 0:
                    change_rate_str = f"↑{avg_change_rate:.2f}%"
                else:
                    change_rate_str = f"↓{abs(avg_change_rate):.2f}%"
            else:
                if avg_change < 0:
                    change_rate_str = f"↓{abs(avg_change_rate):.2f}%"
                else:
                    change_rate_str = f"↑{avg_change_rate:.2f}%"
            
            print(f"{metric_name:<25} {change_str:<15} {change_rate_str:<15}")
    
    print("\n" + "=" * 120)
    print("关键发现:")
    print("=" * 120)
    
    # 分析关键指标
    final_epoch = common_epochs[-1]
    record_before = epoch_data_before[final_epoch]
    record_after = epoch_data_after[final_epoch]
    
    print(f"\n最终Epoch {final_epoch} 对比:")
    
    # Bag准确率
    stage0_acc_before = record_before.get('stage0_bag_acc', 0)
    stage0_acc_after = record_after.get('stage0_bag_acc', 0)
    stage1_acc_before = record_before.get('stage1_bag_acc', 0)
    stage1_acc_after = record_after.get('stage1_bag_acc', 0)
    
    print(f"  Stage0 Bag准确率: {stage0_acc_before:.2f}% → {stage0_acc_after:.2f}% (变化: {stage0_acc_after - stage0_acc_before:+.2f}%)")
    print(f"  Stage1 Bag准确率: {stage1_acc_before:.2f}% → {stage1_acc_after:.2f}% (变化: {stage1_acc_after - stage1_acc_before:+.2f}%)")
    
    # 损失
    loss_before = record_before.get('loss', 0)
    loss_after = record_after.get('loss', 0)
    print(f"  总损失: {loss_before:.4f} → {loss_after:.4f} (变化: {loss_after - loss_before:+.4f})")
    
    # IoU
    mean_iou_before = record_before.get('stage1_mean_ious', 0)
    mean_iou_after = record_after.get('stage1_mean_ious', 0)
    print(f"  Stage1 平均IoU: {mean_iou_before:.4f} → {mean_iou_after:.4f} (变化: {mean_iou_after - mean_iou_before:+.4f})")
    
    # 训练效率
    time_before = record_before.get('time', 0)
    time_after = record_after.get('time', 0)
    print(f"  训练时间: {time_before:.2f}s → {time_after:.2f}s (变化: {time_after - time_before:+.2f}s)")
    
    memory_before = record_before.get('memory', 0)
    memory_after = record_after.get('memory', 0)
    print(f"  内存使用: {memory_before:.0f}MB → {memory_after:.0f}MB (变化: {memory_after - memory_before:+.0f}MB)")
    
    print("\n" + "=" * 120)


if __name__ == '__main__':
    # 设置日志路径
    log_path_before = '/home/mengchao/workspace/p2bfov/TOV_mmdetection/work_dir/temp/head2/20260214_104702.log.json'
    log_path_before_name = 'head2 (优化前)'
    
    log_path_after = '/home/mengchao/workspace/p2bfov/TOV_mmdetection/work_dir/temp/head3/20260217_165824.log.json'
    log_path_after_name = 'head3 (优化后)'
    
    # 执行对比分析
    compare_logs(log_path_before, log_path_before_name, log_path_after, log_path_after_name)
