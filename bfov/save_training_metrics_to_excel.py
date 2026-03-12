import json
import pandas as pd
import numpy as np
from pathlib import Path

def load_log_data(log_path):
    """加载JSON日志文件数据"""
    data = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('{"env_info'):
                try:
                    record = json.loads(line.strip())
                    data.append(record)
                except json.JSONDecodeError:
                    continue
    return data

def align_data_by_epoch(all_data):
    """按epoch对齐三个头部的数据"""
    
    # 获取所有epoch
    all_epochs = set()
    for head_name, data in all_data.items():
        for record in data:
            epoch = record.get('epoch', 0)
            all_epochs.add(epoch)
    
    # 按epoch排序
    all_epochs = sorted(all_epochs)
    
    # 创建对齐后的数据
    aligned_data = {}
    for head_name in all_data.keys():
        aligned_data[head_name] = {}
        
        # 为每个epoch创建空记录
        for epoch in all_epochs:
            aligned_data[head_name][epoch] = {}
    
    # 填充数据 - 使用每个epoch的最后一个iter的数据
    for head_name, data in all_data.items():
        # 按epoch和iter排序
        sorted_data = sorted(data, key=lambda x: (x.get('epoch', 0), x.get('iter', 0)))
        
        # 为每个epoch保留最后一个iter的数据
        for record in sorted_data:
            epoch = record.get('epoch', 0)
            if epoch in aligned_data[head_name]:
                # 使用最后一个iter的数据
                aligned_data[head_name][epoch] = record
    
    return aligned_data, all_epochs

def save_metrics_to_excel():
    """将训练指标保存到Excel文件"""
    
    # 定义三个头部的日志文件路径
    log_files = {
        'head1': '/home/mengchao/workspace/P2BFoV/P2BNet-main/TOV_mmdetection/bfov/temp_log/head1/20260213_161455.log.json',
        'head2': '/home/mengchao/workspace/P2BFoV/P2BNet-main/TOV_mmdetection/bfov/temp_log/head2/20260214_104702.log.json',
        'head3': '/home/mengchao/workspace/P2BFoV/P2BNet-main/TOV_mmdetection/bfov/temp_log/head3/20260217_165824.log.json',
        'head4': '/home/mengchao/workspace/P2BFoV/P2BNet-main/TOV_mmdetection/bfov/temp_log/head4/20260228_214635.log.json'
    }
    
    # 输出文件路径
    output_dir = '/home/mengchao/workspace/P2BFoV/P2BNet-main/TOV_mmdetection/bfov/analysis_results'
    excel_path = f'{output_dir}/training_metrics_analysis.xlsx'
    
    # 创建输出目录
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    all_data = {}
    for head_name, log_path in log_files.items():
        print(f"正在加载 {head_name} 的数据...")
        data = load_log_data(log_path)
        all_data[head_name] = data
        print(f"  {head_name}: 共 {len(data)} 条训练记录")
    
    # 按epoch对齐数据
    aligned_data, all_epochs = align_data_by_epoch(all_data)
    print(f"数据对齐完成，共有 {len(all_epochs)} 个epoch")
    
    # 定义要分析的指标（您指定的关键指标）
    metrics = [
        'loss',
        'stage0_loss_instance_mil',
        'stage0_bag_acc',
        'stage1_loss_instance_mil',
        'stage1_bag_acc',
        'stage1_neg_loss',
        'grad_norm',
        'lr'
    ]
    
    # Stage0 IoU相关指标
    stage0_iou_metrics = [
        'stage0_mean_ious',
        'stage0_s',
        'stage0_m',
        'stage0_l',
        'stage0_h'
    ]
    
    # Stage1 IoU相关指标
    stage1_iou_metrics = [
        'stage1_mean_ious',
        'stage1_s',
        'stage1_m',
        'stage1_l',
        'stage1_h'
    ]
    
    # 创建Excel写入器
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        
        # 为每个指标创建一个sheet页
        for metric in metrics:
            # 创建DataFrame
            df_data = {'Epoch': []}
            
            # 添加每个头部的数据
            for head_name in aligned_data.keys():
                df_data[head_name] = []
            
            # 按epoch填充数据
            for epoch in all_epochs:
                df_data['Epoch'].append(epoch)
                
                for head_name in aligned_data.keys():
                    if epoch in aligned_data[head_name] and metric in aligned_data[head_name][epoch]:
                        df_data[head_name].append(aligned_data[head_name][epoch][metric])
                    else:
                        df_data[head_name].append(None)
            
            # 创建DataFrame
            df = pd.DataFrame(df_data)
            
            # 设置Epoch为索引
            df.set_index('Epoch', inplace=True)
            
            # 添加指标描述
            metric_descriptions = {
                'loss': '总训练损失函数值',
                'stage0_loss_instance_mil': '第一阶段多实例学习损失',
                'stage0_bag_acc': '第一阶段包级别分类准确率',
                'stage1_loss_instance_mil': '第二阶段多实例学习损失',
                'stage1_bag_acc': '第二阶段包级别分类准确率',
                'stage1_neg_loss': '第二阶段负样本损失',
                'grad_norm': '梯度向量的范数',
                'lr': '训练过程中的学习率变化'
            }
            
            description = metric_descriptions.get(metric, '训练指标')
            
            # 保存到Excel
            df.to_excel(writer, sheet_name=metric, index=True)
            
            # 获取工作表并添加描述
            worksheet = writer.sheets[metric]
            worksheet.cell(row=1, column=1, value=f'{metric} - {description}')
            
            print(f"已保存指标: {metric}")
        
        # 创建Stage0 IoU整合sheet页 - 按类别组织
        stage0_iou_data = {'Epoch': []}
        
        # 按类别定义列：平均iou、小物体iou、中物体iou、大物体iou、高iou
        iou_categories = [
            ('stage0_mean_ious', '平均IoU'),
            ('stage0_s', '小物体IoU'),
            ('stage0_m', '中物体IoU'),
            ('stage0_l', '大物体IoU'),
            ('stage0_h', '超大IoU')
        ]
        
        # 为每个类别创建列（每个head一列）
        for metric, category_name in iou_categories:
            for head_name in aligned_data.keys():
                stage0_iou_data[f'{head_name}_{category_name}'] = []
        
        # 按epoch填充数据
        for epoch in all_epochs:
            stage0_iou_data['Epoch'].append(epoch)
            
            # 为每个类别和head填充数据
            for metric, category_name in iou_categories:
                for head_name in aligned_data.keys():
                    if epoch in aligned_data[head_name] and metric in aligned_data[head_name][epoch]:
                        stage0_iou_data[f'{head_name}_{category_name}'].append(aligned_data[head_name][epoch][metric])
                    else:
                        stage0_iou_data[f'{head_name}_{category_name}'].append(None)
        
        # 创建DataFrame
        stage0_iou_df = pd.DataFrame(stage0_iou_data)
        stage0_iou_df.set_index('Epoch', inplace=True)
        
        # 保存到Excel
        stage0_iou_df.to_excel(writer, sheet_name='Stage0_IoU_Metrics', index=True)
        worksheet = writer.sheets['Stage0_IoU_Metrics']
        worksheet.cell(row=1, column=1, value='Stage0 IoU相关指标整合')
        print("已保存Stage0 IoU整合页")
        
        # 创建Stage1 IoU整合sheet页 - 按类别组织
        stage1_iou_data = {'Epoch': []}
        
        # 按类别定义列：平均iou、小物体iou、中物体iou、大物体iou、高iou
        iou_categories = [
            ('stage1_mean_ious', '平均IoU'),
            ('stage1_s', '小物体IoU'),
            ('stage1_m', '中物体IoU'),
            ('stage1_l', '大物体IoU'),
            ('stage1_h', '超大IoU')
        ]
        
        # 为每个类别创建列（每个head一列）
        for metric, category_name in iou_categories:
            for head_name in aligned_data.keys():
                stage1_iou_data[f'{head_name}_{category_name}'] = []
        
        # 按epoch填充数据
        for epoch in all_epochs:
            stage1_iou_data['Epoch'].append(epoch)
            
            # 为每个类别和head填充数据
            for metric, category_name in iou_categories:
                for head_name in aligned_data.keys():
                    if epoch in aligned_data[head_name] and metric in aligned_data[head_name][epoch]:
                        stage1_iou_data[f'{head_name}_{category_name}'].append(aligned_data[head_name][epoch][metric])
                    else:
                        stage1_iou_data[f'{head_name}_{category_name}'].append(None)
        
        # 创建DataFrame
        stage1_iou_df = pd.DataFrame(stage1_iou_data)
        stage1_iou_df.set_index('Epoch', inplace=True)
        
        # 保存到Excel
        stage1_iou_df.to_excel(writer, sheet_name='Stage1_IoU_Metrics', index=True)
        worksheet = writer.sheets['Stage1_IoU_Metrics']
        worksheet.cell(row=1, column=1, value='Stage1 IoU相关指标整合')
        print("已保存Stage1 IoU整合页")
        
        # 创建汇总sheet页
        summary_data = {}
        for head_name, data in all_data.items():
            if data:
                final_record = data[-1]
                first_record = data[0]
                
                summary_data[f'{head_name}_final'] = {}
                summary_data[f'{head_name}_change'] = {}
                
                for metric in metrics:
                    if metric in final_record and metric in first_record:
                        summary_data[f'{head_name}_final'][metric] = final_record[metric]
                        summary_data[f'{head_name}_change'][metric] = final_record[metric] - first_record[metric]
        
        # 创建汇总DataFrame
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=True)
        print("已保存汇总页")
    
    print(f"\nExcel文件已保存到: {excel_path}")
    print("每个指标对应一个sheet页，方便您进行数据分析")
    
    # 显示文件信息
    file_size = os.path.getsize(excel_path) / 1024  # KB
    print(f"文件大小: {file_size:.2f} KB")

if __name__ == '__main__':
    save_metrics_to_excel()