import torch
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.formatting.rule import CellIsRule

# ==================== 修复后的核心算法 ====================

def robust_log_map(q, points, eps=1e-8):
    """
    稳健的球面Log映射（切空间投影）
    处理sin_distance过小的情况
    """
    dot_products = torch.sum(points * q.unsqueeze(0), dim=-1)
    dot_products = torch.clamp(dot_products, -0.999999, 0.999999)
    
    distances = torch.arccos(dot_products)
    sin_distances = torch.sin(distances)
    
    # 数值稳定性：当sin很小时使用一阶近似
    safe_sin = torch.where(
        sin_distances < eps,
        torch.ones_like(sin_distances) * eps,  # 避免除零
        sin_distances
    )
    
    # 切向量 = (p - cos* q) / sin
    tangent_numerators = points - dot_products.unsqueeze(-1) * q.unsqueeze(0)
    tangent_vectors = tangent_numerators / safe_sin.unsqueeze(-1)
    
    # 标记数值警告
    numerical_warning = torch.any(sin_distances < eps).item()
    
    return tangent_vectors, distances, sin_distances, numerical_warning

def iterative_initialization(points, weights, n_iter=3):
    """
    改进的初始化：迭代球面平均（比单次欧几里得平均更稳健）
    """
    # 从欧几里得加权平均开始
    q = (points * weights.unsqueeze(-1)).sum(dim=0)
    q = q / (torch.norm(q) + 1e-12)
    
    # 迭代精炼
    for _ in range(n_iter):
        # 投影到切空间
        dots = torch.sum(points * q.unsqueeze(0), dim=-1, keepdim=True)
        tangent_vecs = points - dots * q.unsqueeze(0)
        
        # 切空间加权平均
        avg_tangent = (tangent_vecs * weights.unsqueeze(-1)).sum(dim=0)
        
        # 指数映射回球面（近似）
        q_new = q + avg_tangent
        q_norm = torch.norm(q_new)
        if q_norm > 1e-12:
            q = q_new / q_norm
    
    return q

def spherical_weighted_average_v2(
    points, 
    weights, 
    max_iter=100, 
    tol=1e-8,
    step_size_mode='adaptive',  # 'adaptive', 'fixed', 'line_search'
    initial_step=1.0,
    min_step=0.1,
    verbose=False
):
    """
    修复后的球面加权平均算法
    
    改进点：
    1. 稳健的切空间计算（处理sin≈0）
    2. 步长控制（自适应衰减或线搜索）
    3. 改进的初始化
    4. 详细的调试信息
    """
    if points.dim() == 2:
        points = points.unsqueeze(0)
        weights = weights.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size, num_points, _ = points.shape
    
    # 改进的初始化
    q = iterative_initialization(points[0], weights[0])
    if not squeeze_output:
        q = q.unsqueeze(0)
    
    # 调试信息
    debug_info = {
        'initial_q': q.clone(),
        'iterations': [],
        'residual_history': [],
        'step_history': [],
        'warnings': []
    }
    
    converged = False
    current_step = initial_step
    
    for iter_idx in range(max_iter):
        # 稳健的Log映射
        tangent_vectors, distances, sin_dists, num_warning = robust_log_map(
            q.squeeze(0) if not squeeze_output else q, 
            points[0] if not squeeze_output else points,
            eps=1e-8
        )
        
        if num_warning:
            debug_info['warnings'].append(f"Iter {iter_idx}: sin_distance too small")
        
        # 切空间加权平均
        w_normalized = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
        u_total = (w_normalized.unsqueeze(-1) * tangent_vectors).sum(dim=0 if squeeze_output else 1)
        u_norm = torch.norm(u_total)
        
        # 记录迭代信息
        debug_info['iterations'].append({
            'iter': iter_idx,
            'residual': u_norm.item(),
            'max_dist': distances.max().item() if squeeze_output else distances[0].max().item(),
            'min_sin': sin_dists.min().item() if squeeze_output else sin_dists[0].min().item(),
            'step': current_step
        })
        debug_info['residual_history'].append(u_norm.item())
        debug_info['step_history'].append(current_step)
        
        # 收敛检查
        if u_norm < tol:
            converged = True
            if verbose:
                print(f"Converged at iter {iter_idx}, residual: {u_norm.item():.2e}")
            break
        
        # 步长控制策略
        if step_size_mode == 'adaptive':
            # 自适应衰减：随迭代次数指数衰减，但保持最小值
            current_step = max(min_step, initial_step * (0.9 ** iter_idx))
        elif step_size_mode == 'fixed':
            current_step = initial_step
        elif step_size_mode == 'line_search':
            # 简单的Armijo线搜索
            alpha = 1.0
            for _ in range(5):  # 最多尝试5次
                q_test = q + alpha * u_total.unsqueeze(0) if not squeeze_output else q + alpha * u_total
                q_test = q_test / (torch.norm(q_test, dim=-1, keepdim=True) + 1e-12)
                
                # 检查是否更接近目标（残差减小）
                new_dots = torch.sum(points * q_test.unsqueeze(1), dim=-1)
                new_dists = torch.arccos(torch.clamp(new_dots, -0.999999, 0.999999))
                new_residual = (w_normalized * new_dists).sum()
                
                if new_residual < u_norm:
                    break
                alpha *= 0.5
            current_step = alpha
        
        # 更新位置（带步长）
        q_new = q + current_step * (u_total.unsqueeze(0) if not squeeze_output else u_total)
        q_norm = torch.norm(q_new, dim=-1, keepdim=True)
        
        if torch.any(q_norm < 1e-12):
            debug_info['warnings'].append(f"Iter {iter_idx}: q_new norm too small")
            break
            
        q = q_new / q_norm
    
    debug_info['final_q'] = q.clone()
    
    if squeeze_output:
        return q.squeeze(0), iter_idx + 1, converged, debug_info
    return q, iter_idx + 1, converged, debug_info

# ==================== 数据生成与测试 ====================

def generate_test_data_v2(n_samples=100, seed=42):
    """生成更具挑战性的测试数据"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    data_list = []
    
    scenarios = [
        ('clustered', 15),      # 聚集，简单
        ('uniform', 25),        # 均匀，中等
        ('dominant', 25),        # 主导权重，中等
        ('wide_spread', 25),     # 大角度分布，困难
        ('extreme', 10)          # 极端情况（近对跖点）
    ]
    
    sample_id = 0
    for scenario_name, count in scenarios:
        for _ in range(count):
            if scenario_name == 'clustered':
                center = torch.randn(3)
                center = center / center.norm()
                points = []
                for _ in range(4):
                    noise = torch.randn(3) * 0.2
                    p = center + noise
                    p = p / p.norm()
                    points.append(p)
                points = torch.stack(points)
                weights = torch.tensor([0.4, 0.3, 0.2, 0.1])
                
            elif scenario_name == 'uniform':
                center = torch.randn(3)
                center = center / center.norm()
                points = []
                for angle_deg in [0, 30, 60, 90]:
                    angle = np.radians(angle_deg)
                    tangent = torch.randn(3)
                    tangent = tangent - tangent.dot(center) * center
                    if tangent.norm() > 0:
                        tangent = tangent / tangent.norm()
                    p = torch.cos(torch.tensor(angle)) * center + torch.sin(torch.tensor(angle)) * tangent
                    points.append(p)
                points = torch.stack(points)
                weights = torch.ones(4) / 4
                
            elif scenario_name == 'dominant':
                center = torch.randn(3)
                center = center / center.norm()
                points = []
                for _ in range(4):
                    noise = torch.randn(3) * 0.4
                    p = center + noise
                    p = p / p.norm()
                    points.append(p)
                points = torch.stack(points)
                weights = torch.tensor([0.85, 0.05, 0.05, 0.05])
                
            elif scenario_name == 'wide_spread':
                # 90-100度分布
                center = torch.tensor([1., 0., 0.])
                angles = [0, 45, 80, 95]
                points = []
                base_tangent = torch.tensor([0., 1., 0.])
                for angle_deg in angles:
                    angle = np.radians(angle_deg)
                    p = torch.cos(torch.tensor(angle)) * center + torch.sin(torch.tensor(angle)) * base_tangent
                    p = p / p.norm()
                    points.append(p)
                points = torch.stack(points)
                weights = torch.tensor([0.35, 0.35, 0.2, 0.1])
                
            elif scenario_name == 'extreme':
                # 近对跖点（最困难）
                p1 = torch.tensor([1., 0., 0.])
                p2 = torch.tensor([-0.95, 0.3122, 0.])  # 约107度
                p3 = torch.tensor([0.5, 0.866, 0.])
                p4 = torch.tensor([0.6, -0.8, 0.])
                points = torch.stack([p1, p2, p3, p4])
                weights = torch.tensor([0.45, 0.45, 0.05, 0.05])
            
            # 计算真值（使用非常严格的参数）
            with torch.no_grad():
                gt, _, _, _ = spherical_weighted_average_v2(
                    points, weights, 
                    max_iter=500, tol=1e-12, 
                    step_size_mode='adaptive', initial_step=0.5
                )
            
            data_list.append({
                'sample_id': sample_id,
                'scenario': scenario_name,
                'points': points,
                'weights': weights,
                'ground_truth': gt
            })
            sample_id += 1
    
    return data_list

def compute_geodesic_error(pred, gt):
    """计算测地线误差（度）"""
    cos_sim = torch.clamp(torch.dot(pred, gt), -1, 1)
    error_rad = torch.arccos(cos_sim)
    return error_rad.item() * 180 / np.pi

def run_comparison_study(data_list):
    """对比原算法和修复后的算法"""
    results = []
    
    print("Running comparison study...")
    
    for sample in tqdm(data_list):
        sample_id = sample['sample_id']
        scenario = sample['scenario']
        points = sample['points']
        weights = sample['weights']
        gt = sample['ground_truth']
        
        # 计算输入特征
        pairwise_angles = []
        for i in range(4):
            for j in range(i+1, 4):
                cos_ij = torch.clamp(torch.dot(points[i], points[j]), -1, 1)
                angle = torch.arccos(cos_ij) * 180 / np.pi
                pairwise_angles.append(angle.item())
        max_angle = max(pairwise_angles)
        min_angle = min(pairwise_angles)
        
        # 测试原算法（大步长=1.0）
        try:
            pred_old, iters_old, conv_old, debug_old = spherical_weighted_average_v2(
                points, weights, max_iter=50, tol=1e-6, 
                step_size_mode='fixed', initial_step=1.0
            )
            error_old = compute_geodesic_error(pred_old, gt)
        except Exception as e:
            pred_old, iters_old, conv_old, error_old = None, 50, False, 180.0
        
        # 测试修复后算法（自适应步长）
        try:
            pred_new, iters_new, conv_new, debug_new = spherical_weighted_average_v2(
                points, weights, max_iter=50, tol=1e-6,
                step_size_mode='adaptive', initial_step=1.0, min_step=0.1
            )
            error_new = compute_geodesic_error(pred_new, gt)
        except Exception as e:
            pred_new, iters_new, conv_new, error_new = None, 50, False, 180.0
        
        # 记录结果
        result = {
            'sample_id': sample_id,
            'scenario': scenario,
            'max_input_angle': max_angle,
            'min_input_angle': min_angle,
            
            # 原算法结果
            'old_error_deg': error_old,
            'old_converged': conv_old,
            'old_iterations': iters_old,
            'old_final_residual': debug_old['residual_history'][-1] if debug_old['residual_history'] else 1.0,
            
            # 修复后结果
            'new_error_deg': error_new,
            'new_converged': conv_new,
            'new_iterations': iters_new,
            'new_final_residual': debug_new['residual_history'][-1] if debug_new['residual_history'] else 1.0,
            
            # 对比
            'improvement': error_old - error_new,
            'converge_improved': (not conv_old) and conv_new,
            'error_reduced': error_new < error_old * 0.5  # 误差减半
        }
        
        # 添加迭代历史（前5次）
        for i in range(min(5, len(debug_old.get('residual_history', [])))):
            result[f'old_res_{i}'] = debug_old['residual_history'][i]
            result[f'new_res_{i}'] = debug_new['residual_history'][i]
        
        results.append(result)
    
    return pd.DataFrame(results)

def export_comparison_excel(df, filename='spherical_comparison_fixed.xlsx'):
    """导出对比结果到Excel"""
    print(f"Exporting to {filename}...")
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # 主表
        df.to_excel(writer, sheet_name='Comparison Results', index=False)
        
        # 按场景汇总
        summary = df.groupby('scenario').agg({
            'old_error_deg': ['mean', 'max'],
            'new_error_deg': ['mean', 'max'],
            'old_converged': 'mean',
            'new_converged': 'mean',
            'old_iterations': 'mean',
            'new_iterations': 'mean',
            'converge_improved': 'sum',
            'error_reduced': 'sum'
        }).round(4)
        summary.to_excel(writer, sheet_name='Scenario Summary')
        
        # 失败案例对比
        failed_old = df[(df['old_converged'] == False) & (df['new_converged'] == True)]
        if not failed_old.empty:
            failed_old.to_excel(writer, sheet_name='Fixed Cases', index=False)
        
        # 仍然失败的案例
        still_failed = df[(df['old_converged'] == False) & (df['new_converged'] == False)]
        if not still_failed.empty:
            still_failed.to_excel(writer, sheet_name='Still Failed', index=False)
    
    # 格式化
    wb = openpyxl.load_workbook(filename)
    ws = wb['Comparison Results']
    
    # 标题行格式
    header_fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
    header_font = Font(color='FFFFFF', bold=True)
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
    
    # 条件格式化：高亮改进显著的行
    red_fill = PatternFill(start_color='FF0000', end_color='FF0000', fill_type='solid')
    green_fill = PatternFill(start_color='00FF00', end_color='00FF00', fill_type='solid')
    
    # 找到关键列的索引
    old_err_col = None
    new_err_col = None
    for idx, cell in enumerate(ws[1], 1):
        if cell.value == 'old_error_deg':
            old_err_col = openpyxl.utils.get_column_letter(idx)
        if cell.value == 'new_error_deg':
            new_err_col = openpyxl.utils.get_column_letter(idx)
    
    if old_err_col and new_err_col:
        # 红色：修复后仍然误差大
        ws.conditional_formatting.add(
            f'{new_err_col}2:{new_err_col}{ws.max_row}',
            CellIsRule(operator='greaterThan', formula=['10'], fill=red_fill)
        )
        # 绿色：修复成功且误差小
        ws.conditional_formatting.add(
            f'{new_err_col}2:{new_err_col}{ws.max_row}',
            CellIsRule(operator='lessThan', formula=['0.1'], fill=green_fill)
        )
    
    wb.save(filename)
    print(f"Saved: {filename}")
    
    # 打印统计
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Total samples: {len(df)}")
    print(f"Original converged: {df['old_converged'].sum()}/{len(df)} ({df['old_converged'].mean():.1%})")
    print(f"Fixed converged: {df['new_converged'].sum()}/{len(df)} ({df['new_converged'].mean():.1%})")
    print(f"Convergence improved: {df['converge_improved'].sum()} cases")
    print(f"Error reduced by half: {df['error_reduced'].sum()} cases")
    print(f"Mean error (old): {df['old_error_deg'].mean():.2f}°")
    print(f"Mean error (new): {df['new_error_deg'].mean():.2f}°")
    print(f"Max error (old): {df['old_error_deg'].max():.2f}°")
    print(f"Max error (new): {df['new_error_deg'].max():.2f}°")

def detailed_analysis_sample(sample, sample_id=0):
    """对单个样本进行详细分析（调试用）"""
    print(f"\n{'='*60}")
    print(f"Detailed Analysis for Sample {sample_id} ({sample['scenario']})")
    print(f"{'='*60}")
    
    points = sample['points']
    weights = sample['weights']
    gt = sample['ground_truth']
    
    print(f"Input angles: ", end="")
    for i in range(4):
        for j in range(i+1, 4):
            cos_ij = torch.clamp(torch.dot(points[i], points[j]), -1, 1)
            angle = torch.arccos(cos_ij) * 180 / np.pi
            print(f"{angle:.1f}° ", end="")
    print()
    
    # 测试不同步长策略
    strategies = [
        ('Fixed(1.0)', 'fixed', 1.0),
        ('Fixed(0.5)', 'fixed', 0.5),
        ('Fixed(0.1)', 'fixed', 0.1),
        ('Adaptive', 'adaptive', 1.0),
        ('LineSearch', 'line_search', 1.0)
    ]
    
    results = []
    for name, mode, step in strategies:
        pred, iters, conv, debug = spherical_weighted_average_v2(
            points, weights, max_iter=100, tol=1e-8,
            step_size_mode=mode, initial_step=step, verbose=False
        )
        error = compute_geodesic_error(pred, gt)
        results.append({
            'Strategy': name,
            'Converged': conv,
            'Iters': iters,
            'Error(deg)': error,
            'Final Residual': debug['residual_history'][-1] if debug['residual_history'] else 999
        })
        
        print(f"{name:12s}: Conv={conv}, Iters={iters:3d}, Error={error:8.4f}°, Residual={debug['residual_history'][-1]:.2e}")
    
    return pd.DataFrame(results)

# ==================== 主程序 ====================

if __name__ == "__main__":
    # 生成数据
    print("Generating test data...")
    data = generate_test_data_v2(n_samples=100, seed=42)
    
    # 可选：对特定样本进行详细分析
    # wide_samples = [d for d in data if d['scenario'] == 'wide_spread']
    # if wide_samples:
    #     detailed_analysis_sample(wide_samples[0], sample_id=wide_samples[0]['sample_id'])
    
    # 运行对比实验
    df_results = run_comparison_study(data)
    
    # 导出到Excel
    export_comparison_excel(df_results, 'spherical_comparison_fixed.xlsx')
    
    # 按场景分析
    print("\nDetailed breakdown by scenario:")
    for scenario in df_results['scenario'].unique():
        subset = df_results[df_results['scenario'] == scenario]