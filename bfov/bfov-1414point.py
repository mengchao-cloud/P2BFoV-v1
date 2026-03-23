import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math

PI = math.pi


# 辅助函数：可视化ERP坐标（与bfov-196point.py保持一致）
def visualize_erp_coords(erp_coords, erp_width=1024, erp_height=512, point_size=2, color=(0, 255, 0), save_path=None, show=True):
    """
    在空白ERP图像上可视化坐标点
    """
    # 处理输入形状
    original_shape = erp_coords.shape
    if erp_coords.ndim == 3:
        erp_coords = erp_coords[np.newaxis, :, :, :]  # 转换为 [N, 14, 14, 2]
    
    N, grid_size, _, _ = erp_coords.shape
    
    # 创建空白图像
    image = np.zeros((erp_height, erp_width, 3), dtype=np.uint8) + 255  # 白色背景
    
    # 为不同的BFOV使用不同颜色
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # 绿色、红色、蓝色
    
    # 在图像上绘制所有BFOV的坐标点
    for idx in range(N):
        coords = erp_coords[idx].reshape(-1, 2)  # 转换为 [196, 2]
        color = colors[idx % len(colors)]
        
        # 将坐标转换为整数像素值
        x_coords = np.round(coords[:, 0]).astype(int)
        y_coords = np.round(coords[:, 1]).astype(int)
        
        # 确保坐标在图像范围内
        x_coords = np.clip(x_coords, 0, erp_width - 1)
        y_coords = np.clip(y_coords, 0, erp_height - 1)
        
        # 绘制点
        for x, y in zip(x_coords, y_coords):
            cv2.circle(image, (x, y), point_size, color, -1)  # 实心圆
    
    # 保存图像
    if save_path is not None:
        cv2.imwrite(save_path, image)
        print(f"图像已保存到: {save_path}")
    
    # 显示图像
    if show:
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"ERP坐标可视化 ({erp_width}×{erp_height})")
        plt.xlabel("像素X坐标")
        plt.ylabel("像素Y坐标")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return image

# 主函数：BFOV转14x14 ERP采样点（保持原始逻辑）
def bfov_roi_to_14x14_erp(bfov_params, erp_width=1024, erp_height=512, grid_size=14):
    """
    Generate 14x14 ERP points from BFOV parameters using PyTorch.
    This version follows the exact same logic as the numpy version bfov_to_spherical_points.
    
    Args:
        bfov_params: Tensor of shape (N, 4) containing BFOV parameters for each ROI.
                    Each row is [lon, lat, fov_u, fov_v] - all in radians:
                    - lon: BFOV center longitude (radians, range [-π, π])
                    - lat: BFOV center latitude (radians, range [-π/2, π/2])
                    - fov_u: Horizontal field of view (radians, longitude direction)
                    - fov_v: Vertical field of view (radians, latitude direction)
        erp_width: ERP image width, default 1024
        erp_height: ERP image height, default 512
        grid_size: Grid size, default 14
    
    Returns:
        points: Tensor of shape (N, 14, 14, 2) containing 14x14 ERP points for each BFOV.
                Each BFOV corresponds to a 14x14 2D array where each element is (x, y) coordinates (float).
    """
    # Get batch size and device
    N = bfov_params.size(0)
    device = bfov_params.device
        
    # Extract BFOV parameters (all in radians)
    lon = bfov_params[:, 0]  # (N,), longitude in radians
    lat = bfov_params[:, 1]  # (N,), latitude in radians
    fov_u = bfov_params[:, 2]  # (N,), horizontal FOV in radians
    fov_v = bfov_params[:, 3]  # (N,), vertical FOV in radians
    
    # Convert longitude/latitude to polar/azimuth angles (same as numpy version)
    phi0 = lon  # Azimuth angle (N,)
    theta0 = PI/2 - lat  # Polar angle (N,)
    
    # 1. Calculate BFOV center unit vector c (x,y,z) - same as numpy version
    c_x = torch.sin(theta0) * torch.cos(phi0)  # (N,)
    c_y = torch.sin(theta0) * torch.sin(phi0)  # (N,)
    c_z = torch.cos(theta0)  # (N,)
    c = torch.stack([c_x, c_y, c_z], dim=1)  # (N, 3)
    
    # 2. Construct local orthogonal basis (e_u, e_v) - same as numpy version
    sin_phi = torch.sin(phi0)  # (N,)
    cos_phi = torch.cos(phi0)  # (N,)
    sin_theta = torch.sin(theta0)  # (N,)
    cos_theta = torch.cos(theta0)  # (N,)
    
    # Longitude direction vector (horizontal)
    lon_dir = torch.stack([-sin_phi, cos_phi, torch.zeros_like(sin_phi)], dim=1)  # (N, 3)
    # Latitude direction vector (vertical)
    lat_dir = torch.stack([-cos_theta*cos_phi, -cos_theta*sin_phi, sin_theta], dim=1)  # (N, 3)
    
    # Ensure orthogonality - same as numpy version
    dot_product = torch.sum(lon_dir * lat_dir, dim=1)  # (N,)
    mask = torch.abs(dot_product) > 1e-6  # (N,)
    
    if mask.any():
        # Recalculate orthogonal basis for problematic cases
        lon_dir_ortho = lon_dir - torch.sum(lon_dir * c, dim=1, keepdim=True) * c
        lon_dir_ortho = lon_dir_ortho / torch.norm(lon_dir_ortho, dim=1, keepdim=True)
        lat_dir_ortho = torch.cross(c, lon_dir_ortho)
        lat_dir_ortho = lat_dir_ortho / torch.norm(lat_dir_ortho, dim=1, keepdim=True)
        
        # Update where mask is True
        lon_dir[mask] = lon_dir_ortho[mask]
        lat_dir[mask] = lat_dir_ortho[mask]
    
    # 3. Calculate tangent plane half-width (U, V) - same as numpy version
    U = torch.tan(fov_u / 2.0)  # Horizontal half-width (N,)
    V = torch.tan(fov_v / 2.0)  # Vertical half-width (N,)
    
    # 4. Calculate grid step - same as numpy version
    delta_u = 2 * U / grid_size  # Horizontal step (N,)
    delta_v = 2 * V / grid_size  # Vertical step (N,)
    
    # 5. Generate grid indices - i对应垂直方向(v), j对应水平方向(u)
    indices = torch.zeros(grid_size, grid_size, 2, device=device, dtype=torch.int64)
    for i in range(grid_size):  # i: 垂直方向(v)索引
        for j in range(grid_size):  # j: 水平方向(u)索引
            indices[i, j, 0] = i  # 行索引 → 垂直方向
            indices[i, j, 1] = j  # 列索引 → 水平方向
    
    # 分离i和j索引
    i_grid = indices[..., 0].float()  # (14, 14) - 垂直方向(v)
    j_grid = indices[..., 1].float()  # (14, 14) - 水平方向(u)
    
    # 6. Calculate u_center and v_center - 修正坐标映射
    # u_center: 水平方向坐标，对应j索引(列)
    # v_center: 垂直方向坐标，对应i索引(行)
    
    # 扩展U, V, delta_u, delta_v到网格形状
    U_expanded = U.view(N, 1, 1)  # (N, 1, 1) - 水平半宽
    V_expanded = V.view(N, 1, 1)  # (N, 1, 1) - 垂直半宽
    delta_u_expanded = delta_u.view(N, 1, 1)  # (N, 1, 1) - 水平步长
    delta_v_expanded = delta_v.view(N, 1, 1)  # (N, 1, 1) - 垂直步长
    
    # 计算中心坐标 - 修正垂直方向映射
    # j对应水平方向(u)，从左到右
    j_plus_half = j_grid + 0.5  # 水平方向网格中心
    
    # i对应垂直方向(v)，反转垂直方向索引，确保从上到下
    # grid_size-1 - i_grid 确保i=0对应BFOV顶部，i=13对应BFOV底部
    vertical_idx = (grid_size - 1 - i_grid) + 0.5  # 垂直方向网格中心（已反转）
    
    # 水平方向(u): 从左到右
    u_center = -U_expanded + j_plus_half.unsqueeze(0) * delta_u_expanded  # (N, 14, 14)
    # 垂直方向(v): 从上到下
    v_center = -V_expanded + vertical_idx.unsqueeze(0) * delta_v_expanded  # (N, 14, 14)
    
    # 7. Expand basis vectors to match grid size
    c_expand = c.unsqueeze(1).unsqueeze(1).expand(N, grid_size, grid_size, 3)  # (N, 14, 14, 3)
    e_u_expand = lon_dir.unsqueeze(1).unsqueeze(1).expand(N, grid_size, grid_size, 3)  # (N, 14, 14, 3)
    e_v_expand = lat_dir.unsqueeze(1).unsqueeze(1).expand(N, grid_size, grid_size, 3)  # (N, 14, 14, 3)
    
    # 8. Calculate points on tangent plane - same as numpy version
    x_plane = c_expand + u_center.unsqueeze(-1) * e_u_expand + v_center.unsqueeze(-1) * e_v_expand  # (N, 14, 14, 3)
    
    # 9. Project back to unit sphere - same as numpy version
    x_sphere = x_plane / torch.norm(x_plane, dim=-1, keepdim=True)  # (N, 14, 14, 3)
    
    # 10. Convert spherical coordinates to lat/lon (degrees for ERP conversion)
    x = x_sphere[..., 0]
    y = x_sphere[..., 1]
    z = x_sphere[..., 2]
    
    theta = torch.acos(z)  # Polar angle
    phi = torch.atan2(y, x)  # Azimuth angle
    
    # Convert to lat/lon (degrees)
    lat_deg = 90.0 - (theta * 180.0 / PI)
    lon_deg = phi * 180.0 / PI
    
    # 11. Convert lat/lon to ERP coordinates - same as numpy version
    erp_x = (lon_deg + 180.0) / 360.0 * erp_width
    erp_y = (90.0 - lat_deg) / 180.0 * erp_height
    
    # 12. Stack to get final ERP points
    erp_points = torch.stack([erp_x, erp_y], dim=-1)  # (N, 14, 14, 2)
    
    return erp_points

# 示例代码：生成三个BFOV并可视化
if __name__ == "__main__":
    print("生成三个BFOV参数示例...")
    
    # 生成三个不同的BFOV参数 [lon, lat, fov_u, fov_v]，单位：弧度
    # 1. 中心在 (0, 0)，水平90度，垂直60度
    bfov1 = torch.tensor([0, -PI/2.2, PI/2, PI/3], dtype=torch.float32)
    
    # 2. 中心在 (45度, 30度)，水平60度，垂直45度
    bfov2 = torch.tensor([PI, -PI/6, PI/3, PI/4], dtype=torch.float32)
    
    # 3. 中心在 (-90度, 0度)，水平120度，垂直90度
    bfov3 = torch.tensor([-PI/2, PI/3, 2*PI/3, PI/2], dtype=torch.float32)
    
    # 组合成批次
    bfov_params = torch.stack([bfov1, bfov2, bfov3])
    print(f"BFOV参数形状: {bfov_params.shape}")
    print("BFOV参数 (弧度):")
    for i, bfov in enumerate(bfov_params):
        print(f"BFOV {i+1}: lon={bfov[0]:.4f}, lat={bfov[1]:.4f}, fov_u={bfov[2]:.4f}, fov_v={bfov[3]:.4f}")
    
    # 调用BFOV转采样点函数
    print("\n调用bfov_roi_to_14x14_erp函数生成采样点...")
    erp_points = bfov_roi_to_14x14_erp(bfov_params)
    
    print(f"\n生成的ERP点形状: {erp_points.shape}")
    print("每个BFOV包含14x14=196个采样点")
    
    # 将BFOV参数和ERP点打印到文本文件
    print("\n将BFOV参数和ERP点打印到文本文件...")
    output_file = './bfov/roi_result/erp_points_output.txt'
    
    with open(output_file, 'w') as f:
        # 写入文件头信息
        f.write("=== BFOV 14x14 ERP 采样点输出 ===\n")
        f.write("格式: BFOV 参数 [lon(弧度), lat(弧度), fov_u(弧度), fov_v(弧度)]\n")
        f.write("      14x14 ERP 点坐标 (x, y)\n\n")
        
        # 为每个BFOV写入参数和采样点
        for b_idx in range(erp_points.shape[0]):
            # 写入BFOV参数
            f.write(f"=== BFOV {b_idx + 1} ===\n")
            bfov_param = bfov_params[b_idx]
            f.write(f"BFOV 参数: [{bfov_param[0]:.6f}, {bfov_param[1]:.6f}, {bfov_param[2]:.6f}, {bfov_param[3]:.6f}]\n")
            f.write(f"(lon={bfov_param[0]*180/PI:.2f}° lat={bfov_param[1]*180/PI:.2f}° fov_u={bfov_param[2]*180/PI:.2f}° fov_v={bfov_param[3]*180/PI:.2f}°)\n\n")
            
            # 写入14x14的ERP点坐标，以二维格式排列
            f.write("14x14 ERP 点坐标:\n")
            f.write("    ")
            for j in range(14):
                f.write(f"{j:4d} ")
            f.write("    列索引\n")
            
            for i in range(14):
                f.write(f"{i:2d}  ")  # 行索引
                for j in range(14):
                    x, y = erp_points[b_idx, i, j]
                    f.write(f"({x:.2f},{y:.2f}) ")
                f.write(f"  行={i}\n")
            f.write("\n" + "="*50 + "\n\n")
    
    print(f"BFOV参数和ERP点已保存到: {output_file}")
    
    # 将PyTorch张量转换为NumPy数组以便可视化
    erp_points_np = erp_points.cpu().numpy()
    
    # 可视化结果
    print("\n可视化采样点...")
    save_path = './bfov/roi_result/bfov_samples_visualization.png'
    visualize_erp_coords(erp_points_np, point_size=3, save_path=save_path, show=False)
    
    print("\n处理完成！可视化结果已保存。")