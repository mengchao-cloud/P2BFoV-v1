import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import os

PI = math.pi


def visualize_erp_coords(erp_coords, erp_width=1024, erp_height=512, point_size=2, color=(0, 255, 0), save_path=None, show=True):
    """
    在空白ERP图像上可视化坐标点
    """
    original_shape = erp_coords.shape
    if erp_coords.ndim == 3:
        erp_coords = erp_coords[np.newaxis, :, :, :]
    
    N, grid_size, _, _ = erp_coords.shape
    
    image = np.zeros((erp_height, erp_width, 3), dtype=np.uint8) + 255
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    
    for idx in range(N):
        coords = erp_coords[idx].reshape(-1, 2)
        color = colors[idx % len(colors)]
        
        x_coords = np.round(coords[:, 0]).astype(int)
        y_coords = np.round(coords[:, 1]).astype(int)
        
        x_coords = np.clip(x_coords, 0, erp_width - 1)
        y_coords = np.clip(y_coords, 0, erp_height - 1)
        
        for x, y in zip(x_coords, y_coords):
            cv2.circle(image, (x, y), point_size, color, -1)
    
    if save_path is not None:
        cv2.imwrite(save_path, image)
        print(f"图像已保存到: {save_path}")
    
    if show:
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"ERP坐标可视化 ({erp_width}×{erp_height})")
        plt.xlabel("像素X坐标")
        plt.ylabel("像素Y坐标")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return image


def rbfov_roi_to_14x14_erp(rbfov_params, erp_width=1024, erp_height=512, grid_size=14):
    """
    将RBFoV参数转换为14×14 ERP采样点
    
    Args:
        rbfov_params: Tensor of shape (N, 5) containing RBFoV parameters.
                    Each row is [lon, lat, fov_u, fov_v, angle] - all in radians:
                    - lon: RBFoV center longitude (radians, range [-π, π])
                    - lat: RBFoV center latitude (radians, range [-π/2, π/2])
                    - fov_u: Horizontal field of view (radians)
                    - fov_v: Vertical field of view (radians)
                    - angle: Rotation angle around the center point (radians)
        erp_width: ERP image width, default 1024
        erp_height: ERP image height, default 512
        grid_size: Grid size, default 14
    
    Returns:
        erp_points: Tensor of shape (N, 14, 14, 2) containing ERP coordinates
    """
    N = rbfov_params.size(0)
    device = rbfov_params.device
    
    # 提取参数
    lon = rbfov_params[:, 0]      # 经度 (N,)
    lat = rbfov_params[:, 1]      # 纬度 (N,)
    fov_u = rbfov_params[:, 2]    # 水平视场 (N,)
    fov_v = rbfov_params[:, 3]    # 垂直视场 (N,)
    angle = rbfov_params[:, 4]    # 旋转角度 (N,)
    
    # 转换为球坐标
    phi0 = lon                     # 方位角 (N,)
    theta0 = PI / 2 - lat          # 极角 (N,)
    
    # 1. 计算中心点单位向量 c
    c_x = torch.sin(theta0) * torch.cos(phi0)
    c_y = torch.sin(theta0) * torch.sin(phi0)
    c_z = torch.cos(theta0)
    c = torch.stack([c_x, c_y, c_z], dim=1)  # (N, 3)
    
    # 2. 构建局部正交基（未旋转）
    sin_phi = torch.sin(phi0)
    cos_phi = torch.cos(phi0)
    sin_theta = torch.sin(theta0)
    cos_theta = torch.cos(theta0)
    
    # 经度方向向量（水平，指向东）
    lon_dir = torch.stack([-sin_phi, cos_phi, torch.zeros_like(sin_phi)], dim=1)  # (N, 3)
    # 纬度方向向量（垂直，指向北）
    lat_dir = torch.stack([-cos_theta * cos_phi, -cos_theta * sin_phi, sin_theta], dim=1)  # (N, 3)
    
    # 确保正交性
    dot_product = torch.sum(lon_dir * lat_dir, dim=1)
    mask = torch.abs(dot_product) > 1e-6
    
    if mask.any():
        lon_dir_ortho = lon_dir - torch.sum(lon_dir * c, dim=1, keepdim=True) * c
        lon_dir_ortho = lon_dir_ortho / torch.norm(lon_dir_ortho, dim=1, keepdim=True)
        lat_dir_ortho = torch.cross(c, lon_dir_ortho)
        lat_dir_ortho = lat_dir_ortho / torch.norm(lat_dir_ortho, dim=1, keepdim=True)
        
        lon_dir[mask] = lon_dir_ortho[mask]
        lat_dir[mask] = lat_dir_ortho[mask]
    
    # 3. 应用旋转角度：旋转局部坐标系
    # 绕中心点位置向量c旋转angle角度
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    
    # 旋转后的水平方向向量
    lon_dir_rot = cos_angle.unsqueeze(1) * lon_dir + sin_angle.unsqueeze(1) * lat_dir
    # 旋转后的垂直方向向量
    lat_dir_rot = -sin_angle.unsqueeze(1) * lon_dir + cos_angle.unsqueeze(1) * lat_dir
    
    # 4. 计算切平面半宽
    U = torch.tan(fov_u / 2.0)  # 水平半宽
    V = torch.tan(fov_v / 2.0)  # 垂直半宽
    
    # 5. 计算网格步长
    delta_u = 2 * U / grid_size
    delta_v = 2 * V / grid_size
    
    # 6. 生成网格索引
    indices = torch.zeros(grid_size, grid_size, 2, device=device, dtype=torch.int64)
    for i in range(grid_size):
        for j in range(grid_size):
            indices[i, j, 0] = i
            indices[i, j, 1] = j
    
    i_grid = indices[..., 0].float()  # 垂直方向索引
    j_grid = indices[..., 1].float()  # 水平方向索引
    
    # 7. 计算网格中心坐标（切平面上）
    U_expanded = U.view(N, 1, 1)
    V_expanded = V.view(N, 1, 1)
    delta_u_expanded = delta_u.view(N, 1, 1)
    delta_v_expanded = delta_v.view(N, 1, 1)
    
    j_plus_half = j_grid + 0.5
    vertical_idx = (grid_size - 1 - i_grid) + 0.5  # 反转垂直方向
    
    u_center = -U_expanded + j_plus_half.unsqueeze(0) * delta_u_expanded
    v_center = -V_expanded + vertical_idx.unsqueeze(0) * delta_v_expanded
    
    # 8. 扩展基向量到网格尺寸
    c_expand = c.unsqueeze(1).unsqueeze(1).expand(N, grid_size, grid_size, 3)
    e_u_expand = lon_dir_rot.unsqueeze(1).unsqueeze(1).expand(N, grid_size, grid_size, 3)
    e_v_expand = lat_dir_rot.unsqueeze(1).unsqueeze(1).expand(N, grid_size, grid_size, 3)
    
    # 9. 计算切平面上的点
    x_plane = c_expand + u_center.unsqueeze(-1) * e_u_expand + v_center.unsqueeze(-1) * e_v_expand
    
    # 10. 投影到单位球面
    x_sphere = x_plane / torch.norm(x_plane, dim=-1, keepdim=True)
    
    # 11. 转换为经纬度（度数）
    x = x_sphere[..., 0]
    y = x_sphere[..., 1]
    z = x_sphere[..., 2]
    
    theta = torch.acos(z)
    phi = torch.atan2(y, x)
    
    lat_deg = 90.0 - (theta * 180.0 / PI)
    lon_deg = phi * 180.0 / PI
    
    # 12. 转换为ERP坐标
    erp_x = (lon_deg + 180.0) / 360.0 * erp_width
    erp_y = (90.0 - lat_deg) / 180.0 * erp_height
    
    # 13. 堆叠结果
    erp_points = torch.stack([erp_x, erp_y], dim=-1)
    
    return erp_points


def erp_to_lonlat(erp_x, erp_y, erp_width=1024, erp_height=512):
    """将ERP像素坐标转换为经纬度（度数）"""
    lon = (erp_x / erp_width) * 360.0 - 180.0
    lat = 90.0 - (erp_y / erp_height) * 180.0
    return lon, lat


def lonlat_to_xyz(lon_deg, lat_deg):
    """将经纬度（度数）转换为三维单位向量"""
    lon = lon_deg * PI / 180.0
    lat = lat_deg * PI / 180.0
    
    theta = PI / 2 - lat
    phi = lon
    
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    
    return torch.stack([x, y, z], dim=-1)


if __name__ == "__main__":
    print("=== RBFoV 14×14 ERP 采样点生成 ===")
    
    # 生成示例RBFoV参数 [lon, lat, fov_u, fov_v, angle]（弧度）
    # 1. 中心在 (0°, 0°)，水平60°，垂直40°，无旋转
    rbfov1 = torch.tensor([0, 0, PI/3, PI/4.5, -PI/3], dtype=torch.float32)
    
    # 2. 中心在 (45°, 30°)，水平90°，垂直60°，旋转45°
    rbfov2 = torch.tensor([PI/4, PI/6, PI/2, PI/3, PI/4], dtype=torch.float32)
    
    # 3. 中心在 (-90°, 45°)，水平120°，垂直80°，旋转-30°
    rbfov3 = torch.tensor([-PI/2, PI/4, 2*PI/3, 4*PI/9, -2*PI/3], dtype=torch.float32)
    
    rbfov_params = torch.stack([rbfov1, rbfov2, rbfov3])
    print(f"\nRBFoV参数形状: {rbfov_params.shape}")
    
    print("\nRBFoV参数（弧度）:")
    for i, rbfov in enumerate(rbfov_params):
        print(f"RBFoV {i+1}: lon={rbfov[0]:.4f}, lat={rbfov[1]:.4f}, fov_u={rbfov[2]:.4f}, fov_v={rbfov[3]:.4f}, angle={rbfov[4]:.4f}")
        print(f"          (经度={rbfov[0]*180/PI:.1f}°, 纬度={rbfov[1]*180/PI:.1f}°, 水平视场={rbfov[2]*180/PI:.1f}°, 垂直视场={rbfov[3]*180/PI:.1f}°, 旋转角={rbfov[4]*180/PI:.1f}°)")
    
    # 调用转换函数
    print("\n调用rbfov_roi_to_14x14_erp函数...")
    erp_points = rbfov_roi_to_14x14_erp(rbfov_params)
    
    print(f"\n生成的ERP点形状: {erp_points.shape}")
    print(f"每个RBFoV包含 14×14=196 个采样点")
    
    # 保存结果到文本文件
    output_file = './bfov/roi_result/rbfov_erp_points_output.txt'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("=== RBFoV 14x14 ERP 采样点输出 ===\n")
        f.write("格式: RBFoV 参数 [lon(弧度), lat(弧度), fov_u(弧度), fov_v(弧度), angle(弧度)]\n")
        f.write("      14x14 ERP 点坐标 (x, y)\n\n")
        
        for b_idx in range(erp_points.shape[0]):
            f.write(f"=== RBFoV {b_idx + 1} ===\n")
            rbfov_param = rbfov_params[b_idx]
            f.write(f"RBFoV 参数: [{rbfov_param[0]:.6f}, {rbfov_param[1]:.6f}, {rbfov_param[2]:.6f}, {rbfov_param[3]:.6f}, {rbfov_param[4]:.6f}]\n")
            f.write(f"(lon={rbfov_param[0]*180/PI:.2f}° lat={rbfov_param[1]*180/PI:.2f}° fov_u={rbfov_param[2]*180/PI:.2f}° fov_v={rbfov_param[3]*180/PI:.2f}° angle={rbfov_param[4]*180/PI:.2f}°)\n\n")
            
            f.write("14x14 ERP 点坐标:\n")
            f.write("    ")
            for j in range(14):
                f.write(f"{j:4d} ")
            f.write("    列索引\n")
            
            for i in range(14):
                f.write(f"{i:2d}  ")
                for j in range(14):
                    x, y = erp_points[b_idx, i, j]
                    f.write(f"({x:.2f},{y:.2f}) ")
                f.write(f"  行={i}\n")
            f.write("\n" + "="*60 + "\n\n")
    
    print(f"\nRBFoV参数和ERP点已保存到: {output_file}")
    
    # 可视化结果
    erp_points_np = erp_points.cpu().numpy()
    save_path = './bfov/roi_result/rbfov_samples_visualization.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    visualize_erp_coords(erp_points_np, point_size=3, save_path=save_path, show=False)
    
    print("\n处理完成！可视化结果已保存。")
