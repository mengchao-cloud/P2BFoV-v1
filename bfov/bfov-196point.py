import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import torch

def spherical_to_latlon(spherical_points):
    """
    将单位球面上的三维坐标转换为经纬度
    
    参数:
        spherical_points: numpy 数组，形状为 [N, M, 3] 或 [M, 3]，每个点为 (x,y,z)
    
    返回:
        latlon: numpy 数组，形状为 [N, M, 2] 或 [M, 2]，每个点为 (纬度, 经度)，单位为度
                - 纬度范围：[-90°, 90°]
                - 经度范围：[-180°, 180°]
    """
    # 处理输入形状
    original_shape = spherical_points.shape
    if spherical_points.ndim == 2:
        spherical_points = spherical_points[np.newaxis, :, :]  # 转换为 [1, M, 3]
    
    N, M, _ = spherical_points.shape
    latlon = np.zeros((N, M, 2))
    
    x = spherical_points[..., 0]
    y = spherical_points[..., 1]
    z = spherical_points[..., 2]
    
    # 计算极角 theta（从 z 轴向下，范围 [0, π]）
    theta = np.arccos(z)
    
    # 计算方位角 phi（绕 z 轴，范围 [-π, π]）
    phi = np.arctan2(y, x)
    
    # 转换为经纬度（度）
    # 纬度：90° - (theta * 180° / π)，范围 [-90°, 90°]
    latlon[..., 0] = 90.0 - (theta * 180.0 / np.pi)
    
    # 经度：phi * 180° / π，范围 [-180°, 180°]
    latlon[..., 1] = phi * 180.0 / np.pi
    
    # 恢复原始形状
    if len(original_shape) == 2:
        latlon = latlon[0]
    
    return latlon

def latlon_to_erp(latlon, erp_width=1024, erp_height=512):
    """
    将经纬度转换为ERP图像坐标
    
    参数:
        latlon: numpy 数组，形状为 [N, M, 2] 或 [M, 2]，每个点为 (纬度, 经度)，单位为度
        erp_width: ERP图像宽度，默认 1024
        erp_height: ERP图像高度，默认 512
    
    返回:
        erp_coords: numpy 数组，形状为 [N, M, 2] 或 [M, 2]，每个点为 (x, y)，单位为像素
                - x 范围：[0, erp_width)
                - y 范围：[0, erp_height)
    """
    # 处理输入形状
    original_shape = latlon.shape
    if latlon.ndim == 2:
        latlon = latlon[np.newaxis, :, :]  # 转换为 [1, M, 2]
    
    N, M, _ = latlon.shape
    erp_coords = np.zeros((N, M, 2))
    
    lat = latlon[..., 0]
    lon = latlon[..., 1]
    
    # 经度转ERP x坐标：[-180°, 180°] → [0, erp_width)，从左到右
    erp_coords[..., 0] = (lon + 180.0) / 360.0 * erp_width
    
    # 纬度转ERP y坐标：[90°, -90°] → [0, erp_height)，从上到下
    erp_coords[..., 1] = (90.0 - lat) / 180.0 * erp_height
    
    # 恢复原始形状
    if len(original_shape) == 2:
        erp_coords = erp_coords[0]
    
    return erp_coords

def visualize_erp_coords(erp_coords, erp_width=1024, erp_height=512, point_size=2, color=(0, 255, 0), save_path=None, show=True):
    """
    在空白ERP图像上可视化坐标点
    
    参数:
        erp_coords: numpy 数组，形状为 [N, M, 2] 或 [M, 2]，每个点为 (x, y)，单位为像素
        erp_width: ERP图像宽度，默认 1024
        erp_height: ERP图像高度，默认 512
        point_size: 点的大小，默认 2
        color: 点的颜色 (B, G, R)，默认绿色 (0, 255, 0)
        save_path: 保存图像的路径，如果为 None 则不保存
        show: 是否显示图像，默认 True
    
    返回:
        image: 绘制了坐标点的ERP图像
    """
    # 处理输入形状
    original_shape = erp_coords.shape
    if erp_coords.ndim == 2:
        erp_coords = erp_coords[np.newaxis, :, :]  # 转换为 [1, M, 2]
    
    N, M, _ = erp_coords.shape
    
    # 创建空白图像
    image = np.zeros((erp_height, erp_width, 3), dtype=np.uint8) + 255  # 白色背景
    
    # 在图像上绘制所有BFOV的坐标点
    for idx in range(N):
        coords = erp_coords[idx]
        
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

def calculate_tangent_plane_size(fov_u_deg, fov_v_deg):
    """
    计算单位球坐标系下切平面的尺寸
    
    参数:
        fov_u_deg: 水平视场角（度）
        fov_v_deg: 垂直视场角（度）
    
    返回:
        dict: 包含切平面尺寸信息的字典
            - width: 切平面水平尺寸
            - height: 切平面垂直尺寸
            - area: 切平面面积
            - area_sqrt: 切平面面积的平方根（尺度）
            - fov_u_deg: 输入的水平视场角（度）
            - fov_v_deg: 输入的垂直视场角（度）
    """
    # 将视场角转换为弧度
    fov_u = np.deg2rad(fov_u_deg)
    fov_v = np.deg2rad(fov_v_deg)
    
    # 计算切平面半宽
    U = np.tan(fov_u / 2.0)  # 水平半宽
    V = np.tan(fov_v / 2.0)  # 垂直半宽
    
    # 计算切平面实际尺寸
    plane_width = 2 * U    # 水平尺寸
    plane_height = 2 * V   # 垂直尺寸
    plane_area = plane_width * plane_height  # 面积
    plane_area_sqrt = np.sqrt(plane_area) *1024/(2*np.pi)   # 面积的平方根（尺度）
    
    return {
        'width': plane_width,
        'height': plane_height,
        'area': plane_area,
        'area_sqrt': plane_area_sqrt,
        'fov_u_deg': fov_u_deg,
        'fov_v_deg': fov_v_deg
    }

def bfov_to_14x14_erp(bfov_params, erp_width=1024, erp_height=512, grid_size=14):
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
    theta0 = np.pi/2 - lat  # Polar angle (N,)
    
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
    
    # 5. Generate grid indices exactly like numpy version - using manual approach to avoid indexing issues
    # numpy版本顺序: for i in range(grid_size): for j in range(grid_size):
    # 手动创建与numpy双重for循环完全相同顺序的索引
    indices = torch.zeros(grid_size, grid_size, 2, device=device, dtype=torch.int64)
    for i in range(grid_size):
        for j in range(grid_size):
            indices[i, j, 0] = i
            indices[i, j, 1] = j
    
    # 分离i和j索引
    i_grid = indices[..., 0].float()  # (14, 14)
    j_grid = indices[..., 1].float()  # (14, 14)
    
    # 6. Calculate u_center and v_center exactly like numpy version
    # numpy版本: u_center = -U + (i + 0.5) * delta_u
    # numpy版本: v_center = -V + (j + 0.5) * delta_v
    
    # 扩展U, V, delta_u, delta_v到网格形状
    U_expanded = U.view(N, 1, 1)  # (N, 1, 1)
    V_expanded = V.view(N, 1, 1)  # (N, 1, 1)
    delta_u_expanded = delta_u.view(N, 1, 1)  # (N, 1, 1)
    delta_v_expanded = delta_v.view(N, 1, 1)  # (N, 1, 1)
    
    # 与numpy版本完全相同的计算方式
    # 先计算(i + 0.5)和(j + 0.5)
    i_plus_half = i_grid + 0.5  # (14, 14)
    j_plus_half = j_grid + 0.5  # (14, 14)
    
    # 然后计算u_center和v_center，确保广播正确
    u_center = -U_expanded + i_plus_half.unsqueeze(0) * delta_u_expanded  # (N, 14, 14)
    v_center = -V_expanded + j_plus_half.unsqueeze(0) * delta_v_expanded  # (N, 14, 14)
    
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
    lat_deg = 90.0 - (theta * 180.0 / np.pi)
    lon_deg = phi * 180.0 / np.pi
    
    # 11. Convert lat/lon to ERP coordinates - same as numpy version
    erp_x = (lon_deg + 180.0) / 360.0 * erp_width
    erp_y = (90.0 - lat_deg) / 180.0 * erp_height
    
    # 12. Stack to get final ERP points
    erp_points = torch.stack([erp_x, erp_y], dim=-1)  # (N, 14, 14, 2)
    
    return erp_points

def save_erp_grid_to_txt(erp_grid, txt_file_path, grid_size=14):
    """
    Save ERP grid points to text file with the same format as the numpy version.
    
    Args:
        erp_grid: Tensor or numpy array of shape (14, 14, 2) or (N, 14, 14, 2) containing ERP coordinates
        txt_file_path: Path to save the text file
        grid_size: Grid size, default 14
    """
    # Convert to numpy array if it's a PyTorch tensor
    if hasattr(erp_grid, 'cpu'):
        erp_grid = erp_grid.cpu().numpy()
    
    # If it's a batch, process each BFOV
    if erp_grid.ndim == 4:
        for bfov_idx in range(erp_grid.shape[0]):
            single_grid = erp_grid[bfov_idx]
            single_file_path = txt_file_path.replace('.txt', f'_{bfov_idx+1}.txt')
            save_erp_grid_to_txt(single_grid, single_file_path, grid_size)
        return
    
    # Save single BFOV
    with open(txt_file_path, 'w') as f:
        # Write file header
        f.write("BFOV ERP坐标 ({}×{}网格格式)\n".format(grid_size, grid_size))
        f.write("="*80 + "\n")
        f.write("水平方向 (i): 经度方向 (0-{}) - 从左到右\n".format(grid_size-1))
        f.write("垂直方向 (j): 纬度方向 (0-{}) - 从下到上（在txt中为从上到下）\n".format(grid_size-1))
        f.write("每个点格式：(x, y) - ERP图像上的像素坐标\n")
        f.write("\n")
        f.write("相对位置说明：\n")
        f.write("- txt文件的第一行对应切平面顶部（纬度较高）\n")
        f.write("- txt文件的最后一行对应切平面底部（纬度较低）\n")
        f.write("- 每行内的点从左到右对应切平面的水平方向（经度方向）\n")
        f.write("="*80 + "\n\n")
        
        # Write points in the same format as the original function
        # Reverse j order to match the original format
        for j in reversed(range(grid_size)):
            for i in range(grid_size):
                x, y = erp_grid[i, j]
                f.write("({:.8f}, {:.8f}) ".format(x, y))
            f.write("\n")

def bfov_to_spherical_points(bfov_params, grid_size=14):
    """
    将批量的 BFOV 参数转换为单位球面上的采样点 (X,Y,Z) 坐标
    
    参数:
        bfov_params: numpy 数组，形状为 [N, 4]，N 是 BFOV 的数量，每行格式为：
                     [lon, lat, fov_u, fov_v]
                     - lon: BFOV 中心经度（度，范围[-180°, 180°]）
                     - lat: BFOV 中心纬度（度，范围[-90°, 90°]）
                     - fov_u: 水平视场角（度），对应经度方向（-180°到180°）
                     - fov_v: 垂直视场角（度），对应纬度方向（-90°到90°）
        grid_size: 采样网格大小，默认 14（14×14=196个点）
    
    返回:
        points: numpy 数组，形状为 [N, grid_size*grid_size, 3]，每个 BFOV 对应 grid_size² 个 (x,y,z) 点
    """
    # 参数校验
    if not isinstance(bfov_params, np.ndarray):
        bfov_params = np.array(bfov_params)
    if bfov_params.ndim != 2 or bfov_params.shape[1] != 4:
        raise ValueError("bfov_params 必须是形状为 [N, 4] 的数组，每行格式：[lon, lat, fov_u, fov_v]")
    
    N = bfov_params.shape[0]  # BFOV 数量
    all_points = []  # 存储所有 BFOV 的采样点
    
    # 遍历每个 BFOV
    for idx in range(N):
        lon, lat, fov_u_deg, fov_v_deg = bfov_params[idx]
        
        # 将经纬度转换为球坐标系的极角和方位角
        # 经度 → 方位角 phi0（弧度）
        phi0 = np.deg2rad(lon)
        # 纬度 → 极角 theta0（弧度），极角是从z轴向下的角度
        theta0 = np.deg2rad(90 - lat)
        # 视场角转换为弧度
        fov_u = np.deg2rad(fov_u_deg)
        fov_v = np.deg2rad(fov_v_deg)
        
        # 1. 计算 BFOV 中心的单位向量 c (x,y,z)
        c_x = np.sin(theta0) * np.cos(phi0)
        c_y = np.sin(theta0) * np.sin(phi0)
        c_z = np.cos(theta0)
        c = np.array([c_x, c_y, c_z])  # 形状 [3]
        
        # 2. 构造切平面的局部正交基 (e_u, e_v)
        # e_u 对应水平方向（经度方向，-180°到180°），对应 fov_u
        # e_v 对应垂直方向（纬度方向，-90°到90°），对应 fov_v
        
        # 计算经度方向的单位向量（水平方向）
        # 经度方向：绕 z 轴旋转的方向，向量为 [-sin(phi), cos(phi), 0]
        sin_phi = np.sin(phi0)
        cos_phi = np.cos(phi0)
        lon_dir = np.array([-sin_phi, cos_phi, 0.0])
        
        # 计算纬度方向的单位向量（垂直方向）
        # 纬度方向：从赤道到极点的方向，向量为 [-cos(theta)*cos(phi), -cos(theta)*sin(phi), sin(theta)]
        sin_theta = np.sin(theta0)
        cos_theta = np.cos(theta0)
        lat_dir = np.array([-cos_theta*cos_phi, -cos_theta*sin_phi, sin_theta])
        
        # 归一化方向向量，确保正交
        e_u = lon_dir  # 水平基（经度方向，对应 fov_u）
        e_v = lat_dir  # 垂直基（纬度方向，对应 fov_v）
        
        # 检查正交性并调整
        if np.abs(np.dot(e_u, e_v)) > 1e-6:
            # 如果不够正交，重新计算
            e_u = lon_dir - np.dot(lon_dir, c) * c
            e_u = e_u / np.linalg.norm(e_u)
            e_v = np.cross(c, e_u)
            e_v = e_v / np.linalg.norm(e_v)
        
        # 3. 计算切平面矩形半宽 (U, V)
        # U 对应水平方向（经度方向，fov_u），V 对应垂直方向（纬度方向，fov_v）
        U = np.tan(fov_u / 2.0)
        V = np.tan(fov_v / 2.0)
        
        # 4. 计算网格的步长
        # grid_size 由函数参数传入，默认值为14
        delta_u = 2 * U / grid_size  # 水平步长
        delta_v = 2 * V / grid_size  # 垂直步长
        
        # 5. 遍历网格，计算每个网格中心的坐标
        sample_points = []  # 存储当前 BFOV 的 grid_size×grid_size 个点
        for i in range(grid_size):  # 水平方向（经度方向）
            for j in range(grid_size):  # 垂直方向（纬度方向）
                # 第 (i,j) 个网格的中心坐标
                # 注意：j=0 对应切平面底部（纬度较低），j=13 对应切平面顶部（纬度较高）
                u_center = -U + (i + 0.5) * delta_u
                v_center = -V + (j + 0.5) * delta_v
                
                # 6. 切平面点投影回单位球面
                x_plane = c + u_center * e_u + v_center * e_v
                x_sphere = x_plane / np.linalg.norm(x_plane)  # 归一化到单位球面
                sample_points.append(x_sphere)
        
        # 将当前 BFOV 的 196 个点存入结果
        all_points.append(np.array(sample_points))
    
    # 转换为 numpy 数组，形状 [N, 196, 3]
    return np.array(all_points)

# ------------------------------
# 测试示例（使用你提供的参数，转换为弧度）
# ------------------------------
def verify_pytorch_vs_numpy():
    """
    Verify that the PyTorch version produces the same results as the numpy version.
    This function compares the output of bfov_to_14x14_erp with bfov_to_spherical_points + latlon_to_erp.
    """
    print("验证PyTorch版本与NumPy版本的一致性...")
    
    # 使用简单的测试参数
    bfov_batch = np.array([
        [-120, 50, 70, 60],  # BFOV 1
        [45, 30, 40, 20]     # BFOV 2
    ])
    
    grid_size = 14
    
    # 1. 使用NumPy版本计算ERP坐标
    print("使用NumPy版本计算ERP坐标...")
    spherical_points_np = bfov_to_spherical_points(bfov_batch, grid_size=grid_size)
    latlon_points_np = spherical_to_latlon(spherical_points_np)
    erp_coords_np = latlon_to_erp(latlon_points_np)
    
    # 将NumPy版本的ERP坐标转换为14x14网格格式，与PyTorch版本的输出格式一致
    N = erp_coords_np.shape[0]
    erp_grid_np = np.zeros((N, grid_size, grid_size, 2))
    for bfov_idx in range(N):
        bfov_erp = erp_coords_np[bfov_idx, :, :]
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                erp_grid_np[bfov_idx, i, j] = bfov_erp[idx]
    
    print(f"NumPy版本输出形状: {erp_grid_np.shape}")
    
    try:
        # 2. 使用PyTorch版本计算ERP坐标
        print("使用PyTorch版本计算ERP坐标...")
        import torch
        
        # 将BFOV参数转换为弧度张量
        bfov_params_rad = np.deg2rad(bfov_batch)
        bfov_params_tensor = torch.tensor(bfov_params_rad, dtype=torch.float32)
        
        # 使用PyTorch函数计算ERP坐标
        erp_grid_torch = bfov_to_14x14_erp(bfov_params_tensor, grid_size=grid_size)
        
        print(f"PyTorch版本输出形状: {erp_grid_torch.shape}")
        
        # 3. 比较两个版本的结果
        erp_grid_torch_np = erp_grid_torch.cpu().numpy()
        
        # 计算最大差异
        max_diff = np.max(np.abs(erp_grid_torch_np - erp_grid_np))
        
        print(f"两个版本的最大差异: {max_diff}")
        
        if max_diff < 1e-6:
            print("✅ PyTorch版本与NumPy版本结果一致!")
        else:
            print("❌ PyTorch版本与NumPy版本结果不一致!")
            
            # 打印一些具体差异
            print("前几个点的差异:")
            for i in range(3):
                for j in range(3):
                    np_val = erp_grid_np[0, i, j]
                    torch_val = erp_grid_torch_np[0, i, j]
                    diff = np.abs(np_val - torch_val)
                    if np.any(diff > 1e-6):
                        print(f"  ({i},{j}): NumPy={np_val}, PyTorch={torch_val}, Diff={diff}")
                        
    except ImportError:
        print("⚠️  PyTorch库未安装，无法进行比较测试")


if __name__ == "__main__":
    # 参数：经度, 纬度, 水平视场, 垂直视场（单位：度）
    lon1 = -120     # 经度 -120°
    lat1 = 50     # 纬度 50°
    fov_u1 = 70   # 水平视场 70°（经度方向）
    fov_v1 = 60   # 垂直视场 60°（纬度方向）
    
    lon2 = 45     # 经度 45°
    lat2 = 30     # 纬度 30°
    fov_u2 = 40   # 水平视场 40°（经度方向）
    fov_v2 = 20   # 垂直视场 20°（纬度方向）
    
    # 批量输入（示例：2 个 BFOV，直接使用度作为单位）
    bfov_batch = np.array([
        [lon1, lat1, fov_u1, fov_v1],          # BFOV 1
        [lon2, lat2, fov_u2, fov_v2]            # BFOV 2
    ])
    
    # 设置网格大小
    grid_size = 14  # 可调整的网格大小，默认14×14=196个点
    
    # 生成采样点
    spherical_points = bfov_to_spherical_points(bfov_batch, grid_size=grid_size)
    
    # 计算并输出切平面尺寸（单位球坐标系下）
    print("\n==================================================")
    print("切平面尺寸计算（单位球坐标系下）：")
    for idx in range(bfov_batch.shape[0]):
        lon, lat, fov_u_deg, fov_v_deg = bfov_batch[idx]
        
        # 使用新函数计算切平面尺寸
        plane_info = calculate_tangent_plane_size(fov_u_deg, fov_v_deg)
        
        print(f"BFOV {idx+1}:")
        print(f"  水平视场角：{plane_info['fov_u_deg']:.1f}°，垂直视场角：{plane_info['fov_v_deg']:.1f}°")
        print(f"  网格大小：{grid_size}×{grid_size}={grid_size*grid_size}个采样点")
        print(f"  切平面水平尺寸：{plane_info['width']:.4f}")
        print(f"  切平面垂直尺寸：{plane_info['height']:.4f}")
        print(f"  切平面面积：{plane_info['area']:.4f}")
        print(f"  切平面面积平方根（尺度）：{plane_info['area_sqrt']:.4f}")

    latlon_points = spherical_to_latlon(spherical_points)
    erp_coords = latlon_to_erp(latlon_points)

    # ------------------------------
    # 新功能：将ERP坐标按指定网格格式输出到txt文件
    # ------------------------------
    print("\n" + "="*50)
    print("将ERP坐标按{}×{}格式输出到txt文件：".format(grid_size, grid_size))
    
    # 创建保存目录（如果不存在）
    txt_save_dir = "./bfov/bfov-7*7/erp_coords_txt"
    os.makedirs(txt_save_dir, exist_ok=True)
    
    # 处理每个BFOV的ERP坐标
    for bfov_idx in range(erp_coords.shape[0]):
        # 获取当前BFOV的ERP坐标
        bfov_erp = erp_coords[bfov_idx, :, :]  # 形状：(grid_size², 2)
        
        # 显式地按照切平面上的i,j顺序排列ERP坐标
        # 使用从主函数传递下来的grid_size参数
        bfov_erp_grid = np.zeros((grid_size, grid_size, 2))
        
        # 按切平面采样顺序重新排列坐标
        # 原始采样顺序：for i in range(grid_size): for j in range(grid_size):
        # 其中j=0对应切平面底部（纬度较低），j=13对应切平面顶部（纬度较高）
        for i in range(grid_size):
            for j in range(grid_size):
                # 计算在一维数组中的索引
                idx = i * grid_size + j
                bfov_erp_grid[i, j] = bfov_erp[idx]
        
        # 保存到txt文件
        txt_file_path = os.path.join(txt_save_dir, f"bfov_{bfov_idx+1}_erp_coords_{grid_size}x{grid_size}.txt")
        
        with open(txt_file_path, 'w') as f:
            # 写入文件头信息
            f.write("BFOV {} 的ERP坐标 ({}×{}网格格式)\n".format(bfov_idx+1, grid_size, grid_size))
            f.write("="*80 + "\n")
            f.write("水平方向 (i): 经度方向 (0-{}) - 从左到右\n".format(grid_size-1))
            f.write("垂直方向 (j): 纬度方向 (0-{}) - 从下到上（在txt中为从上到下）\n".format(grid_size-1))
            f.write("每个点格式：(x, y) - ERP图像上的像素坐标\n")
            f.write("\n")
            f.write("相对位置说明：\n")
            f.write("- txt文件的第一行对应切平面顶部（纬度较高）\n")
            f.write("- txt文件的最后一行对应切平面底部（纬度较低）\n")
            f.write("- 每行内的点从左到右对应切平面的水平方向（经度方向）\n")
            f.write("="*80 + "\n\n")
            
            # 按切平面上的相对位置输出
            # 关键：反转j的顺序，使txt文件中的上下方向与切平面一致
            # j从grid_size-1到0：先输出切平面顶部，再输出底部
            for j in reversed(range(grid_size)):
                for i in range(grid_size):
                    x, y = bfov_erp_grid[i, j]
                    f.write("({:.8f}, {:.8f}) ".format(x, y))
                f.write("\n")
        
        print(f"BFOV {bfov_idx+1} 的ERP坐标已保存到：{txt_file_path}")
    
    # ------------------------------
    # 新功能：可视化ERP坐标
    # ------------------------------
    print("\n" + "="*50)
    print("可视化ERP坐标点：")
    # 保存图像到当前目录
    save_path = "./bfov/bfov-7*7/bfov_erp_visualization.png"
    visualize_erp_coords(erp_coords, save_path=save_path, show=True)

    # ------------------------------
    # PyTorch版本示例
    # ------------------------------
    print("\n" + "="*50)
    print("使用PyTorch版本的bfov_to_14x14_erp函数：")
    
    # 将BFOV参数转换为弧度并转为PyTorch张量
    bfov_params_rad = np.deg2rad(bfov_batch)
    bfov_params_tensor = torch.tensor(bfov_params_rad, dtype=torch.float32)
    
    # 使用PyTorch函数计算ERP坐标
    erp_points_pytorch = bfov_to_14x14_erp(bfov_params_tensor)
    
    print(f"PyTorch版本输出形状: {erp_points_pytorch.shape}")
    
    # 将PyTorch结果保存到txt文件
    txt_save_dir = "./bfov/bfov-7*7/erp_coords_txt_pytorch"
    os.makedirs(txt_save_dir, exist_ok=True)
    
    for bfov_idx in range(erp_points_pytorch.shape[0]):
        txt_file_path = os.path.join(txt_save_dir, f"bfov_{bfov_idx+1}_erp_coords_pytorch_{grid_size}x{grid_size}.txt")
        # 保存单个BFOV的结果
        save_erp_grid_to_txt(erp_points_pytorch[bfov_idx], txt_file_path, grid_size)
        print(f"BFOV {bfov_idx+1} 的PyTorch ERP坐标已保存到：{txt_file_path}")

    # ------------------------------
    # 验证PyTorch版本与NumPy版本的一致性
    # ------------------------------
    verify_pytorch_vs_numpy()