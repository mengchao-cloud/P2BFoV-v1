import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PANDORA.PRDA.lib.ReuseGPUImageRecorder import ReuseGPUImageRecorder

import json

# 配置
ERP_WIDTH = 1920
ERP_HEIGHT = 960
# JSON文件路径
JSON_FILE_PATH = './bfov/frame-core/json/train_coco.json'
# 图像路径（可以直接修改这里的路径来选择不同的图像）
IMAGE_PATH = './bfov/frame-core/7l2TC.jpg'
# 从图像路径中提取文件名
IMAGE_FILENAME = os.path.basename(IMAGE_PATH)
# 输出路径
OUTPUT_PATH = f'./bfov/frame-core/vis/{IMAGE_FILENAME.split(".")[0]}_bfov.jpg'
OUTPUT_VECTOR_PATH = f'./bfov/frame-core/vis/{IMAGE_FILENAME.split(".")[0]}_bfov.pdf'

# 从JSON文件中读取BFOV数据
BFoV_PARAMS = []

def load_bfov_from_json(json_path, image_filename):
    """
    从JSON文件中加载指定图像的BFOV数据
    
    参数:
    json_path: JSON文件路径
    image_filename: 图像文件名
    
    返回:
    bfov_params: BFOV参数列表
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # 创建图像ID到文件名的映射
        image_id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # 找到目标图像的ID
        target_image_id = None
        for img_id, filename in image_id_to_file.items():
            if os.path.basename(filename) == image_filename:
                target_image_id = img_id
                break
        
        if target_image_id is None:
            print(f"警告：在JSON文件中未找到图像 {image_filename}")
            return []
        
        # 提取该图像的BFOV标注
        bfov_params = []
        for ann in coco_data['annotations']:
            if ann['image_id'] == target_image_id:
                if 'bfov' in ann:
                    bfov_params.append(ann['bfov'])
        
        return bfov_params
    except Exception as e:
        print(f"读取JSON文件时发生错误: {e}")
        return []

# 加载BFOV数据
BFoV_PARAMS = load_bfov_from_json(JSON_FILE_PATH, IMAGE_FILENAME)

def get_bfov_boundary_points(bfov_params):
    """
    获取BFOV的边界点

    参数:
    bfov_params: BFOV参数 [center_x, center_y, fov_x, fov_y]，弧度制

    返回:
    points: BFOV边界点的numpy数组
    """
    # 提取BFOV参数并转换为角度制
    center_x, center_y, fov_x, fov_y = bfov_params
    center_x_deg = center_x * 180 / np.pi
    center_y_deg = center_y * 180 / np.pi
    fov_x_deg = fov_x * 180 / np.pi
    fov_y_deg = fov_y * 180 / np.pi

    # 验证BFOV参数，确保视场角在合理范围内
    max_fov_deg = 170
    fov_x_deg = max(-max_fov_deg, min(max_fov_deg, fov_x_deg))
    fov_y_deg = max(-max_fov_deg, min(max_fov_deg, fov_y_deg))

    # 确保视场角大于0
    fov_x_deg = max(1, abs(fov_x_deg))
    fov_y_deg = max(1, abs(fov_y_deg))

    try:
        # 创建ReuseGPUImageRecorder实例
        BFoV = ReuseGPUImageRecorder(ERP_WIDTH, ERP_HEIGHT,
                                    view_angle_w=fov_x_deg, view_angle_h=fov_y_deg,
                                    long_side=ERP_WIDTH, device='cuda')

        # 获取BFOV的边界点
        Px, Py = BFoV._sample_points(center_x, center_y, border_only=True)

        # 将GPU张量转换为numpy数组
        Px = Px.cpu().numpy()
        Py = Py.cpu().numpy()

        if Px.shape[0] > 0:
            # 将坐标转换为整数
            points = np.array([[px, py] for px, py in zip(Px, Py)])

            # 直接使用ReuseGPUImageRecorder返回的边界点顺序
            # 不进行排序，以正确处理跨越图像边界的情况

            return points
        else:
            return None
    except Exception as e:
        # 处理异常，记录错误但不中断程序
        print(f"获取BFOV边界点时发生错误: {e}")
        return None

def visualize_bfov_on_image(img, bfov_params, color=(0, 255, 0)):
    """
    在图像上可视化单个BFOV

    参数:
    img: 输入图像
    bfov_params: BFOV参数 [center_x, center_y, fov_x, fov_y]，弧度制
    color: BFOV边界颜色

    返回:
    img: 绘制后的图像
    """
    # 提取BFOV参数并转换为角度制
    center_x, center_y, fov_x, fov_y = bfov_params
    center_x_deg = center_x * 180 / np.pi
    center_y_deg = center_y * 180 / np.pi
    fov_x_deg = fov_x * 180 / np.pi
    fov_y_deg = fov_y * 180 / np.pi
    
    # 验证BFOV参数，确保视场角在合理范围内
    # 限制视场角最大为170度，避免tan函数计算问题
    max_fov_deg = 170
    fov_x_deg = max(-max_fov_deg, min(max_fov_deg, fov_x_deg))
    fov_y_deg = max(-max_fov_deg, min(max_fov_deg, fov_y_deg))
    
    # 确保视场角大于0
    fov_x_deg = max(1, abs(fov_x_deg))
    fov_y_deg = max(1, abs(fov_y_deg))
    
    try:
        # 创建ReuseGPUImageRecorder实例
        BFoV = ReuseGPUImageRecorder(ERP_WIDTH, ERP_HEIGHT, 
                                    view_angle_w=fov_x_deg, view_angle_h=fov_y_deg, 
                                    long_side=ERP_WIDTH, device='cuda')
        
        # 将中心点角度转换为弧度制（ImageRecorder需要）
        center_x_rad = center_x
        center_y_rad = center_y
        
        # 获取BFOV的边界点
        Px, Py = BFoV._sample_points(center_x_rad, center_y_rad, border_only=True)
        
        # 将GPU张量转换为numpy数组
        Px = Px.cpu().numpy()
        Py = Py.cpu().numpy()
        
        # 绘制BFOV边界
        # 绘制边界点
        if Px.shape[0] > 0:
            # 将坐标转换为整数
            points = np.array([[int(px), int(py)] for px, py in zip(Px, Py)])
            # 绘制点
            for point in points:
                cv2.circle(img, tuple(point), 2, color, -1)
        
        # 绘制BFOV中心点
        center_px = int((center_x_rad + np.pi) / (2 * np.pi) * ERP_WIDTH)
        center_py = int((np.pi/2 - center_y_rad) / np.pi * ERP_HEIGHT)
        cv2.circle(img, (center_px, center_py), 3, color, -1)
    except Exception as e:
        # 处理异常，记录错误但不中断程序
        print(f"可视化BFOV时发生错误: {e}")
        print(f"弧度制BFOV参数: center_x={bfov_params[0]:.4f}, center_y={bfov_params[1]:.4f}, fov_x={bfov_params[2]:.4f}, fov_y={bfov_params[3]:.4f}")
        # 转换为角度制打印
        center_x_deg = bfov_params[0] * 180 / np.pi
        center_y_deg = bfov_params[1] * 180 / np.pi
        fov_x_deg = bfov_params[2] * 180 / np.pi
        fov_y_deg = bfov_params[3] * 180 / np.pi
        print(f"角度制BFOV参数: center_x={center_x_deg:.2f}°, center_y={center_y_deg:.2f}°, fov_x={fov_x_deg:.2f}°, fov_y={fov_y_deg:.2f}°")
        # 使用简单的矩形绘制作为备选方案
        center_x, center_y, fov_x, fov_y = bfov_params
        # 将球面坐标转换为平面坐标
        center_px = int((center_x + np.pi) / (2 * np.pi) * ERP_WIDTH)
        center_py = int((np.pi/2 - center_y) / np.pi * ERP_HEIGHT)
        width_px = int(fov_x / (2 * np.pi) * ERP_WIDTH)
        height_px = int(fov_y / np.pi * ERP_HEIGHT)
        
        # 确保坐标在图像范围内
        x1 = max(0, center_px - width_px // 2)
        y1 = max(0, center_py - height_px // 2)
        x2 = min(ERP_WIDTH, center_px + width_px // 2)
        y2 = min(ERP_HEIGHT, center_py + height_px // 2)
        
        # 绘制简单矩形
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        
        # 绘制中心点
        cv2.circle(img, (center_px, center_py), 3, color, -1)

    return img

def project_bfov_to_plane(img, bfov_params):
    """
    将BFOV投影成切平面无畸变图
    基于single_level_roi_extractor.py中的bfov_roi_to_14x14_erp逻辑
    切平面分辨率计算基于球的周长等于原始图像宽度

    参数:
    img: 输入ERP图像
    bfov_params: BFOV参数 [center_x, center_y, fov_x, fov_y]，弧度制

    返回:
    plane_img: 切平面无畸变图
    """
    # 提取BFOV参数（弧度制）
    lon, lat, fov_u, fov_v = bfov_params
    PI = np.pi

    # 球的周长等于原始图像宽度
    sphere_circumference = ERP_WIDTH
    
    # 根据FOV角度计算切平面分辨率
    # 水平分辨率：基于水平FOV占球周长的比例
    plane_width = int(sphere_circumference * (fov_u / (2 * PI)))
    # 垂直分辨率：基于垂直FOV与水平FOV的比例
    aspect_ratio = fov_v / fov_u
    plane_height = int(plane_width * aspect_ratio)
    
    # 确保分辨率合理
    plane_width = max(16, plane_width)
    plane_height = max(16, plane_height)

    # 转换经度/纬度为极角/方位角
    phi0 = lon  # 方位角
    theta0 = PI/2 - lat  # 极角

    # 1. 计算BFOV中心单位向量c (x,y,z)
    c_x = np.sin(theta0) * np.cos(phi0)
    c_y = np.sin(theta0) * np.sin(phi0)
    c_z = np.cos(theta0)
    c = np.array([c_x, c_y, c_z])
    
    # 归一化确保单位向量精度
    c = c / np.linalg.norm(c)

    # 2. 构造局部正交基 (e_u, e_v)
    sin_phi = np.sin(phi0)
    cos_phi = np.cos(phi0)
    sin_theta = np.sin(theta0)
    cos_theta = np.cos(theta0)

    # 经度方向向量（水平）
    lon_dir = np.array([-sin_phi, cos_phi, 0])
    lon_dir = lon_dir / np.linalg.norm(lon_dir)

    # 纬度方向向量（垂直）
    lat_dir = np.array([-cos_theta*cos_phi, -cos_theta*sin_phi, sin_theta])
    lat_dir = lat_dir / np.linalg.norm(lat_dir)

    # 确保正交性
    dot_product = np.dot(lon_dir, lat_dir)
    if abs(dot_product) > 1e-6:
        # 重新计算正交基
        lon_dir_ortho = lon_dir - np.dot(lon_dir, c) * c
        lon_dir_ortho = lon_dir_ortho / np.linalg.norm(lon_dir_ortho)
        lat_dir_ortho = np.cross(c, lon_dir_ortho)
        lat_dir_ortho = lat_dir_ortho / np.linalg.norm(lat_dir_ortho)
        lon_dir, lat_dir = lon_dir_ortho, lat_dir_ortho

    # 3. 计算切平面半宽 (U, V)
    max_angle = PI/2 - 1e-4  # 接近但小于π/2的值
    fov_u_half = np.clip(fov_u / 2.0, -max_angle, max_angle)
    fov_v_half = np.clip(fov_v / 2.0, -max_angle, max_angle)

    U = np.tan(fov_u_half)  # 水平半宽
    V = np.tan(fov_v_half)  # 垂直半宽

    # 4. 计算网格步长
    delta_u = 2 * U / plane_width  # 水平步长
    delta_v = 2 * V / plane_height  # 垂直步长

    # 5. 生成网格坐标
    j_grid, i_grid = np.meshgrid(np.arange(plane_width), np.arange(plane_height))

    # 6. 计算u_center和v_center
    # 扩展U, V, delta_u, delta_v到网格形状
    U_expanded = U
    V_expanded = V
    delta_u_expanded = delta_u
    delta_v_expanded = delta_v

    # 计算中心坐标
    j_plus_half = j_grid + 0.5  # 水平方向网格中心
    vertical_idx = (plane_height - 1 - i_grid) + 0.5  # 垂直方向网格中心（已反转）

    # 水平方向(u): 从左到右
    u_center = -U_expanded + j_plus_half * delta_u_expanded
    # 垂直方向(v): 从上到下
    v_center = -V_expanded + vertical_idx * delta_v_expanded

    # 7. 计算切平面上的点
    # 扩展基向量到网格大小
    c_expand = np.tile(c, (plane_height, plane_width, 1))
    e_u_expand = np.tile(lon_dir, (plane_height, plane_width, 1))
    e_v_expand = np.tile(lat_dir, (plane_height, plane_width, 1))

    # 计算切平面上的点
    u_center_expand = np.expand_dims(u_center, axis=2)
    v_center_expand = np.expand_dims(v_center, axis=2)
    x_plane = c_expand + u_center_expand * e_u_expand + v_center_expand * e_v_expand

    # 8. 投影回单位球面
    norm_x_plane = np.linalg.norm(x_plane, axis=2, keepdims=True)
    x_sphere = x_plane / norm_x_plane

    # 9. 转换球面坐标到经纬度（度数）
    x = x_sphere[..., 0]
    y = x_sphere[..., 1]
    z = x_sphere[..., 2]

    # 添加数值稳定性处理
    z_clamped = np.clip(z, -1.0 + 1e-6, 1.0 - 1e-6)

    theta = np.arccos(z_clamped)  # 极角
    phi = np.arctan2(y, x)  # 方位角

    # 转换为经纬度（度数）
    lat_deg = 90.0 - (theta * 180.0 / PI)
    lon_deg = phi * 180.0 / PI

    # 10. 转换经纬度到ERP坐标
    erp_x = (lon_deg + 180.0) / 360.0 * ERP_WIDTH
    erp_y = (90.0 - lat_deg) / 180.0 * ERP_HEIGHT

    # 11. 确保坐标在图像范围内
    erp_x = np.clip(erp_x, 0, ERP_WIDTH - 1)
    erp_y = np.clip(erp_y, 0, ERP_HEIGHT - 1)

    # 12. 填充切平面图像
    plane_img = np.zeros((plane_height, plane_width, 3), dtype=np.uint8)
    
    # 使用双线性插值获取像素值
    for i in range(plane_height):
        for j in range(plane_width):
            px = int(erp_x[i, j])
            py = int(erp_y[i, j])
            plane_img[i, j] = img[py, px]

    return plane_img

def main():
    """
    在指定图像上绘制BFOV
    """
    print("开始可视化BFOV...")

    # 创建输出目录
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    # 创建切平面输出目录
    plane_output_dir = os.path.join(os.path.dirname(OUTPUT_PATH), "planes")
    os.makedirs(plane_output_dir, exist_ok=True)

    # 检查图像是否存在
    if not os.path.exists(IMAGE_PATH):
        print(f"错误：图像不存在 {IMAGE_PATH}")
        return

    # 读取图像
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"错误：无法读取图像 {IMAGE_PATH}")
        return

    # 调整图像尺寸
    img = cv2.resize(img, (ERP_WIDTH, ERP_HEIGHT))
    
    # 保存原始图像副本用于生成切平面
    original_img = img.copy()

    # 生成切平面无畸变图（从原始图像生成）
    print("开始生成切平面无畸变图...")
    for i, bfov_params in enumerate(BFoV_PARAMS):
        plane_img = project_bfov_to_plane(original_img, bfov_params)
        if plane_img is not None:
            plane_output_path = os.path.join(plane_output_dir, f"bfov_{i:02d}.jpg")
            cv2.imwrite(plane_output_path, plane_img)
            print(f"切平面图像保存完成：{plane_output_path}")
        else:
            print(f"生成BFOV {i} 的切平面图像失败")

    print("所有切平面图像生成完成！")

    # 绘制所有BFOV（用于JPG输出）
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    for i, bfov_params in enumerate(BFoV_PARAMS):
        color = colors[i % len(colors)]
        img = visualize_bfov_on_image(img, bfov_params, color=color)

    # 保存结果图像
    cv2.imwrite(OUTPUT_PATH, img)
    print(f"BFOV可视化完成！结果保存到: {OUTPUT_PATH}")

    # 保存为矢量图 - 使用原始图像，直接绘制矢量BFOV
    # 将BGR图像转换为RGB
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # 创建Matplotlib图形
    fig, ax = plt.subplots(figsize=(ERP_WIDTH/100, ERP_HEIGHT/100), dpi=100)
    ax.imshow(original_img_rgb)
    ax.axis('off')  # 关闭坐标轴

    # 直接在Matplotlib中绘制BFOV边界（只绘制点）
    matplotlib_colors = ['green', 'blue', 'red', 'yellow', 'purple', 'cyan']
    for i, bfov_params in enumerate(BFoV_PARAMS):
        color = matplotlib_colors[i % len(matplotlib_colors)]
        
        # 提取BFOV参数并转换为角度制
        center_x, center_y, fov_x, fov_y = bfov_params
        center_x_deg = center_x * 180 / np.pi
        center_y_deg = center_y * 180 / np.pi
        fov_x_deg = fov_x * 180 / np.pi
        fov_y_deg = fov_y * 180 / np.pi
        
        # 验证BFOV参数，确保视场角在合理范围内
        max_fov_deg = 170
        fov_x_deg = max(-max_fov_deg, min(max_fov_deg, fov_x_deg))
        fov_y_deg = max(-max_fov_deg, min(max_fov_deg, fov_y_deg))
        
        # 确保视场角大于0
        fov_x_deg = max(1, abs(fov_x_deg))
        fov_y_deg = max(1, abs(fov_y_deg))
        
        try:
            # 创建ReuseGPUImageRecorder实例
            BFoV = ReuseGPUImageRecorder(ERP_WIDTH, ERP_HEIGHT, 
                                        view_angle_w=fov_x_deg, view_angle_h=fov_y_deg, 
                                        long_side=ERP_WIDTH, device='cuda')
            
            # 获取BFOV的边界点
            Px, Py = BFoV._sample_points(center_x, center_y, border_only=True)
            
            # 将GPU张量转换为numpy数组
            Px = Px.cpu().numpy()
            Py = Py.cpu().numpy()
            
            # 绘制边界点（矢量点）
            if Px.shape[0] > 0:
                ax.plot(Px, Py, color=color, marker='o', markersize=2, linestyle='none')
        except Exception as e:
            # 处理异常，记录错误但不中断程序
            print(f"绘制矢量图BFOV时发生错误: {e}")

    # 保存为PDF矢量图
    plt.savefig(OUTPUT_VECTOR_PATH, format='pdf', bbox_inches='tight', pad_inches=0)
    print(f"矢量图保存完成！结果保存到: {OUTPUT_VECTOR_PATH}")
    plt.close(fig)

if __name__ == '__main__':
    main()