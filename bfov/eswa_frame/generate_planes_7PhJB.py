#!/usr/bin/env python3
"""
为 7PhJB.jpg 的18组BFoV生成切平面无畸变图
参考 bfov\frame-core\bfov-vis-qie.py
"""

import os
import sys
import numpy as np
import cv2

# 添加PANDORA路径用于 ReuseGPUImageRecorder
sys.path.append(os.path.join(os.path.dirname(__file__), '../../PANDORA/PRDA/lib'))
from PANDORA.PRDA.lib.ReuseGPUImageRecorder import ReuseGPUImageRecorder

# ====== 从 draw_bfov_7PhJB.py 中复制的 FIXED_BFOV_ANNOTATIONS ======

# 固定的FOV参数组合（9种）
FIXED_FOV_PARAMS = [
    (0.2612193871, 0.1306096936),
    (0.1847100000, 0.1847100000),
    (0.1306096936, 0.2612193871),
    (0.6791704065, 0.3395852032),
    (0.4802460000, 0.4802460000),
    (0.3395852032, 0.6791704065),
    (1.2016091807, 0.6008045903),
    (0.8496660000, 0.8496660000),
    (0.6008045903, 1.2016091807),
]

# person的中心点
PERSON_THETA = -0.0801760625134895
PERSON_PHI = -0.16853335589570229

# chair的中心点
CHAIR_THETA = 0.8492117641734908
CHAIR_PHI = -1.0750137361502572

# 生成person和chair的固定FOV标注
FIXED_BFOV_ANNOTATIONS = []
# person: category_id=34
for fov_x, fov_y in FIXED_FOV_PARAMS:
    FIXED_BFOV_ANNOTATIONS.append([34, PERSON_THETA, PERSON_PHI, fov_x, fov_y])
# chair: category_id=25
for fov_x, fov_y in FIXED_FOV_PARAMS:
    FIXED_BFOV_ANNOTATIONS.append([25, CHAIR_THETA, CHAIR_PHI, fov_x, fov_y])

# ====== 配置 ======
ERP_WIDTH = 1920
ERP_HEIGHT = 960
IMAGE_PATH = './7PhJB.jpg'
OUTPUT_DIR = './planes'

# ====== 切平面投影函数（参考 bfov-vis-qie.py）======
def project_bfov_to_plane(img, bfov_params):
    """
    将BFOV投影成切平面无畸变图
    基于single_level_roi_extractor.py中的bfov_roi_to_14x14_erp逻辑
    切平面分辨率计算基于球的周长等于原始图像宽度

    参数:
    img: 输入ERP图像
    bfov_params: BFOV参数 [category_id, theta, phi, fov_x, fov_y]，弧度制

    返回:
    plane_img: 切平面无畸变图
    """
    # 提取BFOV参数（弧度制），跳过category_id
    lon, lat, fov_u, fov_v = bfov_params[1:5]
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
    # 计算中心坐标
    j_plus_half = j_grid + 0.5  # 水平方向网格中心
    vertical_idx = (plane_height - 1 - i_grid) + 0.5  # 垂直方向网格中心（已反转）

    # 水平方向(u): 从左到右
    u_center = -U + j_plus_half * delta_u
    # 垂直方向(v): 从上到下
    v_center = -V + vertical_idx * delta_v

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
    
    # 使用双线性插值获取像素值（这里简化为最近邻）
    for i in range(plane_height):
        for j in range(plane_width):
            px = int(erp_x[i, j])
            py = int(erp_y[i, j])
            plane_img[i, j] = img[py, px]

    return plane_img

# ====== 主函数 ======
def main():
    print("开始生成18组BFOV的切平面无畸变图...")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    
    # 显示标注信息
    print(f"图像尺寸: {ERP_WIDTH} x {ERP_HEIGHT}")
    print(f"总BFoV数量: {len(FIXED_BFOV_ANNOTATIONS)}")
    print(f"  person: {len(FIXED_BFOV_ANNOTATIONS[:9])} 组")
    print(f"  chair: {len(FIXED_BFOV_ANNOTATIONS[9:])} 组")

    # 生成切平面无畸变图
    print("\n开始生成切平面无畸变图...")
    for i, bfov_params in enumerate(FIXED_BFOV_ANNOTATIONS):
        category_id = bfov_params[0]
        theta, phi, fov_x, fov_y = bfov_params[1:5]
        
        # 确定类别名称和输出文件名
        category_name = 'person' if category_id == 34 else 'chair'
        idx_in_category = i if category_id == 34 else i - 9
        
        # 打印信息
        theta_deg = theta * 180 / np.pi
        phi_deg = phi * 180 / np.pi
        fov_x_deg = fov_x * 180 / np.pi
        fov_y_deg = fov_y * 180 / np.pi
        print(f"  [{i+1:02d}] {category_name}_{idx_in_category+1:02d}")
        print(f"       中心: ({theta_deg:.2f}°, {phi_deg:.2f}°)")
        print(f"       FOV: ({fov_x_deg:.2f}° x {fov_y_deg:.2f}°)")
        
        # 生成切平面
        try:
            plane_img = project_bfov_to_plane(img, bfov_params)
            if plane_img is not None:
                plane_output_path = os.path.join(OUTPUT_DIR, f"{category_name}_{idx_in_category+1:02d}.jpg")
                cv2.imwrite(plane_output_path, plane_img)
                print(f"       保存到: {plane_output_path}")
            else:
                print(f"       失败: 切平面为None")
        except Exception as e:
            print(f"       失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n所有切平面图像生成完成！")

if __name__ == '__main__':
    main()
