import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PANDORA.RoIoU.libs.tools import rotate_image, ro_Shpbbox, tools
from PANDORA.RoIoU.libs.ImageRecorder import ImageRecorder

def rotate_image_3d(img, angles):
    """对ERP图像进行三维旋转"""
    erp_h, erp_w = img.shape[:2]
    roll, pitch, yaw = angles
    rotated_img = img.copy()
    
    if abs(roll) > 1e-6:
        rotated_img = rotate_image(rotated_img, roll, np.array([1, 0, 0], dtype=np.float64))
    if abs(pitch) > 1e-6:
        rotated_img = rotate_image(rotated_img, pitch, np.array([0, 1, 0], dtype=np.float64))
    if abs(yaw) > 1e-6:
        rotated_img = rotate_image(rotated_img, yaw, np.array([0, 0, 1], dtype=np.float64))
    
    return rotated_img

def transform_rbfov_params(rbfov_params, rotation_angles, erp_w, erp_h):
    """
    最终修正版：应用 Case 5 的逻辑 (Roll+, Pitch-, Yaw+)
    计算RBFoV在三维旋转后的新参数
    """
    center_x, center_y, fov_x, fov_y, angle = rbfov_params
    roll, pitch, yaw = rotation_angles
    
    t = tools(erp_w, erp_h)
    
    # ========== 1. 确定变换角度 ==========
    # 根据测试结果 Case 5: Pitch 取反，Roll 和 Yaw 不变
    roll_t = roll
    pitch_t = -pitch  # 关键修正
    yaw_t = yaw
    
    # ========== 2. 变换中心点坐标 ==========
    # 计算原始中心点的像素坐标
    center_px = (center_x + 180) / 360 * erp_w
    center_py = (90 - center_y) / 180 * erp_h
    
    # 转为三维单位向量
    center_xyz = np.array(t.pxpy2xyz([center_px + 0.5, center_py + 0.5]))
    center_xyz = center_xyz / np.linalg.norm(center_xyz)
    
    # 应用旋转矩阵
    if abs(roll_t) > 1e-6:
        center_xyz = np.array(t.roll_T(np.array([1, 0, 0]), center_xyz, roll_t))
    if abs(pitch_t) > 1e-6:
        center_xyz = np.array(t.roll_T(np.array([0, 1, 0]), center_xyz, pitch_t))
    if abs(yaw_t) > 1e-6:
        center_xyz = np.array(t.roll_T(np.array([0, 0, 1]), center_xyz, yaw_t))
        
    center_xyz = center_xyz / np.linalg.norm(center_xyz)
    
    # 转回经纬度
    px_new, py_new = t.xyz2pxpy(center_xyz)
    new_center_x = (px_new / erp_w) * 360 - 180
    new_center_y = 90 - (py_new / erp_h) * 180
    
    # ========== 3. 变换方向向量以计算新角度 ==========
    # 重新获取原始中心点向量
    orig_xyz = np.array(t.pxpy2xyz([center_px + 0.5, center_py + 0.5]))
    orig_xyz = orig_xyz / np.linalg.norm(orig_xyz)
    
    # 计算原始位置的局部坐标系 (东方向、北方向)
    east_orig = np.cross([0, 1, 0], orig_xyz)
    if np.linalg.norm(east_orig) < 1e-10: east_orig = np.array([1, 0, 0])
    else: east_orig = east_orig / np.linalg.norm(east_orig)
    
    north_orig = np.cross(orig_xyz, east_orig)
    north_orig = north_orig / np.linalg.norm(north_orig)
    
    # 构建原始方向向量
    rad_angle = np.deg2rad(angle)
    dir_vec = np.cos(rad_angle) * east_orig + np.sin(rad_angle) * north_orig
    
    # 对方向向量应用同样的旋转
    if abs(roll_t) > 1e-6:
        dir_vec = np.array(t.roll_T(np.array([1, 0, 0]), dir_vec, roll_t))
    if abs(pitch_t) > 1e-6:
        dir_vec = np.array(t.roll_T(np.array([0, 1, 0]), dir_vec, pitch_t))
    if abs(yaw_t) > 1e-6:
        dir_vec = np.array(t.roll_T(np.array([0, 0, 1]), dir_vec, yaw_t))
        
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    
    # 计算新位置处的局部坐标系
    east_new = np.cross([0, 1, 0], center_xyz)
    if np.linalg.norm(east_new) < 1e-10: east_new = np.array([1, 0, 0])
    else: east_new = east_new / np.linalg.norm(east_new)
        
    north_new = np.cross(center_xyz, east_new)
    north_new = north_new / np.linalg.norm(north_new)
    
    # 投影计算新角度
    dot_e = np.dot(dir_vec, east_new)
    dot_n = np.dot(dir_vec, north_new)
    
    new_angle = np.rad2deg(np.arctan2(dot_n, dot_e))
    new_angle = (new_angle + 180) % 360 - 180
    
    return np.array([new_center_x, new_center_y, fov_x, fov_y, new_angle])

def draw_rbfov(img, rbfov_params, color=(0, 255, 0)):
    """绘制RBFoV"""
    erp_h, erp_w = img.shape[:2]
    center_x, center_y, fov_x, fov_y, angle = rbfov_params
    
    BFoV = ImageRecorder(erp_w, erp_h, view_angle_w=fov_x, view_angle_h=fov_y, long_side=erp_w)
    
    center_x_rad = center_x / 180 * np.pi
    center_y_rad = center_y / 180 * np.pi
    
    Px, Py = BFoV._sample_points(center_x_rad, center_y_rad, border_only=True)
    
    rbfov_rad = np.array([center_x_rad, center_y_rad, fov_x, fov_y, angle])
    Px, Py = ro_Shpbbox(rbfov_rad, Px, Py, erp_w, erp_h)
    
    img = BFoV.draw_Sphbbox(img, Px, Py, border_only=True, color=color)
    
    center_px = int((center_x + 180) / 360 * erp_w)
    center_py = int((90 - center_y) / 180 * erp_h)
    cv2.circle(img, (center_px, center_py), 5, color, -1)
    
    return img

def main():
    input_path = "./bfov/3derp/7PhJB.jpg"
    output_dir = os.path.dirname(input_path)
    
    if not os.path.exists(input_path): return

    img = cv2.imread(input_path)
    erp_h, erp_w = img.shape[:2]
    
    # 参数设置
    rbfov = np.array([0, 0, 60, 40, 45])  # 中心(0,0), FoV 60x40, Angle 45
    rotation_params = [30, -45, 60]        # Roll=30, Pitch=45, Yaw=60
    
    print(f"原始 RBFoV: {rbfov}")
    print(f"旋转参数: Roll={rotation_params[0]}, Pitch={rotation_params[1]}, Yaw={rotation_params[2]}")
    
    # 1. 绘制原始框 (红色)
    img_orig = draw_rbfov(img.copy(), rbfov, color=(0, 0, 255))
    cv2.imwrite(os.path.join(output_dir, "step1_original.jpg"), img_orig)
    
    # 2. 旋转图像
    rotated_img = rotate_image_3d(img, rotation_params)
    
    # 3. 计算变换后的参数 (应用 Case 5 逻辑)
    new_rbfov = transform_rbfov_params(rbfov, rotation_params, erp_w, erp_h)
    print(f"变换后 RBFoV: {new_rbfov}")
    
    # 4. 绘制变换后的框 (绿色)
    img_rot = draw_rbfov(rotated_img, new_rbfov, color=(0, 255, 0))
    
    save_path = os.path.join(output_dir, "final_correct_result.jpg")
    cv2.imwrite(save_path, img_rot)
    print(f"\n成功！结果已保存至: {save_path}")

if __name__ == '__main__':
    main()