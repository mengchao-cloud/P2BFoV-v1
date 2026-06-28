#!/usr/bin/env python3
"""
绘制图像 7PhJB.jpg 上的 person 和 chair 的中心点

使用方法：
conda run -n pointobb python draw_center_7PhJB.py
"""

import os
import sys
import numpy as np
import cv2

# person的中心点
PERSON_THETA = -0.0801760625134895
PERSON_PHI = -0.16853335589570229

# chair的中心点
CHAIR_THETA = 0.8492117641734908
CHAIR_PHI = -1.0750137361502572


def draw_center_points(img, center_points, img_w, img_h, color=(0, 255, 0)):
    """
    在图像上绘制中心点
    
    参数:
    img: 输入图像
    center_points: 中心点列表，每个元素为 (theta, phi)
    img_w: 图像宽度
    img_h: 图像高度
    color: 点的颜色 (B, G, R)
    
    返回:
    img: 绘制后的图像
    """
    for i, (theta, phi) in enumerate(center_points):
        # 将弧度转换为角度
        theta_deg = theta * 180 / np.pi
        phi_deg = phi * 180 / np.pi
        
        # 计算像素坐标
        cx = int((theta_deg + 180) / 360 * img_w)
        cy = int((90 - phi_deg) / 180 * img_h)
        
        print(f"中心点 {i+1}: 经纬度 ({theta_deg:.2f}°, {phi_deg:.2f}°) -> 像素坐标 ({cx}, {cy})")
        
        if 0 <= cy < img_h and 0 <= cx < img_w:
            # 绘制中心点：白色边框 + 彩色填充
            cv2.circle(img, (cx, cy), 8, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.circle(img, (cx, cy), 8, color, -1, lineType=cv2.LINE_AA)
    
    return img


def main():
    # 配置参数
    img_path = './7PhJB.jpg'  # 图像路径
    output_path = './7PhJB_center.jpg'  # 输出路径
    
    # 检查图像是否存在
    if not os.path.exists(img_path):
        print(f"错误: 图像文件不存在: {img_path}")
        return
    
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"错误: 无法读取图像: {img_path}")
        return
    
    img_h, img_w = img.shape[:2]
    print(f"图像尺寸: {img_w} x {img_h}")
    
    # 定义要绘制的中心点
    center_points = [
        (PERSON_THETA, PERSON_PHI),  # person
        (CHAIR_THETA, CHAIR_PHI),    # chair
    ]
    
    print("\n=== 绘制中心点 ===")
    # 绘制中心点（绿色）
    img_with_center = draw_center_points(img.copy(), center_points, img_w, img_h, color=(0, 255, 0))
    
    # 保存图像
    cv2.imwrite(output_path, img_with_center)
    print(f"\n中心点绘制完成，保存到: {os.path.abspath(output_path)}")


if __name__ == '__main__':
    main()
