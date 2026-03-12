#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像分辨率调整脚本
将图像从1920*920调整为1024*512
"""

from PIL import Image
import os
import sys

def resize_image(input_path, output_path, new_size=(1024, 512)):
    """
    调整图像分辨率
    
    参数:
        input_path (str): 输入图像路径
        output_path (str): 输出图像路径
        new_size (tuple): 新的尺寸 (width, height)
    """
    try:
        # 打开原始图像
        with Image.open(input_path) as img:
            print(f"原始图像信息: {img.size} {img.mode}")
            
            # 调整图像大小
            resized_img = img.resize(new_size, Image.LANCZOS)  # 使用LANCZOS插值方法保持高质量
            print(f"调整后图像信息: {resized_img.size} {resized_img.mode}")
            
            # 保存调整后的图像
            resized_img.save(output_path, quality=95, optimize=True)
            print(f"图像已成功保存到: {output_path}")
            
    except Exception as e:
        print(f"处理图像时出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # 设置输入和输出路径
    input_image = "/home/mengchao/workspace/P2BFoV/P2BNet-main/TOV_mmdetection/bfov/test/7PhJB.jpg"
    output_image = "/home/mengchao/workspace/P2BFoV/P2BNet-main/TOV_mmdetection/bfov/test/7PhJB_resized_1024x512.jpg"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_image):
        print(f"错误: 输入文件不存在: {input_image}")
        sys.exit(1)
    
    # 执行图像调整
    resize_image(input_image, output_image)
