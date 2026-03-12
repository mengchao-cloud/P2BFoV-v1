import numpy as np
import cv2
import os
import sys
from pathlib import Path

# 添加本地lib目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
from ImageRecorder import ImageRecorder

def visualize_bfov_direct(image_path, bfov_params_list, output_path=None):
    """
    在指定图像上直接绘制BFOV
    
    参数:
    image_path: 图像文件路径，如果为None则创建空白图像
    bfov_params_list: BFOV参数列表，每个元素为 [center_x, center_y, fov_x, fov_y] (弧度制)
    output_path: 输出图像路径，如果为None则显示图像而不保存
    """
    # 读取图像或创建空白图像
    if image_path is None:
        # 创建1920x960的空白图像（白色背景）
        img_w, img_h = 1920, 960
        img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255  # 白色背景
        print(f"创建空白图像: {img_w} x {img_h}")
    else:
        # 读取指定图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"错误: 无法读取图像: {image_path}")
            return
        
        # 获取图像尺寸
        img_h, img_w = img.shape[:2]
        print(f"处理图像: {image_path}")
        print(f"图像尺寸: {img_w} x {img_h}")
    
    print(f"BFOV数量: {len(bfov_params_list)}")
    
    # 为每个BFOV分配不同的颜色
    colors = [
        (0, 255, 0),   # 绿色
        (255, 0, 0),   # 蓝色
        (0, 0, 255),   # 红色
        (255, 255, 0), # 青色
        (255, 0, 255), # 品红色
        (0, 255, 255), # 黄色
        (128, 0, 0),   # 深蓝色
        (0, 128, 0),   # 深绿色
        (0, 0, 128),   # 深红色
        (128, 128, 0), # 橄榄色
    ]
    
    # 绘制每个BFOV
    for i, bfov_params in enumerate(bfov_params_list):
        # 提取BFOV参数 [center_x, center_y, fov_x, fov_y] (弧度制)
        center_x_rad, center_y_rad, fov_x_rad, fov_y_rad = bfov_params
        
        # 转换为角度制用于显示
        center_x_deg = center_x_rad * 180 / np.pi
        center_y_deg = center_y_rad * 180 / np.pi
        fov_x_deg = fov_x_rad * 180 / np.pi
        fov_y_deg = fov_y_rad * 180 / np.pi
        
        # 检查FOV参数是否为负值，如果是则跳过绘制
        if fov_x_deg <= 0 or fov_y_deg <= 0:
            print(f"  BFOV {i+1}: 中心({center_x_deg:.2f}°, {center_y_deg:.2f}°), "
                  f"FOV({fov_x_deg:.2f}°x{fov_y_deg:.2f}°) - FOV参数为负，跳过绘制")
            continue
        
        print(f"  BFOV {i+1}: 中心({center_x_deg:.2f}°, {center_y_deg:.2f}°), "
              f"FOV({fov_x_deg:.2f}°x{fov_y_deg:.2f}°)")
        
        # 选择颜色（循环使用颜色列表）
        color = colors[i % len(colors)]
        
        # 使用ImageRecorder绘制BFOV
        try:
            # 为每个BFOV创建新的ImageRecorder实例（使用角度制FOV参数）
            BFoV = ImageRecorder(img_w, img_h, 
                               view_angle_w=fov_x_deg, 
                               view_angle_h=fov_y_deg, 
                               long_side=img_w)
            
            # 1. 获取BFOV的边界点
            Px, Py = BFoV._sample_points(center_x_rad, center_y_rad, border_only=True)
            
            # 2. 使用原始的draw_BFoV方法绘制BFOV
            img = BFoV.draw_BFoV(img, Px, Py, border_only=True, color=color)
            
            # 3. 绘制BFOV中心点
            center_px = int((center_x_rad + np.pi) / (2 * np.pi) * img_w)
            center_py = int((np.pi/2 - center_y_rad) / np.pi * img_h)
            # cv2.circle(img, (center_px, center_py), 16, color, -1)  # 增大中心点半径到8
            
            # 4. 添加BFOV编号标签
            # label = f"BFOV{i+1}"
            # cv2.putText(img, label, (center_px + 5, center_py), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        except Exception as e:
            print(f"    警告: 绘制BFOV {i+1} 时出错: {e}")
            continue
    
    # 保存或显示结果图像
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"结果已保存到: {output_path}")
    else:
        # 显示图像
        cv2.imshow('BFOV Visualization', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    """
    主函数：在这里直接指定图像路径和BFOV参数
    """
    # ================================
    # 在这里修改您的配置
    # ================================
    
    # 1. 指定图像路径（如果为None则创建空白图像）
    image_path = '/home/mengchao/workspace/P2BFoV/P2BNet-main/TOV_mmdetection/360indoor-short/images_short/7PhJB.jpg'  # 设置为None创建空白图像，或指定具体图像路径
    
    # 2. 指定BFOV参数列表（弧度制）
    # 每个BFOV参数格式: [center_x, center_y, fov_x, fov_y]
    # center_x: 经度（-π 到 π）
    # center_y: 纬度（-π/2 到 π/2）
    # fov_x: 水平视场角（弧度）
    # fov_y: 垂直视场角（弧度）
    
    bfov_params_list = [
        # # BFOV 1: 示例参数
        # [0.0867, -0.1587, 0.2094, 0.2793],  # 中心在(0,0)，FOV 45°x45°
        # [0.0867, -0.1587, 0.2094*2, 0.2793/2],
        # [0.0867, -0.1587, 0.2094/2, 0.2793*2],
        [0.8492117641734908,-1.0750137361502572,0.3490658503988659*1.1,0.2792526803190927/1.1],
        [0.8492117641734908+0.3490658503988659*0.2,-1.0750137361502572,0.3490658503988659*1.3,0.2792526803190927/1.3],
        [0.8492117641734908+0.3490658503988659*0.2,-1.0750137361502572,0.3490658503988659/1.3,0.2792526803190927*1.3],
        [0.8492117641734908,-1.0750137361502572+0.2792526803190927*0.2,0.3490658503988659*0.8,0.2792526803190927/0.8],
        [0.8492117641734908,-1.0750137361502572+0.2792526803190927*0.2,0.3490658503988659/0.8,0.2792526803190927*0.8],
        

        # [0.8492117641734908,-1.0750137361502572,0.3490658503988659*2,0.2792526803190927/2],
        # [0.8492117641734908,-1.0750137361502572,0.3490658503988659/2,0.2792526803190927*2],
        # [-2.541090307825494,-0.2372556951929793,0.2792526803190927,2.0943951023931953],
        # [-2.541090307825494,-0.2372556951929793,0.2792526803190927*2,2.0943951023931953/2],
        # [-2.541090307825494,-0.2372556951929793,0.2792526803190927/1.5,2.0943951023931953*1.2],
        # [1.867, -0.6587, 0.2094, 0.2793],  # 中心在(0,0)，FOV 45°x45°
        # [1.967, 2.6587, 0.2094*2, 0.2793],
        # [-1.867, -1.6587, 0.2094, 0.2793/2],
        # [2.867, 1.6587, 0.2094*1.5, 0.2793*1.5]
        # [0.8492117641734908,-1.0750137361502572,0.0000001,0.0000001],     
        # [-2.541090307825494,-0.2372556951929793,0.0000001,0.0000001],  
        # BFOV 2: 另一个示例

    ]
    
    # 3. 指定输出路径（可选，如果为None则显示图像而不保存）
    output_path = '/home/mengchao/workspace/P2BFoV/P2BNet-main/TOV_mmdetection/360indoor-short/visualize/bfov_visualization_result_neg.jpg'
    
    # ================================
    # 配置结束
    # ================================
    
    # 检查图像文件是否存在（仅在指定了图像路径时）
    if image_path is not None and not os.path.exists(image_path):
        print(f"错误: 图像文件不存在: {image_path}")
        print("请检查image_path配置是否正确")
        return
    
    print("开始直接可视化BFOV...")
    print(f"图像文件: {image_path}")
    print(f"BFOV参数数量: {len(bfov_params_list)}")
    print("-" * 50)
    
    # 执行可视化
    visualize_bfov_direct(image_path, bfov_params_list, output_path)
    
    print("-" * 50)
    print("BFOV直接可视化完成!")

if __name__ == '__main__':
    main()