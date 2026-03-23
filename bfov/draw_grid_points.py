import re
from PIL import Image, ImageDraw, ImageFont

# 配置文件路径和行范围
ROI_FILE_PATH = '/home/mengchao/workspace/P2BFoV/P2BNet-main/TOV_mmdetection/bfov/roi_result/roi_info.txt'
START_LINE = 51  # 开始行
END_LINE = 246   # 结束行

# 从文件读取指定范围的行
def read_target_lines(file_path, start_line, end_line):
    target_lines = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if start_line <= line_num <= end_line:
                target_lines.append(line.strip())
    return target_lines

# 解析每行中的网格索引
def parse_grid_indices(lines):
    indices = []
    # 正则表达式：匹配网格索引
    pattern = r'\((\d+),(\d+)\): \(([\d\.]+), ([\d\.]+)\)'
    
    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            # 提取网格索引
            h_idx = int(match.group(1))
            w_idx = int(match.group(2))
            indices.append((h_idx, w_idx))
    
    return indices

# 根据网格索引绘制点
def draw_grid_points(indices, output_path):
    # 创建1024x512的空白图像
    img = Image.new('RGB', (1024, 512), color='white')
    draw = ImageDraw.Draw(img)
    
    # 计算网格在图像上的显示范围
    # 使用图像的中间区域来显示网格
    margin = 100
    grid_width = 1024 - 2 * margin
    grid_height = 512 - 2 * margin
    
    # 找到网格索引的范围
    min_h = min(h for h, w in indices)
    max_h = max(h for h, w in indices)
    min_w = min(w for h, w in indices)
    max_w = max(w for h, w in indices)
    h_count = max_h - min_h + 1
    w_count = max_w - min_w + 1
    
    # 计算每个网格单元的大小
    cell_width = grid_width / (w_count - 1) if w_count > 1 else grid_width
    cell_height = grid_height / (h_count - 1) if h_count > 1 else grid_height
    
    # 绘制网格线（可选）
    for w in range(w_count):
        x = margin + w * cell_width
        draw.line((x, margin, x, margin + grid_height), fill='lightgray', width=1)
    for h in range(h_count):
        y = margin + h * cell_height
        draw.line((margin, y, margin + grid_width, y), fill='lightgray', width=1)
    
    # 绘制所有网格点
    drawn_points = []
    for h_idx, w_idx in indices:
        # 计算点在图像上的位置
        x = margin + (w_idx - min_w) * cell_width
        y = margin + (h_idx - min_h) * cell_height
        drawn_points.append((x, y))
        
        # 绘制点
        draw.ellipse((x-5, y-5, x+5, y+5), fill='red', outline='black', width=2)
        
        # 显示网格索引
        draw.text((x+8, y-8), f'({h_idx},{w_idx})', fill='black', font=None)
    
    # 保存图像
    img.save(output_path)
    print(f"图像已保存到: {output_path}")
    print(f"共绘制了 {len(drawn_points)} 个点")
    print(f"网格尺寸: {h_count}x{w_count}")

# 主函数
def main():
    print(f"正在读取 {ROI_FILE_PATH} 文件的第 {START_LINE}-{END_LINE} 行...")
    
    # 读取目标行
    target_lines = read_target_lines(ROI_FILE_PATH, START_LINE, END_LINE)
    print(f"成功读取 {len(target_lines)} 行数据")
    
    # 解析网格索引
    indices = parse_grid_indices(target_lines)
    print(f"成功解析 {len(indices)} 个网格点")
    
    # 绘制网格点
    output_path = '/home/mengchao/workspace/P2BFoV/P2BNet-main/TOV_mmdetection/bfov/grid_points.png'
    draw_grid_points(indices, output_path)
    
    print("绘制完成！")

if __name__ == "__main__":
    main()