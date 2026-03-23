import re
from PIL import Image, ImageDraw

# 配置文件路径和行范围
ROI_FILE_PATH = './bfov/roi_result/roi_info.txt'
START_LINE = 51  # 开始行
END_LINE = 246   # 结束行

# 从文件读取指定范围的行
def read_target_lines(file_path, start_line, end_line):
    target_lines = []
    with open(file_path, 'r') as f:
        # 遍历文件，行号从1开始
        for line_num, line in enumerate(f, 1):
            if start_line <= line_num <= end_line:
                target_lines.append(line.strip())
    return target_lines

# 解析每行中的浮点数坐标
def parse_coordinates(lines):
    coordinates = []
    # 正则表达式：匹配每行中后面的两个浮点数
    pattern = r'\([^)]+\): \(([\d\.]+), ([\d\.]+)\)'
    
    for line in lines:
        match = re.match(pattern, line)
        if match:
            # 提取两个浮点数作为坐标
            x = float(match.group(1))
            y = float(match.group(2))
            coordinates.append((x, y))
    
    return coordinates

# 绘制坐标点
def draw_points(coordinates, output_path):
    # 创建1024x512的空白图像
    img = Image.new('RGB', (1024, 512), color='white')
    draw = ImageDraw.Draw(img)
    
    # 绘制所有坐标点，使用大尺寸红色点
    for x, y in coordinates:
        # 确保坐标在图像范围内
        if 0 <= x <= 1024 and 0 <= y <= 512:
            draw.ellipse((x-5, y-5, x+5, y+5), fill='red', outline='black', width=3)
    
    # 保存图像
    img.save(output_path)
    print(f"图像已保存到: {output_path}")
    print(f"共绘制了 {len(coordinates)} 个点")

# 主函数
def main():
    print(f"正在读取 {ROI_FILE_PATH} 文件的第 {START_LINE}-{END_LINE} 行...")
    
    # 读取目标行
    target_lines = read_target_lines(ROI_FILE_PATH, START_LINE, END_LINE)
    print(f"成功读取 {len(target_lines)} 行数据")
    
    # 解析坐标
    coordinates = parse_coordinates(target_lines)
    print(f"成功解析 {len(coordinates)} 个坐标点")
    
    # 显示坐标范围
    if coordinates:
        xs = [x for x, y in coordinates]
        ys = [y for x, y in coordinates]
        print(f"X坐标范围: {min(xs):.2f} - {max(xs):.2f}")
        print(f"Y坐标范围: {min(ys):.2f} - {max(ys):.2f}")
    
    # 绘制并保存图像
    output_path = './bfov/coordinates_plot.png'
    draw_points(coordinates, output_path)
    
    print("绘制完成！")

if __name__ == "__main__":
    main()