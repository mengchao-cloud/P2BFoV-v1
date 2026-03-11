import numpy as np

class BFOVAreaCalculator:
    """
    计算BFOV（球坐标系视场）面积的工具类
    使用精确的球面矩形面积公式
    """
    
    def calculate_area(self, fov_x, fov_y, unit='degrees'):
        """
        计算BFOV面积
        
        参数:
            fov_x (float or list): 水平视场角
            fov_y (float or list): 垂直视场角
            unit (str): 输入参数的单位 ('degrees' 或 'radians')
            
        返回:
            float or list: BFOV面积，单位为球面度 (steradians)
        """
        # 处理列表输入
        is_list = isinstance(fov_x, list) or isinstance(fov_y, list)
        if is_list:
            # 确保两个参数都是列表
            fov_x_list = fov_x if isinstance(fov_x, list) else [fov_x] * len(fov_y)
            fov_y_list = fov_y if isinstance(fov_y, list) else [fov_y] * len(fov_x)
            
            # 计算每个BFOV的面积
            areas = []
            for fx, fy in zip(fov_x_list, fov_y_list):
                area = self._single_area(fx, fy, unit)
                areas.append(area)
            return areas
        else:
            # 单个BFOV面积计算
            return self._single_area(fov_x, fov_y, unit)
    
    def _single_area(self, fov_x, fov_y, unit):
        """
        计算单个BFOV的面积
        """
        # 转换为弧度
        if unit == 'degrees':
            fov_x_rad = np.deg2rad(fov_x)
            fov_y_rad = np.deg2rad(fov_y)
        elif unit == 'radians':
            fov_x_rad = fov_x
            fov_y_rad = fov_y
        else:
            raise ValueError("unit must be 'degrees' or 'radians'")
        
        # 使用球面矩形面积公式
        area = 4 * np.arccos(-np.sin(fov_x_rad / 2) * np.sin(fov_y_rad / 2)) - 2 * np.pi
        return area


def example_usage():
    """
    示例用法
    """
    # 创建计算器实例
    calculator = BFOVAreaCalculator()
    
    print("=== BFOV面积计算示例 ===")
    print()
    
    # 示例1: 单个BFOV面积计算
    print("1. 单个BFOV面积计算:")
    fov_x = 60.0  # 水平视场角，单位：度
    fov_y = 40.0  # 垂直视场角，单位：度
    area = calculator.calculate_area(fov_x, fov_y)
    print(f"   水平视场: {fov_x}°，垂直视场: {fov_y}°，面积: {area:.4f} 球面度")
    print()
    
    # 示例2: 多个BFOV面积计算（使用列表）
    print("2. 多个BFOV面积计算:")
    fov_x_list = [30.0, 60.0, 90.0, 120.0]  # 水平视场角列表，单位：度
    fov_y_list = [20.0, 40.0, 60.0, 80.0]    # 垂直视场角列表，单位：度
    
    areas = calculator.calculate_area(fov_x_list, fov_y_list)
    
    for i, (fx, fy, area) in enumerate(zip(fov_x_list, fov_y_list, areas), 1):
        print(f"   第{i}组: 水平视场: {fx}°，垂直视场: {fy}°，面积: {area:.4f} 球面度")
    print()
    
    # 示例3: 使用弧度作为单位（单个）
    print("3. 使用弧度作为单位（单个）:")
    fov_x_rad = np.pi / 3  # 60度，水平视场角
    fov_y_rad = np.pi / 4.5  # 40度，垂直视场角
    area = calculator.calculate_area(fov_x_rad, fov_y_rad, unit='radians')
    print(f"   水平视场: {fov_x_rad:.4f} rad，垂直视场: {fov_y_rad:.4f} rad，面积: {area:.4f} 球面度")
    print()
    
    # 示例4: 使用弧度作为单位（多组）
    print("4. 使用弧度作为单位（多组）:")
    # 定义用户指定的弧度制BFOV参数
    fov_x_rad_list = [0.5, 1.0,3.14]  # 水平视场角（弧度）
    fov_y_rad_list = [0.5, 1.0,3.14]  # 垂直视场角（弧度）
    
    # 转换为角度便于查看（可选）
    fov_x_deg_list = np.rad2deg(fov_x_rad_list)
    fov_y_deg_list = np.rad2deg(fov_y_rad_list)
    
    # 计算面积
    areas = calculator.calculate_area(fov_x_rad_list, fov_y_rad_list, unit='radians')
    
    # 输出结果
    print("   弧度制BFOV面积计算结果:")
    print("   ---------------------------")
    print("   | 水平视场(rad) | 垂直视场(rad) | 水平视场(°) | 垂直视场(°) | 面积(球面度) |")
    print("   ---------------------------")
    for fx_rad, fy_rad, fx_deg, fy_deg, area in zip(fov_x_rad_list, fov_y_rad_list, fov_x_deg_list, fov_y_deg_list, areas):
        print(f"   |     {fx_rad:.4f}     |     {fy_rad:.4f}     |    {fx_deg:.1f}°    |    {fy_deg:.1f}°    |    {area:.4f}    |")
    print("   ---------------------------")


if __name__ == "__main__":
    example_usage()