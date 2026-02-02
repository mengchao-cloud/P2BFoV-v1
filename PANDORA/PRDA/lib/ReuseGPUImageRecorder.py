import torch
import numpy as np


class ReuseGPUImageRecorder(object):
    """
    复用版本的GPUImageRecorder类，支持参数动态更新
    通过实例复用大幅降低显存占用
    """

    def __init__(self, sphereW, sphereH, view_angle_w=64, view_angle_h=64, long_side=640, device='cuda'):
        self.sphereW = sphereW
        self.sphereH = sphereH
        self.device = device
        self._long_side = long_side
        
        # 🔑 新增：存储FOV参数，支持动态更新
        self._view_angle_w = view_angle_w
        self._view_angle_h = view_angle_h
        
        # 初始化核心张量
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化核心计算组件"""
        fov_w, fov_h = self._view_angle_w, self._view_angle_h
        
        # 限制视场角的最小值，避免计算出过大的图像尺寸
        min_fov = 1.0  # 最小视场角为1度
        fov_w = max(min_fov, fov_w)
        fov_h = max(min_fov, fov_h)

        if fov_w >= fov_h:
            self._imgW = self._long_side
            self._imgH = int(np.tan(fov_h / 360 * np.pi) * self._imgW / float(np.tan(fov_w / 360 * np.pi)))
        else:
            self._imgH = self._long_side
            self._imgW = int(np.tan(fov_w / 360 * np.pi) * self._imgH / float(np.tan(fov_h / 360 * np.pi)))

        # 限制图像尺寸的最大值，避免内存溢出
        max_img_size = 2048  # 最大图像尺寸为2048
        self._imgW = min(max_img_size, max(1, self._imgW))
        self._imgH = min(max_img_size, max(1, self._imgH))
        
        TX, TY = self._meshgrid()
        R, ANGy = self._compute_radius(self._view_angle_w, TY)

        self._R = R
        self._ANGy = ANGy
        self._Z = TX
    
    # 🔑 新增：属性访问器和更新方法
    @property
    def view_angle_w(self):
        """获取水平视场角"""
        return self._view_angle_w
    
    @view_angle_w.setter
    def view_angle_w(self, value):
        """设置水平视场角并重新初始化组件"""
        if value != self._view_angle_w:
            self._view_angle_w = value
            self._initialize_components()
    
    @property
    def view_angle_h(self):
        """获取垂直视场角"""
        return self._view_angle_h
    
    @view_angle_h.setter
    def view_angle_h(self, value):
        """设置垂直视场角并重新初始化组件"""
        if value != self._view_angle_h:
            self._view_angle_h = value
            self._initialize_components()
    
    @property
    def long_side(self):
        """获取长边长度"""
        return self._long_side
    
    @long_side.setter
    def long_side(self, value):
        """设置长边长度并重新初始化组件"""
        if value != self._long_side:
            self._long_side = value
            self._initialize_components()

    def _meshgrid(self):
        """在GPU上构造网格点
        :returns: TX, TY (torch.Tensor)
        """
        # 确保imgW和imgH至少为1，避免torch.arange参数错误
        self._imgW = max(1, self._imgW)
        self._imgH = max(1, self._imgH)
        
        if self._imgW >= self._imgH:
            offset = int((self._imgW - self._imgH)/2)
            # 在GPU上创建网格
            x_range = torch.arange(self._imgW, device=self.device)
            y_range = torch.arange(offset, self._imgH + offset, device=self.device)
            # 移除indexing参数，使用默认方式
            TX, TY = torch.meshgrid(x_range, y_range)
        else:
            offset = int((self._imgH - self._imgW)/2)
            # 在GPU上创建网格
            # 确保起始值小于结束值
            start = offset
            end = self._imgW + offset
            if start >= end:
                # 如果起始值大于等于结束值，创建一个最小的有效范围
                start = 0
                end = max(1, self._imgW)
            
            x_range = torch.arange(start, end, device=self.device)
            y_range = torch.arange(self._imgH, device=self.device)
            # 移除indexing参数，使用默认方式
            TX, TY = torch.meshgrid(x_range, y_range)

        TX = TX.to(torch.float32) - 0.5
        TX -= self._long_side / 2

        TY = TY.to(torch.float32) - 0.5
        TY -= self._long_side / 2
        return TX, TY

    def _compute_radius(self, view_angle, TY):
        """在GPU上计算半径和角度
        :returns: R, ANGy (torch.Tensor)
        """
        _view_angle = torch.tensor(np.pi * view_angle / 180., device=self.device, dtype=torch.float32)
        r = self._imgW / 2 / torch.tan(_view_angle / 2)
        R = torch.sqrt(torch.pow(TY, 2) + r**2)
        ANGy = torch.atan(-TY / r)

        return R, ANGy

    def _sample_points(self, x, y, border_only=False):
        """在GPU上生成采样点
        :param x: 经度旋转角度（弧度）
        :param y: 纬度旋转角度（弧度）
        :param border_only: 是否只生成边界点
        :returns: Px, Py (torch.Tensor) - ERP图像上的采样点坐标
        """
        angle_x, angle_y = self._direct_camera(x, y, border_only)
        Px = (angle_x + np.pi) / (2 * np.pi) * self.sphereW + 0.5
        Py = (np.pi / 2 - angle_y) / np.pi * self.sphereH + 0.5
        INDx = Px < 1
        Px[INDx] += self.sphereW
        return Px, Py  # torch.Tensor

    def _direct_camera(self, rotate_x, rotate_y, border_only=False):
        """在GPU上计算相机方向
        :param rotate_x: 经度旋转角度（弧度）
        :param rotate_y: 纬度旋转角度（弧度）
        :param border_only: 是否只计算边界点
        :returns: angle_x, angle_y (torch.Tensor) - 计算后的角度
        """
        if border_only:
            # 只处理边界点
            top_edge = torch.hstack([self._ANGy[0, :], self._ANGy[-1, :], self._ANGy[:, 0], self._ANGy[:, -1]])
            angle_y = top_edge + rotate_y
            
            top_Z = torch.hstack([self._Z[0, :], self._Z[-1, :], self._Z[:, 0], self._Z[:, -1]])
            Z = top_Z
            
            top_R = torch.hstack([self._R[0, :], self._R[-1, :], self._R[:, 0], self._R[:, -1]])
            R = top_R
        else:
            # 处理所有点
            angle_y = self._ANGy + rotate_y
            Z = self._Z  # Z = TX
            R = self._R

        X = torch.sin(angle_y) * R
        Y = - torch.cos(angle_y) * R

        INDn = torch.abs(angle_y) > np.pi / 2

        angle_x = torch.atan(Z / -Y)
        RZY = torch.sqrt(torch.pow(Y, 2) + torch.pow(Z, 2))
        angle_y = torch.atan(X / RZY)

        angle_x[INDn] += np.pi
        angle_x += rotate_x

        INDy = angle_y < -np.pi / 2
        angle_y[INDy] = -np.pi - angle_y[INDy]
        angle_x[INDy] = angle_x[INDy] + np.pi

        INDx = angle_x <= -np.pi
        angle_x[INDx] += 2 * np.pi
        INDx = angle_x > np.pi
        angle_x[INDx] -= 2 * np.pi
        return angle_x, angle_y