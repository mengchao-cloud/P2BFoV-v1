import torch
import math
from mmcv.runner import force_fp32

from mmdet.models.builder import ROI_EXTRACTORS
from .base_roi_extractor import BaseRoIExtractor

# Define PI constant for compatibility
def get_pi():
    if hasattr(torch, 'pi'):
        return torch.pi
    else:
        return torch.tensor(math.pi)

PI = get_pi()


@ROI_EXTRACTORS.register_module()
class SingleRoIExtractor(BaseRoIExtractor):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (List[int]): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0. Default: 56.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56,
                 init_cfg=None):
        super(SingleRoIExtractor, self).__init__(roi_layer, out_channels,
                                                 featmap_strides, init_cfg)
        self.finest_scale = finest_scale
        self.batch_count = 0  # Track batch number to only print first batch

    # def map_roi_levels(self, rois, num_levels):
    #     """Map rois to corresponding feature levels by scales.

    #     - scale < finest_scale * 2: level 0
    #     - finest_scale * 2 <= scale < finest_scale * 4: level 1
    #     - finest_scale * 4 <= scale < finest_scale * 8: level 2
    #     - scale >= finest_scale * 8: level 3

    #     Args:
    #         rois (Tensor): Input RoIs, shape (k, 5).
    #         num_levels (int): Total level number.

    #     Returns:
    #         Tensor: Level index (0-based) of each RoI, shape (k, )
    #     """
    #     scale = torch.sqrt(
    #         (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
    #     target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
    #     target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
    #     return target_lvls

   
    def calculate_tangent_plane_scales(self, fov_u, fov_v):
        # add by mc
        """
        批量计算单位球坐标系下切平面的尺度（结合图像大小）
        
        参数:
            fov_u: 水平视场角（弧度）
                - 支持torch.Tensor标量或张量
                - 支持与fov_v相同形状的PyTorch张量
                - 支持fov_u和fov_v都是一维张量，长度相同
            fov_v: 垂直视场角（弧度）
                - 要求与fov_u形状兼容
        
        返回:
            torch.Tensor: 每个输入对应的切平面尺度
                尺度 = sqrt(切平面面积) * 1024/(2*PI)
                - 输出形状与输入形状相同
        """
        # 添加数值稳定性处理：限制tan函数的输入，避免接近π/2
        max_angle = PI/2 - 1e-4  # 接近但小于π/2的值
        fov_u_half = torch.clamp(fov_u / 2.0, min=-max_angle, max=max_angle)
        fov_v_half = torch.clamp(fov_v / 2.0, min=-max_angle, max=max_angle)
        
        # 计算切平面半宽（向量化计算，支持批量输入）
        U = torch.tan(fov_u_half)  # 水平半宽
        V = torch.tan(fov_v_half)  # 垂直半宽
        
        # 计算切平面水平和垂直尺寸
        plane_width = 2 * U
        plane_height = 2 * V
        
        # 计算面积平方根并结合图像大小得到实际尺度
        image_scale = 1024 / (2 * PI)
        scales = torch.sqrt(plane_width * plane_height) * image_scale
        
        return scales

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        # scale = torch.sqrt(
        #     (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        # add by mc
        scale = self.calculate_tangent_plane_scales(rois[:, 3], rois[:, 4])

        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls




    def bfov_roi_to_14x14_erp(self, bfov_params, erp_width=1024, erp_height=512, grid_size=14):
        """
        Generate 14x14 ERP points from BFOV parameters using PyTorch.
        This version follows the exact same logic as the numpy version bfov_to_spherical_points.
        
        Args:
            bfov_params: Tensor of shape (N, 4) containing BFOV parameters for each ROI.
                        Each row is [lon, lat, fov_u, fov_v] - all in radians:
                        - lon: BFOV center longitude (radians, range [-π, π])
                        - lat: BFOV center latitude (radians, range [-π/2, π/2])
                        - fov_u: Horizontal field of view (radians, longitude direction)
                        - fov_v: Vertical field of view (radians, latitude direction)
            erp_width: ERP image width, default 1024
            erp_height: ERP image height, default 512
            grid_size: Grid size, default 14
        
        Returns:
            points: Tensor of shape (N, 14, 14, 2) containing 14x14 ERP points for each BFOV.
                    Each BFOV corresponds to a 14x14 2D array where each element is (x, y) coordinates (float).
        """
        # Get batch size and device
        N = bfov_params.size(0)
        device = bfov_params.device
            
        # Extract BFOV parameters (all in radians)
        lon = bfov_params[:, 0]  # (N,), longitude in radians
        lat = bfov_params[:, 1]  # (N,), latitude in radians
        fov_u = bfov_params[:, 2]  # (N,), horizontal FOV in radians
        fov_v = bfov_params[:, 3]  # (N,), vertical FOV in radians
        
        # Convert longitude/latitude to polar/azimuth angles (same as numpy version)
        phi0 = lon  # Azimuth angle (N,)
        theta0 = PI/2 - lat  # Polar angle (N,)
        
        # 1. Calculate BFOV center unit vector c (x,y,z) - same as numpy version
        c_x = torch.sin(theta0) * torch.cos(phi0)  # (N,)
        c_y = torch.sin(theta0) * torch.sin(phi0)  # (N,)
        c_z = torch.cos(theta0)  # (N,)
        c = torch.stack([c_x, c_y, c_z], dim=1)  # (N, 3)
        
        # 添加归一化确保单位向量精度，提高数值稳定性
        c = c / torch.norm(c, dim=1, keepdim=True)  # (N, 3)
        
        # 2. Construct local orthogonal basis (e_u, e_v) - same as numpy version
        sin_phi = torch.sin(phi0)  # (N,)
        cos_phi = torch.cos(phi0)  # (N,)
        sin_theta = torch.sin(theta0)  # (N,)
        cos_theta = torch.cos(theta0)  # (N,)
        
        # Longitude direction vector (horizontal)
        lon_dir = torch.stack([-sin_phi, cos_phi, torch.zeros_like(sin_phi)], dim=1)  # (N, 3)
        lon_dir = lon_dir / torch.norm(lon_dir, dim=1, keepdim=True)  # 归一化，确保单位向量
        
        # Latitude direction vector (vertical)
        lat_dir = torch.stack([-cos_theta*cos_phi, -cos_theta*sin_phi, sin_theta], dim=1)  # (N, 3)
        lat_dir = lat_dir / torch.norm(lat_dir, dim=1, keepdim=True)  # 归一化，确保单位向量
        
        # Ensure orthogonality - same as numpy version
        dot_product = torch.sum(lon_dir * lat_dir, dim=1)  # (N,)
        mask = torch.abs(dot_product) > 1e-6  # (N,)
        
        if mask.any():
            # Recalculate orthogonal basis for problematic cases
            lon_dir_ortho = lon_dir - torch.sum(lon_dir * c, dim=1, keepdim=True) * c
            lon_dir_ortho = lon_dir_ortho / torch.norm(lon_dir_ortho, dim=1, keepdim=True)
            lat_dir_ortho = torch.cross(c, lon_dir_ortho)
            lat_dir_ortho = lat_dir_ortho / torch.norm(lat_dir_ortho, dim=1, keepdim=True)
            
            # Update where mask is True
            lon_dir[mask] = lon_dir_ortho[mask]
            lat_dir[mask] = lat_dir_ortho[mask]
        
        # 3. Calculate tangent plane half-width (U, V) - same as numpy version
        # 添加数值稳定性处理：限制tan函数的输入，避免接近π/2
        max_angle = PI/2 - 1e-4  # 接近但小于π/2的值
        fov_u_half = torch.clamp(fov_u / 2.0, min=-max_angle, max=max_angle)
        fov_v_half = torch.clamp(fov_v / 2.0, min=-max_angle, max=max_angle)
        
        U = torch.tan(fov_u_half)  # Horizontal half-width (N,)
        V = torch.tan(fov_v_half)  # Vertical half-width (N,)
        
        # 4. Calculate grid step - same as numpy version
        delta_u = 2 * U / grid_size  # Horizontal step (N,)
        delta_v = 2 * V / grid_size  # Vertical step (N,)
        
        # 5. Generate grid indices - i对应垂直方向(v), j对应水平方向(u)
        indices = torch.zeros(grid_size, grid_size, 2, device=device, dtype=torch.int64)
        for i in range(grid_size):  # i: 垂直方向(v)索引
            for j in range(grid_size):  # j: 水平方向(u)索引
                indices[i, j, 0] = i  # 行索引 → 垂直方向
                indices[i, j, 1] = j  # 列索引 → 水平方向
        
        # 分离i和j索引
        i_grid = indices[..., 0].float()  # (14, 14) - 垂直方向(v)
        j_grid = indices[..., 1].float()  # (14, 14) - 水平方向(u)
        
        # 6. Calculate u_center and v_center - 修正坐标映射
        # u_center: 水平方向坐标，对应j索引(列)
        # v_center: 垂直方向坐标，对应i索引(行)
        
        # 扩展U, V, delta_u, delta_v到网格形状
        U_expanded = U.view(N, 1, 1)  # (N, 1, 1) - 水平半宽
        V_expanded = V.view(N, 1, 1)  # (N, 1, 1) - 垂直半宽
        delta_u_expanded = delta_u.view(N, 1, 1)  # (N, 1, 1) - 水平步长
        delta_v_expanded = delta_v.view(N, 1, 1)  # (N, 1, 1) - 垂直步长
        
        # 计算中心坐标 - 修正垂直方向映射
        # j对应水平方向(u)，从左到右
        j_plus_half = j_grid + 0.5  # 水平方向网格中心
        
        # i对应垂直方向(v)，反转垂直方向索引，确保从上到下
        # grid_size-1 - i_grid 确保i=0对应BFOV顶部，i=13对应BFOV底部
        vertical_idx = (grid_size - 1 - i_grid) + 0.5  # 垂直方向网格中心（已反转）
        
        # 水平方向(u): 从左到右
        u_center = -U_expanded + j_plus_half.unsqueeze(0) * delta_u_expanded  # (N, 14, 14)
        # 垂直方向(v): 从上到下
        v_center = -V_expanded + vertical_idx.unsqueeze(0) * delta_v_expanded  # (N, 14, 14)
        
        # 7. Expand basis vectors to match grid size
        c_expand = c.unsqueeze(1).unsqueeze(1).expand(N, grid_size, grid_size, 3)  # (N, 14, 14, 3)
        e_u_expand = lon_dir.unsqueeze(1).unsqueeze(1).expand(N, grid_size, grid_size, 3)  # (N, 14, 14, 3)
        e_v_expand = lat_dir.unsqueeze(1).unsqueeze(1).expand(N, grid_size, grid_size, 3)  # (N, 14, 14, 3)
        
        # 8. Calculate points on tangent plane - same as numpy version
        x_plane = c_expand + u_center.unsqueeze(-1) * e_u_expand + v_center.unsqueeze(-1) * e_v_expand  # (N, 14, 14, 3)
        
        # 9. Project back to unit sphere - same as numpy version
        x_sphere = x_plane / torch.norm(x_plane, dim=-1, keepdim=True)  # (N, 14, 14, 3)
        
        # 10. Convert spherical coordinates to lat/lon (degrees for ERP conversion)
        x = x_sphere[..., 0]
        y = x_sphere[..., 1]
        z = x_sphere[..., 2]
        
        # 添加数值稳定性处理：限制z的取值范围，避免acos函数导数过大
        z_clamped = torch.clamp(z, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        
        theta = torch.acos(z_clamped)  # Polar angle
        phi = torch.atan2(y, x)  # Azimuth angle
        
        # Convert to lat/lon (degrees)
        lat_deg = 90.0 - (theta * 180.0 / PI)
        lon_deg = phi * 180.0 / PI
        
        # 11. Convert lat/lon to ERP coordinates - same as numpy version
        erp_x = (lon_deg + 180.0) / 360.0 * erp_width
        erp_y = (90.0 - lat_deg) / 180.0 * erp_height
        
        # 12. Stack to get final ERP points
        erp_points = torch.stack([erp_x, erp_y], dim=-1)  # (N, 14, 14, 2)
        
        # 对erp_points添加小的L2正则化，提高数值稳定性
        # 这有助于在反向传播时限制梯度大小
        erp_points = erp_points + (erp_points * 1e-6) * 0.01
        
        return erp_points

    

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        # rois: 张量，形状为[sum(num_gt[i]*M), 5]，sum求和的是一个批次所有图像的边界框数量
        """Forward function."""
        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        expand_dims = (-1, self.out_channels * out_size[0] * out_size[1])
        if torch.onnx.is_in_onnx_export():
            # Work around to export mask-rcnn to onnx
            roi_feats = rois[:, :1].clone().detach()
            roi_feats = roi_feats.expand(*expand_dims)
            roi_feats = roi_feats.reshape(-1, self.out_channels, *out_size)
            roi_feats = roi_feats * 0
        else:
            roi_feats = feats[0].new_zeros(
                rois.size(0), self.out_channels, *out_size)
        # TODO: remove this when parrots supports
        if torch.__version__ == 'parrots':
            roi_feats.requires_grad = True

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            
            # Use custom 14x14 sampling for single level feature maps
            
            # Step 1: Generate 14x14 points from RoI
            # 注意：这里输入的是BFOV参数，格式为[N, 5] = [batch_index, lon, lat, fov_u, fov_v]
            # 需要提取BFOV参数部分（去掉batch_index）
            # bfov_roi_to_14x14_erp返回的是ERP图像上的像素坐标 [N, 14, 14, 2]
            roi_points = self.bfov_roi_to_14x14_erp(rois[:, 1:])
            
            # Step 2: Convert to feature map coordinates
            # 将ERP像素坐标转换为特征图坐标（考虑特征图步长）
            points_feat = roi_points / self.featmap_strides[0]
            
            # Step 3: Normalize for grid_sample
            # grid_sample要求坐标在[-1, 1]范围内
            H_feat = feats[0].shape[2]  # 特征图高度
            W_feat = feats[0].shape[3]  # 特征图宽度
            grid = points_feat.clone()
            
            # 水平坐标归一化：将[0, W_feat-1]映射到[-1, 1]
            grid[..., 0] = 2.0 * grid[..., 0] / (W_feat - 1) - 1.0
            
            # 垂直坐标归一化：将[0, H_feat-1]映射到[-1, 1]
            grid[..., 1] = 2.0 * grid[..., 1] / (H_feat - 1) - 1.0
            
            # Step 4: Sample features (memory-efficient version)
            N = rois.size(0)
            C = feats[0].shape[1]  # Number of channels
            sampled_feats = feats[0].new_zeros(N, C, 14, 14)
            
            # Process each ROI individually to save memory
            for i in range(N):
                # Get grid for current ROI (shape: (1, 14, 14, 2))
                roi_grid = grid[i].unsqueeze(0)
                
                # 获取当前ROI对应的批次索引
                batch_idx = rois[i][0].long()
                
                # 提取对应批次的特征图 [1, C, H, W]
                feat = feats[0][batch_idx].unsqueeze(0)
                
                # Sample features for current ROI (shape: (1, C, 14, 14))
                # 使用border填充代替zeros填充，减少边缘区域的梯度异常
                roi_sampled_feats = torch.nn.functional.grid_sample(
                    feat,  # 使用对应批次的特征图
                    roi_grid,
                    mode='bilinear',
                    padding_mode='border',  # 改为border填充，提高边缘区域的梯度稳定性
                    align_corners=False
                )
                
                # Store in result tensor
                sampled_feats[i] = roi_sampled_feats[0]
            
            # Step 5: Pool to 7x7
            roi_feats = torch.nn.functional.avg_pool2d(
                sampled_feats, kernel_size=2, stride=2
            )
            
            return roi_feats
            
            # Original single level roi_layers call (commented out)
            # return self.roi_layers[0](feats[0], rois)

        target_lvls = self.map_roi_levels(rois, num_levels)

        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        for i in range(num_levels):
            mask = target_lvls == i
            if torch.onnx.is_in_onnx_export():
                # For onnx export, use the original roi_layers for compatibility
                mask = mask.float().unsqueeze(-1)
                # select target level rois and reset the rest rois to zero.
                rois_i = rois.clone().detach()
                rois_i *= mask
                mask_exp = mask.expand(*expand_dims).reshape(roi_feats.shape)
                roi_feats_t = self.roi_layers[i](feats[i], rois_i)
                roi_feats_t *= mask_exp
                roi_feats += roi_feats_t
                continue
            inds = mask.nonzero(as_tuple=False).squeeze(1)
            if inds.numel() > 0:
                rois_ = rois[inds]
               
                
                # Using custom 14x14 sampling to get 7x7 features
                # Step 1: Generate 14x14 points from RoI
                # 注意：这里输入的是BFOV参数，格式为[N, 5] = [batch_index, lon, lat, fov_u, fov_v]
                # 需要提取BFOV参数部分（去掉batch_index）
                roi_points = self.bfov_roi_to_14x14_erp(rois_[:, 1:])
                
               
                
                # Step 2: Convert to feature map coordinates
                points_feat = roi_points / self.featmap_strides[i]
                
                # Step 3: Normalize for grid_sample
                H_feat = feats[i].shape[2]
                W_feat = feats[i].shape[3]
                grid = points_feat.clone()
                grid[..., 0] = 2.0 * grid[..., 0] / (W_feat - 1) - 1.0
                grid[..., 1] = 2.0 * grid[..., 1] / (H_feat - 1) - 1.0
                
                # Step 4: Sample features (memory-efficient version)
                N = rois_.size(0)
                C = feats[i].shape[1]  # Number of channels
                sampled_feats = feats[i].new_zeros(N, C, 14, 14)
                
                # Process each ROI individually to save memory
                for j in range(N):
                    # Get grid for current ROI (shape: (1, 14, 14, 2))
                    roi_grid = grid[j].unsqueeze(0)
                    
                    # 获取当前ROI对应的批次索引
                    batch_idx = rois_[j][0].long()
                    
                    # 提取对应批次的特征图 [1, C, H, W]
                    feat = feats[i][batch_idx].unsqueeze(0)
                    
                    # Sample features for current ROI (shape: (1, C, 14, 14))
                    # 使用border填充代替zeros填充，减少边缘区域的梯度异常
                    roi_sampled_feats = torch.nn.functional.grid_sample(
                        feat,  # 使用对应批次的特征图
                        roi_grid,
                        mode='bilinear',
                        padding_mode='border',  # 改为border填充，提高边缘区域的梯度稳定性
                        align_corners=False
                    )
                    
                    # Store in result tensor
                    sampled_feats[j] = roi_sampled_feats[0]
                
                # Step 5: Pool to 7x7
                roi_feats_t = torch.nn.functional.avg_pool2d(
                    sampled_feats, kernel_size=2, stride=2
                )
                
                # Original roi_layers call (commented out)
                # roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t
            else:
                # Sometimes some pyramid levels will not be used for RoI
                # feature extraction and this will cause an incomplete
                # computation graph in one GPU, which is different from those
                # in other GPUs and will cause a hanging error.
                # Therefore, we add it to ensure each feature pyramid is
                # included in the computation graph to avoid runtime bugs.
                roi_feats += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.
        return roi_feats