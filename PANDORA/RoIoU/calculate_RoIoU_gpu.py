import torch
import numpy as np

PI = torch.tensor(np.pi, dtype=torch.float32)

def theta_phi_to_xyz(theta, phi):
    theta = theta.unsqueeze(1) if len(theta.shape) == 1 else theta
    phi = phi.unsqueeze(1) if len(phi.shape) == 1 else phi
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    return torch.cat([x, y, z], dim=1)

def roll_T(n, xyz, gamma=0):
    n = n.squeeze() if len(n.shape) > 2 else n
    xyz = xyz.squeeze() if len(xyz.shape) > 2 else xyz
    gamma = gamma.squeeze() if len(gamma.shape) > 1 else gamma
    gamma = gamma.to(n.device)

    cos_g = torch.cos(gamma)
    sin_g = torch.sin(gamma)
    N = len(gamma) if gamma.numel() > 1 else 1

    # 标准Z轴旋转矩阵
    rot_mat = torch.zeros((N, 3, 3), device=n.device)
    rot_mat[:, 0, 0] = cos_g
    rot_mat[:, 0, 1] = -sin_g
    rot_mat[:, 1, 0] = sin_g
    rot_mat[:, 1, 1] = cos_g
    rot_mat[:, 2, 2] = 1.0

    if len(xyz.shape) == 1:
        xyz = xyz.unsqueeze(0)
    xyz_rot = torch.bmm(xyz.unsqueeze(1), rot_mat).squeeze(1)
    return xyz_rot

def roArrayVector(theta, phi, v, ang):
    theta = theta.unsqueeze(1) if len(theta.shape) == 1 else theta
    phi = phi.unsqueeze(1) if len(phi.shape) == 1 else phi
    ang = ang.unsqueeze(1) if len(ang.shape) == 1 else ang
    c_xyz = theta_phi_to_xyz(theta, phi)
    pp_xyz = roll_T(c_xyz, v, ang)
    return pp_xyz

class SphGPU:
    """修复版GPU球面IoU计算，对齐修复后的CPU版"""
    def __init__(self, device=None):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.PI = PI.to(self.device)
        print(f"[DEBUG] SphGPU初始化，设备: {self.device}")

    def area(self, fov_x, fov_y):
        """修正：标准球面矩形面积公式"""
        fov_x = fov_x.squeeze() if len(fov_x.shape) > 1 else fov_x
        fov_y = fov_y.squeeze() if len(fov_y.shape) > 1 else fov_y
        
        sin_x = torch.sin(fov_x / 2)
        sin_y = torch.sin(fov_y / 2)
        sin_term = sin_x * sin_y
        sin_term = torch.clamp(sin_term, 0, 1)
        area = 4 * torch.arcsin(sin_term)
        area = torch.clamp(area, 1e-10, None)
        
        print(f"[DEBUG] area - 面积范围: min={area.min().item():.8f}, max={area.max().item():.8f}")
        return area

    def getNormal(self, bbox):
        """对齐修复后的CPU版法向量计算"""
        bbox = bbox.to(self.device)
        theta = bbox[:, [0]]
        phi = bbox[:, [1]]
        fov_x = bbox[:, [2]]
        fov_y = bbox[:, [3]]
        angle = bbox[:, [4]]

        # 标准球面坐标系基础向量
        V_lookat = torch.cat([
            torch.cos(phi) * torch.cos(theta),
            torch.cos(phi) * torch.sin(theta),
            torch.sin(phi)
        ], dim=1)
        
        V_right = torch.cat([
            -torch.sin(theta),
            torch.cos(theta),
            torch.zeros_like(theta)
        ], dim=1)
        
        V_up = torch.cat([
            -torch.sin(phi) * torch.cos(theta),
            -torch.sin(phi) * torch.sin(theta),
            torch.cos(phi)
        ], dim=1)

        # 法向量计算
        fov_x_half = fov_x / 2
        fov_y_half = fov_y / 2
        
        N_left = V_lookat * torch.cos(fov_x_half) - V_right * torch.sin(fov_x_half)
        N_right = V_lookat * torch.cos(fov_x_half) + V_right * torch.sin(fov_x_half)
        N_up = V_lookat * torch.cos(fov_y_half) + V_up * torch.sin(fov_y_half)
        N_down = V_lookat * torch.cos(fov_y_half) - V_up * torch.sin(fov_y_half)

        # 旋转+归一化
        N_left = roArrayVector(theta, phi, N_left, angle)
        N_right = roArrayVector(theta, phi, N_right, angle)
        N_up = roArrayVector(theta, phi, N_up, angle)
        N_down = roArrayVector(theta, phi, N_down, angle)
        
        N_left = N_left / (torch.norm(N_left, dim=1, keepdim=True) + 1e-10)
        N_right = N_right / (torch.norm(N_right, dim=1, keepdim=True) + 1e-10)
        N_up = N_up / (torch.norm(N_up, dim=1, keepdim=True) + 1e-10)
        N_down = N_down / (torch.norm(N_down, dim=1, keepdim=True) + 1e-10)

        N = torch.stack([N_left, N_right, N_up, N_down], dim=0)
        
        # 顶点向量
        V1 = torch.cross(N_left, N_up, dim=1)
        V2 = torch.cross(N_down, N_left, dim=1)
        V3 = torch.cross(N_up, N_right, dim=1)
        V4 = torch.cross(N_right, N_down, dim=1)
        V = torch.stack([V1, V2, V3, V4], dim=0)
        V = V / (torch.norm(V, dim=2, keepdim=True) + 1e-10)

        # 边向量
        E1 = torch.stack([N_left, N_up], dim=1)
        E2 = torch.stack([N_down, N_left], dim=1)
        E3 = torch.stack([N_up, N_right], dim=1)
        E4 = torch.stack([N_right, N_down], dim=1)
        E = torch.stack([E1, E2, E3, E4], dim=0)
        
        return N, V, E

    def interArea(self, orders, E):
        """核心修复：交集面积公式，对齐CPU版"""
        if len(E.shape) == 2:
            E = E.unsqueeze(0)
        
        vec1 = E[:, 0, :]
        vec2 = E[:, 1, :]
        
        dot = torch.sum(vec1 * vec2, dim=1)
        dot = torch.clamp(dot, -1, 1)
        angles = torch.acos(dot)

        inter_res = torch.zeros(len(orders), device=self.device)
        loop = 0
        idx = torch.where(orders > 1)[0]
        
        print(f"[DEBUG] interArea - 有效orders数量: {len(idx)}, orders值: {orders[idx[:10]].cpu().numpy()}")
        
        for i in idx:
            j = int(orders[i].item())
            # 修正公式：(j-2)*PI - 角度和（避免负数）
            angle_sum = torch.sum(angles[loop:loop+j])
            inter_res[i] = (j - 2) * self.PI - angle_sum
            loop += j
        
        inter_res = torch.clamp(inter_res, 1e-10, None)
        print(f"[DEBUG] interArea - 交集面积: min={inter_res.min().item():.8f}, max={inter_res.max().item():.8f}")
        return inter_res

    def remove_outer_points(self, dets, gt):
        """优化：减少无效计算，提升GPU性能"""
        N_dets, V_dets, E_dets = self.getNormal(dets)
        N_gt, V_gt, E_gt = self.getNormal(gt)

        N_res = torch.cat([N_dets, N_gt], dim=0)
        V_res = torch.cat([V_dets, V_gt], dim=0)
        E_res = torch.cat([E_dets, E_gt], dim=0)

        # 高效扩展（匹配CPU版）
        N_dets_expand = N_dets.repeat_interleave(4, dim=0)
        N_gt_expand = N_gt.tile((4, 1, 1))

        # 叉乘+归一化
        tmp1 = torch.cross(N_dets_expand, N_gt_expand, dim=-1)
        tmp1 = tmp1 / (torch.norm(tmp1, dim=-1, keepdim=True) + 1e-10)
        tmp2 = torch.cross(N_gt_expand, N_dets_expand, dim=-1)
        tmp2 = tmp2 / (torch.norm(tmp2, dim=-1, keepdim=True) + 1e-10)

        # 堆叠
        V_res = torch.cat([V_res, tmp1, tmp2], dim=0)
        N_concat1 = torch.cat([N_dets_expand, N_gt_expand], dim=-1).view(16, -1, 2, 3)
        N_concat2 = torch.cat([N_gt_expand, N_dets_expand], dim=-1).view(16, -1, 2, 3)
        E_res = torch.cat([E_res, N_concat1, N_concat2], dim=0)

        # 矩阵乘法
        V_perm = V_res.permute(1, 0, 2)
        N_perm = N_res.permute(1, 2, 0)
        res = torch.bmm(V_perm, N_perm)

        # 合理筛选阈值（平衡精度和速度）
        value = torch.all(res >= -0.1, dim=2)
        valid_num = torch.sum(value).item()
        print(f"[DEBUG] remove_outer_points - 有效点: {valid_num}/{value.numel()} ({valid_num/value.numel()*100:.2f}%)")
        
        return value, V_res, E_res

    def computeInter(self, dets, gt):
        dets, gt = dets.to(self.device), gt.to(self.device)
        value, V_res, E_res = self.remove_outer_points(dets, gt)

        ind0, ind1 = torch.where(value)
        print(f"[DEBUG] computeInter - 索引数量: ind0={len(ind0)}, ind1={len(ind1)}")
        
        if len(ind0) == 0:
            return torch.zeros(len(dets), device=self.device)

        ind1 = torch.clamp(ind1, 0, len(E_res)-1)
        ind0 = torch.clamp(ind0, 0, E_res.shape[1]-1)
        E_final = E_res[ind1, ind0, :, :]

        orders = torch.bincount(ind0, minlength=len(dets))
        inter = self.interArea(orders, E_final)
        
        print(f"[DEBUG] computeInter - 最终交集面积: min={inter.min().item():.8f}, max={inter.max().item():.8f}")
        return inter

    def sphIoU(self, dets, gt):
        if dets.numel() == 0 or gt.numel() == 0:
            return torch.zeros((len(dets), len(gt)), device=self.device)
        
        dets, gt = dets.to(self.device), gt.to(self.device)
        d_size, g_size = len(dets), len(gt)
        
        # 批量扩展
        dets_expand = dets.repeat_interleave(g_size, dim=0)
        gt_expand = gt.repeat(d_size, 1)
        
        # 计算面积和交集
        area_A = self.area(dets_expand[:,2], dets_expand[:,3])
        area_B = self.area(gt_expand[:,2], gt_expand[:,3])
        inter = self.computeInter(dets_expand, gt_expand)
        
        # 计算IoU
        union = area_A + area_B - inter
        union = torch.clamp(union, 1e-10, None)
        iou = inter / union
        iou = torch.clamp(iou, 0, 1)
        iou_matrix = iou.reshape(d_size, g_size)
        
        print(f"[DEBUG] sphIoU - 最终IoU: min={iou_matrix.min().item():.8f}, max={iou_matrix.max().item():.8f}, mean={iou_matrix.mean().item():.8f}")
        return iou_matrix