import copy
import math
from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from mmdet.core.bbox import bbox_xyxy_to_cxcywh
from mmdet.core import bbox_cxcywh_to_xyxy
import torch
import numpy as np
#下面这个球bboxiou的需要重写成bfoviou,目前暂时放在这里
from mmdet.core.bbox.iou_calculators import bbox_overlaps
#球面坐标按照-pi-pi，-pi/2-pi/2

from ..builder import build_head
from sphdet.iou.sph_iou_api import sph2pob_efficient_iou

import sys
import os

if hasattr(torch, 'pi'):
    PI = torch.pi
else:
    PI = torch.tensor(math.pi)




def constrain_spherical_coords(centers):
        """
        球面坐标（经纬度，弧度制）边界约束工具函数
        支持两种输入格式：
        1. 形状为 [N, 2] 的张量：[:,0]为经度λ（∈[-π, π]），[:,1]为纬度φ（∈[-π/2, π/2]）
        2. 形状为 [..., 4] 的张量：[...,0]为经度，[...,1]为纬度，[...,2:4]为视场角
        
        输出：约束后的球面坐标张量，形状与输入一致
        """
        constrained_centers = centers.clone()
        
        # 获取经度和纬度（支持任意前导维度）
        λ = constrained_centers[..., 0]
        φ = constrained_centers[..., 1]
        
        # 1. 经度约束：超出[-π, π]时循环映射（±2π修正）
        λ = torch.remainder(λ + PI, 2 * PI) - PI
        constrained_centers[..., 0] = λ
        
        # 2. 纬度约束：超出[-π/2, π/2]时，越过极点并同步修正经度
        # 情况1：纬度 > π/2（越过北极点）
        phi_over_upper = φ > PI / 2
        if phi_over_upper.any():
            new_phi_upper = PI - φ[phi_over_upper]
            new_lambda_upper = λ[phi_over_upper] + PI  # 经度+π
            # 约束新经度
            new_lambda_upper = torch.remainder(new_lambda_upper + PI, 2 * PI) - PI
            # 赋值修正后的坐标
            φ[phi_over_upper] = new_phi_upper
            λ[phi_over_upper] = new_lambda_upper
        
        # 情况2：纬度 < -π/2（越过南极点）
        phi_over_lower = φ < -PI / 2
        if phi_over_lower.any():
            new_phi_lower = -PI - φ[phi_over_lower]
            new_lambda_lower = λ[phi_over_lower] + PI  # 经度+π
            # 约束新经度
            new_lambda_lower = torch.remainder(new_lambda_lower + PI, 2 * PI) - PI
            # 赋值修正后的坐标
            φ[phi_over_lower] = new_phi_lower
            λ[phi_over_lower] = new_lambda_lower
        
        # 更新约束后的坐标
        constrained_centers[..., 0] = λ
        constrained_centers[..., 1] = φ
        
        return constrained_centers


def calculate_spherical_iou(pseudo_boxes, gt_bfov, device=None, sph_calculator=None):
        """
        计算球面矩形 IoU，替代原有的平面 IoU 计算
        支持无角度的球面矩形：[theta, phi, fov_x, fov_y]
        与 bbox_overlaps 保持一致的输出形状：[N, M]
        
        Args:
            pseudo_boxes: 伪框，格式：
                - 形状：(N, 4)
                - 格式：[theta, phi, fov_x, fov_y]
                - 类型：torch.Tensor 或 numpy.ndarray
            gt_bfov: 真实框，格式：
                - 形状：(M, 4)
                - 格式：[theta, phi, fov_x, fov_y]
                - 类型：torch.Tensor 或 numpy.ndarray
            device: 输出设备，默认与 pseudo_boxes 相同
            sph_calculator: Sph 实例，默认自动创建
            
        Returns:
            iou: 球面 IoU 结果，形状：(N, M)，与 bbox_overlaps 保持一致
                - 类型：与 pseudo_boxes 相同（torch.Tensor 或 numpy.ndarray）
                - 设备：与 device 参数或 pseudo_boxes 相同
        """
        # 1. 创建 Sph 实例（如果未提供）
        if sph_calculator is None:
            sph_calculator = Sph()
        
        # 2. 处理 gt_bfov 输入格式
        if isinstance(gt_bfov, list):
            # 列表 → 合并为张量
            gt_bfov = torch.cat(gt_bfov)
        
        # 3. 处理设备和数据类型
        is_tensor = isinstance(pseudo_boxes, torch.Tensor)
        if is_tensor:
            if device is None:
                device = pseudo_boxes.device
            # 转换为 numpy 数组处理
            pseudo_boxes_np = pseudo_boxes.detach().cpu().numpy()
            gt_bfov_np = gt_bfov.detach().cpu().numpy()
        else:
            # 已为 numpy 数组
            pseudo_boxes_np = pseudo_boxes
            gt_bfov_np = gt_bfov
            if device is not None:
                print("Warning: device parameter is ignored for numpy input")
        
        # 4. 获取输入形状
        assert len(pseudo_boxes_np.shape) == 2, f"pseudo_boxes must be 2D, got {len(pseudo_boxes_np.shape)}D"
        assert len(gt_bfov_np.shape) == 2, f"gt_bfov must be 2D, got {len(gt_bfov_np.shape)}D"
        N, _ = pseudo_boxes_np.shape
        M, _ = gt_bfov_np.shape
        
        # 5. 添加角度维度：角度为0，形状变为 (N, 5) 和 (M, 5)
        # dets 和 gt 格式：[theta, phi, fov_x, fov_y, angle]
        pseudo_boxes_with_angle = np.concatenate([
            pseudo_boxes_np, 
            np.zeros((N, 1))  # 添加角度，全部为0
        ], axis=1)  # (N, 5)
        
        gt_bfov_with_angle = np.concatenate([
            gt_bfov_np, 
            np.zeros((M, 1))  # 添加角度，全部为0
        ], axis=1)  # (M, 5)
        
        # 6. 计算球面 IoU，输出形状 (N, M)，与 bbox_overlaps 保持一致
        iou_np = np.zeros((N, M))
        for i in range(N):
            det = pseudo_boxes_with_angle[i].reshape(1, 5)
            for j in range(M):
                gt = gt_bfov_with_angle[j].reshape(1, 5)
                iou_value = sph_calculator.sphIoU(det, gt)[0, 0]
                iou_np[i, j] = iou_value
        
        # 7. 转换回原始类型和设备
        if is_tensor:
            iou_np = torch.from_numpy(iou_np).to(device)
               
        return iou_np


def gen_proposals_from_cfg(gt_points, proposal_cfg, img_meta):
    # gt_points的形状应该是列表，长度为N，每个元素是一个张量，形状为[num_gt[i],2],下面可能存在逻辑错误，或者维度不匹配的问题
    if isinstance(gt_points, torch.Tensor):
        # 将形状为 [N, num_gt, 2] 的张量转换为长度为N的列表
        gt_points = [gt_points[i] for i in range(gt_points.shape[0])]
    base_scales = proposal_cfg['base_scales']
    base_ratios = proposal_cfg['base_ratios']
    shake_ratio = proposal_cfg['shake_ratio']

          
    #基础提案列表和有效提案列表
    base_proposal_list = []
    proposals_valid_list = []
    #对中心点进行遍历
    for i in range(len(gt_points)):


        img_h, img_w, _ = img_meta[i]['img_shape']
        # 基础尺度，改成基础FoV,弧度制
        base = math.pi/180
        base_proposals = []
        #生成基础尺度a*尺度比b个初始提案
        for scale in base_scales:
            scale = scale * base
            for ratio in base_ratios:
                #创建一个跟gtpoint[i]（第i张图像中所有的gt点张量）相同工作环境的张量，
                # 之后多次循环使得单个gt点生成a*b个张量，每个张量形状为[1,2]
                # 确保scale * ratio和scale / ratio都小于pi，如果大于pi的话要赋值为pi
                pi_limit = math.pi - 1e-6
                fov_x = min(scale * ratio, pi_limit)
                fov_y = min(scale / ratio, pi_limit)
                base_proposals.append(gt_points[i].new_tensor([[fov_x, fov_y]]))
        #将a*b个张量拼接成一个张量，形状为[a*b,2]，位置在base_proposal的维度0上
        base_proposals = torch.cat(base_proposals)
        #在维度0上重复张量len(gt_points[i])次在维度1重复1次，这里重复一次的原因是，提案的数量和种类的区别，
        #前者重复多次代表的是同一张图像上有多个点同一个点有多个提案，后者代表的不是数量而是类别和形状，就跟三棵树四棵树，不同于三棵树四个栗子，
        #目的是给同一张图片上的所有gt点都匹配上全尺度和全比例的提案，此时的张量形状为（gt_num_i*a*b,2）
        base_proposals = base_proposals.repeat((len(gt_points[i]), 1))
        #gt_points[i]形状从[gt_num_i,2]变成[gt_num_i*a*b,2]
        base_center = torch.repeat_interleave(gt_points[i], len(base_scales) * len(base_ratios), dim=0)

        #这一部分的是对中心进行抖动的工作，原框架为了防止抖动超越边界，后面需要对其进行裁剪
        #换到球面上的话需要考虑中心跨越水平的正负180度的界限度经线或者跨越90度纬线
        if shake_ratio is not None:
            # 新增：形状校验
            assert base_center.ndim == 2 and base_center.shape[1] == 2, \
                f"base_center需为[N, 2]形状，当前形状为{base_center.shape}"
            assert base_proposals.ndim == 2 and base_proposals.shape[1] == 2, \
                f"base_proposals需为[N, 2]形状，当前形状为{base_proposals.shape}"
            assert base_center.shape[0] == base_proposals.shape[0], \
                f"base_center批次维度{base_center.shape[0]}与base_proposals批次维度{base_proposals.shape[0]}不匹配"
            
            #这里需要改动的地方是，沿着球面抖动的应该是一个跟中心坐标和抖动弧度有关的函数
            # 抖动方式1：沿着与该点处经线垂直的过球心的大圆弧进行左右抖动，
            
            lambda0 = base_center[:, 0]  # 原始经度（弧度，直接提取）
            phi0 = base_center[:, 1]    # 原始纬度（弧度，直接提取）
            alpha = shake_ratio * base_proposals[:, 0] # 水平抖动角度（弧度，直接提取）

            # 计算新纬度φ2（左右抖动纬度一致，弧度制）
            sin_phi2 = torch.sin(phi0) * torch.cos(alpha)
            # 限制sin_phi2范围在[-1,1]，避免浮点误差导致arcsin返回NaN
            sin_phi2 = torch.clamp(sin_phi2, -1.0 + 1e-6, 1.0 - 1e-6)
            phi2 = torch.arcsin(sin_phi2)  # 新纬度（弧度，无需转换）

            # 2. 计算新经度λ2（使用atan2解决取值范围问题，修正球面偏移逻辑）
            # 避免分母为0：给cos(phi0)添加微小偏移
            cos_phi0 = torch.cos(phi0) + 1e-8
            # 球面经度偏移量：使用atan2获取[-π, π]范围的偏移，解决arctan截断问题
            delta_lambda = torch.atan2(torch.sin(alpha), cos_phi0 * torch.cos(alpha))
            lambda2_l = lambda0 + delta_lambda  # 左抖动新经度
            lambda2_r = lambda0 - delta_lambda  # 右抖动新经度

            # 直接赋值（全程弧度制，无需转角度）
            base_center_l = torch.zeros_like(base_center)
            base_center_r = torch.zeros_like(base_center)
             # 下面的是抖动之后的坐标张量
            base_center_l[:, 0] = lambda2_l  # 左抖动新经度（弧度）
            base_center_l[:, 1] = phi2       # 左抖动新纬度（弧度）
            base_center_r[:, 0] = lambda2_r  # 右抖动新经度（弧度）
            base_center_r[:, 1] = phi2       # 右抖动新纬度（弧度） 

            # 抖动方式2：沿着经线上下抖动；
            base_center_t = torch.zeros_like(base_center)
            base_center_d = torch.zeros_like(base_center)
            base_center_t[:, 0] = base_center[:, 0]  # 上抖动的经度与原始一致
            base_center_d[:, 0] = base_center[:, 0]  # 下抖动的经度与原始一致
            base_center_t[:, 1] = base_center[:, 1] - shake_ratio * base_proposals[:, 1]
            base_center_d[:, 1] = base_center[:, 1] + shake_ratio * base_proposals[:, 1]
            
            # 抖动之后需要约束坐标在[-pi, pi]和[-pi/2, pi/2]之间，超出部分要跨越边界进行反向
            # 往左抖动超界需要加上2*pi，往右超界需要减去2*pi
            # 往上抖动超界pi/2-超界角度，往下超界-pi/2+超界角度
            base_center_l = constrain_spherical_coords(base_center_l)
            base_center_r = constrain_spherical_coords(base_center_r)
            base_center_t = constrain_spherical_coords(base_center_t)
            base_center_d = constrain_spherical_coords(base_center_d)

            #为各个各个中心点分配提案
            #将形状为[N, 2]的提案增加一个维度[N,1, 2]然后将中间的1对应的维度重复5次，对应四个抖动和原始中心
            base_proposals = base_proposals.unsqueeze(1).repeat((1, 5, 1))#[N, 5, 2]
            # 中心合并形状是[N, 5, 2]
            base_center = torch.stack([base_center, base_center_l, base_center_r, base_center_t, base_center_d], dim=1)
        #中心与提案在最后一个维度拼接起来原本各自张量都是[N, 5, 4]现在变成[N, 5, 4]，[x1, y1, w1, h1]
        base_proposals = torch.cat([base_center, base_proposals], dim=-1)
        #将所有的proposal展平成一维列表，里面的每一个proposal都是[x1, y1, w1, h1]
        #没有抖动的话[N, 4]，抖动的话[N, 5, 4]
        base_proposals = base_proposals.reshape(-1, 4)
        proposals_valid = base_proposals.new_full(
            (*base_proposals.shape[:-1], 1), 1, dtype=torch.long).reshape(-1, 1)
        # 对坐标进行有效性筛选，但是我们前边已经验证了不筛选，这里直接append
        proposals_valid_list.append(proposals_valid)
        
        base_proposal_list.append(base_proposals)
        # 输出返回的张量的形状和意义
        # base_proposal_list: 列表，每个元素是一个张量，形状为[N, 5, 4]，N是批量大小，5是每个中心抖动后的5个提案，4是提案信息[x1, y1, w1, h1]
        # proposals_valid_list: 列表，每个元素是一个张量，形状为[N, 5, 1]，N是批量大小，5是每个中心对应5个提案，1是是否有效（1表示有效，0表示无效）
    return base_proposal_list, proposals_valid_list,

def gen_negative_proposals(gt_points, proposal_cfg, aug_generate_proposals, img_meta):
    # aug_generate_proposals,列表长度为N,每个元素为[num_gt[i] * k * (1+4*S), 4]
    #gt_points,列表长度为N,每个元素为[gt_num_i, 2]
    num_neg_gen = proposal_cfg['gen_num_neg']
    if num_neg_gen == 0:
        return None, None
    neg_proposal_list = []
    neg_weight_list = []
    for i in range(len(gt_points)):
        # aug_generate_proposals上一阶段生成的正样本提案
        # pos_box是张量 [num_gt[i] * k * (1+4*S), 4]
        pos_box = aug_generate_proposals[i]
        #生成中心坐标水平方向在-pi到pi之间，垂直方向在-pi/2到pi/2之间的负提案
        #宽高分别在1-180之间
        # 1. 生成中心坐标（弧度）：水平(-π, π)，垂直(-π/2, π/2)
        # center_x是张量 [num_neg_gen]
        # center_y是张量 [num_neg_gen]  
        center_x = -PI + torch.rand(num_neg_gen) * (2 * PI)  # 水平：-π ~ π 弧度
        center_y = -PI/2 + torch.rand(num_neg_gen) * PI      # 垂直：-π/2 ~ π/2 弧度

        # 2. 生成宽高（先取1°~180°的角度值，再转换为弧度）
        # 步骤1：生成1~180度的随机角度（浮点数）
        # angle_w是张量 [num_neg_gen]
        # angle_h是张量 [num_neg_gen]
        angle_w = 1 + torch.rand(num_neg_gen) * 159  # 角度：1° ~ 180°
        angle_h = 1 + torch.rand(num_neg_gen) * 159

        # 步骤2：角度转弧度（核心修正：角度→弧度，π弧度=180°）
        w = angle_w * (PI / 180)  # 宽：1°→π/180 弧度，180°→π 弧度
        h = angle_h * (PI / 180)  # 高：同上

        # 3. 拼接负提案并转移到目标设备
        # neg_bboxes是张量 [num_neg_gen, 4]
        neg_bboxes = torch.stack([center_x, center_y, w, h], dim=1).to(gt_points[0].device)

        iou = calculate_spherical_iou_gpu(neg_bboxes, pos_box)

        # [num_neg_gen, num_pos] （布尔值张量）类似有效性掩码
        neg_weight = ((iou < 0.3).sum(dim=1) == iou.shape[1])
        # neg_bboxes是张量 [num_neg_gen, 4],
        # neg_proposal_list最终形状为[N, num_neg_gen, 4]
        neg_proposal_list.append(neg_bboxes)
        # neg_weight是张量 [num_neg_gen, 1]
        # neg_weight_list最终形状为[N, num_neg_gen, 1]
        neg_weight_list.append(neg_weight)

        # neg_proposal_list最终形状为[N, num_neg_gen, 4]
        # neg_weight_list最终形状为[N, num_neg_gen, 1]
    return neg_proposal_list, neg_weight_list

# def fine_proposals_from_cfg(pseudo_boxes, fine_proposal_cfg, img_meta, stage):
#     gen_mode = fine_proposal_cfg['gen_proposal_mode']
#     # cut_mode = fine_proposal_cfg['cut_mode']
#     cut_mode = None
#     if isinstance(fine_proposal_cfg['base_ratios'], tuple):
#         base_ratios = fine_proposal_cfg['base_ratios'][stage - 1]
#         shake_ratio = fine_proposal_cfg['shake_ratio'][stage - 1]
#     else:
#         base_ratios = fine_proposal_cfg['base_ratios']
#         shake_ratio = fine_proposal_cfg['shake_ratio']
#     if gen_mode == 'fix_gen':
#         proposal_list = []
#         proposals_valid_list = []
#         # 遍历批次图像，每个图像生成提案
#         for i in range(len(img_meta)):
#             pps = []
#             #上一个阶段的加权伪框，中心点加水平垂直方向的视场
#             #  'pseudo_boxes': list[torch.Tensor], 长度为N, 每个元素形状: [num_gt[i], 4], 数据类型: float32， 伪真实框列表，用于下一阶段训练
#             #base_boxes: 形状为 [num_gt[i], 4]
#             base_boxes = pseudo_boxes[i]
            
#             for ratio_w in base_ratios:
#                 for ratio_h in base_ratios:
#                     #base_boxes_: 形状为 [num_gt[i], 4]
#                     base_boxes_ = base_boxes.clone() 
#                     base_boxes_[:, 2] *= ratio_w
#                     base_boxes_[:, 3] *= ratio_h
#                     # 确保视场角不超过pi
#                     pi_limit = math.pi - 1e-6
#                     pi_tensor = torch.tensor(pi_limit, device=base_boxes_.device)
#                     base_boxes_[:, 2] = torch.clamp(base_boxes_[:, 2], max=pi_tensor)
#                     base_boxes_[:, 3] = torch.clamp(base_boxes_[:, 3], max=pi_tensor)
#                     # 第一次循环 (ratio_w=0.75, ratio_h=0.75)
#                         # base_boxes_: [num_gt_i, 4]  # 调整视场角后
#                         # pps.append(...) → pps: [ [num_gt_i, 1, 4] ]  # 列表长度=1

#                     # # 第二次循环 (ratio_w=0.75, ratio_h=1.0)
#                         # pps.append(...) → pps: [ [num_gt_i, 1, 4], [num_gt_i, 1, 4] ]  # 列表长度=2
#                     # pps是一个列表，每个元素是一个张量，形状为 [num_gt_i, 1, 4]
#                     pps.append(base_boxes_.unsqueeze(1))
            
#             # pps是一个列表长度为K*K，每个元素是一个张量，形状为 [num_gt_i, 1, 4]
#             # 按照维度1拼接起来，形状为 [num_gt_i, K*K, 4]
#             pps_old = torch.cat(pps, dim=1)
            
#             if shake_ratio is not None:
#                 # [num_gt[i], K*K, 1, 4]
#                 pps_new = []
#                 # pps_old_reshape: [num_gt_i, K, 1, 4]  # 添加了一个新维度
#                 # pps_new: [ [num_gt_i, K, 1, 4] ]  # 包含原始基础提案
#                 # 最终的pps_new: [
#                 #     [num_gt_i, K, 1, 4],  # 原始提案
#                 #     [num_gt_i, K, 1, 4],  # 抖动提案1
#                 #     [num_gt_i, K, 1, 4],  # 抖动提案2
#                 #     共有1+4*中心抖动次数个元素
#                 pps_new.append(pps_old.reshape(*pps_old.shape[0:2], -1, 4))#这里执行之后pps_old形状为[num_gt_i, K, 4]
                
#                 for ratio in shake_ratio:
#                     # pps_old: 形状为[num_gt_[i], K, 4]，num_gt_i是中心点数量，K是ratio_w*ratio_h的组合数，4是提案信息[x1, y1, w1, h1]
#                     pps=pps_old
#                     # 下面是抖动获取另外四个中心点的坐标
#                     # pps_center: 形状为[num_gt_[i], K, 2]
#                     # pps_wh: 形状为[num_gt_[i], K, 2]
#                     pps_center = pps[:, :, :2]
#                     pps_wh = pps[:, :, 2:4]
                    
                    
#                     # 正确定义变量形状（参考gen_proposals_from_cfg函数）
#                     pps_x_l = pps_center.clone()  # 形状：[num_gt_[i], K, 2]
#                     pps_x_r = pps_center.clone()  # 形状：[num_gt_[i], K, 2]
#                     pps_y_t = pps_center.clone()  # 形状：[num_gt_[i], K, 2]
#                     pps_y_d = pps_center.clone()  # 形状：[num_gt_[i], K, 2]
                    
#                     lambda0 = pps_center[:, :, 0]
#                     phi0 = pps_center[:, :,1]
#                     # 水平抖动角度（弧度，直接提取）
#                     alpha = ratio * pps_wh[:, :, 0]
#                     sin_phi2 = torch.sin(phi0) * torch.cos(alpha)
#                     # 限制sin_phi2范围在[-1,1]，避免浮点误差导致arcsin返回NaN
#                     sin_phi2 = torch.clamp(sin_phi2, -1.0 + 1e-6, 1.0 - 1e-6)
#                     phi2 = torch.arcsin(sin_phi2)  # 新纬度（弧度，无需转换）

#                     # 计算新经度λ2
#                     cos_phi0 = torch.cos(phi0) + 1e-8
#                     delta_lambda = torch.atan2(torch.sin(alpha), cos_phi0 * torch.cos(alpha))
#                     lambda2_l = lambda0 + delta_lambda  # 左抖动新经度
#                     lambda2_r = lambda0 - delta_lambda  # 右抖动新经度

#                     # 直接赋值（参考gen_proposals_from_cfg函数）
#                     # 下面的是抖动之后的坐标张量
#                     pps_x_l[:, :, 0] = lambda2_l  # 左抖动新经度（弧度）
#                     pps_x_l[:, :, 1] = phi2       # 左抖动新纬度（弧度）
#                     pps_x_r[:, :, 0] = lambda2_r  # 右抖动新经度（弧度）
#                     pps_x_r[:, :, 1] = phi2       # 右抖动新纬度（弧度） 

#                     # 抖动方式2：沿着经线上下抖动；
#                     pps_y_t[:, :, 0] = pps_center[:, :, 0]  # 上抖动的经度与原始一致
#                     pps_y_d[:, :, 0] = pps_center[:, :, 0]  # 下抖动的经度与原始一致
#                     pps_y_t[:, :, 1] = pps_center[:, :, 1] + ratio * pps_wh[:, :, 1]  # 上抖动
#                     pps_y_d[:, :, 1] = pps_center[:, :, 1] - ratio * pps_wh[:, :, 1]  # 下抖动

                    
#                     # 抖动之后需要约束坐标在[-pi, pi]和[-pi/2, pi/2]之间，超出部分要跨越边界进行反向
#                     # 往左抖动超界需要加上2*pi，往右超界需要减去2*pi
#                     # 往上抖动超界pi/2-超界角度，往下超界-pi/2+超界角度
#                     # 抖动之后需要约束坐标在[-pi, pi]和[-pi/2, pi/2]之间
#                     # pps_x_l 形状：[num_gt_[i], K, 2]
#                     # pps_x_r 形状：[num_gt_[i], K, 2]
#                     # pps_y_t 形状：[num_gt_[i], K, 2]
#                     # pps_y_d 形状：[num_gt_[i], K, 2]
#                     pps_x_l = constrain_spherical_coords(pps_x_l)
#                     pps_x_r = constrain_spherical_coords(pps_x_r)
#                     pps_y_t = constrain_spherical_coords(pps_y_t)
#                     pps_y_d = constrain_spherical_coords(pps_y_d)

#                     pps_center_l = pps_x_l  # 左抖动中心坐标
#                     pps_center_r = pps_x_r  # 右抖动中心坐标
#                     pps_center_t = pps_y_t  # 上抖动中心坐标
#                     pps_center_d = pps_y_d  # 下抖动中心坐标
#                     # pps_center: [num_gt_i, K, 4, 2]
#                     pps_center = torch.stack([pps_center_l, pps_center_r, pps_center_t, pps_center_d], dim=2)

#                     # pps_wh原始形状为[num_gt_i, K, 2]
#                     #pps_wh.unsqueeze(2)，[num_gt_i, K, 1, 2]
#                     #expand(*pps_center.shape[:3], 2)，最终形状[num_gt_i, K, 4, 2]
#                     pps_wh = pps_wh.unsqueeze(2).expand(*pps_center.shape[:3], 2)

#                     # pps_wh: [num_gt_i, K, 4, 4]
#                     pps = torch.cat([pps_center, pps_wh], dim=-1)

#                     # [num_gt_i, K*4, 4]
#                     pps = pps.reshape(pps.shape[0], -1, 4)
#                         # # 第518行重塑后
#                         # pps: [num_gt_i, K*4, 4]  # 抖动方向与宽高比组合合并
#                         # # 第523行再次重塑
#                         # reshaped_pps: [num_gt_i, K, 4, 4]  # 恢复抖动方向维度
#                         # # 添加到列表后
#                         # pps_new: [ [num_gt_i, K, 1, 4], [num_gt_i, K, 4, 4] ]  # 包含原始提案和抖动提案，这是一个ratio抖动的结束
#                             #最终的pps_new: [
#                             #     [num_gt_i, K, 1, 4],  # 原始提案
#                             #     [num_gt_i, K, 4, 4]  # 抖动ratio1
#                             #     [num_gt_i, K, 4, 4]  # 抖动ratio2
#                             #     共有1+4*中心抖动次数个元素
#                     pps_new.append(pps.reshape(*pps_old.shape[0:2], -1, 4))
#                 #S是中心抖动比例，每个比例有四个抖动方向
#                 # 拼接抖动结果，形状为[num_gt[i],k,1+4*S,4]
#                 pps_new = torch.cat(pps_new, dim=2)
#             else:
#                 pps_new = pps_old
#             h, w, _ = img_meta[i]['img_shape']
#             #设置有效性掩码全是1
#             # pps_new: [num_gt[i],k,1+4*S,4]
#             # (*pps_new.shape[0:3], 1) → 基于 pps_new 的前3个维度 [num_gt[i], k, 1+4*S] ，
#             # 添加一个大小为1的新维度，最终形状为 [num_gt[i], k, 1+4*S, 1]
#             #将有效性掩码添加到列表中单个元素形状为[num_gt[i], k, 1+4*S, 1]
#             proposals_valid_list.append(pps_new.new_full((*pps_new.shape[0:3], 1), 1, dtype=torch.long).reshape(-1, 1))
            
            
#             # 重塑后形状： [num_gt[i] * k * (1+4*S), 4]
#             #然后添加进列表
#             # 阶段0生成的伪框加权结果只保留了一个
#             num_gt_i = pps_new.shape[0]
#             k = pps_new.shape[1]
#             s = pps_new.shape[2]
#             total_proposals = num_gt_i * k * s


#             proposal_list.append(pps_new.reshape(-1, 4))
#             # 最终输出结果
#             # proposal_list,列表长度为N,每个元素为[num_gt[i] * k * (1+4*S), 4]
#             # proposals_valid_list: 列表长度为N,每个元素为[num_gt[i], k, 1+4*S, 1]
#             proposals_valid_list
#     return proposal_list, proposals_valid_list
def fine_proposals_from_cfg(pseudo_boxes, fine_proposal_cfg, img_meta, stage):
    gen_mode = fine_proposal_cfg['gen_proposal_mode']
    cut_mode = None
    
    if isinstance(fine_proposal_cfg['base_ratios'], tuple):
        base_ratios = fine_proposal_cfg['base_ratios'][stage - 1]
        shake_ratio = fine_proposal_cfg['shake_ratio'][stage - 1]
    else:
        base_ratios = fine_proposal_cfg['base_ratios']
        shake_ratio = fine_proposal_cfg['shake_ratio']
    
    if gen_mode == 'fix_gen':
        proposal_list = []
        proposals_valid_list = []
        
        for i in range(len(img_meta)):
            pps = []
            base_boxes = pseudo_boxes[i]  # [num_gt[i], 4]
            
            # 生成不同宽高比的基础提案
            for ratio_w in base_ratios:
                for ratio_h in base_ratios:
                    base_boxes_ = base_boxes.clone()
                    base_boxes_[:, 2] *= ratio_w  # 水平FOV
                    base_boxes_[:, 3] *= ratio_h  # 垂直FOV
                    
                    # 限制视场角不超过pi
                    pi_limit = math.pi - 1e-6
                    pi_tensor = torch.tensor(pi_limit, device=base_boxes_.device)
                    base_boxes_[:, 2] = torch.clamp(base_boxes_[:, 2], max=pi_tensor)
                    base_boxes_[:, 3] = torch.clamp(base_boxes_[:, 3], max=pi_tensor)
                    
                    pps.append(base_boxes_.unsqueeze(1))
            
            # pps_old: [num_gt[i], K, 4], K = len(base_ratios)^2
            pps_old = torch.cat(pps, dim=1)
            
            if shake_ratio is not None:
                pps_new = []
                # 原始提案 [num_gt[i], K, 1, 4]
                pps_new.append(pps_old.reshape(*pps_old.shape[0:2], -1, 4))
                
                for ratio in shake_ratio:
                    # pps_old: [num_gt[i], K, 4]
                    pps = pps_old
                    pps_center = pps[:, :, :2]  # [num_gt[i], K, 2] (lon, lat)
                    pps_wh = pps[:, :, 2:4]     # [num_gt[i], K, 2] (width, height)
                    
                    # 提取经纬度
                    lambda0 = pps_center[:, :, 0]  # 经度
                    phi0 = pps_center[:, :, 1]     # 纬度
                    
                    # 计算抖动步长（弧度）
                    delta_lon = ratio * pps_wh[:, :, 0]  # 水平方向步长
                    delta_lat = ratio * pps_wh[:, :, 1]  # 垂直方向步长
                    
                    # 初始化四个方向的中心坐标 [num_gt[i], K, 2]
                    pps_x_l = pps_center.clone()  # 左
                    pps_x_r = pps_center.clone()  # 右
                    pps_y_t = pps_center.clone()  # 上
                    pps_y_d = pps_center.clone()  # 下
                    
                    # === 修改后的抖动逻辑：直接经纬度加减 ===
                    # 水平抖动（左/右）：只变经度，纬度保持不变
                    pps_x_l[:, :, 0] = lambda0 - delta_lon  # 左减
                    pps_x_l[:, :, 1] = phi0                 # 纬度不变
                    
                    pps_x_r[:, :, 0] = lambda0 + delta_lon  # 右加
                    pps_x_r[:, :, 1] = phi0                 # 纬度不变
                    
                    # 垂直抖动（上/下）：只变纬度，经度保持不变
                    pps_y_t[:, :, 0] = lambda0              # 经度不变
                    pps_y_t[:, :, 1] = phi0 + delta_lat     # 上加
                    
                    pps_y_d[:, :, 0] = lambda0              # 经度不变
                    pps_y_d[:, :, 1] = phi0 - delta_lat     # 下减
                    
                    # 约束坐标在有效范围内
                    pps_x_l = constrain_spherical_coords(pps_x_l)
                    pps_x_r = constrain_spherical_coords(pps_x_r)
                    pps_y_t = constrain_spherical_coords(pps_y_t)
                    pps_y_d = constrain_spherical_coords(pps_y_d)
                    
                    # 堆叠四个方向: [num_gt[i], K, 4, 2]
                    pps_center_shaked = torch.stack([pps_x_l, pps_x_r, pps_y_t, pps_y_d], dim=2)
                    
                    # 扩展宽高以匹配: [num_gt[i], K, 4, 2]
                    pps_wh_expanded = pps_wh.unsqueeze(2).expand(*pps_center_shaked.shape[:3], 2)
                    
                    # 拼接中心和宽高: [num_gt[i], K, 4, 4]
                    pps_shaked = torch.cat([pps_center_shaked, pps_wh_expanded], dim=-1)
                    
                    # 重塑为 [num_gt[i], K*4, 4] 然后恢复维度为 [num_gt[i], K, 4, 4]
                    pps_shaked = pps_shaked.reshape(pps_shaked.shape[0], -1, 4)
                    pps_shaked = pps_shaked.reshape(*pps_old.shape[0:2], -1, 4)
                    
                    pps_new.append(pps_shaked)
                
                # 拼接所有抖动结果: [num_gt[i], K, 1+4*S, 4]
                pps_new = torch.cat(pps_new, dim=2)
            else:
                pps_new = pps_old
            
            h, w, _ = img_meta[i]['img_shape']
            
            # 生成有效性掩码
            proposals_valid_list.append(
                pps_new.new_full((*pps_new.shape[0:3], 1), 1, dtype=torch.long).reshape(-1, 1)
            )
            
            # 重塑为列表: [num_gt[i] * K * (1+4*S), 4]
            proposal_list.append(pps_new.reshape(-1, 4))
        
        return proposal_list, proposals_valid_list


def calculate_spherical_iou_gpu(pseudo_boxes, gt_bfov, device=None, sph_calculator=None):
    """
    计算球面矩形 IoU，使用 sph2pob_efficient_iou 方法
    支持无角度的球面矩形：[theta, phi, fov_x, fov_y]
    与 bbox_overlaps 保持一致的输出形状：[N, M]
    
    Args:
        pseudo_boxes: 伪框，格式：
            - 形状：(N, 4)
            - 格式：[theta, phi, fov_x, fov_y]
            - 类型：torch.Tensor 或 numpy.ndarray
        gt_bfov: 真实框，格式：
            - 形状：(M, 4)
            - 格式：[theta, phi, fov_x, fov_y]
            - 类型：torch.Tensor 或 numpy.ndarray
        device: 输出设备，默认与 pseudo_boxes 相同
        sph_calculator: 未使用，为保持接口一致而保留
        
    Returns:
        iou: 球面 IoU 结果，形状：(N, M)，与 bbox_overlaps 保持一致
            - 类型：与 pseudo_boxes 相同（torch.Tensor 或 numpy.ndarray）
            - 设备：与 device 参数或 pseudo_boxes 相同
    """
    # 1. 处理 gt_bfov 输入格式
    if isinstance(gt_bfov, list):
        # 列表 → 合并为张量
        gt_bfov = torch.cat(gt_bfov)
    
    # 2. 处理设备和数据类型
    is_tensor = isinstance(pseudo_boxes, torch.Tensor)
    if is_tensor:
        if device is None:
            device = pseudo_boxes.device
        # 确保输入是张量
        if not isinstance(gt_bfov, torch.Tensor):
            gt_bfov = torch.tensor(gt_bfov, device=device)
        pseudo_boxes_tensor = pseudo_boxes.to(device)
        gt_bfov_tensor = gt_bfov.to(device)
    else:
        # 转换为张量处理
        if device is None:
            device = torch.device('cpu')
        pseudo_boxes_tensor = torch.tensor(pseudo_boxes, device=device)
        gt_bfov_tensor = torch.tensor(gt_bfov, device=device)
    
    # 3. 获取输入形状
    assert len(pseudo_boxes_tensor.shape) == 2, f"pseudo_boxes must be 2D, got {len(pseudo_boxes_tensor.shape)}D"
    assert len(gt_bfov_tensor.shape) == 2, f"gt_bfov must be 2D, got {len(gt_bfov_tensor.shape)}D"
    N, _ = pseudo_boxes_tensor.shape
    M, _ = gt_bfov_tensor.shape
    
    # 4. 将弧度转换为度
    pseudo_boxes_deg = torch.rad2deg(pseudo_boxes_tensor)
    gt_bfov_deg = torch.rad2deg(gt_bfov_tensor)
    
    # 5. 使用 sph2pob_efficient_iou 计算 IoU
    # 由于 sph2pob_efficient_iou 支持 is_aligned=False，直接计算 (N, M) 的 IoU 矩阵
    # print(f"开始计算球面IoU (GPU)，共{N}个检测框和{M}个GT框")
    iou_tensor = sph2pob_efficient_iou(pseudo_boxes_deg, gt_bfov_deg, is_aligned=False)
    # print("球面IoU计算完成")
    
    # 6. 转换回原始类型
    if not is_tensor:
        iou_tensor = iou_tensor.cpu().numpy()
       
    return iou_tensor
















@DETECTORS.register_module()
class P2BFoV(TwoStageDetector):
    def __init__(self,
                 backbone,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 bbox_head=None,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        #上面的是P2BFoV类的构造函数，用于初始化模型，后买你的参数是需要接收的参数
        # 下面的super调用父类 TwoStageDetector 的构造函数来初始化继承的组件
        super(P2BFoV, self).__init__(
            backbone=backbone,
            neck=neck,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.num_stages = roi_head.num_stages
        if bbox_head is not None:
            self.with_bbox_head = True
            self.bbox_head = build_head(bbox_head)
    ###这里可能需要改一下train.py因为添加了新的数据输入
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bfov,
                      gt_points,
                      gt_true_bboxes=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):


        # 开始特征提取
        x = self.extract_feat(img)  

        #这里需要注意下全景ERP特征提取的时候和普通平面不一样，要么两边padding或者改变卷积核
        #下面两个是初始提案和精细提案的配置信息
        base_proposal_cfg = self.train_cfg.get('base_proposal',
                                               self.test_cfg.rpn)
        fine_proposal_cfg = self.train_cfg.get('fine_proposal',
                                               self.test_cfg.rpn)
        losses = dict()#定义损失字典函数
        
        # if len(img_metas) > 0:
            # h,w = img_metas[0]['pad_shape'][0],img_metas[0]['pad_shape'][1]
            # meta = img_metas[0]
            # print(f"ori_shape: {meta['ori_shape']}")
            # print(f"img_shape: {meta['img_shape']}") 
            # print(f"pad_shape: {meta['pad_shape']}")
            # print(f"scale_factor: {meta['scale_factor']}")
            # print(f"h_w: {meta['pad_shape'][0]}/{meta['pad_shape'][1]}")

        for stage in range(self.num_stages):
            if stage == 0:
                #generate_proposals长度为N列表，每个元素是一个张量，形状为[num_gt[i]*M, 4]
                #num_gt[i]是每张图的真实标签数量，M是每个标签的扩展候选提案数量，4是提案信息[x1, y1, w1, h1]
                # proposals_valid_list长度为N列表，每个元素是一个张量，形状为[num_gt[i]*M, 1]
                #num_gt[i]是每张图的真实标签数量，1是是否有效（1表示有效，0表示无效）

                generate_proposals, proposals_valid_list = gen_proposals_from_cfg(gt_points, base_proposal_cfg,
                                                                                  img_meta=img_metas)
                # 生成一个与真实标签数量相同的张量存储初始权重为1
                dynamic_weight = torch.cat(gt_labels).new_ones(len(torch.cat(gt_labels)))
                # 阶段0暂时不生成负提案和负提案的权重
                neg_proposal_list, neg_weight_list = None, None
                # 下一阶段的伪提案
                #generate_proposals长度为N列表，每个元素是一个张量，形状为[num_gt[i]*M, 4]
                pseudo_boxes = generate_proposals




                    
            else:
                #  'pseudo_boxes': list[torch.Tensor], 长度为N, 每个元素形状: [num_gt[i], 4], 数据类型: float32， 伪真实框列表，用于下一阶段训练
                # proposal_list,列表长度为N,每个元素为[num_gt[i] * k * (1+4*S), 4]
                # proposals_valid_list: 列表长度为N,每个元素为[num_gt[i], k, 1+4*S, 1]
                generate_proposals, proposals_valid_list = fine_proposals_from_cfg(pseudo_boxes, fine_proposal_cfg,
                                                                                   img_meta=img_metas,
                                                                                   stage=stage)
                # 生成负提案及其权重
                # generate_proposals,列表长度为N,每个元素为[num_gt[i] * k * (1+4*S), 4]
                # neg_proposal_list最终形状为[N, num_neg_gen, 4]
                # neg_weight_list最终形状为[N, num_neg_gen, 1]
                neg_proposal_list, neg_weight_list = gen_negative_proposals(gt_points, fine_proposal_cfg,
                                                                            generate_proposals,
                                                                            img_meta=img_metas)
            
            roi_losses, pseudo_boxes, dynamic_weight = self.roi_head.forward_train(stage, x, img_metas,
                                                                                   pseudo_boxes,
                                                                                   generate_proposals,
                                                                                   proposals_valid_list,
                                                                                   neg_proposal_list, neg_weight_list,
                                                                                   gt_true_bboxes, gt_labels,
                                                                                   dynamic_weight,
                                                                                   gt_bboxes_ignore, gt_masks,
                                                                                   gt_bfov,
                                                                                   gt_bboxes,
                                                                                   gt_points,
                                                                                   **kwargs
                                                                                    )
            if stage == 0:
                pseudo_boxes_out = pseudo_boxes
                dynamic_weight_out = dynamic_weight
            for key, value in roi_losses.items():
                losses[f'stage{stage}_{key}'] = value
        return losses
        
    def simple_test(self, img, img_metas, gt_bboxes, gt_anns_id,  gt_labels,gt_bfov,gt_points,
                    gt_bboxes_ignore=None, proposals=None, rescale=False):
        """Test without augmentation."""
        base_proposal_cfg = self.train_cfg.get('base_proposal',
                                               self.test_cfg.rpn)
        fine_proposal_cfg = self.train_cfg.get('fine_proposal',
                                               self.test_cfg.rpn)
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        for stage in range(self.num_stages):
            # gt_points形状为[N, num_gt[i], 2]
            # gt_points = [bbox_xyxy_to_cxcywh(b)[:, :2] for b in gt_bboxes]
            if stage == 0:
                generate_proposals, proposals_valid_list = gen_proposals_from_cfg(gt_points, base_proposal_cfg,
                                                                                  img_meta=img_metas)
            else:
                generate_proposals, proposals_valid_list = fine_proposals_from_cfg(pseudo_boxes, fine_proposal_cfg,
                                                                                   img_meta=img_metas, stage=stage)
            # pseudo_bboxes = [i[:, :4] for i in det_bboxes]
            #bbox_results ：嵌套列表，外层长度为N，中层长度为num_classes，最内层每个元素是一个张量，形状为[num_gt[i], 5]，5个值是： [x1, y1, w, h, score]
            test_result, pseudo_boxes = self.roi_head.simple_test(stage,
                                                                  x, generate_proposals, proposals_valid_list,
                                                                  gt_bboxes,
                                                                  gt_labels,
                                                                  gt_anns_id,
                                                                  img_metas,
                                                                  gt_bfov,
                                                                  proposals=None,
                                                                  rescale=rescale)


        #bbox_results ：嵌套列表，外层长度为N，中层长度为num_classes，最内层每个元素是一个张量，形状为[num_gt[i], 5]，5个值是： [x1, y1, w, h, score]

        return test_result