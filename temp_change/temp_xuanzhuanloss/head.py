import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from mmdet.core import bbox2result, build_assigner, build_sampler, multi_apply
from sphdet.bbox.box_formator import bbox2roi
from ..builder import HEADS, build_head, build_roi_extractor
from .standard_roi_head import StandardRoIHead
from .cascade_roi_head import CascadeRoIHead
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from .test_mixins import BBoxTestMixin, MaskTestMixin
# from mmdet.core.bbox import bbox_xyxy_to_cxcyw
import copy
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
import os
import sys
from sphdet.iou.sph2pob_efficient import sph2pob_efficient

# 导入MaskToBox类
from mask_to_box import MaskToBox

from PANDORA.PRDA.lib.GPUImageRecorder import GPUImageRecorder
from PANDORA.RoIoU.calculate_RoIoU import Sph
if hasattr(torch, 'pi'):
    PI = torch.pi
else:
    PI = torch.tensor(math.pi)



@HEADS.register_module()
class P2BFoVHead(StandardRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self, bbox_roi_extractor, num_stages, bbox_head, top_k=7, with_atten=None, **kwargs):
        super(P2BFoVHead, self).__init__(bbox_roi_extractor=bbox_roi_extractor, bbox_head=bbox_head, **kwargs)
        self.threshold = 0.3
        self.merge_mode = 'weighted_clsins'
        self.test_mean_iou = False
        self.sum_iou = 0
        self.sum_num = 0
        self.num_stages = num_stages
        self.topk1 = top_k  # 7
        self.topk2 = top_k  # 7

        self.featmap_stride = bbox_roi_extractor.featmap_strides
        self.with_atten = with_atten


    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            # 将生成的候选提案分配给gt
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            # 从分配的gt中采样正负样本
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        # self.cdb = build_head(dict(type='ConvConcreteDB', cfg=None, planes=256))
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

 
    def forward_train(self,
                      stage, x, img_metas,
                        proposal_list_base, proposals_list,
                        proposals_valid_list,
                        neg_proposal_list, neg_weight_list,
                        gt_true_bboxes, gt_labels,
                        dynamic_weight,
                        gt_bboxes_ignore, gt_masks,
                        gt_bfov,
                        gt_bboxes,
                        gt_points,
                        **kwargs
                      ):
        #  mask_left_list, mask_right_list掩码是列表格式长度为N，每个元素是一个张量，形状为[num_gt[i]*M, H, W]，N是批量大小，H是高度，W是宽度
        # gt_points:列表，长度为N，批次数据每个元素对应一张图像gt_points[i]:张量，形状[num_gt_i, 2]，第 i 张图像的标注点，2表示(x, y)坐标
        # proposal_list_base：列表，长度为N,批次提案每个元素对应一张图像
        # proposal_list_base[i]：张量，形状[M*num_gt[i], 4]，第i张图像所有gt点生成的所有提案
        # proposals_valid_list 列表 长度为 N 批次提案有效性，每个元素对应一张图像
        # proposals_valid_list[i]：张量，形状[M*num_gt[i], 1]，提案是否有效的掩码


        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, proposal_list_base, proposals_list, proposals_valid_list,
                                                    neg_proposal_list,
                                                    neg_weight_list,
                                                    gt_points, gt_labels, dynamic_weight,
                                                    img_metas, 
                                                    stage,
                                                    gt_bfov,gt_bboxes)

            losses.update(bbox_results['loss_instance_mil'])
        return losses, bbox_results['pseudo_boxes'], bbox_results['dynamic_weight'],bbox_results
        # bbox_results 是一个字典，包含目标检测的核心结果数据
            # 结构如下：
            # {
            #     # 基础特征与得分
            #     'cls_score':       # torch.Tensor, 形状: [num_gt, M, num_class], 数据类型: float32
            #                        # 分类得分，每个真实框对应M个候选框的num_class个类别概率
            #     'ins_score':       # torch.Tensor, 形状: [num_gt, M, num_class], 数据类型: float32
            #                        # 实例得分，每个真实框对应M个候选框的num_class个实例概率
            #     'bbox_pred':       # torch.Tensor或None, 形状: [num_gt, M, 4] (如果不为None), 数据类型: float32
            #                        # 边界框预测偏移量，这里实际是空的
            #     'bbox_feats':      # torch.Tensor, 形状: [sum(num_gt*M), C, H_roi, W_roi], 数据类型: float32
            #                        # 边界框特征，融合了左右区域的掩码特征
            #     'num_instance':    # int, 形状: [] (标量), 数据类型: int
            #                        # 整个批次的真实实例数量
            #     
            #     # 伪真实框
            #     'pseudo_boxes':    # list[torch.Tensor], 长度为N, 每个元素形状: [num_gt_i, 4], 数据类型: float32
            #                        # 伪真实框列表，用于下一阶段训练
            #     
            #     # 动态权重
            #     'dynamic_weight':  # torch.Tensor, 形状: [sum(num_gt_i),], 数据类型: float32
            #                        # 动态权重，经过维度求和处理
            #     
            #     # 损失信息
            #     'loss_instance_mil':  # dict, 包含以下键值对:
            #         {
            #             'loss_instance_mil':  # torch.Tensor, 数据类型: float32
            #                                   # 实例MIL损失
            #             'bag_acc':            # torch.Tensor, 数据类型: float32
            #                                   # Bag级别的准确率
            #             'neg_loss':           # torch.Tensor或None, 数据类型: float32
            #                                   # 负样本损失（如果存在）
            #             'mean_ious':          # torch.Tensor, 数据类型: float32
            #                                   # 总平均IoU
            #             's':                  # torch.Tensor, 数据类型: float32
            #                                   # 小尺度IoU均值（视场角乘积 < 0.5²）
            #             'm':                  # torch.Tensor, 数据类型: float32
            #                                   # 中尺度IoU均值（0.5² < 视场角乘积 < 1²）
            #             'l':                  # torch.Tensor, 数据类型: float32
            #                                   # 大尺度IoU均值（1² < 视场角乘积 < 2²）
            #             'h':                  # torch.Tensor, 数据类型: float32
            #                                   # 超大尺度IoU均值（视场角乘积 > 2²）
            #         }
            # }



 
    def _bbox_forward_train(self, x, proposal_list_base, proposals_list, proposals_valid_list, neg_proposal_list,
                            neg_weight_list, gt_points,
                            gt_labels,
                            cascade_weight,
                            img_metas, stage,
                            gt_bfov,gt_bboxes):
        """Run forward function and calculate loss for box head in training."""


        




        device = proposals_list[0].device if proposals_list else 'cpu'        
        # 修复：从列表第一个元素获取设备
        # # 在关键位置添加设备检查
        # proposals_list是一个列表（对应一个批次内的各个图像），每个元素是一个张量，这个张量是[num_gt[i]，M, 5]，num_gt[i]表示第i个图像有多少个gt
        # bboxes_list: 边界框列表，每个元素是张量[num_gt[i]*M, 5]，M是每个图像的掩码数量
        #这一部分是主体保留关键需要改动的
        bboxes_list = proposals_list
        
        # 自动检测边界框维度（支持4维或5维BFoV）
        if len(bboxes_list) > 0 and bboxes_list[0].ndim > 0:
            box_version = bboxes_list[0].shape[-1]
        else:
            box_version = 5  # 默认使用5维
        rois = bbox2roi(bboxes_list, box_version=box_version)

        # 🔑 关键修改：调用_bbox_forward_single函数（新功能）
        bbox_results = self._bbox_forward_single(x, rois, gt_points, stage)


        #获取批次中真实目标的总数num_gt
        num_instance = bbox_results['num_instance']
        gt_labels = torch.cat(gt_labels)

        # print(f"bbox_results['cls_score'].shape: {bbox_results['cls_score'].shape}")
        # print(f"torch.cat(proposals_valid_list).shape: {torch.cat(proposals_valid_list).shape}")
        proposals_valid_list = torch.cat(proposals_valid_list).reshape(
            *bbox_results['cls_score'].shape[:2], 1)
        
        # 阶段0的时候先不管负提案
        if neg_proposal_list is not None:
            # neg_proposal_list最终形状为[N, num_neg_gen, 4]
            # neg_weight_list最终形状为[N, num_neg_gen, 1]
            device = neg_proposal_list[0].device if neg_proposal_list else 'cpu'
            
            device = x[0].device

            neg_bboxes_list = neg_proposal_list

            # 自动检测负提案边界框维度（支持4维或5维BFoV）
            if len(neg_bboxes_list) > 0 and neg_bboxes_list[0].ndim > 0:
                neg_box_version = neg_bboxes_list[0].shape[-1]
            else:
                neg_box_version = 5  # 默认使用5维
            neg_rois = bbox2roi(neg_bboxes_list, box_version=neg_box_version)

            neg_bbox_results = self._bbox_forward_single(x, neg_rois, None, stage)

            neg_cls_scores = neg_bbox_results['cls_score']
            neg_weights = torch.cat(neg_weight_list)
            

        else:
            neg_cls_scores = None
            neg_weights = None
        #实际上是没有预测框生成的，因为计算得分的时候没有要求输出reg_box
        reg_box = bbox_results['bbox_pred']

        if reg_box is not None:
            # 自动检测边界框维度（支持4维或5维BFoV）
            proposals_cat = torch.cat(proposals_list)
            box_dim = proposals_cat.shape[-1]
            boxes_pred = self.bbox_head.bbox_coder.decode(proposals_cat.reshape(-1, box_dim),
                                                          reg_box.reshape(-1, box_dim)).reshape(reg_box.shape)
        else:
            boxes_pred = None

        proposals_list_to_merge = proposals_list
        if stage == self.num_stages - 1:
            retrain_weights = None ##TO
        else:
            retrain_weights = None
        # - pseudo_boxes ：列表，长度为N，每个元素形状为 [num_gt[i], 5]，球面中心加水平垂直视场角度加角度，作为下一个阶段的伪gt框
        # - mean_ious ：列表，长度为4，每个元素为一个尺度的球面IoU
        # - filtered_boxes ：列表，长度为N，每个元素形状为 [num_gt[i], k, 5]，球面中心加水平垂直视场角度加角度，保留得分高的k个提案用作后续生成精细提案
        # - filtered_scores ：列表，长度为N，每个元素形状为 [num_gt[i], k]   
        pseudo_boxes, mean_ious, filtered_boxes, filtered_scores, dynamic_weight = self.merge_box(bbox_results,
                                                                                                  proposals_list_to_merge,
                                                                                                  proposals_valid_list,
                                                                                                  gt_labels,
                                                                                                  gt_bboxes,
                                                                                                  gt_bfov,
                                                                                                  img_metas, stage)

        bbox_results.update(pseudo_boxes=pseudo_boxes)
        bbox_results.update(dynamic_weight=dynamic_weight.sum(dim=-1))
        pseudo_boxes = torch.cat(pseudo_boxes)
        #一下参数实际并未使用
        #- boxes_pred ：预测边界框
        # - torch.cat(proposal_list_base) ：基础提案列表
        # - retrain_weights ：重训练权重
        #并没有用到跟gt的交互比来计算损失
        loss_instance_mil = self.bbox_head.loss_mil(stage, bbox_results['cls_score'], bbox_results['ins_score'],
                                                    proposals_valid_list,
                                                    neg_cls_scores, neg_weights,
                                                    boxes_pred, gt_labels,
                                                    torch.cat(proposal_list_base), label_weights=cascade_weight,
                                                    retrain_weights=retrain_weights)
        
        loss_instance_mil.update({"mean_ious": mean_ious[-1]})
        loss_instance_mil.update({"s": mean_ious[0]})
        loss_instance_mil.update({"m": mean_ious[1]})
        loss_instance_mil.update({"l": mean_ious[2]})
        loss_instance_mil.update({"h": mean_ious[3]})

        bbox_results.update(loss_instance_mil=loss_instance_mil)

        return bbox_results
        # bbox_results 是一个字典，包含目标检测的核心结果数据
            # 结构如下：
            # {
            #     # 基础特征与得分
            #     'cls_score':       # torch.Tensor, 形状: [num_gt, M, num_class], 数据类型: float32
            #                        # 分类得分，每个真实框对应M个候选框的num_class个类别概率
            #     'ins_score':       # torch.Tensor, 形状: [num_gt, M, num_class], 数据类型: float32
            #                        # 实例得分，每个真实框对应M个候选框的num_class个实例概率
            #     'bbox_pred':       # torch.Tensor或None, 形状: [num_gt, M, 4] (如果不为None), 数据类型: float32
            #                        # 边界框预测偏移量，这里实际是空的
            #     'bbox_feats':      # torch.Tensor, 形状: [sum(num_gt*M), C, H_roi, W_roi], 数据类型: float32
            #                        # 边界框特征，融合了左右区域的掩码特征
            #     'num_instance':    # int, 形状: [] (标量), 数据类型: int
            #                        # 整个批次的真实实例数量
            #     
            #     # 伪真实框
            #     'pseudo_boxes':    # list[torch.Tensor], 长度为N, 每个元素形状: [num_gt_i, 4], 数据类型: float32
            #                        # 伪真实框列表，用于下一阶段训练
            #     
            #     # 动态权重
            #     'dynamic_weight':  # torch.Tensor, 形状: [sum(num_gt_i),], 数据类型: float32
            #                        # 动态权重，经过维度求和处理
            #     
            #     # 损失信息
            #     'loss_instance_mil':  # dict, 包含以下键值对:
            #         {
            #             'loss_instance_mil':  # torch.Tensor, 数据类型: float32
            #                                   # 实例MIL损失
            #             'bag_acc':            # torch.Tensor, 数据类型: float32
            #                                   # Bag级别的准确率
            #             'neg_loss':           # torch.Tensor或None, 数据类型: float32
            #                                   # 负样本损失（如果存在）
            #             'mean_ious':          # torch.Tensor, 数据类型: float32
            #                                   # 总平均IoU
            #             's':                  # torch.Tensor, 数据类型: float32
            #                                   # 小尺度IoU均值（视场角乘积 < 1²）
            #             'm':                  # torch.Tensor, 数据类型: float32
            #                                   # 中尺度IoU均值（1² < 视场角乘积 < 2²）
            #             'l':                  # torch.Tensor, 数据类型: float32
            #                                   # 大尺度IoU均值（2² < 视场角乘积 < 3²）
            #             'h':                  # torch.Tensor, 数据类型: float32
            #                                   # 超大尺度IoU均值（视场角乘积 > 3²）
            #         }
            # }

        
        




    def _bbox_forward_single(self, x, rois, gt_points, stage):
        """
        不区分左右的Box head forward函数
        
        Args:
            x: 特征图列表
            rois: ROI信息，形状为 [sum(num_gt[i]*M), 5]，[batch_ind, bfov四个参数]
            gt_points: 标注点列表，长度为N
            stage: 训练阶段
            mask_list: 掩码列表，每个元素是形状为 [M_i, H, W] 的张量
            mask_valid_list: 掩码有效性列表，每个元素形状为 [M_i, 1]
            
        Returns:
            bbox_results: 包含分类得分、实例得分、回归框等的字典
        """
        # 提取ROI特征
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        
        # 分类和回归
        cls_score, ins_score, reg_box = self.bbox_head(bbox_feats, stage)

        # positive sample
        if gt_points is not None:
            if isinstance(gt_points, (list, tuple)):
                num_gt = torch.cat(gt_points).shape[0]
            else:
                num_gt = gt_points.shape[0] if gt_points.ndim > 0 else 0
            
            assert num_gt != 0, f'num_gt = 0 {gt_points}'

            cls_score = cls_score.view(num_gt, -1, cls_score.shape[-1])
            ins_score = ins_score.view(num_gt, -1, ins_score.shape[-1])
            if reg_box is not None:
                reg_box = reg_box.view(num_gt, -1, reg_box.shape[-1])

            bbox_results = dict(
                cls_score=cls_score, ins_score=ins_score, bbox_pred=reg_box, bbox_feats=bbox_feats, num_instance=num_gt)
            return bbox_results
        # negative sample
        else:
            bbox_results = dict(
                cls_score=cls_score, ins_score=ins_score, bbox_pred=reg_box, bbox_feats=bbox_feats, num_instance=None)
            return bbox_results


    def merge_box_single(self, cls_score, ins_score, dynamic_weight, gt_bboxes, gt_labels, proposals, img_metas, stage):

        # 直接设置merge_mode，移除冗余条件判断
        merge_mode = 'weighted_clsins_topk'

        # 支持5维BFoV: [theta, phi, fov_x, fov_y, angle]
        box_dim = proposals.shape[-1]
        proposals = proposals.reshape(cls_score.shape[0], cls_score.shape[1], box_dim)
        h, w, c = img_metas['img_shape']
        num_gt, num_gen = proposals.shape[:2]

        if merge_mode == 'weighted_clsins_topk':
            if stage == 0:
                k = self.topk1
            else:
                k = self.topk2
            dynamic_weight_, idx = dynamic_weight.topk(k=k, dim=1)
            weight = dynamic_weight_.unsqueeze(2).repeat([1, 1, box_dim])
            weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)
            
            # Get filtered boxes
            filtered_boxes = proposals[torch.arange(proposals.shape[0]).unsqueeze(1), idx]
            
            # Process coordinates: lon/lat to 3D then weight merge
            # Extract lon/lat (first two dimensions) and fovx/fovy (next two dimensions)
            lon_lat = filtered_boxes[:, :, :2]  # (num_gt, k, 2)
            fovs = filtered_boxes[:, :, 2:4]  # (num_gt, k, 2)
            
            # Convert lon/lat to 3D spherical coordinates
            lon = lon_lat[:, :, 0]  # longitude in radians
            lat = lon_lat[:, :, 1]  # latitude in radians
            
            # Spherical to 3D unit vector conversion
            x = torch.cos(lat) * torch.cos(lon)
            y = torch.cos(lat) * torch.sin(lon)
            z = torch.sin(lat)
            points_3d = torch.stack([x, y, z], dim=-1)  # (num_gt, k, 3)
            
            # Use spherical weighted average algorithm
            weights_spherical = weight[:, :, 0]
            center_3d = self.spherical_weighted_average(points_3d, weights_spherical)
            
            # 3D spherical to lon/lat conversion
            lat_merged = torch.asin(center_3d[:, 2])  # latitude in radians
            lon_merged = torch.atan2(center_3d[:, 1], center_3d[:, 0])  # longitude in radians
            
            # Weighted merge for fovx/fovy
            fovs_weighted = (fovs * weight[:, :, 2:4]).sum(dim=1)  # (num_gt, 2)
            
            # Combine to form final boxes (5维BFoV: [theta, phi, fov_x, fov_y, angle])
            if box_dim == 5:
                # 角度维度取均值（通常都是0）
                angles = filtered_boxes[:, :, 4].mean(dim=1, keepdim=True)  # (num_gt, 1)
                boxes = torch.cat([lon_merged.unsqueeze(1), lat_merged.unsqueeze(1), fovs_weighted, angles], dim=1)
            else:
                boxes = torch.cat([lon_merged.unsqueeze(1), lat_merged.unsqueeze(1), fovs_weighted], dim=1)
            
            # 添加球面坐标约束，确保坐标在有效范围内
            # 只对经度和纬度部分进行约束
            lon_lat = boxes[:, :2]
            constrained_lon_lat = self.constrain_spherical_coords(lon_lat)
            # 保留fovx、fovy和angle部分
            boxes = torch.cat([constrained_lon_lat, boxes[:, 2:]], dim=1)

            filtered_scores = dict(cls_score=cls_score[torch.arange(proposals.shape[0]).unsqueeze(1), idx],
                                ins_score=ins_score[torch.arange(proposals.shape[0]).unsqueeze(1), idx],
                                dynamic_weight=dynamic_weight_)
           

            
            # - boxes ：列表，长度为N，每个元素形状为 [num_gt[i], 5]，球面中心加水平垂直视场角度加角度，作为下一个阶段的伪gt框
            # - filtered_boxes ：列表，长度为N，每个元素形状为 [num_gt[i], k, 5]，球面中心加水平垂直视场角度加角度，保留得分高的k个提案用作后续生成精细提案
            # - filtered_scores ：列表，长度为N，每个元素形状为 [num_gt[i], k] 

            return boxes, filtered_boxes, filtered_scores






    def merge_box(self, bbox_results, proposals_list, proposals_valid_list, gt_labels, gt_bboxes, gt_bfov, img_metas, stage):
        cls_scores = bbox_results['cls_score']
        ins_scores = bbox_results['ins_score']
        num_instances = bbox_results['num_instance']
        # num_gt = len(gt_labels)
    
        # num_gt * num_box * num_class
        if stage < 1:
            cls_scores = cls_scores.softmax(dim=-1)
        else:
            cls_scores = cls_scores.sigmoid()

        #利用有效列表过滤无效提案的得分和权重
        ins_scores = ins_scores.softmax(dim=-2) * proposals_valid_list
        ins_scores = F.normalize(ins_scores, dim=1, p=1)
        cls_scores = cls_scores * proposals_valid_list
        dynamic_weight = (cls_scores * ins_scores)
        dynamic_weight = dynamic_weight[torch.arange(len(cls_scores)), :, gt_labels]
        cls_scores = cls_scores[torch.arange(len(cls_scores)), :, gt_labels]
        ins_scores = ins_scores[torch.arange(len(cls_scores)), :, gt_labels]
        # split batch
        batch_gt = [len(b) for b in gt_bboxes]
        cls_scores = torch.split(cls_scores, batch_gt)
        ins_scores = torch.split(ins_scores, batch_gt)
        gt_labels = torch.split(gt_labels, batch_gt)
        dynamic_weight_list = torch.split(dynamic_weight, batch_gt)
        if not isinstance(proposals_list, list):
            proposals_list = torch.split(proposals_list, batch_gt)
        stage_ = [stage for _ in range(len(cls_scores))]
        # 返回融合后的边界框和动态权重
        # proposal_list：列表，长度为N,批次提案每个元素对应一张图像
        # proposal_list[i]：张量，形状[M*num_gt[i], 5]，第i张图像所有gt点生成的所有提案
        boxes, filtered_boxes, filtered_scores = multi_apply(self.merge_box_single, cls_scores, ins_scores,
                                                             dynamic_weight_list,
                                                             gt_bboxes,
                                                             gt_labels,
                                                             proposals_list,
                                                             img_metas, stage_)

        



        # - boxes ：列表，长度为N，每个元素形状为 [num_gt[i], 5]，球面中心加水平垂直视场角度加角度，作为下一个阶段的伪gt框
        # - filtered_boxes ：列表，长度为N，每个元素形状为 [num_gt[i], k, 5]，球面中心加水平垂直视场角度加角度，保留得分高的k个提案用作后续生成精细提案
        # - filtered_scores ：列表，长度为N，每个元素形状为 [num_gt[i], k]   
        
        #为了计算球面IoU，需要将gt_bfov转换为gt_bboxes的格式，这里的box其实是bfov格式
        #将box拼接起来，形状[sum(num_gt[i]), 5]
        pseudo_boxes = torch.cat(boxes).detach()
        # gt_bfov是长度为N的列表，每个元素形状为[num_gt[i], 4或5]，球面中心加水平垂直视场角度，真实球面视场
        temp_gt_bfov = torch.cat(gt_bfov)
        
        # 如果gt_bfov是4维，添加第五个维度（角度）设置为0，确保与pseudo_boxes维度一致
        if temp_gt_bfov.shape[-1] == 4 and pseudo_boxes.shape[-1] == 5:
            num_gt = temp_gt_bfov.shape[0]
            angles = torch.zeros(num_gt, 1, device=temp_gt_bfov.device)
            temp_gt_bfov = torch.cat([temp_gt_bfov, angles], dim=1)
        
        # iou1的形状是[sum(num_gt[i]),]，每个元素都是0-1之间的浮点数，代表球面IoU
        iou1 = self.calculate_spherical_iou_gpu(pseudo_boxes, temp_gt_bfov)
        



        
        scale = temp_gt_bfov[:, 2] * temp_gt_bfov[:, 3]
        mean_iou_s = iou1[scale < 0.5 ** 2].sum() / (len(iou1[scale < 0.5 ** 2]) + 1e-5)
        mean_iou_m = iou1[(scale > 0.5 ** 2) * (scale < 1 ** 2)].sum() / (len(
            iou1[(scale > 0.5 ** 2) * (scale < 1 ** 2)]) + 1e-5)
        mean_iou_l = iou1[(scale > 1 ** 2) * (scale < 2 ** 2)].sum() / (len(
            iou1[(scale > 1 ** 2) * (scale < 2 ** 2)]) + 1e-5)
        mean_iou_h = iou1[scale > 2 ** 2].sum() / (len(iou1[scale > 2 ** 2]) + 1e-5)

        mean_ious_all = iou1.mean()
        
        mean_ious = [mean_iou_s, mean_iou_m, mean_iou_l, mean_iou_h,mean_ious_all]

        if self.test_mean_iou and stage == 1:
            self.sum_iou += iou1.sum()
            self.sum_num += len(iou1)
            # time.sleep(0.01)  # 这里为了查看输出变化，实际使用不需要sleep
            print('\r', self.sum_iou / self.sum_num, end='', flush=True)
        # batch_gt = [len(b) for b in gt_bboxes]
        #这里的pseudo_boxes是张量
        pseudo_boxes = torch.split(pseudo_boxes, batch_gt)
        #返回的数据信息
        # - pseudo_boxes ：列表，长度为N，每个元素形状为 [num_gt[i], 5]，球面中心加水平垂直视场角度加角度，作为下一个阶段的伪gt框
        # - mean_ious ：列表，长度为4，每个元素为一个尺度的球面IoU
        # - filtered_boxes ：列表，长度为N，每个元素形状为 [num_gt[i], k, 5]，球面中心加水平垂直视场角度加角度，保留得分高的k个提案用作后续生成精细提案
        # - filtered_scores ：列表，长度为N，每个元素形状为 [num_gt[i], k]   
        return list(pseudo_boxes), mean_ious, list(filtered_boxes), list(filtered_scores), dynamic_weight.detach()




    def constrain_spherical_coords(self, centers):
        """
        球面坐标（经纬度，弧度制）边界约束工具函数
        输入：centers - 形状为 [N, 2] 的张量，[:,0]为经度λ（∈[-π, π]），[:,1]为纬度φ（∈[-π/2, π/2]）
        输出：约束后的球面坐标张量，形状与输入一致
        """
        constrained_centers = centers.clone()
        λ = constrained_centers[:, 0]
        φ = constrained_centers[:, 1]
        
        # 1. 经度约束：超出[-π, π]时循环映射（±2π修正）
        λ = torch.remainder(λ + PI, 2 * PI) - PI
        constrained_centers[:, 0] = λ
        
        # 2. 纬度约束：超出[-π/2, π/2]时，越过极点并同步修正经度
        # 情况1：纬度 > π/2（越过北极点）
        phi_over_upper = φ > PI / 2
        new_phi_upper = PI - φ[phi_over_upper]
        new_lambda_upper = λ[phi_over_upper] + PI  # 经度±π（等价，此处用+π后统一约束）
        # 情况2：纬度 < -π/2（越过南极点）
        phi_over_lower = φ < -PI / 2
        new_phi_lower = -PI - φ[phi_over_lower]
        new_lambda_lower = λ[phi_over_lower] + PI  # 经度±π（等价，此处用+π后统一约束）
        
        # 赋值修正后的纬度
        constrained_centers[phi_over_upper, 1] = new_phi_upper
        constrained_centers[phi_over_lower, 1] = new_phi_lower
        # 赋值修正后的经度（并再次约束经度范围）
        constrained_centers[phi_over_upper, 0] = new_lambda_upper
        constrained_centers[phi_over_lower, 0] = new_lambda_lower
        constrained_centers[:, 0] = torch.remainder(constrained_centers[:, 0] + PI, 2 * PI) - PI
        
        return constrained_centers


    def spherical_weighted_average(self, points, weights, max_iter=100, tol=1e-8):
        """
        三维笛卡尔加权平均算法
        
        Args:
            points: 3D点，形状为 [batch_size, num_points, 3]
            weights: 权重，形状为 [batch_size, num_points]
            max_iter: 最大迭代次数（兼容旧接口，不再使用）
            tol: 收敛容忍度（兼容旧接口，不再使用）
            
        Returns:
            weighted_points: 笛卡尔加权平均点，形状为 [batch_size, 3]
        """
        batch_size, num_points, _ = points.shape
        
        # 直接进行三维笛卡尔加权平均
        # 将权重扩展为 [batch_size, num_points, 1] 以便与 points 相乘
        weights_expanded = weights.unsqueeze(2)
        
        # 计算加权和
        weighted_sum = torch.sum(weights_expanded * points, dim=1)  # [batch_size, 3]
        
        # 计算权重总和（用于归一化）
        weights_sum = weights.sum(dim=1, keepdim=True)  # [batch_size, 1]
        
        # 进行归一化，避免除以零
        weighted_average = weighted_sum / (weights_sum + tol)  # [batch_size, 3]
        
        return weighted_average



    def simple_test(self,
                    stage,
                    x,
                    proposal_list,
                    proposals_valid_list,
                    gt_bboxes,
                    gt_labels,
                    gt_anns_id,
                    img_metas,
                    gt_bfov,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        # pseudo_boxes ：列表，长度为N，每个元素形状为 [num_gt[i], 4]，球面中心加水平垂直视场角度，作为下一个阶段的伪gt框
        # - det_bboxes ： [N, num_gt[i], 6] 6个值是： [bfov参数(4) + 置信度权重(1) + 标注ID(1)]
        # - det_labels ： [N, num_gt[i]] → 检测框对应的标签
        det_bboxes, det_labels, pseudo_bboxes = self.simple_test_bboxes(
            x, img_metas, proposal_list, proposals_valid_list, gt_bboxes, gt_labels, gt_anns_id, stage, gt_bfov, self.test_cfg,
            rescale=rescale)
        
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]
        # pseudo_bboxes = [i[:, :4] for i in det_bboxes]
        #bbox_results ：嵌套列表，外层长度为N，中层长度为num_classes，最内层每个元素是一个张量，形状为[num_gt[i], 5]，5个值是： [x1, y1, x2, y2, score]
       
        
        return bbox_results, pseudo_bboxes
    def simple_test_bboxes(self,
                            x,
                            img_metas,
                            proposals,
                            proposals_valid_list,
                            gt_bboxes,
                            gt_labels,
                            gt_anns_id,
                            stage,
                            gt_bfov,
                            rcnn_test_cfg,
                            rescale=False):
        # get origin input shape to support onnx dynamic input shape
        # img_shape 形状是[N,3],3shape是[height, width, channel]，
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        #proposals长度为N列表，每个元素是一个张量，形状为[num_gt[i]*M, 4]，格式是bfov



        device = proposals[0].device if proposals else 'cpu'


        # 将掩码转换为边界框列表
        bboxes_list = proposals
        
        # 自动检测边界框维度（支持4维或5维BFoV）
        if len(bboxes_list) > 0 and bboxes_list[0].ndim > 0:
            box_version = bboxes_list[0].shape[-1]
        else:
            box_version = 5  # 默认使用5维
        
        # 获取ROI特征
        rois = bbox2roi(bboxes_list, box_version=box_version)

        bbox_results = self._bbox_forward_single(x, rois, None, stage)
        
        # 在测试阶段，我们需要将cls_score和ins_score重塑为 [num_gt, M, num_classes] 形状
        # 首先计算每个图像的gt数量
        num_gts = [len(bbox) for bbox in gt_bboxes]
        num_proposals = [len(bfov) for bfov in proposals]
        
        # 重塑cls_score和ins_score
        cls_score = bbox_results['cls_score'].split(num_proposals, dim=0)
        ins_score = bbox_results['ins_score'].split(num_proposals, dim=0)
        
        # 计算每个gt对应的提案数量
        M = num_proposals[0] // num_gts[0]
        
        # 重塑为 [num_gt, M, num_classes] 形状
        cls_score = [score.view(num_gt, M, -1) for score, num_gt in zip(cls_score, num_gts)]
        ins_score = [score.view(num_gt, M, -1) for score, num_gt in zip(ins_score, num_gts)]
        
        # 合并为张量
        bbox_results['cls_score'] = torch.cat(cls_score, dim=0)
        bbox_results['ins_score'] = torch.cat(ins_score, dim=0)
        
        # 设置num_instance
        bbox_results['num_instance'] = sum(num_gts)
        
        # 在测试阶段，我们使用mask_valid_list而不是proposals_valid_list
        # 因为mask_valid_list与生成的掩码一一对应
        valid_list = torch.cat(proposals_valid_list).reshape(
            *bbox_results['cls_score'].shape[:2], 1)
        # def merge_box(self, bbox_results, proposals_list, proposals_valid_list, gt_labels, gt_bboxes, gt_bfov, img_metas, stage):
        #返回的数据信息
        # - pseudo_boxes ：列表，长度为N，每个元素形状为 [num_gt[i], 5]，球面中心加水平垂直视场角度加角度，作为下一个阶段的伪gt框
        # - mean_ious ：列表，长度为4，每个元素为一个尺度的球面IoU
        # - filtered_boxes ：列表，长度为N，每个元素形状为 [num_gt[i], k, 5]，球面中心加水平垂直视场角度加角度，保留得分高的k个提案用作后续生成精细提案
        # - filtered_scores ：列表，长度为N，每个元素形状为 [num_gt[i], k]   
        pseudo_boxes, mean_ious, filtered_boxes, filtered_scores, dynamic_weight = self.merge_box(bbox_results,
                                                                                                    proposals,
                                                                                                    valid_list,
                                                                                                    torch.cat(gt_labels),
                                                                                                    gt_bboxes,
                                                                                                    gt_bfov,
                                                                                                    img_metas, stage)
        # pseudo_boxes ：列表，长度为N，每个元素形状为 [num_gt[i], 4]，球面中心加水平垂直视场角度，作为下一个阶段的伪gt框
        pseudo_boxes_out = copy.deepcopy(pseudo_boxes)
       
        
        # - det_bboxes ： [N, num_gt[i], 6] 6个值是： [bfov参数(4) + 置信度权重(1) + 标注ID(1)]
        # - det_labels ： [N, num_gt[i]] → 检测框对应的标签
        det_bboxes, det_labels = self.pseudobox_to_result(pseudo_boxes, gt_labels, dynamic_weight, gt_anns_id,
                                                            scale_factors, rescale)
        
        
        return det_bboxes, det_labels, pseudo_boxes_out  

    def pseudobox_to_result(self, pseudo_boxes, gt_labels, dynamic_weight, gt_anns_id, scale_factors, rescale):
        # pseudo_boxes ：列表，长度为N，每个元素形状为 [num_gt[i], 5]，球面中心加水平垂直视场角度加角度，
        # gt_labels ：列表，长度为N，每个元素形状为 [num_gt[i]]，每个元素是gt的类别标签
        det_bboxes = []
        det_labels = []
        batch_gt = [len(b) for b in gt_labels]
        dynamic_weight = torch.split(dynamic_weight, batch_gt)
        for i in range(len(pseudo_boxes)):
            boxes = pseudo_boxes[i]
            labels = gt_labels[i]
            #这里他妈的该死的缩放，把结果全弄乱了
            # if rescale and boxes.shape[0] > 0:
            #     scale_factor = boxes.new_tensor(scale_factors[i]).unsqueeze(0).repeat(
            #         1,
            #         boxes.size(-1) // 4)
            #     boxes /= scale_factor

            # 去掉角度维度，只保留前4个维度 [theta, phi, fov_x, fov_y]
            boxes = boxes[:, :4]
            
            boxes = torch.cat([boxes, dynamic_weight[i].sum(dim=1, keepdim=True)], dim=1)
            gt_anns_id_single = gt_anns_id[i]
            boxes = torch.cat([boxes, gt_anns_id_single.unsqueeze(1)], dim=1)
            det_bboxes.append(boxes)
            det_labels.append(labels)
                # - det_bboxes ： [N, num_gt[i], 6] 6个值是： [bfov参数(4) + 置信度权重(1) + 标注ID(1)]
                # - det_labels ： [N, num_gt[i]] → 检测框对应的标签
        
        return det_bboxes, det_labels

    def calculate_spherical_iou_gpu(self, pseudo_boxes, gt_bfov, device=None, sph_calculator=None):
        """
        计算球面矩形 IoU，使用 sph2pob_efficient_iou 方法
        支持4维或5维的球面矩形：
        - 4维：[theta, phi, fov_x, fov_y]
        - 5维：[theta, phi, fov_x, fov_y, angle]
        
        Args:
            pseudo_boxes: 伪框，格式：列表，长度为N，每个元素形状为 [M_i, 4或5]
                
                - 格式：[theta, phi, fov_x, fov_y] 或 [theta, phi, fov_x, fov_y, angle]
                - 类型：torch.Tensor 或 numpy.ndarray
            gt_bfov: 真实框，格式：列表，长度为N，每个元素形状为 [num_gt[i], 4或5]
                - 格式：[theta, phi, fov_x, fov_y] 或 [theta, phi, fov_x, fov_y, angle]
                - 类型：torch.Tensor 或 numpy.ndarray
            device: 输出设备，默认与 pseudo_boxes 相同
            sph_calculator: 未使用，为保持接口一致而保留
            
        Returns:
            iou1: 球面 IoU 结果，形状：(N,)，与 bbox_overlaps(..., is_aligned=True) 保持一致
                - 类型：与 pseudo_boxes 相同（torch.Tensor 或 numpy.ndarray）
                - 设备：与 device 参数或 pseudo_boxes 相同
        """
        from sphdet.iou.sph_iou_api import sph2pob_efficient_iou
        
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
        
        # 3. 验证输入形状是否匹配（对齐计算）
        assert pseudo_boxes_tensor.shape == gt_bfov_tensor.shape, \
            f"pseudo_boxes shape {pseudo_boxes_tensor.shape} must match gt_bfov shape {gt_bfov_tensor.shape} for aligned calculation"
        assert len(pseudo_boxes_tensor.shape) == 2, \
            f"Input must be 2D tensors, got {len(pseudo_boxes_tensor.shape)}D tensors"
        
        # 4. 将弧度转换为度
        pseudo_boxes_deg = torch.rad2deg(pseudo_boxes_tensor)
        gt_bfov_deg = torch.rad2deg(gt_bfov_tensor)
        
        # 5. 使用 sph2pob_efficient_iou 计算 IoU（对齐计算）
        # N = pseudo_boxes_tensor.shape[0]
        # print(f"开始计算球面IoU（对齐），共{N}个框对")
        iou_tensor = sph2pob_efficient_iou(pseudo_boxes_deg, gt_bfov_deg, is_aligned=True)
        # print("球面IoU（对齐）计算完成")
        
        # 6. 转换回原始类型和设备
        if not is_tensor:
            iou_tensor = iou_tensor.cpu().numpy()
        
        
        return iou_tensor


   