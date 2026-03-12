# 基础配置文件继承，这里只继承了默认运行时配置
_base_ = [
    # '../base/faster_rcnn_r50_fpn_1x_tinycoco.py',  # 注释掉的基础Faster R-CNN配置
    # '../../_base_/datasets/TinyCOCO/TinyCOCO_detection.py',  # 注释掉的TinyCOCO数据集配置
    # '../../../configs/_base_/schedules/schedule_1x.py',  # 注释掉的1x学习率调度配置
    '../../../configs/_base_/default_runtime.py'  # 继承默认运行时配置（日志、检查点等）
]

# 归一化配置，使用GroupNorm，32个组，需要梯度更新
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)  # add

# 调试模式开关，False表示正常训练
debug = True

# 模型设置
num_stages = 2  # 模型阶段数，这里设置为2

# 模型整体配置字典
model = dict(
    type='P2BFoV',  # 模型类型为P2BNet
    pretrained='torchvision://resnet50',  # 使用torchvision预训练的ResNet50权重
    backbone=dict(  # 骨干网络配置
        type='ResNet',  # 骨干网络类型为ResNet
        depth=50,  # ResNet深度为50
        num_stages=4,  # 骨干网络阶段数为4
        out_indices=(0, 1, 2, 3),  # 输出特征图的索引
        frozen_stages=1,  # 冻结第一个阶段的参数
        norm_cfg=dict(type='BN', requires_grad=True),  # 批归一化配置
        norm_eval=True,  # 评估时冻结归一化层
        style='pytorch',  # 网络风格为PyTorch格式
        # conv_cfg=dict(type='Conv', padding_mode='circular')  # 添加卷积配置，启用circular padding
    ),
    
    neck=dict(  # 颈部网络配置（特征金字塔FPN）
        type='FPN',  # 类型为FPN
        in_channels=[256, 512, 1024, 2048],  # 输入特征图通道数（对应ResNet各阶段输出）
        out_channels=256,  # 输出特征图通道数
        start_level=0,  # 开始使用的特征图层级
        add_extra_convs='on_input',  # 在输入特征图上添加额外卷积
        num_outs=4,  # 输出特征图数量（这里为4）
        norm_cfg=norm_cfg,  # 使用前面定义的GroupNorm配置
        # conv_cfg=dict(type='Conv', padding_mode='circular')  # 添加卷积配置，启用circular padding
    ),
    
    roi_head=dict(  # ROI头配置
        type='P2BFoVHead',  # 类型为P2BFoVHead
        num_stages=num_stages,  # 阶段数，使用前面定义的2
        # top_k=7,  # 顶部k值选择
        top_k=7,  # 顶部k值选择
        with_atten=False,  # 不使用注意力机制
        
        # ROI特征提取器配置
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',  # 单个ROI提取器
            # 这里或许要根据实际数据集调整输出大小
            roi_layer=dict(type='RoIAlign', output_size=7),  # 使用RoIAlign，输出大小7x7
            out_channels=256,  # 输出通道数
            featmap_strides=[4, 8, 16, 32]),  # 特征图步长
        
        # 边界框头配置
        bbox_head=dict(
            type='Shared2FCInstanceMILHead',  # 共享两个全连接层的实例MIL头
            num_stages=num_stages,  # 阶段数
            with_loss_pseudo=False,  # 不使用伪标签损失
            in_channels=256,  # 输入通道数
            fc_out_channels=1024,  # 全连接层输出通道数
            roi_feat_size=7,  # ROI特征大小

            num_classes=37,  # 类别数（COCO数据集为37类）
            num_ref_fcs=0,  # 参考全连接层数量
            # conv_cfg=dict(type='Conv', padding_mode='circular'),  # 添加循环padding配置
            
            # 边界框编码器配置
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',  # 基于delta的边界框编码器
                target_means=[0., 0., 0., 0.],  # 目标均值
                target_stds=[0.1, 0.1, 0.2, 0.2]),  # 目标标准差
            
            reg_class_agnostic=True,  # 回归与类别无关
            loss_type='MIL',  # 损失类型为MIL
            
            # 第一个MIL损失配置
            loss_mil1=dict(
                type='MILLoss',  # MIL损失
                binary_ins=False,  # 不是二值实例
                loss_weight=0.25,  # 损失权重
                loss_type='binary_cross_entropy'),  # 使用二元交叉熵
            
            # 第二个MIL损失配置
            loss_mil2=dict(
                type='MILLoss',  # MIL损失
                binary_ins=False,  # 不是二值实例
                loss_weight=0.25,  # 损失权重
                loss_type='gfocal_loss'),)  # 使用gfocal损失
    ),
    
    # 模型训练配置
    train_cfg=dict(
        base_proposal=dict(  # 基础 proposal 配置
            # base_scales=[6, 12, 24, 48, 92, 180],  # 基础尺fov度
            # base_ratios=[1 / 3, 1 / 2, 1 / 1.5, 1.0, 1.5, 2.0, 3.0],  # 基础长宽比
            base_scales=[6, 18, 36, 72, 108, 144],  # 基础尺fov度
            base_ratios=[ 1 / 1.5, 1.0, 1.5],  # 基础长宽比
            shake_ratio=None,  # 不使用抖动比例
            cut_mode='symmetry',  # 裁剪模式为对称
            gen_num_neg=0),  # 生成负样本数量
        
        fine_proposal=dict(  # 精细 proposal 配置
            gen_proposal_mode='fix_gen',  # proposal生成模式
            cut_mode=None,  # 不使用裁剪模式
            shake_ratio=[0.1],  # 抖动比例
            base_ratios=[ 1.2, 1.3, 0.8, 0.7],  # 基础长宽比
            # base_ratios=[1, 1.2,0.8],  # 基础长宽比
            iou_thr=0.,  # IOU阈值
            gen_num_neg=500,  # 生成负样本数量
        ),
        rcnn=None  # RCNN训练配置为None
    ),
    
    # 模型测试配置
    test_cfg=dict(
        # rpn=None,  # RPN测试配置为None
        # rcnn=None,  # RCNN测试配置为None
        rpn=dict(  # RPN测试配置（根据项目需求补充，若无特殊需求可设默认值）
            nms_pre=1000,  # NMS前保留的proposal数量
            max_per_img=1000,  # 每张图最终保留的proposal数量
            nms=dict(type='nms', iou_threshold=0.7)  # RPN的NMS配置
        ),
        rcnn=dict(  # 关键：补充RCNN测试配置，解决None属性问题
            score_thr=0,  # 推理时的置信度阈值（过滤低置信度预测）
            nms=dict(type='nms', iou_threshold=0.1),  # 检测框的NMS配置
            max_per_img=100  # 每张图最终保留的检测框数量
        )


    ))

# 数据集设置每个GPU的样本数
dataset_type = 'CocoFmtDataset'  # 数据集类型为COCO格式数据集
# data_root = '360indoor/'  # 数据根目录
data_root = '../360indoor/'

# 图像归一化配置
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],  # RGB通道均值
    std=[58.395, 57.12, 57.375],  # RGB通道标准差
    to_rgb=True)  # 转换为RGB格式

# 训练数据处理流水线
train_pipeline = [
    dict(type='LoadImageFromFile'),  # 从文件加载图像
    dict(type='LoadAnnotations', with_bbox=True),  # 加载标注，包含边界框
    dict(type='Resize',  # 调整图像大小
        #  img_scale=[(2000, 480), (2000, 576), (2000, 688), (2000, 864), (2000, 1000), (2000, 1200)],  # 多尺度
         img_scale=[(1024, 512)],  # 图像尺度不能动，如果改变的话会直接导致经纬度坐标失真
         multiscale_mode='value',  # 多尺度模式为指定值
         keep_ratio=True),  # 保持长宽比
    # 随机翻转，调试模式关闭
    dict(type='RandomFlip', flip_ratio=0.) if not debug else dict(type='RandomFlip', flip_ratio=0.),
    dict(type='Normalize',** img_norm_cfg),  # 归一化处理
    dict(type='Pad', size_divisor=32),  # 填充图像至32的倍数
    dict(type='DefaultFormatBundle'),  # 默认格式打包
    # 收集所需的键值，gt_points是弧度经纬度坐标，'gt_bfov', 'gt_points',N是每个批次的图像数量，num_gt[i]是第i个图像中标注点的数量，
    # gt_bfov形状是列表，每个元素是一个张量，形状是(num_gt[i],4)，gt_points形状是列表，每个元素是一个张量，形状是(num_gt[i],2)
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels',  'gt_bfov', 'gt_points']),
]

# 测试尺度
test_scale = 512# 改进后的调用方式

# 测试数据处理流水线
test_pipeline = [
    dict(type='LoadImageFromFile'),  # 从文件加载图像
    dict(type='LoadAnnotations', with_bbox=True),  # 加载标注，包含边界框
    dict(
        type='MultiScaleFlipAug',  # 多尺度翻转增强
        img_scale=(1024, test_scale),  # 图像尺度
        flip=False,  # 不翻转
        transforms=[  # 变换列表
            dict(type='Resize', keep_ratio=True),  # 调整大小，保持比例
            dict(type='RandomFlip'),  # 随机翻转
            dict(type='Normalize', **img_norm_cfg),  # 归一化
            dict(type='Pad', size_divisor=32),  # 填充至32的倍数
            dict(type='DefaultFormatBundle'),  # 默认格式打包
            # 收集所需的键值
            dict(type='Collect',
                  keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bfov', 'gt_points','gt_anns_id']),
        ])
]

# 数据加载配置
data = dict(
    samples_per_gpu=1,  # 每个GPU的样本数
    workers_per_gpu=4,  # 每个GPU的工作进程数
    shuffle=False if debug else None,  # 调试模式不打乱数据顺序
    train=dict(  # 训练集配置
        type=dataset_type,  # 数据集类型
        ann_file=data_root + "ann/train_coco.json",  # 标注文件路径
        img_prefix=data_root + 'images/',  # 图像前缀路径
        pipeline=train_pipeline,  # 使用训练流水线
    ),
    val=dict(  # 验证集配置
        samples_per_gpu=1,  # 每个GPU的样本数
        type=dataset_type,  # 数据集类型
        ann_file=data_root + "ann/test_coco.json",  # 标注文件路径
        img_prefix=data_root + 'images/',  # 图像前缀路径
        pipeline=test_pipeline,  # 使用测试流水线
        test_mode=False,  # 不是测试模式
    ),
    test=dict(  # 测试集配置
        type=dataset_type,  # 数据集类型
        ann_file=data_root + "ann/test_coco.json",  # 标注文件路径
        img_prefix=data_root + 'images/',  # 图像前缀路径
        pipeline=test_pipeline))  # 使用测试流水线

# 检查配置，当出现NaN时不停止
check = dict(stop_while_nan=False)  # add by hui

# 优化器配置
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# 优化器配置（梯度裁剪）
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# 学习率配置
lr_config = dict(
    policy='step',  # 学习率策略为step
    warmup='linear',  # 线性预热
    warmup_iters=500,  # 预热迭代次数
    warmup_ratio=0.001,  # 预热学习率比例
    step=[8, 11])  # 在第8和11个epoch调整学习率

# 运行器配置
runner = dict(type='EpochBasedRunner', max_epochs=12)  # 基于epoch的运行器，最大12个epoch
work_dir='work_dir/P2BFoV/'  # 工作目录



#这里需要确定是否需要评估bfov和point，暂时搁置
# 评估配置
evaluation = dict(
    interval=12,  # 评估间隔（每12个epoch）
    # metric='bfov',  # 评估指标为bfov，这里之所以还用bbox是因为保留整体逻辑联通
    # 但是背后的处理层面都改成了球面iou计算逻辑
    metric='bbox',  # 评估指标为边界框
    save_result_file=work_dir + '_' + str(test_scale) + '_latest_result.json',  # 结果保存文件
    do_first_eval=False,  # 不进行首次评估
    do_final_eval=True,  # 进行最终评估
)