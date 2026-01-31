import os
import sys
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import torch

# 设置配置文件路径
config_file = 'configs2/COCO/P2BFoV/P2BFoV_r50_fpn_1x_coco_ms.py'

# 加载配置
cfg = Config.fromfile(config_file)

# 修改配置，仅运行1个iteration
cfg.runner = dict(type='IterBasedRunner', max_iters=1)  # 只运行1个iteration
cfg.total_epochs = 1
cfg.log_config.interval = 1
cfg.checkpoint_config.interval = 1
cfg.gpu_ids = [0]  # 手动添加gpu_ids属性，使用第0个GPU

# 构建数据集
print("构建数据集...")
dataset = [build_dataset(cfg.data.train)]

# 构建模型
print("构建模型...")
model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model.init_weights()

# 添加CLASSES属性
model.CLASSES = dataset[0].CLASSES

# 开始训练（只运行1个iteration）
print("开始测试训练...")
train_detector(model, dataset, cfg, distributed=False, validate=False)

print("✅ 训练测试成功！程序能正常跑通！")