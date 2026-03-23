
# Prerequisites
install environment following
```shell script
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab


# 1. 先安装CUDA 11.1工具包（如果还没安装）
conda install -c nvidia cudatoolkit=11.1
# 2. 使用pip安装匹配的PyTorch（注意URL中的cu111对应CUDA 11.1）
# 1. 彻底卸载所有PyTorch相关包
pip uninstall -y torch torchvision torchaudio

# 2. 安装完整的PyTorch 1.9.0+cu111
pip install --no-cache-dir torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# 3. 安装mmcv-full 1.4.0（对应CUDA 11.1）
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# install mmdetection

pip uninstall pycocotools   # sometimes need to source deactivate before, for 
pip install -r requirements/build.txt
pip install -v -e . --user  # or try "python setup.py develop" if get still got pycocotools error
conda install scikit-image  # or pip install scikit-image
# #下面这个库可能会不匹配根据情况调整如果报错的话
# pip install yapf==0.40.0


#开始训练
工作空间
cd /mnt/c/mengchao/shared/wsl/P2BFoV/P2BNet-main/TOV_mmdetection$ 

# 正式训练
python tools/train.py configs2/COCO/P2BFoV/P2BFoV_r50_fpn_1x_coco_ms.py --work-dir work_dir/P2BFoV/ --gpu-ids 0


查看谁在用gpu
 for g in 0 1 2 3; do echo -e "\n===== GPU $g 占用信息 ====="; nvidia-smi -i $g | awk '/^|    [0-9]+ +N\/A +N\/A/ && /python/ {pid=$5; mem=$12; cmd="ps -o user= -p " pid; cmd | getline user; close(cmd); print "用户: " user "\nPID: " pid "\n显存占用: " mem "\n---"}'; done

# 最终环境
#兼容mmcv1.6以上，目前装的是2.3.2
conda create -n pointobb python=3.7 -y
conda activate pointobb
# install pytorch
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html


# install mmcv
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# install mmdetection
pip uninstall pycocotools   # sometimes need to source deactivate before, for 
pip install -r requirements/build.txt
pip install -v -e . --user  # or try "python setup.py develop" if get still got pycocotools error
chmod +x tools/dist_train.sh
```

```shell script
conda install scikit-image  # or pip install scikit-image
```
# 正式训练
```shell script

<!-- 单卡训练 -->
python tools/train.py configs2/COCO/P2BFoV/P2BFoV_r50_fpn_1x_coco_ms.py --work-dir work_dir/P2BFoV/ --gpu-ids 0

<!-- 两张卡分布式训练 -->
bash tools/dist_train.sh configs2/COCO/P2BFoV/P2BFoV_r50_fpn_1x_coco_ms.py 2 

CUDA_VISIBLE_DEVICES=2,3 bash tools/dist_train.sh configs2/COCO/P2BFoV/P2BFoV_r50_fpn_1x_coco_ms.py 2

<!-- 两张卡接着中断前训练 -->
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh configs2/COCO/P2BFoV/P2BFoV_r50_fpn_1x_coco_ms.py 2 \
    --work-dir work_dir/P2BFoV/ \
    --resume-from work_dir/P2BFoV/epoch_13.pth

<!-- 指定gpu-id的分布式训练 -->
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh configs2/COCO/P2BFoV/P2BFoV_r50_fpn_1x_coco_ms.py 4 


<!-- # 验证方法 跳过训练直接验证 -->
 python tools/train.py \
    configs2/COCO/P2BFoV/P2BFoV_r50_fpn_1x_coco_ms.py \
    --work-dir work_dir/P2BFoV/ \
    --gpu-ids 0 \
    --cfg-options load_from=work_dir/P2BFoV/epoch_8.pth \
    runner.max_epochs=0 evaluation.interval=1

<!-- # 验证方法多卡验证 -->
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh configs2/COCO/P2BFoV/P2BFoV_r50_fpn_1x_coco_ms.py 2 \
    --work-dir work_dir/P2BFoV/ \
    --cfg-options "load_from=work_dir/P2BFoV/epoch_16.pth" "runner.max_epochs=0" "evaluation.interval=1"



<!-- # 获取结果文件转换为coco格式 -->
python bfov/temp_json/bfov2coco/merge_detection_results.py 
<!-- # 可视化结果 -->
python bfov/temp_json/bfov2coco/bfov_batch_visualization.py 




 <!-- 只杀死训练相关的进程 -->
    pkill -9 -u mengchao -f "python.*train.py"
<!-- 只杀死推理相关的进程 -->
    pkill -9 -u mengchao -f "python.*test.py"
 <!-- 查看当前 mengchao 用户的进程 -->
    ps -u mengchao -u
<!-- 杀死特定进程（如果知道 PID） -->
    kill -9 <PID>

<!-- 1. 先查看当前进程 -->
    ps -u mengchao -u
<!-- 2. 确认无误后，杀死所有进程 -->
    pkill -9 -u mengchao
 <!-- 3. 验证进程已被杀死 -->
    ps -u mengchao -u
```






每次训练之后需要保存好训练日志、推理日志，推理结果json，配置文件以及权重文件方便后续返回分析结果

temp_change里面保存的都是待测试的训练模块，到时候直接复制即可
head0的精度为0.015目前怀疑有几处问题
head1相较于0去掉了左右特征合并部分，添加了实例复用，大大优化了显存爆炸问题
head2相较于1增加了球面加权的计算逻辑（后续可能考虑有关球面损失）
head2-1相较于1增加了球面加权的计算逻辑同时添加了实例复用，大大优化了显存爆炸问题
head3相较于2-1调整了掩码部分分辨率降低其显存占用
head4相较于2-1优化了尺度设定，使其不用填充保留好跨边界分析能力
    (由于全景图不能进行常规填充所以我直接把尺度调整成64的倍数，这样缩放环节以及特征提取环节都可以直接进行不用额外pad成32的倍数)
    (在这个地方查到了pbr阶段的设置数量，)
head4-1相较于2-1不用多尺度了，直接固定尺度，同时优化了特征过滤的时候使用使用的HW为pad_HW
head4-2相较于3使用调试模式训练，不用反转同时训练7个epoch

待改写的head
head5相较于4修改了mask_to_bbox的逻辑，使其对于大视场保留主体性(很大的区域直接选择左右部分最大的box)不用直接全选掩码区域，但那时配置文件可能需要改一下
head6相较于head4-2改变了掩码的生成逻辑，在生成掩码之后对单个掩码进行处理，找出掩码最厚的区域，厚度为max_thickness，然后过滤掉厚度小于0.3*max_thickness的掩码,但是这个方法无法跨越边界的情况

head7:改动很大，去掉了循环卷积，球面加权，并且检查出来的问题出现在伪框转结果的地方，由于缩放因子会让box放大回到原来的图像，所以这里已经处理，并且采用head6的mask_to_bbox逻辑,而且部分放开尺度，调整基础尺度跨度，之前存在代码混乱的问题，导致每次改动都无法生效
head7-1:在head7的基础上进行means_iou输出补充，同时扩大基础iou跨度以及数量
head7-2:相较于head7-1，增加了循环卷积的配置(效果稍微变差了)
head8:改变获取提案特征的方式为切平面提取