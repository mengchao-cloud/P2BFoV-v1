
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
<!-- 单卡训练 -->
python tools/train.py configs2/COCO/P2BFoV/P2BFoV_r50_fpn_1x_coco_ms.py --work-dir work_dir/P2BFoV/ --gpu-ids 0

<!-- 两张卡分布式训练 -->
bash tools/dist_train.sh configs2/COCO/P2BFoV/P2BFoV_r50_fpn_1x_coco_ms.py 2 --work-dir ../work_dir/P2BFoV/

<!-- 指定gpu-id的分布式训练 -->
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh configs2/COCO/P2BFoV/P2BFoV_r50_fpn_1x_coco_ms.py 2 --work-dir ../work_dir/P2BFoV/


# 验证方法 跳过训练直接验证
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs2/COCO/P2BFoV/P2BFoV_r50_fpn_1x_coco_ms.py \
    --work-dir work_dir/P2BFoV/ \
    --gpu-ids 0 \
    --cfg-options load_from=work_dir/P2BFoV/epoch_12.pth \
    runner.max_epochs=0 evaluation.interval=1