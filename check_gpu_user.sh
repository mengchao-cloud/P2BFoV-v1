#!/bin/bash
# 遍历0-3号GPU，逐个查询占用用户
for gpu_id in 0 1 2 3; do
    echo -e "\n===== GPU $gpu_id 占用信息 ====="
    # 提取该GPU的计算进程（排除Xorg等系统进程）
    nvidia-smi -i $gpu_id | grep -E 'C +' | grep -v 'Xorg' | awk '{
        pid=$5; mem=$12;
        # 查询进程所属用户
        cmd="ps -o user= -p " pid;
        cmd | getline user;
        close(cmd);
        # 输出结果
        print "用户: " user "\nPID: " pid "\n显存占用: " mem "\n---";
    }'
    # 若该GPU无计算进程，提示闲置
    if [ $? -ne 0 ]; then
        echo "该GPU无用户计算进程（仅系统进程）"
    fi
done

# for g in 0 1 2 3; do echo -e "\n===== GPU $g 占用信息 ====="; nvidia-smi -i $g | awk '/^|    [0-9]+ +N\/A +N\/A/ && /python/ {pid=$5; mem=$12; cmd="ps -o user= -p " pid; cmd | getline user; close(cmd); print "用户: " user "\nPID: " pid "\n显存占用: " mem "\n---"}'; done