import time
import py3nvml

# 初始化NVML库
py3nvml.py3nvml.nvmlInit()
# 监控GPU0和GPU1，定义索引列表
MONITOR_GPUS = [0, 1]
# 监控刷新间隔（秒），可按需修改
INTERVAL = 2

try:
    while True:
        monitor_msg = ""  # 拼接GPU0和GPU1的监控信息
        # 遍历监控GPU0、GPU1
        for gpu_idx in MONITOR_GPUS:
            # 获取对应GPU句柄
            gpu_handle = py3nvml.py3nvml.nvmlDeviceGetHandleByIndex(gpu_idx)
            # 显存信息（已用/总，单位MB，原始为字节需转换）
            mem_info = py3nvml.py3nvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            used_mem = mem_info.used / 1024 / 1024
            total_mem = mem_info.total / 1024 / 1024
            # CUDA核心利用率（即CUDA算力，%）
            cuda_util = py3nvml.py3nvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
            # 拼接单卡监控信息
            monitor_msg += f"GPU{gpu_idx}: 显存{used_mem:.0f}/{total_mem:.0f}MB | CUDA算力{cuda_util}%  |  "
        
        # 单行覆盖打印，去掉末尾多余的"|  "，end='\r'实现刷新不刷屏
        print(monitor_msg.rstrip("|  "), end='\r')
        # 监控间隔
        time.sleep(INTERVAL)

except KeyboardInterrupt:
    # 捕获Ctrl+C，优雅退出并释放NVML资源
    py3nvml.py3nvml.nvmlShutdown()
    print("\nGPU监控已停止，资源已释放")
except Exception as e:
    # 异常情况也强制释放资源，避免资源泄漏
    py3nvml.py3nvml.nvmlShutdown()
    print(f"\n监控异常终止：{str(e)}，资源已释放")