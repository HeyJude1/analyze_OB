"""
设计智能体需使用的工具函数
"""
import os
import re
# 读取硬件信息需要的库
import cpuinfo
from py3nvml import py3nvml
def get_env_var(name: str) -> str:
    """ 从环境变量中获取值，如果没有设置则报错 """
    if name not in os.environ:
        raise ValueError(f"Environment variable {name} not set.")
    return os.environ[name]

def get_function_name(prompt: str, execution_model: str) -> str:
    """ 根据执行模型选择对应正则表达式来提取函数名 """
    # 匹配 GPU 和 CPU 函数的正则表达式
    GPU_FUNCTION_NAME_PATTERN = re.compile(r"__global__ void ([a-zA-Z0-9_]+)\(")
    CPU_FUNCTION_NAME_PATTERN = re.compile(r"\s*[a-zA-Z_]+ ([a-zA-Z0-9_]+)\(")
    if execution_model in ['cuda']:
        match = GPU_FUNCTION_NAME_PATTERN.match(prompt.splitlines()[-1])
    else:
        match = CPU_FUNCTION_NAME_PATTERN.match(prompt.splitlines()[-1])
    if match is None:
        raise ValueError(f"Could not find function name in prompt: {prompt}")
    return match.group(1)

def postprocess(prompt: str, output: str) -> str:
    """ 处理模型输出，去除 markdown 标记等内容 """
    # remove leading ```, ```cpp, and trailing ```
    output = output.strip().lstrip("```cpp").lstrip("```").rstrip("```") # 去掉 markdown 代码块标记
    # 如果输出包含了 prompt 自身，去掉
    if output.startswith(prompt):
        output = output[len(prompt):]
    return output

"""
读取硬件信息
CPU:版本型号、核心数量、架构、指令集扩展、缓存大小、频率
GPU:版本型号、计算能力、CUDA 版本、显存、带宽
"""
def cpu_info():
    # cpu版本型号
    getcpuinfo = cpuinfo.get_cpu_info()
    cpu_version = getcpuinfo['brand_raw']

    # cpu核心数量
    # import psutil
    # cpu_core_num = psutil.cpu_count(logical=True) # 逻辑核心数
    # physical_cores = psutil.cpu_count(logical=False)  # 物理核心数
    cpu_core_num = getcpuinfo['count'] # 逻辑核心

    # cpu架构
    cpu_arch = getcpuinfo['arch']

    # cpu支持的指令集
    cpu_flags =  getcpuinfo['flags']

    # cpu缓存大小(MB)
    cpu_l1_cache_data = getcpuinfo['l1_data_cache_size'] / (1024*1024) # l1数据缓存大小
    cpu_l1_cache_instruction = getcpuinfo['l1_instruction_cache_size'] / (1024*1024) # l1指令缓存大小
    cpu_l2_cache = getcpuinfo['l2_cache_size'] / (1024*1024) # l2缓存大小
    cpu_l3_cache = getcpuinfo['l3_cache_size'] / (1024*1024) # l3缓存大小

    # cpu频率
    cpu_hz_advertised = getcpuinfo['hz_advertised_friendly']
    cpu_hz_actual = getcpuinfo['hz_actual_friendly']

#     res = f"""CPU架构为:{cpu_version}
# CPU逻辑核心数:{cpu_core_num}
# CPU架构为:{cpu_arch}
# CPU支持的指令集:{', '.join(cpu_flags)}
# CPU L1数据缓存大小:{cpu_l1_cache_data}MB
# CPU L1指令缓存大小:{cpu_l1_cache_instruction}MB
# CPU L2缓存大小:{cpu_l2_cache}MB
# CPU L3缓存大小:{cpu_l3_cache}MB
# CPU 广告Hz:{cpu_hz_advertised}
# CPU 实际Hz:{cpu_hz_actual}
# """
    res = f"""CPU Architecture: {cpu_version}
Number of CPU Logical Cores: {cpu_core_num}
CPU Architecture: {cpu_arch}
Supported CPU Instruction Sets: {', '.join(cpu_flags)}
CPU L1 Data Cache Size: {cpu_l1_cache_data}MB
CPU L1 Instruction Cache Size: {cpu_l1_cache_instruction}MB
CPU L2 Cache Size: {cpu_l2_cache}MB
CPU L3 Cache Size: {cpu_l3_cache}MB
CPU Advertised Frequency: {cpu_hz_advertised}
CPU Actual Frequency: {cpu_hz_actual}
"""
    
    return res
def gpu_info():
    py3nvml.nvmlInit()
    # 获取GPU数量
    device_count = py3nvml.nvmlDeviceGetCount()
    if device_count == 0:
        return "This device does not have a GPU."
    res = f"There are a total of {device_count} GPUs, and the information is as follows:\n"
    for i in range(device_count):
        handle = py3nvml.nvmlDeviceGetHandleByIndex(i)

        # 获取GPU版本型号
        gpu_version = py3nvml.nvmlDeviceGetName(handle)

        # 获取CUDA版本
        cuda_version = py3nvml.nvmlSystemGetDriverVersion()

        # 获取计算能力
        compute_capability = py3nvml.nvmlDeviceGetCudaComputeCapability(handle)

        # 获取显存信息
        memory_info = py3nvml.nvmlDeviceGetMemoryInfo(handle)
        
        # 获取 GPU 带宽
        clock_info = py3nvml.nvmlDeviceGetClockInfo(handle, py3nvml.NVML_CLOCK_MEM)

#         res += f"""GPU名称为:{gpu_version}
# CUDA 版本: {cuda_version}
# 计算能力: {compute_capability}
# 显存总大小: {memory_info.total / (1024 ** 2)} MB
# GPU 带宽: {clock_info / 1000} MHz
# --------------------------------------------------
# """

        res += f"""GPU Model: {gpu_version}
CUDA Version: {cuda_version}
Compute Capability: {compute_capability}
Total VRAM Size: {memory_info.total / (1024 ** 2)} MB
GPU Bandwidth: {clock_info / 1000} MHz
--------------------------------------------------
""" 
    # 清理
    py3nvml.nvmlShutdown()
    return res