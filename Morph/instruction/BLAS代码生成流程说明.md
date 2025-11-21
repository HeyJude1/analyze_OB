# BLAS算子代码生成流程完整说明

## 概述

本文档说明了基于OpenBLAS优化策略的LLM代码生成完整流程，包括3个主要组件的修改和使用方法。

## 修改内容总结

### 1. ✅ 创建了 prompts1.json

**位置**: `/home/dgc/mjs/project/analyze_OB/Morph/prompts1.json`

**内容**: 包含10个BLAS算子的20个prompt（每个算子Serial和OpenMP两种并行模型）

**格式示例**:
```json
{
    "problem_type": "blas",
    "language": "cpp", 
    "name": "00_blas_gemm",
    "parallelism_model": "serial",
    "prompt": "typedef long BLASLONG;\n\n/* Matrix-Matrix Multiplication... */"
}
```

**特点**:
- 基于 `prompts.json` 的标准格式
- 包含完整的函数签名和注释
- 提供了具体的使用示例
- 支持Serial和OpenMP两种并行模型

### 2. ✅ 修改了 kg_config.json

**位置**: `/home/dgc/mjs/project/analyze_OB/KG/kg_config.json`

**新增配置**:
```json
{
  "optimization_results": {
    "output_dir": "/home/dgc/mjs/project/analyze_OB/op_results"
  }
}
```

**作用**: 为 `Operator_op2.py` 指定优化策略输出目录

### 3. ✅ 修改了 Operator_op2.py

**位置**: `/home/dgc/mjs/project/analyze_OB/KG/Operator_op2.py`

**主要修改**:
- 添加了 `--source` 参数支持指定源代码文件
- 添加了 `--output_dir` 参数支持自定义输出目录
- 支持从 `kg_config.json` 读取 `optimization_results.output_dir` 配置
- 输出路径格式: `/op_results/算子名称/算子名称.json`
- 自动创建算子专用目录

**新的命令行用法**:
```bash
# 使用默认配置
python Operator_op2.py --source /path/to/01_gemm.c

# 指定输出目录
python Operator_op2.py --source /path/to/01_gemm.c --output_dir /custom/path
```

### 4. ✅ 创建了 llmgenv4.py

**位置**: `/home/dgc/mjs/project/analyze_OB/Morph/functions/llmgenv4.py`

**基于**: `llmgenv3.py`

**新增功能**:
1. **优化策略加载**: 自动加载 `Operator_op2.py` 生成的优化策略JSON文件
2. **策略目录配置**: 支持 `--strategy_dir` 参数指定策略目录
3. **增强的Prompt模板**: 在原有基础上添加优化策略字段

**新的Prompt模板**:
```
Complete the C++ function {function_name}. Only write the body of the function {function_name}.

```cpp
{prompt}
```
Below are some potential optimization strategies you can use for code generation on this platform:
{op_strategy}

Additional optimization recommendations from analysis:
{optimization_strategies}
```

**优化策略格式化**: 
- 提取 `final_strategies` 字段
- 包含核心模式、上下文模式、得分、实现建议等
- 友好的文本格式展示

## 完整使用流程

### 步骤1: 准备OpenBLAS源码

确保在 `openblas_output/` 目录下有10个算子的源码文件：
```
openblas_output/
├── 01_gemm.c
├── 02_gemv.c
├── 03_axpby.c
├── 04_dot.c
├── 05_swap.c
├── 06_copy.c
├── 07_nrm2.c
├── 08_asum.c
├── 09_iamax.c
└── 10_iamin.c
```

### 步骤2: 生成优化策略

为每个算子运行 `Operator_op2.py`：

```bash
cd /home/dgc/mjs/project/analyze_OB/KG

# 为GEMM生成优化策略
python Operator_op2.py --source ../Morph/openblas_output/01_gemm.c

# 为GEMV生成优化策略  
python Operator_op2.py --source ../Morph/openblas_output/02_gemv.c

# ... 其他算子
```

**输出结果**:
```
/home/dgc/mjs/project/analyze_OB/op_results/
├── 01_gemm/
│   └── 01_gemm.json
├── 02_gemv/
│   └── 02_gemv.json
└── ...
```

### 步骤3: 使用LLM生成优化代码

```bash
cd /home/dgc/mjs/project/analyze_OB/Morph

python functions/llmgenv4.py \
    --input prompts1.json \
    --output results/blas_optimized_code.json \
    --strategy_dir /home/dgc/mjs/project/analyze_OB/op_results
```

### 步骤4: 一键运行完整流程

使用提供的自动化脚本：

```bash
cd /home/dgc/mjs/project/analyze_OB/Morph
python run_blas_generation.py
```

## 输出文件说明

### 1. 优化策略文件 (Operator_op2.py输出)

**路径**: `/home/dgc/mjs/project/analyze_OB/op_results/{算子名称}/{算子名称}.json`

**关键字段**:
```json
{
  "final_strategies": [
    {
      "name": "策略名称",
      "core_patterns": ["核心模式列表"],
      "contextual_patterns": ["上下文模式列表"], 
      "optimization_context": {
        "cluster_size": 数值,
        "pattern_counts": {"模式": 频次}
      },
      "score": 得分,
      "implementation": "实现建议",
      "impact": "预期影响"
    }
  ]
}
```

### 2. 生成代码文件 (llmgenv4.py输出)

**路径**: `/home/dgc/mjs/project/analyze_OB/Morph/results/blas_optimized_code.json`

**格式**:
```json
[
  {
    "problem_type": "blas",
    "name": "00_blas_gemm", 
    "parallelism_model": "serial",
    "prompt": "原始prompt",
    "temperature": 0.0,
    "outputs": ["生成的C++代码"]
  }
]
```

## 关键特性

### 1. 智能策略匹配
- 根据算子名称自动匹配对应的优化策略文件
- 支持策略文件不存在时的优雅降级

### 2. 丰富的优化信息
- 核心模式和上下文模式
- 模式频次统计
- 优化得分和实现建议
- 预期性能影响

### 3. 灵活的配置
- 支持命令行参数覆盖默认配置
- 支持自定义策略目录和输出路径
- 兼容原有的配置文件格式

### 4. 完整的错误处理
- 文件不存在时的警告提示
- 策略加载失败时的降级处理
- 代码生成异常时的错误记录

## 预期效果

### 1. 代码质量提升
通过集成OpenBLAS的优化策略，生成的代码应该包含：
- 循环展开优化
- 缓存友好的访问模式
- 分支预测优化
- SIMD向量化提示
- OpenMP并行化策略

### 2. 性能改进
相比基础版本，优化后的代码应该在以下方面有改进：
- 更好的缓存局部性
- 更高的并行化效率
- 更少的分支预测失误
- 更好的向量化利用

### 3. 策略可追溯性
每个生成的代码都可以追溯到：
- 使用的优化策略
- 策略的得分和置信度
- 具体的优化模式
- 预期的性能影响

## 扩展和定制

### 1. 添加新算子
1. 在 `openblas_output/` 添加源码文件
2. 在 `prompts1.json` 添加对应的prompt
3. 运行 `Operator_op2.py` 生成优化策略
4. 运行 `llmgenv4.py` 生成代码

### 2. 自定义优化策略
1. 修改 `Operator_op2.py` 的策略提取逻辑
2. 调整 `llmgenv4.py` 的策略格式化方法
3. 更新prompt模板以包含新的策略信息

### 3. 集成其他并行模型
1. 在 `prompts1.json` 添加新的并行模型prompt
2. 确保驱动模板支持新的并行模型
3. 测试生成代码的正确性

## 总结

✅ **已完成的工作**:
1. 创建了20个BLAS算子的标准prompt
2. 配置了优化策略输出路径
3. 增强了 `Operator_op2.py` 的命令行支持
4. 实现了基于优化策略的LLM代码生成器
5. 提供了完整的自动化流程脚本

**这套系统实现了从OpenBLAS优化知识到LLM代码生成的完整闭环，为高性能计算代码的自动生成提供了强有力的支持！** 🚀
