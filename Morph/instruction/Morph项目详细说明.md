# Morph 项目详细说明文档

## 项目概述

Morph 是一个**基于LLM的并行代码生成和测试框架**，用于评估大语言模型在不同并行编程模型下生成高性能计算代码的能力。

**核心目标**：
- 使用 LLM 根据自然语言描述生成并行计算代码
- 支持多种并行编程模型（Serial、OpenMP、MPI、CUDA等）
- 自动化测试生成代码的正确性和性能
- 评估 LLM 在高性能计算领域的代码生成能力

---

## 1. 如何输入 Prompt 生成代码？

### 1.1 Prompt 结构

**输入文件**：`prompts.json`

每个 prompt 包含以下字段：
```json
{
    "problem_type": "geometry",           // 问题类型（dense_la、fft、geometry等）
    "language": "cpp",                    // 编程语言
    "name": "13_geometry_closest_pair_2d", // 问题名称（对应benchmarks目录结构）
    "parallelism_model": "omp",           // 并行模型（serial、omp、cuda、kokkos等）
    "prompt": "#include <omp.h>\\n...\\n/* 描述 */\\ndouble closestPair(...) {"
}
```

### 1.2 代码生成流程

**主要代码生成函数**（位于 `functions/` 目录）：

#### **CodeGenv1** - 基础版本
```python
CodeGenv1(input_path="prompts.json", 
          output_path="results/prompts_code.json", 
          temperature=0.2)
```
- 直接将 `prompt` 字段发送给 LLM（如 Qwen）
- LLM 补全函数体
- 保存生成的代码到 JSON

#### **CodeGenv2** - 添加硬件信息
```python
CodeGenv2(input_path="prompts.json", 
          output_path="results/prompts_code.json")
```
- 在prompt基础上添加硬件信息（CPU/GPU规格）
- 提示词模板：
```
Complete the C++ function {function_name}.
Below is the hardware information:
{hardware_info}
```

#### **CodeGenv3** - 添加优化策略
```python
CodeGenv3(input_path="prompts.json", 
          output_path="results/prompts_code.json", 
          temperature=0)
```
- 先用 Agent 分析硬件信息，生成优化策略
- 然后在 prompt 中包含这些优化策略
- 提示词模板：
```
Complete the C++ function {function_name}.
Below are some potential optimization strategies:
{optimization_strategies}
```

**生成流程图**：
```
prompts.json 
  ↓ (读取prompt)
CodeGenv1/v2/v3
  ↓ (调用LLM API)
LLM生成函数体
  ↓ (后处理、提取代码)
prompts_code.json (包含生成的代码)
```

---

## 2. 并行模型详解

Morph 支持 **7种并行编程模型**，每种模型对应不同的驱动文件：

| 并行模型 | 驱动文件 | 编译器 | 编译选项 | 适用场景 |
|---------|---------|--------|---------|---------|
| **serial** | serial-driver.cc | g++ | -O3 | 串行代码，基准对比 |
| **omp** | omp-driver.cc | g++ | -O3 -fopenmp | 共享内存多线程（OpenMP） |
| **mpi** | mpi-driver.cc | mpicxx | -O3 | 分布式内存并行（MPI） |
| **mpi+omp** | mpi-omp-driver.cc | mpicxx | -O3 -fopenmp | 混合并行（MPI+OpenMP） |
| **kokkos** | kokkos-driver.cc | g++ + cmake | -O3 -fopenmp + Kokkos库 | 跨平台并行抽象（CPU/GPU） |
| **cuda** | cuda-driver.cu | nvcc | -arch=sm_80 -O3 | NVIDIA GPU并行 |
| **hip** | hip-driver.cu | hipcc | -O3 | AMD GPU并行 |

### 2.1 并行模型的区别

#### **Serial（串行）**
- 无并行化
- 作为性能基准
- 示例：简单的for循环

#### **OpenMP（共享内存）**
```cpp
#pragma omp parallel for
for (int i = 0; i < n; i++) {
    // 并行执行
}
```
- 多线程并行
- 适用于单机多核CPU
- 共享内存访问

#### **MPI（分布式内存）**
```cpp
MPI_Scatter(...);  // 分发数据
// 各进程计算
MPI_Gather(...);   // 收集结果
```
- 多进程并行
- 适用于集群
- 消息传递通信

#### **MPI+OpenMP（混合）**
- 进程间：MPI通信
- 进程内：OpenMP多线程
- 适用于大规模集群

#### **Kokkos（跨平台抽象）**
```cpp
Kokkos::parallel_for(n, KOKKOS_LAMBDA(int i) {
    // 并行执行
});
```
- 统一的API，可在CPU/GPU上运行
- 编译时选择后端

#### **CUDA（NVIDIA GPU）**
```cpp
__global__ void kernel(...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // GPU计算
}
kernel<<<blocks, threads>>>(...);
```
- NVIDIA GPU专用
- CUDA核函数

#### **HIP（AMD GPU）**
- AMD GPU专用
- 与CUDA语法相似

---

## 3. run-all.py 详解：自动化测试系统

### 3.1 主要功能

`drivers/run-all.py` 是**自动化代码测试系统**，负责：

1. **编译**生成的代码
2. **执行**编译后的程序
3. **验证**输出正确性
4. **记录**结果到JSON

### 3.2 执行流程

```python
python drivers/run-all.py \
    --input_json "../results/prompts_code.json" \
    --output "../results/code_run.json" \
    --include-models serial omp cuda
```

**详细步骤**：

#### **步骤1：加载输入**
- 读取 `prompts_code.json`（包含LLM生成的代码）
- 读取 `launch-configs.json`（CUDA/HIP的kernel配置）
- 读取 `problem-sizes.json`（测试问题规模）

#### **步骤2：对每个生成的代码**
```python
for prompt in prompts:
    # 1. 确定并行模型（serial、omp、cuda等）
    parallelism_model = prompt["parallelism_model"]
    
    # 2. 获取对应的驱动器
    driver = get_driver(prompt, ...)  # CppDriverWrapper
    
    # 3. 编译代码
    build_result = driver.build(prompt)  
    # 返回：BuildOutput(success, message, executable_path)
    
    # 4. 如果编译成功，运行代码
    if build_result.success:
        run_result = driver.run(prompt, build_result)
        # 返回：RunOutput(success, message, correctness, runtime)
    
    # 5. 保存结果
    prompt["outputs"].append({
        "build_success": build_result.success,
        "run_success": run_result.success,
        "correctness": run_result.correctness,
        "runtime": run_result.runtime
    })
```

#### **步骤3：输出结果**
保存到 `code_run.json`：
```json
[
    {
        "name": "00_dense_la_lu_decomp",
        "parallelism_model": "serial",
        "prompt": "...",
        "outputs": [
            {
                "code": "生成的代码",
                "build_success": true,
                "run_success": true,
                "correctness": true,
                "runtime": 0.123
            }
        ]
    }
]
```

### 3.3 关键参数

- `--include-models serial omp cuda`: 只测试指定模型
- `--problem "00_dense_la_lu_decomp"`: 只测试特定问题
- `--problem-type "dense_la"`: 只测试特定类型
- `--build-timeout 30`: 编译超时（秒）
- `--run-timeout 120`: 运行超时（秒）
- `--overwrite`: 覆盖已有结果
- `--dry`: 模拟运行（不真正执行）

---

## 4. cpp/ 目录结构详解

### 4.1 整体结构

```
cpp/
├── benchmarks/          # 测试基准（60个算法问题）
│   ├── dense_la/       # 密集线性代数（00-04）
│   ├── fft/            # 快速傅里叶变换（05-09）
│   ├── geometry/       # 几何计算（10-14）
│   ├── graph/          # 图算法（15-19）
│   ├── histogram/      # 直方图统计（20-24）
│   ├── reduce/         # 归约运算（25-29）
│   ├── scan/           # 扫描/前缀和（30-34）
│   ├── search/         # 搜索算法（35-39）
│   ├── sort/           # 排序算法（40-44）
│   ├── sparse_la/      # 稀疏线性代数（45-49）
│   ├── stencil/        # 模板计算（50-54）
│   └── transform/      # 变换操作（55-59）
│
└── models/             # 并行模型的驱动框架
    ├── serial-driver.cc      # 串行驱动
    ├── omp-driver.cc         # OpenMP驱动
    ├── mpi-driver.cc         # MPI驱动
    ├── mpi-omp-driver.cc     # MPI+OpenMP驱动
    ├── kokkos-driver.cc      # Kokkos驱动
    ├── cuda-driver.cu        # CUDA驱动
    └── hip-driver.cu         # HIP驱动
```

### 4.2 benchmarks/ 详解

#### **目录结构模式**

每个问题（如 `00_dense_la_lu_decomp`）包含以下文件：

```
00_dense_la_lu_decomp/
├── baseline.hpp         # 正确的参考实现（用于验证）
├── cpu.cc              # Serial/OpenMP/MPI/MPI+OpenMP的驱动代码
├── gpu.cu              # CUDA/HIP的驱动代码
└── kokkos.cc           # Kokkos的驱动代码
```

#### **文件说明**

**baseline.hpp** - 参考实现
```cpp
#pragma once
#include <vector>

// 正确的LU分解实现
void correctLuFactorize(std::vector<double> &A, size_t N) {
    // 串行的正确实现
    // 用于验证LLM生成代码的正确性
}
```

**cpu.cc** - CPU驱动代码
```cpp
#include "utilities.hpp"
#include "baseline.hpp"
#include "generated-code.hpp"   // LLM生成的代码

// 定义测试上下文
struct Context {
    std::vector<double> A;
    size_t N;
};

// 初始化测试数据
Context *init() {
    Context *ctx = new Context();
    ctx->N = DRIVER_PROBLEM_SIZE;
    ctx->A.resize(ctx->N * ctx->N);
    reset(ctx);
    return ctx;
}

// 重置数据（随机填充）
void reset(Context *ctx) {
    fillRand(ctx->A, -10.0, 10.0);
    BCAST(ctx->A, DOUBLE);  // MPI广播
}

// 调用LLM生成的函数
void NO_OPTIMIZE compute(Context *ctx) {
    luFactorize(ctx->A, ctx->N);  // 调用生成的代码
}

// 调用正确的基准实现
void NO_OPTIMIZE best(Context *ctx) {
    correctLuFactorize(ctx->A, ctx->N);  // 调用baseline
}

// 验证正确性
bool validate(Context *ctx) {
    // 多次随机测试
    for (int trial = 0; trial < MAX_VALIDATION_ATTEMPTS; trial++) {
        // 1. 准备测试数据
        // 2. 运行baseline得到正确结果
        // 3. 运行生成的代码得到测试结果
        // 4. 比较两者是否相等（允许误差1e-3）
    }
    return true;  // 所有测试通过
}

// 清理资源
void destroy(Context *ctx) {
    delete ctx;
}
```

**gpu.cu / kokkos.cc** - GPU驱动代码
- 结构与 cpu.cc 类似
- 使用 `__global__` 内核（CUDA）或 `Kokkos::View`（Kokkos）
- 包含设备内存管理

### 4.3 models/ 详解

**驱动框架文件**，为每种并行模型提供统一的测试框架。

#### **驱动文件的结构**

以 `serial-driver.cc` 为例：

```cpp
// 包含必要的头文件
#include <iostream>
#include <chrono>
#include "utilities.hpp"

// 定义宏
#define DRIVER_PROBLEM_SIZE (1<<20)  // 默认问题规模
#define MAX_VALIDATION_ATTEMPTS 5     // 验证次数
#define NO_OPTIMIZE __attribute__((noinline))
#define NO_INLINE __attribute__((noinline))

// MPI相关宏（串行时为空）
#define GET_RANK(rank) rank = 0
#define BCAST(data, type) 
#define SYNC()
#define IS_ROOT(rank) true

// 主函数
int main(int argc, char **argv) {
    // 1. 初始化
    Context *ctx = init();
    
    // 2. 预热
    compute(ctx);
    reset(ctx);
    
    // 3. 验证正确性
    bool isCorrect = validate(ctx);
    if (!isCorrect) {
        std::cout << "VALIDATION FAILED" << std::endl;
        return 1;
    }
    
    // 4. 性能测试
    reset(ctx);
    auto start = std::chrono::high_resolution_clock::now();
    compute(ctx);
    auto end = std::chrono::high_resolution_clock::now();
    
    double elapsed = std::chrono::duration<double>(end - start).count();
    std::cout << elapsed << std::endl;
    
    // 5. 清理
    destroy(ctx);
    return 0;
}
```

**不同模型的驱动差异**：

- **omp-driver.cc**: 
  - 相同结构
  - 宏保持不变（OMP在代码中用 `#pragma`）

- **mpi-driver.cc**:
  - 添加 `MPI_Init` / `MPI_Finalize`
  - `BCAST` 宏改为 `MPI_Bcast`
  - `SYNC` 宏改为 `MPI_Barrier`

- **cuda-driver.cu**:
  - 添加设备内存管理
  - kernel 启动配置（blocks、threads）
  - 结果从设备拷回主机

- **kokkos-driver.cc**:
  - 添加 `Kokkos::initialize` / `Kokkos::finalize`
  - 使用 `Kokkos::View` 管理数据

---

## 3. 测试流程详解（run-all.py的工作原理）

### 3.1 完整测试流程

```
加载 prompts_code.json
  ↓
对每个问题的每个并行模型：
  ↓
  ① 创建临时目录
  ↓
  ② 组装完整的源文件：
     - 驱动代码（cpu.cc/gpu.cu/kokkos.cc）
     - LLM生成的代码（写入 generated-code.hpp）
     - 基准代码（baseline.hpp）
  ↓
  ③ 编译
     - Serial/OMP/MPI: g++/mpicxx直接编译
     - CUDA: nvcc编译
     - Kokkos: cmake + make
  ↓
  ④ 如果编译成功，运行可执行文件
     - 捕获标准输出（运行时间）
     - 捕获标准错误
     - 设置超时（默认120秒）
  ↓
  ⑤ 解析运行结果
     - "VALIDATION FAILED" → 正确性失败
     - 数字（如0.123）→ 运行时间
  ↓
  ⑥ 记录结果
  ↓
保存到 code_run.json
```

### 3.2 CppDriverWrapper 的实现

**关键方法**：

#### **build()** - 编译代码
```python
def build(self, prompt: dict) -> BuildOutput:
    # 1. 创建临时目录
    # 2. 提取LLM生成的代码
    # 3. 写入 generated-code.hpp
    # 4. 选择对应的驱动代码（cpu.cc/gpu.cu/kokkos.cc）
    # 5. 根据并行模型选择编译器和选项
    # 6. 执行编译命令
    # 7. 返回 BuildOutput(success, message, executable_path)
```

#### **run()** - 运行代码
```python
def run(self, prompt: dict, build_output: BuildOutput) -> RunOutput:
    # 1. 获取可执行文件路径
    # 2. 准备运行命令（MPI需要mpirun前缀）
    # 3. 执行程序
    # 4. 解析输出：
    #    - 验证失败 → correctness = False
    #    - 成功 → 提取运行时间
    # 5. 返回 RunOutput(success, correctness, runtime)
```

### 3.3 验证机制

**正确性验证**（在驱动代码的 `validate()` 函数中）：
1. 生成随机测试数据
2. 用 baseline 计算正确答案
3. 用 LLM 生成的代码计算测试答案
4. 比较两者（允许浮点误差 1e-3 到 1e-6）
5. 重复多次（默认5次）确保稳定性

---

## 5. 如何添加新算子？

### 5.1 添加算子的完整步骤

#### **步骤1：创建 benchmark 目录**

在 `cpp/benchmarks/` 下选择合适的类别（或创建新类别），创建问题目录：

```bash
cd cpp/benchmarks/dense_la/
mkdir 60_dense_la_matrix_norm
cd 60_dense_la_matrix_norm/
```

#### **步骤2：编写 baseline.hpp**

提供正确的参考实现：

```cpp
#pragma once
#include <vector>
#include <cmath>

/* Compute the Frobenius norm of matrix A.
   A is an NxM matrix stored in row-major.
   Example:
   
   input: [[1, 2], [3, 4]]  (2x2 matrix)
   output: 5.477  (sqrt(1^2 + 2^2 + 3^2 + 4^2))
*/
double correctMatrixNorm(std::vector<double> const& A, size_t N, size_t M) {
    double sum = 0.0;
    for (size_t i = 0; i < N * M; i++) {
        sum += A[i] * A[i];
    }
    return std::sqrt(sum);
}
```

#### **步骤3：编写驱动代码**

**cpu.cc** - Serial/OpenMP/MPI驱动：

```cpp
// Driver for 60_dense_la_matrix_norm for Serial, OpenMP, MPI, and MPI+OpenMP
// /* Compute the Frobenius norm of matrix A.
//    Example:
//    input: [[1, 2], [3, 4]]
//    output: 5.477
// */
// double matrixNorm(std::vector<double> const& A, size_t N, size_t M) {

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "utilities.hpp"
#include "baseline.hpp"
#include "generated-code.hpp"   // LLM生成的代码

struct Context {
    std::vector<double> A;
    size_t N, M;
    double result;
};

void reset(Context *ctx) {
    fillRand(ctx->A, -10.0, 10.0);
    BCAST(ctx->A, DOUBLE);
    ctx->result = 0.0;
}

Context *init() {
    Context *ctx = new Context();
    ctx->N = DRIVER_PROBLEM_SIZE / 100;
    ctx->M = DRIVER_PROBLEM_SIZE / 100;
    ctx->A.resize(ctx->N * ctx->M);
    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    ctx->result = matrixNorm(ctx->A, ctx->N, ctx->M);
}

void NO_OPTIMIZE best(Context *ctx) {
    ctx->result = correctMatrixNorm(ctx->A, ctx->N, ctx->M);
}

bool validate(Context *ctx) {
    const size_t TEST_N = 128;
    const size_t TEST_M = 128;
    
    std::vector<double> A(TEST_N * TEST_M);
    
    int rank;
    GET_RANK(rank);
    
    const size_t numTries = MAX_VALIDATION_ATTEMPTS;
    for (int trialIter = 0; trialIter < numTries; trialIter += 1) {
        // 设置输入
        fillRand(A, -10.0, 10.0);
        BCAST(A, DOUBLE);
        
        // 计算正确结果
        double correct = correctMatrixNorm(A, TEST_N, TEST_M);
        
        // 计算测试结果
        double test = matrixNorm(A, TEST_N, TEST_M);
        SYNC();
        
        // 比较结果
        bool isCorrect = true;
        if (IS_ROOT(rank) && std::abs(correct - test) > 1e-3) {
            isCorrect = false;
        }
        BCAST_PTR(&isCorrect, 1, CXX_BOOL);
        if (!isCorrect) {
            return false;
        }
    }
    
    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}
```

**gpu.cu** - CUDA驱动：

```cpp
// Driver for 60_dense_la_matrix_norm for CUDA
// /* Compute the Frobenius norm of matrix A.
//    Store result in norm.
//    Use CUDA to compute in parallel.
//    Example:
//    input: [[1, 2], [3, 4]]
//    output: 5.477
// */
// __global__ void matrixNorm(const double *A, size_t N, size_t M, double *norm) {

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>
#include <cuda_runtime.h>

#include "utilities.hpp"
#include "baseline.hpp"
#include "generated-code.hpp"

struct Context {
    std::vector<double> h_A;
    double *d_A;
    double *d_norm;
    double h_norm;
    size_t N, M;
};

void reset(Context *ctx) {
    fillRand(ctx->h_A, -10.0, 10.0);
    cudaMemcpy(ctx->d_A, ctx->h_A.data(), ctx->N * ctx->M * sizeof(double), cudaMemcpyHostToDevice);
    ctx->h_norm = 0.0;
}

Context *init() {
    Context *ctx = new Context();
    ctx->N = DRIVER_PROBLEM_SIZE / 100;
    ctx->M = DRIVER_PROBLEM_SIZE / 100;
    ctx->h_A.resize(ctx->N * ctx->M);
    
    cudaMalloc(&ctx->d_A, ctx->N * ctx->M * sizeof(double));
    cudaMalloc(&ctx->d_norm, sizeof(double));
    
    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    // 启动kernel（从launch-configs.json读取配置）
    dim3 blocks(DRIVER_KERNEL_BLOCKS);
    dim3 threads(DRIVER_KERNEL_THREADS);
    
    matrixNorm<<<blocks, threads>>>(ctx->d_A, ctx->N, ctx->M, ctx->d_norm);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&ctx->h_norm, ctx->d_norm, sizeof(double), cudaMemcpyDeviceToHost);
}

void NO_OPTIMIZE best(Context *ctx) {
    ctx->h_norm = correctMatrixNorm(ctx->h_A, ctx->N, ctx->M);
}

bool validate(Context *ctx) {
    // 类似cpu.cc的验证逻辑
    // 但需要处理设备内存的分配和拷贝
}

void destroy(Context *ctx) {
    cudaFree(ctx->d_A);
    cudaFree(ctx->d_norm);
    delete ctx;
}
```

**kokkos.cc** - Kokkos驱动：

```cpp
// Driver for 60_dense_la_matrix_norm for Kokkos
// #include <Kokkos_Core.hpp>
// /* Compute the Frobenius norm of matrix A using Kokkos.
//    Example:
//    input: [[1, 2], [3, 4]]
//    output: 5.477
// */
// double matrixNorm(Kokkos::View<const double**> &A) {

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "kokkos-includes.hpp"
#include "utilities.hpp"
#include "baseline.hpp"
#include "generated-code.hpp"

struct Context {
    Kokkos::View<double**> A;
    std::vector<double> h_A;
    double norm;
    size_t N, M;
};

void reset(Context *ctx) {
    fillRand(ctx->h_A, -10.0, 10.0);
    copyVectorToView2D(ctx->h_A, ctx->A, ctx->N, ctx->M);
    ctx->norm = 0.0;
}

Context *init() {
    Context *ctx = new Context();
    ctx->N = DRIVER_PROBLEM_SIZE / 100;
    ctx->M = DRIVER_PROBLEM_SIZE / 100;
    ctx->h_A.resize(ctx->N * ctx->M);
    ctx->A = Kokkos::View<double**>("A", ctx->N, ctx->M);
    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    Kokkos::View<const double**> AConst = ctx->A;
    ctx->norm = matrixNorm(AConst);
}

void NO_OPTIMIZE best(Context *ctx) {
    ctx->norm = correctMatrixNorm(ctx->h_A, ctx->N, ctx->M);
}

bool validate(Context *ctx) {
    // Kokkos特定的验证逻辑
    // 使用Kokkos::View进行数据管理
}

void destroy(Context *ctx) {
    delete ctx;
}
```

#### **步骤4：添加到 prompts.json**

为每个并行模型添加prompt：

```json
[
    {
        "problem_type": "dense_la",
        "language": "cpp",
        "name": "60_dense_la_matrix_norm",
        "parallelism_model": "serial",
        "prompt": "#include <vector>\\n#include <cmath>\\n\\n/* Compute the Frobenius norm of matrix A.\\n   A is an NxM matrix stored in row-major.\\n   Example:\\n\\n   input: [[1, 2], [3, 4]]\\n   output: 5.477\\n*/\\ndouble matrixNorm(std::vector<double> const& A, size_t N, size_t M) {"
    },
    {
        "problem_type": "dense_la",
        "language": "cpp",
        "name": "60_dense_la_matrix_norm",
        "parallelism_model": "omp",
        "prompt": "#include <omp.h>\\n#include <vector>\\n#include <cmath>\\n\\n/* Compute the Frobenius norm of matrix A.\\n   Use OpenMP to compute in parallel.\\n   Example:\\n\\n   input: [[1, 2], [3, 4]]\\n   output: 5.477\\n*/\\ndouble matrixNorm(std::vector<double> const& A, size_t N, size_t M) {"
    },
    {
        "problem_type": "dense_la",
        "language": "cpp",
        "name": "60_dense_la_matrix_norm",
        "parallelism_model": "cuda",
        "prompt": "/* Compute the Frobenius norm of matrix A.\\n   Store result in norm.\\n   Use CUDA to compute in parallel. The kernel is launched with at least as many threads as elements.\\n   Example:\\n\\n   input: [[1, 2], [3, 4]]\\n   output: 5.477\\n*/\\n__global__ void matrixNorm(const double *A, size_t N, size_t M, double *norm) {"
    },
    {
        "problem_type": "dense_la",
        "language": "cpp",
        "name": "60_dense_la_matrix_norm",
        "parallelism_model": "kokkos",
        "prompt": "#include <Kokkos_Core.hpp>\\n\\n/* Compute the Frobenius norm of matrix A using Kokkos.\\n   Assume Kokkos is already initialized.\\n   Example:\\n\\n   input: [[1, 2], [3, 4]]\\n   output: 5.477\\n*/\\ndouble matrixNorm(Kokkos::View<const double**> &A) {"
    }
]
```

#### **步骤5：生成代码**

```bash
cd /home/dgc/mjs/project/analyze_OB/Morph
python main.py  # 调用 CodeGenv1/v2/v3
```

#### **步骤6：测试代码**

```bash
cd drivers
python run-all.py --input_json "../results/prompts_code.json" \
                  --output "../results/code_run.json" \
                  --include-models serial omp cuda \
                  --problem "60_dense_la_matrix_norm"
```

---

## 6. 问题分类说明

### 6.1 12个问题类别

| 类别 | 编号 | 问题数 | 代表算法 |
|-----|-----|-------|---------|
| **dense_la** | 00-04 | 5 | LU分解、线性求解、GEMM、AXPY、GEMV |
| **fft** | 05-09 | 5 | 逆FFT、DFT、FFT共轭、分离FFT、离线FFT |
| **geometry** | 10-14 | 5 | 凸包、凸包周长、最小三角形、最近点对2D、最近点对1D |
| **graph** | 15-19 | 5 | 边计数、最大连通分量、最高度数、连通分量计数、最短路径 |
| **histogram** | 20-24 | 5 | 像素直方图、0-100分箱、象限计数、首字母计数、四分位计数 |
| **reduce** | 25-29 | 5 | XOR、倒数积、平均值、最小奇数、最小对和 |
| **scan** | 30-34 | 5 | 前缀和、最小扫描、前缀和数组和、逆前缀和、最大连续子数组和 |
| **search** | 35-39 | 5 | 按键搜索、包含判断、最接近π、第一个偶数、XOR包含 |
| **sort** | 40-44 | 5 | 复数按模排序、第k小元素、排序秩、结构体排序、非零排序 |
| **sparse_la** | 45-49 | 5 | 稀疏求解、SpMM、SpMV、稀疏AXPY、稀疏LU |
| **stencil** | 50-54 | 5 | XOR核、边缘核、1D-Jacobi-3点、2D-Jacobi-5点、生命游戏 |
| **transform** | 55-59 | 5 | ReLU、奇数取反、逆偏移、平方、映射函数 |

---

## 7. 项目文件组织

```
Morph/
├── prompts.json              # 问题描述集合（60个算法 × 多种并行模型）
├── main.py                   # 主入口：调用代码生成
├── functions/                # 代码生成函数
│   ├── llmgenv1.py          # CodeGenv1: 基础生成
│   ├── llmgenv2.py          # CodeGenv2: +硬件信息
│   └── llmgenv3.py          # CodeGenv3: +优化策略
├── base_agents.py            # Agent定义（CodeGenerate、PromptGenerate）
├── utils.py                  # 工具函数（硬件信息获取、后处理）
├── drivers/
│   ├── run-all.py           # 自动化测试系统
│   ├── driver_wrapper.py    # 驱动抽象基类
│   └── cpp/
│       ├── cpp_driver_wrapper.py  # C++驱动包装器
│       ├── benchmarks/       # 60个测试算法
│       └── models/           # 7种并行模型的驱动框架
└── results/                  # 实验结果
    ├── prompts_code.json    # LLM生成的代码
    └── code_run.json        # 测试运行结果
```

---

## 8. 实验流程示例

### 8.1 完整实验流程

```bash
# 步骤1：生成代码
cd /home/dgc/mjs/project/analyze_OB/Morph
python main.py  # 修改main.py选择CodeGenv1/v2/v3

# 步骤2：测试代码
cd drivers
python run-all.py \
    --input_json "../results/prompts_code.json" \
    --output "../results/code_run.json" \
    --include-models serial omp cuda

# 步骤3：分析结果
cd ..
python analysis.py  # JSON转CSV + 统计指标
```

### 8.2 评估指标

- **构建成功率** = 编译成功数 / 总问题数
- **运行成功率** = 运行成功数 / 编译成功数
- **正确性** = 验证通过数 / 运行成功数
- **性能** = 生成代码运行时间 / baseline运行时间

---

## 9. 关键技术点

### 9.1 驱动框架的设计哲学

**标准化接口**：所有驱动都实现相同的5个函数
```cpp
Context *init()           // 初始化
void reset(Context *ctx)  // 重置数据
void compute(Context *ctx) // 调用生成的代码
void best(Context *ctx)    // 调用baseline
bool validate(Context *ctx) // 验证正确性
void destroy(Context *ctx)  // 清理
```

**宏抽象**：统一不同并行模型的差异
```cpp
// Serial: 空宏
#define BCAST(data, type)
#define SYNC()

// MPI: 实际的MPI调用
#define BCAST(data, type) MPI_Bcast(data.data(), ...)
#define SYNC() MPI_Barrier(MPI_COMM_WORLD)
```

### 9.2 自动化测试的关键

1. **统一的驱动接口**：所有问题共享相同的测试流程
2. **分离的生成代码**：`generated-code.hpp` 独立，便于替换
3. **多次随机验证**：确保稳定性和正确性
4. **超时保护**：防止死循环或性能极差的代码

---

## 10. 快速开始：添加新算子示例

假设我们要添加"矩阵迹计算"（Matrix Trace）：

```bash
# 1. 创建目录
mkdir -p cpp/benchmarks/dense_la/61_dense_la_matrix_trace
cd cpp/benchmarks/dense_la/61_dense_la_matrix_trace/

# 2. 创建 baseline.hpp
cat > baseline.hpp << 'EOF'
#pragma once
#include <vector>

double correctMatrixTrace(std::vector<double> const& A, size_t N) {
    double trace = 0.0;
    for (size_t i = 0; i < N; i++) {
        trace += A[i * N + i];
    }
    return trace;
}
EOF

# 3. 创建 cpu.cc（复制并修改类似问题）
cp ../00_dense_la_lu_decomp/cpu.cc ./cpu.cc
# 然后编辑cpu.cc，修改函数名和验证逻辑

# 4. 创建 gpu.cu 和 kokkos.cc（可选）
# ...

# 5. 添加到 prompts.json
# 添加对应的prompt条目

# 6. 生成和测试
cd ../../../../
python main.py
cd drivers
python run-all.py --problem "61_dense_la_matrix_trace"
```

---

## 总结

Morph 项目是一个**系统化的LLM代码生成评估框架**，通过：
1. **标准化的测试基准**（60个算法问题）
2. **多样化的并行模型**（7种并行编程范式）
3. **自动化的测试流程**（编译+运行+验证）
4. **灵活的prompt策略**（基础、硬件感知、优化感知）

来全面评估LLM在高性能计算代码生成方面的能力，为研究LLM辅助并行编程提供了完整的实验平台。
