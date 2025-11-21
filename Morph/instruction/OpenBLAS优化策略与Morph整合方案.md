# OpenBLAS优化策略与Morph整合方案

## 1. 整合目标与理解

### 1.1 当前状态分析

**Operator_op2.py 的输出**：
- 输入：源代码（如 gemm.txt）
- 输出：`opinfo2.json`，包含：
  - `patterns_detected`: 检测到的计算流程
  - `search_strategies`: 所有相关的优化策略
  - `final_strategies`: 高分优化策略（score ≥ 0.5，core_patterns ⊆ patterns_detected）

**Morph 的能力**：
- 输入：`prompts.json`（问题描述 + 并行模型）
- 输出：LLM生成的代码 + 自动化测试结果
- 核心：将优化提示整合到prompt中，引导LLM生成更优代码

### 1.2 整合的核心思想

```
OpenBLAS优化策略知识
  ↓
Operator_op2.py（推荐策略）
  ↓
转换为Morph可用的prompt
  ↓
Morph生成并验证代码
  ↓
性能对比与评估
```

---

## 2. 整合方案设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────┐
│  阶段1：知识提取与策略推荐                                   │
│  ├─ analyze_OB/KG/Operator_op2.py                       │
│  │  输入：BLAS算子源代码（gemm.txt等）                      │
│  │  输出：opinfo2.json（优化策略知识）                      │
│  └─ 关键字段：final_strategies（高分策略+optimization_context）│
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  阶段2：Prompt构建                                        │
│  ├─ 新增：blas_prompt_generator.py                       │
│  │  输入：opinfo2.json + 算子列表                          │
│  │  输出：blas_prompts.json                               │
│  └─ 功能：                                                │
│     - 为每个BLAS算子创建prompt                            │
│     - 整合推荐的优化策略到prompt                           │
│     - 生成多种并行模型的变体（serial、omp、cuda）           │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  阶段3：代码生成                                          │
│  ├─ Morph/functions/llmgenv4.py（新增）                  │
│  │  输入：blas_prompts.json                              │
│  │  输出：blas_code.json                                 │
│  └─ 特点：                                                │
│     - 使用CodeGenv3的模式                                 │
│     - 优化策略来自Operator_op2推荐                        │
│     - 支持温度、top_p等参数调优                            │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  阶段4：基准与驱动准备                                     │
│  ├─ 新增：Morph/drivers/cpp/benchmarks/blas/             │
│  │  ├─ 00_blas_gemm/                                    │
│  │  │  ├─ baseline.hpp（OpenBLAS参考实现）                │
│  │  │  ├─ cpu.cc（Serial/OMP/MPI驱动）                   │
│  │  │  └─ gpu.cu（CUDA驱动）                             │
│  │  ├─ 01_blas_gemv/                                    │
│  │  ├─ 02_blas_axpby/                                   │
│  │  └─ ... (共10个算子)                                  │
│  └─ baseline可选方案：                                    │
│     - 方案A：直接调用OpenBLAS库函数                        │
│     - 方案B：提取OpenBLAS核心实现作为参考                  │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  阶段5：自动化测试                                         │
│  ├─ Morph/drivers/run-all.py（复用）                      │
│  │  输入：blas_code.json                                 │
│  │  输出：blas_code_run.json                             │
│  └─ 测试维度：                                            │
│     - 正确性验证（与OpenBLAS结果对比）                     │
│     - 性能测试（运行时间）                                 │
│     - 不同并行模型对比                                     │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  阶段6：结果分析与评估                                     │
│  ├─ 新增：blas_analysis.py                               │
│  └─ 评估指标：                                            │
│     - 编译成功率（分并行模型）                             │
│     - 正确性率                                            │
│     - 性能提升比例（vs baseline）                         │
│     - 优化策略使用率（分析生成代码是否采纳推荐策略）         │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 详细实施方案

### 3.1 阶段1：知识提取（已完成）

**当前能力**：
- `Operator_op2.py` 已能从 OpenBLAS 源码中提取优化策略
- 输出包含：
  - `patterns_detected`: 计算流程（如 prep.index_pointer_init、core.mm_microkernel）
  - `final_strategies`: 推荐的优化策略（含名称、原理、实现、影响、核心模式）

**需要的输入**：
- 为10个BLAS算子各准备一个参考实现（可以是OpenBLAS的简化版本）
- 运行 `Operator_op2.py` 获取每个算子的优化策略

### 3.2 阶段2：Prompt构建器（需新建）

**新建文件**：`analyze_OB/Morph/blas_prompt_generator.py`

**功能**：从 `opinfo2.json` 生成 `blas_prompts.json`

**代码框架**：
```python
#!/usr/bin/env python3
"""
BLAS Prompt生成器
将Operator_op2推荐的优化策略整合到Morph的prompt格式中
"""

import json
from typing import Dict, List, Any

class BLASPromptGenerator:
    def __init__(self):
        # BLAS算子的基本信息
        self.blas_operators = {
            "gemm": {
                "description": "Matrix-matrix multiplication: C = alpha*A*B + beta*C",
                "signature": "void gemm(int M, int N, int K, double alpha, const double *A, const double *B, double beta, double *C)",
                "test_sizes": [128, 256, 512, 1024]
            },
            "gemv": {
                "description": "Matrix-vector multiplication: y = alpha*A*x + beta*y",
                "signature": "void gemv(int M, int N, double alpha, const double *A, const double *x, double beta, double *y)",
                "test_sizes": [1024, 2048, 4096]
            },
            "axpby": {
                "description": "Vector scaling and addition: y = alpha*x + beta*y",
                "signature": "void axpby(int N, double alpha, const double *x, double beta, double *y)",
                "test_sizes": [10000, 100000, 1000000]
            },
            # ... 其他7个算子
        }
    
    def load_optimization_strategies(self, opinfo_path: str) -> Dict[str, Any]:
        """加载优化策略知识"""
        with open(opinfo_path, 'r') as f:
            return json.load(f)
    
    def format_strategies_for_prompt(self, strategies: List[Dict]) -> str:
        """将优化策略格式化为prompt文本"""
        strategy_text = []
        for i, strategy in enumerate(strategies, 1):
            opt_ctx = strategy.get("optimization_context", {})
            core_patterns = opt_ctx.get("core_patterns", [])
            
            text = f"""
策略{i}: {strategy.get('strategy_name', 'Unknown')}
- 层级: {strategy.get('level', 'unknown')}
- 核心思想: {strategy.get('overview', '')}
- 适用场景: {strategy.get('when_to_use', '')}
- 关键操作: {strategy.get('key_actions', '')}
- 注意事项: {strategy.get('cautions', '')}
- 相关计算流程: {', '.join(core_patterns)}
"""
            strategy_text.append(text.strip())
        
        return "\n\n".join(strategy_text)
    
    def generate_prompt(self, 
                       operator_name: str,
                       parallelism_model: str,
                       strategies: List[Dict]) -> Dict[str, Any]:
        """为单个算子的单个并行模型生成prompt"""
        op_info = self.blas_operators[operator_name]
        
        # 基础prompt（根据并行模型调整）
        if parallelism_model == "serial":
            base_prompt = f"""#include <vector>
#include <cmath>

/* {op_info['description']}
   Implement this function efficiently.
   Example usage provided in comments below.
*/
{op_info['signature']} {{"""
        
        elif parallelism_model == "omp":
            base_prompt = f"""#include <omp.h>
#include <vector>
#include <cmath>

/* {op_info['description']}
   Use OpenMP to compute in parallel.
   Example usage provided in comments below.
*/
{op_info['signature']} {{"""
        
        elif parallelism_model == "cuda":
            base_prompt = f"""/* {op_info['description']}
   Use CUDA to compute in parallel.
   The kernel is launched with appropriate grid/block dimensions.
*/
__global__ void {operator_name}_kernel(...) {{"""
        
        # 如果有推荐的优化策略，添加到prompt
        if strategies:
            strategy_section = self.format_strategies_for_prompt(strategies)
            full_prompt = f"""{base_prompt}

// Below are recommended optimization strategies for this operation:
/*
{strategy_section}
*/
"""
        else:
            full_prompt = base_prompt
        
        return {
            "problem_type": "blas",
            "language": "cpp",
            "name": f"00_blas_{operator_name}",
            "parallelism_model": parallelism_model,
            "prompt": full_prompt,
            "operator_name": operator_name,
            "optimization_strategies": strategies
        }
    
    def generate_all_prompts(self, opinfo_files: Dict[str, str]) -> List[Dict]:
        """
        为所有BLAS算子生成prompts
        
        Args:
            opinfo_files: {operator_name: opinfo_json_path}
            例如: {"gemm": "results/gemm_opinfo2.json", ...}
        """
        all_prompts = []
        parallelism_models = ["serial", "omp", "cuda", "kokkos"]
        
        for operator_name in self.blas_operators.keys():
            # 加载该算子的优化策略
            if operator_name in opinfo_files:
                opinfo = self.load_optimization_strategies(opinfo_files[operator_name])
                strategies = opinfo.get("final_strategies", [])
            else:
                strategies = []
            
            # 为每个并行模型生成prompt
            for model in parallelism_models:
                prompt = self.generate_prompt(operator_name, model, strategies)
                all_prompts.append(prompt)
        
        return all_prompts
    
    def save_prompts(self, prompts: List[Dict], output_path: str):
        """保存生成的prompts"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)
        print(f"✅ 已生成 {len(prompts)} 个prompts，保存到: {output_path}")


def main():
    generator = BLASPromptGenerator()
    
    # 定义每个算子的优化策略文件路径
    opinfo_files = {
        "gemm": "../results/gemm_opinfo2.json",
        "gemv": "../results/gemv_opinfo2.json",
        "axpby": "../results/axpby_opinfo2.json",
        # ... 其他算子
    }
    
    # 生成prompts
    prompts = generator.generate_all_prompts(opinfo_files)
    
    # 保存
    generator.save_prompts(prompts, "blas_prompts.json")
    
    print(f"""
下一步：
1. 运行代码生成：
   cd Morph
   python main.py  # 修改为使用 blas_prompts.json

2. 准备baseline实现（见下文方案）

3. 运行测试：
   cd Morph/drivers
   python run-all.py --input_json "../blas_code.json" --output "../blas_code_run.json"
""")

if __name__ == "__main__":
    main()
```

---

## 3. Baseline准备方案

### 3.1 方案A：直接使用OpenBLAS库（推荐）

**优点**：
- 最权威的参考实现
- 性能极优，适合作为性能基准
- 实现简单，维护成本低

**实现**：

**baseline.hpp**：
```cpp
#pragma once
#include <cblas.h>  // OpenBLAS头文件
#include <vector>

/* GEMM baseline using OpenBLAS */
void correctGemm(int M, int N, int K, double alpha, 
                 const double *A, const double *B, 
                 double beta, double *C) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, K, B, N, beta, C, N);
}

/* GEMV baseline using OpenBLAS */
void correctGemv(int M, int N, double alpha,
                 const double *A, const double *x,
                 double beta, double *y) {
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                M, N, alpha, A, N, x, 1, beta, y, 1);
}

/* AXPBY baseline using OpenBLAS */
void correctAxpby(int N, double alpha, const double *x,
                  double beta, double *y) {
    // OpenBLAS: y = alpha*x + beta*y
    cblas_dscal(N, beta, y, 1);      // y = beta*y
    cblas_daxpy(N, alpha, x, 1, y, 1); // y = alpha*x + y
}

// ... 其他7个算子的baseline
```

**编译时链接OpenBLAS**：
```python
# 在 cpp_driver_wrapper.py 中添加
COMPILER_SETTINGS = {
    "serial": {
        "CXX": "g++", 
        "CXXFLAGS": "-std=c++17 -O3 -I/usr/include/openblas -lopenblas"
    },
    # ...
}
```

### 3.2 方案B：简化版纯C++实现

**优点**：
- 无外部依赖
- 便于理解和修改
- 适合教学和研究

**缺点**：
- 性能可能不是最优
- 需要自己实现每个算子

**实现示例**（GEMM）：
```cpp
#pragma once
#include <vector>

void correctGemm(int M, int N, int K, double alpha,
                 const double *A, const double *B,
                 double beta, double *C) {
    // 简单的三重循环实现
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}
```

**推荐**：方案A（使用OpenBLAS）更合适，因为：
1. 你已经从OpenBLAS提取了优化策略
2. 性能对比更有说服力
3. 正确性验证更可靠

---

## 4. 驱动代码适配

### 4.1 创建BLAS驱动模板

**目录结构**：
```
Morph/drivers/cpp/benchmarks/blas/
├── 00_blas_gemm/
│   ├── baseline.hpp
│   ├── cpu.cc
│   ├── gpu.cu
│   └── kokkos.cc
├── 01_blas_gemv/
├── 02_blas_axpby/
├── 03_blas_dot/
├── 04_blas_swap/
├── 05_blas_copy/
├── 06_blas_nrm2/
├── 07_blas_asum/
├── 08_blas_iamax/
├── 09_blas_iamin/
└── README.md
```

### 4.2 GEMM驱动示例

**cpu.cc**：
```cpp
// Driver for 00_blas_gemm for Serial, OpenMP, MPI, and MPI+OpenMP
// /* Matrix-matrix multiplication: C = alpha*A*B + beta*C
//    A is MxK, B is KxN, C is MxN (row-major storage)
//    Example:
//    A = [[1,2],[3,4]], B = [[5,6],[7,8]]
//    C = A*B = [[19,22],[43,50]]
// */
// void gemm(int M, int N, int K, double alpha, const double *A,
//           const double *B, double beta, double *C) {

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "utilities.hpp"
#include "baseline.hpp"
#include "generated-code.hpp"

struct Context {
    std::vector<double> A, B, C;
    int M, N, K;
    double alpha, beta;
};

void reset(Context *ctx) {
    fillRand(ctx->A, -1.0, 1.0);
    fillRand(ctx->B, -1.0, 1.0);
    fillRand(ctx->C, -1.0, 1.0);
    ctx->alpha = 1.0;
    ctx->beta = 0.0;
    BCAST(ctx->A, DOUBLE);
    BCAST(ctx->B, DOUBLE);
    BCAST(ctx->C, DOUBLE);
}

Context *init() {
    Context *ctx = new Context();
    
    // 问题规模从 DRIVER_PROBLEM_SIZE 推导
    // 例如：DRIVER_PROBLEM_SIZE = 1<<20 (1M)
    // GEMM: M=N=K ≈ 100 (使得M*N*K ≈ 1M)
    ctx->M = ctx->N = ctx->K = static_cast<int>(std::cbrt(DRIVER_PROBLEM_SIZE));
    
    ctx->A.resize(ctx->M * ctx->K);
    ctx->B.resize(ctx->K * ctx->N);
    ctx->C.resize(ctx->M * ctx->N);
    
    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    gemm(ctx->M, ctx->N, ctx->K, ctx->alpha,
         ctx->A.data(), ctx->B.data(), ctx->beta, ctx->C.data());
}

void NO_OPTIMIZE best(Context *ctx) {
    correctGemm(ctx->M, ctx->N, ctx->K, ctx->alpha,
                ctx->A.data(), ctx->B.data(), ctx->beta, ctx->C.data());
}

bool validate(Context *ctx) {
    const int TEST_M = 64, TEST_N = 64, TEST_K = 64;
    
    std::vector<double> A(TEST_M * TEST_K);
    std::vector<double> B(TEST_K * TEST_N);
    std::vector<double> C_correct(TEST_M * TEST_N);
    std::vector<double> C_test(TEST_M * TEST_N);
    
    int rank;
    GET_RANK(rank);
    
    const size_t numTries = MAX_VALIDATION_ATTEMPTS;
    for (int trialIter = 0; trialIter < numTries; trialIter += 1) {
        // 设置输入
        fillRand(A, -1.0, 1.0);
        fillRand(B, -1.0, 1.0);
        fillRand(C_correct, -1.0, 1.0);
        C_test = C_correct;
        
        BCAST(A, DOUBLE);
        BCAST(B, DOUBLE);
        BCAST(C_correct, DOUBLE);
        BCAST(C_test, DOUBLE);
        
        // 计算正确结果
        correctGemm(TEST_M, TEST_N, TEST_K, 1.0,
                   A.data(), B.data(), 0.0, C_correct.data());
        
        // 计算测试结果
        gemm(TEST_M, TEST_N, TEST_K, 1.0,
             A.data(), B.data(), 0.0, C_test.data());
        SYNC();
        
        // 验证
        bool isCorrect = true;
        if (IS_ROOT(rank) && !fequal(C_correct, C_test, 1e-5)) {
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

**gpu.cu** - CUDA版本：
```cpp
// Driver for 00_blas_gemm for CUDA
// /* Matrix-matrix multiplication using CUDA
//    Store result in C.
//    Use CUDA to compute in parallel.
// */
// __global__ void gemm_kernel(int M, int N, int K, double alpha,
//                             const double *A, const double *B,
//                             double beta, double *C) {

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <vector>

#include "utilities.hpp"
#include "baseline.hpp"
#include "generated-code.hpp"

struct Context {
    std::vector<double> h_A, h_B, h_C;
    double *d_A, *d_B, *d_C;
    int M, N, K;
    double alpha, beta;
};

void reset(Context *ctx) {
    fillRand(ctx->h_A, -1.0, 1.0);
    fillRand(ctx->h_B, -1.0, 1.0);
    fillRand(ctx->h_C, -1.0, 1.0);
    
    cudaMemcpy(ctx->d_A, ctx->h_A.data(), ctx->M * ctx->K * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_B, ctx->h_B.data(), ctx->K * ctx->N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_C, ctx->h_C.data(), ctx->M * ctx->N * sizeof(double), cudaMemcpyHostToDevice);
}

Context *init() {
    Context *ctx = new Context();
    ctx->M = ctx->N = ctx->K = static_cast<int>(std::cbrt(DRIVER_PROBLEM_SIZE));
    
    ctx->h_A.resize(ctx->M * ctx->K);
    ctx->h_B.resize(ctx->K * ctx->N);
    ctx->h_C.resize(ctx->M * ctx->N);
    
    cudaMalloc(&ctx->d_A, ctx->M * ctx->K * sizeof(double));
    cudaMalloc(&ctx->d_B, ctx->K * ctx->N * sizeof(double));
    cudaMalloc(&ctx->d_C, ctx->M * ctx->N * sizeof(double));
    
    ctx->alpha = 1.0;
    ctx->beta = 0.0;
    
    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    dim3 blocks(DRIVER_KERNEL_BLOCKS);
    dim3 threads(DRIVER_KERNEL_THREADS);
    
    gemm_kernel<<<blocks, threads>>>(ctx->M, ctx->N, ctx->K, ctx->alpha,
                                     ctx->d_A, ctx->d_B, ctx->beta, ctx->d_C);
    cudaDeviceSynchronize();
    
    cudaMemcpy(ctx->h_C.data(), ctx->d_C, ctx->M * ctx->N * sizeof(double), cudaMemcpyDeviceToHost);
}

void NO_OPTIMIZE best(Context *ctx) {
    correctGemm(ctx->M, ctx->N, ctx->K, ctx->alpha,
                ctx->h_A.data(), ctx->h_B.data(), ctx->beta, ctx->h_C.data());
}

bool validate(Context *ctx) {
    // 类似cpu.cc的验证逻辑，但需处理GPU内存
}

void destroy(Context *ctx) {
    cudaFree(ctx->d_A);
    cudaFree(ctx->d_B);
    cudaFree(ctx->d_C);
    delete ctx;
}
```

---

## 5. 代码生成适配

### 5.1 修改 Morph/functions/

**新建**：`llmgenv4.py`（基于 llmgenv3.py）

**关键修改**：

```python
PROMPT_TEMPLATE = """Complete the C++ function {function_name}. Only write the body of the function {function_name}.

```cpp
{prompt}
```

Below are optimization strategies specifically recommended for this operation based on knowledge graph analysis:
{optimization_strategies}

Please implement the function efficiently using these strategies where applicable.
"""

def CodeGenv4(
    input_path: str = "blas_prompts.json",
    output_path: str = "blas_code.json",
    temperature: float = 0.2,
    model: str = "qwen-plus-2025-04-28"
) -> None:
    
    with open(input_path, "r") as f:
        prompts = json.load(f)
    
    outputs = []
    for prompt_obj in tqdm(prompts):
        # 提取优化策略
        strategies = prompt_obj.get("optimization_strategies", [])
        strategy_text = format_optimization_strategies(strategies)
        
        # 构建完整prompt
        full_prompt = PROMPT_TEMPLATE.format(
            function_name=prompt_obj["operator_name"],
            prompt=prompt_obj["prompt"],
            optimization_strategies=strategy_text
        )
        
        # 调用LLM
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=temperature,
            max_tokens=2048
        )
        
        code = response.choices[0].message.content
        code = postprocess(code)  # 提取代码块
        
        # 保存结果
        prompt_obj["outputs"] = [code]
        outputs.append(prompt_obj)
    
    with open(output_path, 'w') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
```

---

## 6. 性能评估与对比

### 6.1 评估维度

创建 `blas_analysis.py`：

```python
class BLASAnalysis:
    def analyze_results(self, code_run_path: str):
        """分析BLAS代码生成与测试结果"""
        with open(code_run_path, 'r') as f:
            results = json.load(f)
        
        metrics = {
            "by_operator": {},      # 按算子统计
            "by_parallelism": {},   # 按并行模型统计
            "overall": {}           # 总体统计
        }
        
        for result in results:
            operator = result["operator_name"]
            model = result["parallelism_model"]
            
            # 统计成功率
            build_success = result["outputs"][0].get("build_success", False)
            run_success = result["outputs"][0].get("run_success", False)
            correctness = result["outputs"][0].get("correctness", False)
            runtime = result["outputs"][0].get("runtime", 0.0)
            
            # ... 统计逻辑
        
        return metrics
    
    def compare_with_baseline(self, results: List[Dict]):
        """与OpenBLAS性能对比"""
        comparisons = []
        
        for result in results:
            # LLM生成代码的运行时间
            llm_time = result["outputs"][0].get("runtime", 0.0)
            
            # OpenBLAS baseline的运行时间（需要单独测试）
            # baseline_time = self.get_baseline_time(result["operator_name"])
            
            # speedup = baseline_time / llm_time
            # comparisons.append({
            #     "operator": result["operator_name"],
            #     "parallelism": result["parallelism_model"],
            #     "speedup": speedup
            # })
        
        return comparisons
    
    def analyze_strategy_adoption(self, code_run_path: str, opinfo_path: str):
        """分析生成代码是否采纳了推荐的优化策略"""
        # 1. 加载推荐的策略
        with open(opinfo_path, 'r') as f:
            opinfo = json.load(f)
        recommended_strategies = opinfo.get("final_strategies", [])
        
        # 2. 加载生成的代码
        with open(code_run_path, 'r') as f:
            code_results = json.load(f)
        
        adoption_analysis = []
        for result in code_results:
            generated_code = result["outputs"][0].get("code", "")
            
            # 3. 检查代码中是否包含策略关键词
            for strategy in recommended_strategies:
                keywords = self._extract_keywords(strategy)
                adopted = any(kw in generated_code for kw in keywords)
                
                adoption_analysis.append({
                    "operator": result["operator_name"],
                    "strategy": strategy["strategy_name"],
                    "adopted": adopted
                })
        
        return adoption_analysis
```

### 6.2 输出报告格式

**blas_evaluation_report.json**：
```json
{
    "overall_metrics": {
        "total_prompts": 40,
        "build_success_rate": 0.85,
        "run_success_rate": 0.95,
        "correctness_rate": 0.90
    },
    "by_operator": {
        "gemm": {
            "serial": {"build": true, "run": true, "correct": true, "time": 0.123},
            "omp": {"build": true, "run": true, "correct": true, "time": 0.045},
            "cuda": {"build": true, "run": true, "correct": true, "time": 0.012}
        },
        // ... 其他算子
    },
    "performance_comparison": {
        "gemm_omp_vs_baseline": {"speedup": 0.85, "note": "略慢于OpenBLAS"},
        "gemm_cuda_vs_baseline": {"speedup": 1.2, "note": "GPU加速有效"}
    },
    "strategy_adoption": {
        "循环展开": {"recommended": 5, "adopted": 3, "rate": 0.6},
        "分块优化": {"recommended": 8, "adopted": 6, "rate": 0.75}
    }
}
```

---

## 7. 完整工作流程

### 7.1 步骤清单

```bash
# ========== 步骤1：提取优化策略 ==========
cd /home/dgc/mjs/project/analyze_OB

# 为每个BLAS算子运行Operator_op2
for op in gemm gemv axpby dot swap copy nrm2 asum iamax iamin; do
    # 1. 准备算子的参考实现（如 KG/gemm.txt）
    # 2. 运行策略推荐
    python KG/Operator_op2.py --source KG/${op}.txt --output results/${op}_opinfo2.json
done

# ========== 步骤2：生成Morph prompts ==========
cd Morph
python blas_prompt_generator.py

# 输出: blas_prompts.json (10算子 × 4并行模型 = 40个prompts)

# ========== 步骤3：准备baseline实现 ==========
# 创建 drivers/cpp/benchmarks/blas/ 目录
# 为每个算子编写 baseline.hpp、cpu.cc、gpu.cu、kokkos.cc

# ========== 步骤4：生成代码 ==========
python main.py  # 调用 CodeGenv4

# 输出: blas_code.json

# ========== 步骤5：运行测试 ==========
cd drivers
python run-all.py \
    --input_json "../blas_code.json" \
    --output "../blas_code_run.json" \
    --include-models serial omp cuda kokkos \
    --build-timeout 60 \
    --run-timeout 180

# 输出: blas_code_run.json

# ========== 步骤6：分析结果 ==========
cd ..
python blas_analysis.py

# 输出: blas_evaluation_report.json
```

### 7.2 预期时间成本

- 步骤1（策略提取）：10个算子 × 2分钟 ≈ 20分钟
- 步骤2（Prompt生成）：< 1分钟
- 步骤3（准备baseline）：首次10算子 × 30分钟 ≈ 5小时（一次性）
- 步骤4（代码生成）：40个prompts × 5秒 ≈ 3分钟
- 步骤5（测试）：40个测试 × 30秒 ≈ 20分钟
- 步骤6（分析）：< 1分钟

**总计**：约6小时（首次），后续迭代约25分钟

---

## 8. 核心创新点与优势

### 8.1 与原Morph的区别

| 维度 | 原Morph | 整合后 |
|-----|---------|--------|
| **优化知识来源** | 无/LLM自主 | 从OpenBLAS真实代码中提取 |
| **策略推荐** | 通用硬件信息 | 针对特定算子的精准策略 |
| **评估维度** | 正确性+性能 | +策略采纳率分析 |
| **知识图谱** | 无 | 有完整的优化策略KG |

### 8.2 整合的价值

1. **知识驱动的代码生成**：
   - 不是让LLM凭空想象优化
   - 而是基于真实、验证过的优化策略

2. **闭环验证**：
   - 策略提取 → 代码生成 → 性能验证 → 策略有效性反馈

3. **可解释性**：
   - 知道LLM被推荐了哪些策略
   - 知道LLM是否采纳了这些策略
   - 知道采纳后的性能提升

4. **迁移性评估**：
   - 评估从OpenBLAS学到的策略能否泛化到其他实现
   - 评估LLM理解和应用优化策略的能力

---

## 9. 下一步行动计划

### 9.1 短期目标（1-2天）

**任务列表**：
- [ ] 创建 `blas_prompt_generator.py`
- [ ] 准备10个BLAS算子的参考实现（txt文件）
- [ ] 运行 `Operator_op2.py` 获取所有算子的优化策略
- [ ] 生成 `blas_prompts.json`

### 9.2 中期目标（3-5天）

**任务列表**：
- [ ] 创建 `drivers/cpp/benchmarks/blas/` 目录结构
- [ ] 为每个算子编写 baseline.hpp（使用OpenBLAS）
- [ ] 编写 cpu.cc、gpu.cu、kokkos.cc 驱动
- [ ] 配置编译选项（链接OpenBLAS）

### 9.3 长期目标（1周+）

**任务列表**：
- [ ] 创建 `llmgenv4.py`（整合优化策略的代码生成）
- [ ] 运行完整的生成-测试流程
- [ ] 创建 `blas_analysis.py`（结果分析）
- [ ] 生成评估报告
- [ ] 论文撰写/实验迭代

---

## 10. 潜在挑战与解决方案

### 10.1 挑战1：Baseline的准确性

**问题**：直接调用OpenBLAS可能过于优化，LLM难以匹敌

**解决方案**：
- 方案A：使用OpenBLAS的"参考实现"版本（reference BLAS）
- 方案B：自己实现简单但正确的baseline（如三重循环GEMM）
- 方案C：设置"性能目标"而非"超越baseline"（如达到baseline的50%即为成功）

### 10.2 挑战2：优化策略的表达

**问题**：从OpenBLAS提取的策略可能过于底层，LLM难以理解

**解决方案**：
- 在 `blas_prompt_generator.py` 中对策略进行"翻译"
- 将技术细节转换为更高层的指导
- 示例：
  - 原始："使用res0-res7八个寄存器累积2x2复数结果"
  - 翻译："使用多个临时变量在循环中累积结果，减少内存访问"

### 10.3 挑战3：不同并行模型的策略适配

**问题**：某些策略只适用于特定并行模型

**解决方案**：
- 在prompt生成时根据 `parallelism_model` 过滤策略
- 示例：
  - Serial/OMP：推荐"循环展开"、"缓存分块"
  - CUDA：推荐"共享内存优化"、"合并访存"
  - Kokkos：推荐"并行模式"、"View优化"

### 10.4 挑战4：测试规模与性能差异

**问题**：测试规模太小可能看不出优化效果

**解决方案**：
- 在 `problem-sizes.json` 中为BLAS算子设置更大的规模
- 多个规模测试：小（验证正确性）、中（观察趋势）、大（性能对比）
- 示例：
```json
{
    "00_blas_gemm": {
        "small": "(1<<10)",   // 32x32 验证用
        "medium": "(1<<14)",  // 128x128
        "large": "(1<<18)"    // 512x512 性能测试用
    }
}
```

---

## 11. 扩展方向

### 11.1 自动化Pipeline

创建端到端自动化脚本：

```bash
#!/bin/bash
# blas_full_pipeline.sh

OPERATORS=("gemm" "gemv" "axpby" "dot" "swap" "copy" "nrm2" "asum" "iamax" "iamin")

# 步骤1：提取所有算子的优化策略
echo "步骤1：提取优化策略..."
for op in "${OPERATORS[@]}"; do
    python ../KG/Operator_op2.py --source ../KG/${op}.txt --output ../results/${op}_opinfo2.json
done

# 步骤2：生成prompts
echo "步骤2：生成Morph prompts..."
python blas_prompt_generator.py

# 步骤3：生成代码
echo "步骤3：LLM生成代码..."
python functions/llmgenv4.py

# 步骤4：测试代码
echo "步骤4：运行测试..."
cd drivers
python run-all.py --input_json "../blas_code.json" --output "../blas_code_run.json"

# 步骤5：分析结果
echo "步骤5：分析结果..."
cd ..
python blas_analysis.py

echo "✅ Pipeline完成！查看 blas_evaluation_report.json"
```

### 11.2 迭代优化

**反馈循环**：
```
策略推荐 → 代码生成 → 性能测试 → 
  ↓ (如果性能不佳)
分析未采纳的策略 → 改进prompt → 重新生成
  ↓ (如果采纳但无效)
质疑策略有效性 → 更新知识图谱
```

---

## 12. 示例：GEMM的完整流程

### 12.1 输入（来自Operator_op2）

**gemm_opinfo2.json**（简化）：
```json
{
    "patterns_detected": [
        {"pattern_type": "prep.index_pointer_init", "name": "索引初始化"},
        {"pattern_type": "core.mm_microkernel", "name": "矩阵乘法微内核"},
        {"pattern_type": "core.tiled_loop", "name": "分块循环"}
    ],
    "final_strategies": [
        {
            "strategy_name": "2x2分块矩阵乘法微内核",
            "level": "algorithm",
            "overview": "通过将大矩阵划分为2x2小块...",
            "when_to_use": "大规模矩阵运算...",
            "key_actions": ["分块循环", "微内核调用", "寄存器累积"],
            "optimization_context": {
                "core_patterns": ["core.mm_microkernel", "core.tiled_loop"],
                "contextual_patterns": {}
            },
            "score": 0.85
        },
        {
            "strategy_name": "循环展开与FMA融合",
            "level": "algorithm",
            "score": 0.78
        }
    ]
}
```

### 12.2 生成的Prompt

**blas_prompts.json**（GEMM-OMP条目）：
```json
{
    "problem_type": "blas",
    "language": "cpp",
    "name": "00_blas_gemm",
    "parallelism_model": "omp",
    "operator_name": "gemm",
    "prompt": "#include <omp.h>\\n#include <vector>\\n\\n/* Matrix-matrix multiplication: C = alpha*A*B + beta*C\\n   Use OpenMP to compute in parallel.\\n*/\\nvoid gemm(int M, int N, int K, double alpha, const double *A, const double *B, double beta, double *C) {",
    "optimization_strategies": [
        {
            "strategy_name": "2x2分块矩阵乘法微内核",
            "overview": "通过将大矩阵划分为2x2小块，提升缓存局部性...",
            "key_actions": ["外层循环按2x2分块", "内层微内核完全展开", "寄存器累积中间结果"],
            "score": 0.85
        }
    ]
}
```

### 12.3 LLM生成的代码

**blas_code.json**（GEMM-OMP结果）：
```json
{
    "name": "00_blas_gemm",
    "parallelism_model": "omp",
    "outputs": [
        "void gemm(int M, int N, int K, double alpha, const double *A, const double *B, double beta, double *C) {\n    // 2x2分块实现\n    #pragma omp parallel for collapse(2)\n    for (int i = 0; i < M; i += 2) {\n        for (int j = 0; j < N; j += 2) {\n            // 微内核：2x2块\n            double c00=0, c01=0, c10=0, c11=0;\n            for (int k = 0; k < K; k++) {\n                // 寄存器累积\n                double a0 = A[i*K+k];\n                double a1 = A[(i+1)*K+k];\n                double b0 = B[k*N+j];\n                double b1 = B[k*N+j+1];\n                c00 += a0*b0; c01 += a0*b1;\n                c10 += a1*b0; c11 += a1*b1;\n            }\n            // 写回\n            C[i*N+j] = alpha*c00 + beta*C[i*N+j];\n            C[i*N+j+1] = alpha*c01 + beta*C[i*N+j+1];\n            C[(i+1)*N+j] = alpha*c10 + beta*C[(i+1)*N+j];\n            C[(i+1)*N+j+1] = alpha*c11 + beta*C[(i+1)*N+j+1];\n        }\n    }\n}"
    ]
}
```

### 12.4 测试结果

**blas_code_run.json**（GEMM-OMP结果）：
```json
{
    "name": "00_blas_gemm",
    "parallelism_model": "omp",
    "outputs": [
        {
            "code": "...",
            "build_success": true,
            "build_message": "Compilation successful",
            "run_success": true,
            "correctness": true,
            "runtime": 0.087,
            "baseline_runtime": 0.065,
            "speedup": 0.75
        }
    ]
}
```

---

## 13. 总结

### 13.1 整合的技术路线

```
OpenBLAS源码分析
  ↓ (Operator_op2)
优化策略知识图谱
  ↓ (blas_prompt_generator)
知识驱动的Prompt
  ↓ (CodeGenv4)
LLM生成优化代码
  ↓ (run-all.py)
自动化测试验证
  ↓ (blas_analysis)
策略有效性评估
```

### 13.2 关键文件清单

**需要创建的新文件**：
1. `Morph/blas_prompt_generator.py` - Prompt生成器
2. `Morph/functions/llmgenv4.py` - 代码生成（整合策略）
3. `Morph/blas_analysis.py` - 结果分析
4. `Morph/drivers/cpp/benchmarks/blas/` - BLAS基准目录
5. `Morph/blas_full_pipeline.sh` - 自动化脚本

**需要修改的文件**：
1. `Morph/main.py` - 添加对 `llmgenv4` 的调用
2. `Morph/drivers/cpp/cpp_driver_wrapper.py` - 添加OpenBLAS编译选项

### 13.3 预期成果

1. **定量评估**：
   - LLM在有/无优化策略指导下的代码生成质量对比
   - 不同并行模型下的性能差异
   - 策略采纳率与性能提升的相关性

2. **定性分析**：
   - 哪些优化策略更容易被LLM理解和应用
   - OpenBLAS策略在不同并行模型间的迁移性
   - LLM生成代码的常见问题模式

3. **知识反馈**：
   - 验证知识图谱中策略的有效性
   - 发现新的优化模式
   - 改进策略推荐算法

---

## 14. 建议的实施优先级

### Phase 1（核心功能，1周）
1. 创建 `blas_prompt_generator.py`
2. 准备3个核心算子（GEMM、GEMV、AXPBY）
3. 编写对应的baseline和驱动
4. 跑通完整流程

### Phase 2（完善扩展，1周）
1. 扩展到全部10个算子
2. 创建 `blas_analysis.py`
3. 优化prompt策略表达
4. 多轮迭代测试

### Phase 3（深度分析，2周）
1. 策略采纳率分析
2. 性能深度对比
3. 不同LLM模型对比（Qwen、GPT等）
4. 撰写技术报告

---

这个整合方案将你的**优化策略知识图谱**与**代码生成验证框架**有机结合，形成了一个完整的"知识→生成→验证→反馈"的闭环系统。
