# OpenBLAS参考内核选择说明

本目录包含从OpenBLAS GENERIC kernel中选择的10个BLAS算子参考实现，用于Morph项目的代码生成基准测试。

## 算子列表与说明

| 序号 | 算子 | 文件名 | 原始文件 | 行数 | 功能描述 | 选择理由 |
|-----|------|--------|---------|------|---------|---------|
| 1 | **GEMM** | 01_gemm.c | dgemm_small_kernel_b0_nn.clean.c | 17 | 矩阵-矩阵乘法<br>C = α·A·B + β·C | 简洁的三重循环实现<br>适合作为baseline |
| 2 | **GEMV** | 02_gemv.c | dgemv_n.clean.c | 25 | 矩阵-向量乘法<br>y = α·A·x + y | 标准的两层循环<br>支持stride访问 |
| 3 | **AXPBY** | 03_axpby.c | daxpby_k.clean.c | 55 | 向量缩放与加法<br>y = α·x + β·y | 包含特殊情况处理<br>体现分支优化 |
| 4 | **DOT** | 04_dot.c | ddot_k.clean.c | 16 | 向量点积<br>result = x·y | 最简单的归约操作<br>易于测试 |
| 5 | **SWAP** | 05_swap.c | dswap_k.clean.c | 18 | 向量交换<br>swap(x, y) | 简单的内存操作<br>测试数据移动 |
| 6 | **COPY** | 06_copy.c | dcopy_k.clean.c | 15 | 向量复制<br>y = x | 最基础的内存拷贝<br>baseline参考 |
| 7 | **NRM2** | 07_nrm2.c | dnrm2_k.clean.c | 31 | 向量2-范数<br>‖x‖₂ | 涉及平方和与开方<br>数值稳定性考验 |
| 8 | **ASUM** | 08_asum.c | dasum_k.clean.c | 15 | 向量绝对值和<br>Σ\|x_i\| | 简单的归约操作<br>测试abs处理 |
| 9 | **IAMAX** | 09_iamax.c | idamax_k.clean.c | 24 | 最大绝对值索引<br>argmax(\|x_i\|) | 归约+索引追踪<br>测试条件判断 |
| 10 | **IAMIN** | 10_iamin.c | idamin_k.clean.c | 24 | 最小绝对值索引<br>argmin(\|x_i\|) | 与IAMAX对偶<br>完善测试集 |

## 算子功能详解

### Level 1 BLAS（向量-向量操作）

#### 1. DOT - 向量点积
```c
double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
```
- **输入**：两个向量 x、y
- **输出**：标量 x·y = Σ(x_i * y_i)
- **计算复杂度**：O(n)
- **关键特性**：归约操作，需要处理stride访问

#### 2. AXPBY - 向量缩放与加法
```c
int daxpby_k(BLASLONG n, double alpha, double *x, BLASLONG inc_x, 
             double beta, double *y, BLASLONG inc_y)
```
- **输入**：向量x、y，标量α、β
- **输出**：y = α·x + β·y
- **计算复杂度**：O(n)
- **关键特性**：特殊情况优化（α=0、β=0、β=1）

#### 3. SWAP - 向量交换
```c
int dswap_k(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, 
            double dummy3, double *x, BLASLONG inc_x, 
            double *y, BLASLONG inc_y, double *dummy, BLASLONG dummy2)
```
- **输入**：向量x、y
- **输出**：交换x和y的内容
- **计算复杂度**：O(n)
- **关键特性**：in-place操作，需要临时变量

#### 4. COPY - 向量复制
```c
int dcopy_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
```
- **输入**：向量x
- **输出**：y = x
- **计算复杂度**：O(n)
- **关键特性**：最基础的内存拷贝

#### 5. NRM2 - 向量2-范数
```c
double dnrm2_k(BLASLONG n, double *x, BLASLONG inc_x)
```
- **输入**：向量x
- **输出**：‖x‖₂ = sqrt(Σx_i²)
- **计算复杂度**：O(n)
- **关键特性**：数值稳定性（避免溢出/下溢）

#### 6. ASUM - 向量绝对值和
```c
double dasum_k(BLASLONG n, double *x, BLASLONG inc_x)
```
- **输入**：向量x
- **输出**：Σ|x_i|
- **计算复杂度**：O(n)
- **关键特性**：归约操作

#### 7. IAMAX - 最大绝对值索引
```c
BLASLONG idamax_k(BLASLONG n, double *x, BLASLONG inc_x)
```
- **输入**：向量x
- **输出**：argmax_i(|x_i|)
- **计算复杂度**：O(n)
- **关键特性**：归约+索引追踪

#### 8. IAMIN - 最小绝对值索引
```c
BLASLONG idamin_k(BLASLONG n, double *x, BLASLONG inc_x)
```
- **输入**：向量x
- **输出**：argmin_i(|x_i|)
- **计算复杂度**：O(n)
- **关键特性**：与IAMAX对偶

### Level 2 BLAS（矩阵-向量操作）

#### 9. GEMV - 矩阵-向量乘法
```c
int dgemv_n(BLASLONG m, BLASLONG n, BLASLONG dummy1, double alpha, 
            double *a, BLASLONG lda, double *x, BLASLONG inc_x, 
            double *y, BLASLONG inc_y, double *buffer)
```
- **输入**：矩阵A (m×n)、向量x、标量α
- **输出**：y = α·A·x + y
- **计算复杂度**：O(mn)
- **关键特性**：矩阵列访问，缓存局部性

### Level 3 BLAS（矩阵-矩阵操作）

#### 10. GEMM - 矩阵-矩阵乘法
```c
int dgemm_small_kernel_b0_nn(BLASLONG M, BLASLONG N, BLASLONG K, 
                             double *A, BLASLONG lda, double alpha, 
                             double *B, BLASLONG ldb, 
                             double *C, BLASLONG ldc)
```
- **输入**：矩阵A (M×K)、矩阵B (K×N)、标量α
- **输出**：C = α·A·B
- **计算复杂度**：O(MNK)
- **关键特性**：三重循环，分块优化的核心

## 代码特点分析

### 优点
1. **简洁易懂**：所有实现都是15-55行，逻辑清晰
2. **纯C实现**：无汇编、无SIMD，适合LLM学习
3. **通用性好**：GENERIC版本，不依赖特定硬件
4. **正确性高**：OpenBLAS官方实现，久经考验
5. **包含优化痕迹**：虽然简洁，但能看到分支优化、stride处理等技巧

### 适合作为参考实现的原因
1. **可测试性强**：函数签名明确，易于编写测试用例
2. **优化空间大**：这些是"小规模"或"通用"版本，有很大的优化空间（分块、向量化、并行化）
3. **教学价值高**：代码简洁，适合LLM学习和改进
4. **覆盖全面**：涵盖Level 1/2/3 BLAS，计算模式多样

## 下一步操作

### 1. 检查代码完整性
```bash
cd openblas_reference_kernels
for file in *.c; do
    echo "=== $file ==="
    head -5 "$file"
    echo "..."
done
```

### 2. 提取核心函数（去除预处理指令）
部分文件开头有 `# 行号 "路径"` 的预处理标记，可以选择保留或删除。

### 3. 运行Operator_op2提取优化策略
```bash
cd /home/dgc/mjs/project/analyze_OB
for i in {01..10}; do
    file="openblas_reference_kernels/${i}_*.c"
    operator=$(basename "$file" | sed 's/[0-9]*_//;s/\.c$//')
    python KG/Operator_op2.py --source "$file" --output "results/${operator}_opinfo2.json"
done
```

### 4. 准备Morph的baseline和驱动
基于这些参考实现，为每个算子编写：
- `baseline.hpp`（调用这些函数或重新实现）
- `cpu.cc`（Serial/OMP/MPI驱动）
- `gpu.cu`（CUDA驱动）

## 预期效果

使用这10个算子可以覆盖：
- **归约操作**：DOT、NRM2、ASUM、IAMAX、IAMIN
- **逐元素操作**：AXPBY、SWAP、COPY
- **矩阵运算**：GEMV、GEMM

这些算子的优化策略包括：
- 循环展开
- 缓存分块
- SIMD向量化
- 并行归约
- 内存访问优化
- 分支预测优化

非常适合评估LLM在高性能计算代码生成方面的能力！
