# 算子 C/RVV 配对校验报告（自动生成）

> 数据源：`C:\Users\MSI\Desktop\cx_operator_tool\operator_extract\openblas_newtest1`｜生成脚本：`tools/op_pairing_audit.py`

## 一句话结论

共审计 **45** 个算子：配对正常 **22**，配对错误 **5**，有 RVV 文件但**无向量指令** **17**，无 `_k.c` **1**。**只有配对正常的算子才能进实验；无向量指令的只能做 C0/C1 对比，不能当 RVV 知识源。**


## 状态分布

| 状态 | 数量 | 含义 |
|---|---:|---|
| ok | 22 | C/RVV 配对正确，可进实验（x86 + RISC-V 双平台） |
| rvv_file_no_intrinsics | 17 | 有 `_k_rvv.c` 但文件里没有 RVV intrinsic/`vsetvl` → 只能做 C0/C1 对比，不能当 RVV 知识源 |
| mispair_k(caxpy_k) | 1 | 配对错误：`_k` 或 `_rvv` 文件里的函数名不属于该算子 |
| mispair_k(cgemm_kernel_n) | 1 | 配对错误：`_k` 或 `_rvv` 文件里的函数名不属于该算子 |
| no_k_file | 1 | 没有找到 `_k.c` 文件（目录结构异常） |
| mispair_k(dgetrf_single) | 1 | 配对错误：`_k` 或 `_rvv` 文件里的函数名不属于该算子 |
| mispair_k(dgemm_kernel) | 1 | 配对错误：`_k` 或 `_rvv` 文件里的函数名不属于该算子 |
| mispair_k(?) | 1 | 配对错误：`_k` 或 `_rvv` 文件里的函数名不属于该算子 |

## 配对错误的算子（必须人工处理）

| 算子 | 目录 | k_file 里实际函数 | rvv_file 里实际函数 |
|---|---|---|---|
| chbmv | 01_chbmv_op | caxpy_k | caxpy_k_rvv |
| chemm | 02_chemm_op | cgemm_kernel_n | cgemm_kernel_n_rvv |
| dgesv | 19_dgesv_op | dgetrf_single | dgetrf_single_rvv |
| dsymm | 29_dsymm_op1 | dgemm_kernel | dgemm_kernel_rvv |
| zdotu | 45_zdotu_op |  |  |

## 可用算子清单（配对正常 + 有日志）

| 算子 | 层级建议 | RVV | 日志通过率 | geomean(Migration/Ref, size≥32) |
|---|---|---|---:|---:|
| chemv | L1/L2/L3 待定 | 有 | 1.0 | 1.0206 |
| damax | L1/L2/L3 待定 | 有 | 1.0 | 1.0285 |
| damin | L1/L2/L3 待定 | 有 | 1.0 | 1.0666 |
| dasum | L1/L2/L3 待定 | 有 | 1.0 | 1.303 |
| daxpby | L1/L2/L3 待定 | 有 | 1.0 | 1.2022 |
| daxpy | L1/L2/L3 待定 | 有 | 1.0 | 1.2134 |
| dcopy | L1/L2/L3 待定 | 有 | 1.0 | 1.2014 |
| ddot | L1/L2/L3 待定 | 有 | 1.0 | 1.1584 |
| dgemm | L1/L2/L3 待定 | 有 | 1.0 | 2.0351 |
| dgemv | L1/L2/L3 待定 | 有 | 1.0 | 2.5729 |
| dmax | L1/L2/L3 待定 | 有 | 1.0 | 1.0501 |
| dmin | L1/L2/L3 待定 | 有 | 1.0 | 1.0517 |
| dnrm2 | L1/L2/L3 待定 | 有 | 1.0 | 0.5574 |
| drot | L1/L2/L3 待定 | 有 | 1.0 | 1.1795 |
| dscal | L1/L2/L3 待定 | 有 | 1.0 | 1.2473 |
| dswap | L1/L2/L3 待定 | 有 | 1.0 | 1.1804 |
| dsymv | L1/L2/L3 待定 | 有 | 1.0 | 1.0077 |
| dtrmm | L1/L2/L3 待定 | 有 | 1.0 | 1.374 |
| idamax | L1/L2/L3 待定 | 有 | 1.0 | 0.187 |
| idamin | L1/L2/L3 待定 | 有 | 1.0 | 0.2042 |
| idmax | L1/L2/L3 待定 | 有 | 1.0 | 0.181 |
| idmin | L1/L2/L3 待定 | 有 | 1.0 | 0.1805 |
