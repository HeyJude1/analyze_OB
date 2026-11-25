# OpOptBench-12 任务清单（按配对审计修正版）

> 2026-11-12｜由 `tools/gen_optbench_list.py` 生成｜依据 `op_pairing_audit.csv` 与 `kg_adequacy_audit.md`

## 一句话

原计划的 12 个算子里 `dger/dtrmv/dsyrk` 的 `_rvv.c` 没有向量指令，不能当 RVV 知识源；本清单换成审计后**配对正常 + 有 RVV 指令 + 有日志**的算子。**S1（6 个）进 KG，S2（6 个）只在评测出现**。


## 任务表

| # | 算子 | 层级 | 角色 | 目录 | 输入 C | RVV 专家实现 | 尺寸 | QEMU 日志 geomean* | 备注 |
|---|---|---|---|---|---|---|---|---:|---|
| 1 | daxpy | L1 | S1 进 KG | 13_daxpy_op | daxpy_k.c | daxpy_k_rvv.c | 1024/16384/1048576 | 1.2134 | — |
| 2 | dasum | L1 | S1 进 KG | 11_dasum_op | dasum_k.c | dasum_k_rvv.c | 1024/16384/1048576 | 1.303 | — |
| 3 | ddot | L1 | S1 进 KG | 15_ddot_op | ddot_k.c | ddot_k_rvv.c | 1024/16384/1048576 | 1.1584 | — |
| 4 | dgemv | L2 | S1 进 KG | 17_dgemv_op | dgemv_k.c | dgemv_k_rvv.c | 256/1024/4096 | 2.5729 | — |
| 5 | dgemm | L3 | S1 进 KG | 16_dgemm_op5 | dgemm_k.c | dgemm_k_rvv.c | 64/128/512 | 2.0351 | — |
| 6 | dtrmm | L3 | S1 进 KG | 37_dtrmm_op1 | dtrmm_k.c | dtrmm_k_rvv.c | 64/128/512 | 1.374 | — |
| 7 | dscal | L1 | S2 留出 | 24_dscal_op | dscal_k.c | dscal_k_rvv.c | 1024/16384/1048576 | 1.2473 | — |
| 8 | dnrm2 | L1 | S2 留出 | 22_dnrm2_op | dnrm2_k.c | dnrm2_k_rvv.c | 1024/16384/1048576 | 0.5574 | — |
| 9 | daxpby | L1 | S2 留出 | 12_daxpby_op | daxpby_k.c | daxpby_k_rvv.c | 1024/16384/1048576 | 1.2022 | — |
| 10 | dsymv | L2 | S2 留出 | 30_dsymv_op | dsymv_k.c | dsymv_k_rvv.c | 256/1024/4096 | 1.0077 | — |
| 11 | chemv | L2 | S2 留出 | 03_chemv_op | chemv_k.c | chemv_k_rvv.c | 256/1024/4096 | 1.0206 | — |
| 12 | dsymm | L3 | S2 留出 | 29_dsymm_op1 | dsymm_k.c | dsymm_k_rvv.c | 64/128/512 | 2.4542 | **无需替换**：`_k.c` 里是 dgemm_kernel 属正常（对称矩阵乘复用 gemm 核，用户文档确认 symm 有 RVV）；pair_status=mispair_k(dgemm_kernel) 为审计误报 |

> \* QEMU 日志的 `Migration/Ref` 几何平均只用于确认「语料可用、函数能跑」，**不是性能数字**，不能进论文性能表。


## 每个任务要准备的文件

```text
OpOptBench/tasks/<op>/
  task.yaml          # 签名、语义不变量、尺寸、flops、容差
  input/generic.c    # 由 <op>_k.c 清洗（去 common.h/BLASLONG 依赖，保留接口）
  ref/lib_ref.c      # x86: 系统 OpenBLAS；RISC-V: 仓库预编译库或 <op>_k_rvv.c
  driver/bench.c     # 改造自 _file/benchmark/_<op>_dgc.c，输出 JSONL
  tests/gen_input.c  # D1-D4 四个隐藏分布
  tests/tolerance    # 参考 vs 参考自校准
```


## S1 / S2 的用途

- **S1（6 个）**：它们的优化实现进入 KG，用来验证「知识确实抓到了」；对应 E1 金标与 E2/E3 的 in-domain 结果。

- **S2（6 个）**：不进 KG，只在评测出现，是**跨算子泛化**的主结果集；匹配论文第 5 章的「未见算子」表。

- dsymm 若人工确认不可用，S2 的 L3 位置用 dcopy/drot 顶替（保持 12 个总数，层级比例略变，论文中如实说明）。


## 下一步（W1 内完成）

1. 人工确认 dsymm 的 `_k.c` 是否可用（看它是否只是复用 dgemm kernel）。
2. ~~写 `clean_generic.py`~~ **已完成**：12 个 `generic.c` 已生成，10/12 可独立编译；`chemv`/`dsymv` 需补 helper kernel（`ccopy_k`/`dcopy_k`、`cgemv_n/t`/`dgemv_n/t`），或把该算子的 helper 文件一起放进 `input/`。
3. 先在 x86 编 C0/C1，再交叉编译到 RISC-V，确认 12/12 能跑通。
4. 用 `_<op>_dgc.c` 改造 driver，输出 JSONL；容差按参考 vs 参考自校准。

## 已完成的首个冒烟结果（daxpy）

| 配置 | n=16384 | n=1048576 |
|---|---:|---:|
| C0 `-O2` | 0.00415 ms | 0.271 ms |
| C1 `-O3 -march=native` | 0.00171 ms | 0.119 ms |
| C1/C0 | **2.4×** | **2.3×** |

正确性 maxdiff=0。→ 说明：① 清洗后的 generic.c 可独立编译运行；② 编译器单独就能带来 2.4×，主指标必须报 **S_C1**。

