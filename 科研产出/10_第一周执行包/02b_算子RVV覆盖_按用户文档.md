# 算子 RVV 覆盖权威表（依据用户《1.1 算子提取工具》整理）

> 来源：`D:\myobsidian\大工程项目\算子迁移\1.1算子提取工具.md` §3.2 与 §7.0.3 的表格（用户自己维护的"是否有 rvv"列），与 Lead 的自动审计交叉核对。
> 结论：**45 个算子里 24 个有 RVV 版本、21 个没有**。自动审计（按 `_rvv.c` 里是否有 intrinsic）会漏判/误判，**以本表为准**。

## 1. 有 RVV 版本（24 个）

| 算子 | 层级 | 备注（来自用户文档） |
|---|---|---|
| hemv | L2 | 复数 Hermitian 矩阵向量，接口需多提取 |
| amin / amax | L1 | d 开头 |
| axpy | L1 | |
| asum | L1 | |
| axpby | L1 | |
| copy | L1 | |
| dot | L1 | |
| gemm | L3 | 核心 kernel 多个 |
| gemv | L2 | 接口提取特殊 |
| idamin / idamax / idmax / idmin | L1 | 索引类 |
| nrm2 | L1 | |
| swap | L1 | |
| symm | L3 | **`_k.c` 里是 dgemm_kernel 属正常**（对称矩阵乘复用 gemm 核），不是错配 |
| symv | L2 | kernel 调用 dcopy_k / SYMCOPY_L / dgemv_n / dgemv_t |
| trmm | L3 | 文档标注"暂时不要"（迁移侧），但 RVV 版本存在，KG 可用 |
| zdot | L1 | 复数点积，接口 zdotu |
| max / min | L1 | |
| rot | L1 | |
| scal | L1 | |

## 2. 无 RVV 版本（21 个）

hbmv、hemm、her、her2、herk、her2k、hpmv、ger、gesv、syr、syr2、syrk、syr2k、tpmv、tpsv、trmv、trsm、trsv、spmv、spr、spr2。

> 这 21 个只能做 C0/C1 对比或作为"S2 留出算子"（不注入 RVV 知识），不能当 RVV 知识源。

## 3. 与本项目 12 算子清单的对应

| 角色 | 算子 | 是否有 RVV | 结论 |
|---|---|---|---|
| S1 进 KG | daxpy, dasum, ddot, dgemv, dgemm, dtrmm | 有（6/6） | 可用 |
| S2 留出 | dscal, dnrm2, daxpby, dsymv, chemv, dsymm | 有（6/6） | 可用；dsymm 正常 |

> 原计划里的 `dger/dtrmv/dsyrk` 属于"无 RVV"名单，**不能进 RVV 知识源**，已从 12 算子清单移除。
> `chemv`/`dsymv` 的 `_k.c` 依赖其他 kernel（`ccopy_k`/`dcopy_k`、`cgemv_n/t`/`dgemv_n/t`、`SYMCOPY_L` 等），构建 `generic.c` 时需把这些 helper 一起放进 `input/`（用户文档 §7.0.3 已列出依赖）。

## 4. 对 KG 抽取的影响

- 有 RVV 的 24 个算子可以做"generic C ↔ RVV"差分抽取（S2 阶段），得到"策略实例"。
- 无 RVV 的 21 个算子只能做 C 侧特征抽取与 C0/C1 对比，**不能生成 RVV 差分知识**。
- KG 里每条策略要标 `arch_evidence: ["x86","rvv"]` 或 `["x86"]`，检索时按目标架构过滤；这正好支撑论文的"跨架构可迁移性"分析。
