# 10 第一周执行包 · 今天就能开始

> 生成日期：2026-11-12｜Lead 直接执行产出（不是建议清单，是真实跑出来的结果）
> 用途：把"第一件事做什么"变成可勾选的动作；所有文件都在本目录或 `tools/`。

## 0. 今天（D1）按顺序做这 3 件

| 顺序 | 动作 | 预计 | 产出/判定 |
|---|---|---|---|
| 1 | **上板跑 30 分钟环境确认**：把 `tools/riscv_env_check.sh` scp 到板子运行（或 `08b` §1.2 的 `board_probe.sh`），重点看板卡型号、`nproc`、**VLEN**、频率档位、是否已有 OpenBLAS | 30 min | `board_report.txt`；VLEN=256 → 按主线；128 → 改 zvl128b；无 `v` → 性能全留 x86 |
| 2 | **看配对审计结果**：打开 `10_第一周执行包/pairing_audit.md` 与 `op_pairing_audit.csv`，确认 22 个可用算子、5 个配对错误、17 个"有 RVV 文件但无向量指令" | 15 min | 确认下方 `02_OpOptBench-12任务清单.md` 里的 12 个算子（第 12 个 dsymm 需人工拍板） |
| 3 | **看 KG 体检结果**：打开 `10_第一周执行包/kg_adequacy_audit.md`，按 `03_Schema_v1.5补丁清单.md` 启动 2 天补丁 | 15 min | 决定"先补 3 个缺口再实验"，不是重做 Schema |

> 三件事没做完，不要开始写 `kg_to_prompt.py`；否则后面实验会把"Schema 缺口"误判成"KG 方法无效"。

## 1. 本目录文件

| 文件 | 内容 |
|---|---|
| `pairing_audit.md` / `op_pairing_audit.csv` | **45 个算子 C/RVV 配对审计**（22 可用、5 错配、17 无向量指令） |
| `kg_adequacy_audit.md` | **现有 KG 建模体检**：206 条策略、805 条关系、4 类关系；10 个 competency questions 的能答/不能答 |
| `02_OpOptBench-12任务清单.md` | 修正后的 12 个实验算子（按审计结果，不是原计划的 12 个） |
| `03_Schema_v1.5补丁清单.md` | 2 天补丁：硬件 Profile、可执行条件、模板+证据+溯源；含 YAML/JSON 样例 |
| 生成脚本 | `../tools/op_pairing_audit.py`、`../tools/kg_adequacy_audit.py`、`../tools/riscv_env_check.sh` |

## 2. 审计改写了原计划的两个地方

1. **原计划的 12 算子不可行**：`dger`、`dtrmv`、`dsyrk` 属于**无 RVV 版本**的 21 个算子（用户《1.1 算子提取工具》§3.2/§7.0.3 已列明），不能当 RVV 知识源；`dsymm` 的 `_k.c` 里是 dgemm kernel **属正常**（对称矩阵乘复用 gemm 核），审计的 mispair 是误报。→ 12 个算子已按用户文档确认，见 `02_OpOptBench-12任务清单.md` 与 `02b_算子RVV覆盖_按用户文档.md`。
2. **KG 不是"建模错了"，而是"知识清单，不可执行"**：206 条策略里只有 28% 有参数，硬件描述只有 0.5% 含数值，关系只有 4 类且无属性，`CONFLICTS_WITH`/`COMPOSES_WITH` 为 0。→ 先打 v1.5 补丁（2 天），再开实验；补丁清单见 `03`。

## 3. 与主计划的关系

- 本包是 [`08_毕设冲刺执行方案.md`](../08_毕设冲刺执行方案.md) 的 W1 具体化；W1 结束时应交付：`board_report.txt`、12 算子清单确认、KG v1.5 冻结。
- [历史 K1 执行细节](../_archive/2026-11-19/08b_RISC-V真机实验协议.md)和[现行实验口径](../03_实验与验收.md)。

## 4. 执行记录（Lead 2026-11-12 已完成，可直接复核）

| 动作 | 结果 |
|---|---|
| 配对审计（45 算子） | **22 可用 / 5 配对错误 / 17 有 RVV 文件但无向量指令 / 1 无 k 文件**；原计划 12 算子不可行，已修正 |
| KG 体检（206 策略/805 关系） | 4 类关系有实例（声明 12）；28% 策略有参数；0.5% 硬件描述含数值；`CONFLICTS_WITH`/`COMPOSES_WITH` 为 0 |
| 清洗 `generic.c`（12 算子） | **10/12 可独立编译**（x86 gcc 14.2）；`chemv`/`dsymv` 需补 helper kernel（`ccopy_k`/`dcopy_k`、`cgemv_n/t`/`dgemv_n/t`） |
| daxpy 端到端冒烟 | 编译 + 正确性（maxdiff=0）+ 计时全部通过：**C0(-O2) 0.00415 ms，C1(-O3 -march=native) 0.00171 ms → 编译器单独带来 2.4×**（n=16384；n=1M 为 0.271→0.119 ms，2.3×） |
| Schema v1.5 模板 | `kg_v1.5/hardware_profiles.yaml`、`condition_vocab.yaml` 已生成，可直接填 |

> 这 4 条结果意味着：**"能不能跑实验"已经不是风险；风险只剩"KG 补丁是否到位"和"真机环境是否正常"**。而 daxpy 的 C1/C0=2.4× 直接说明主指标必须用 **S_C1（相对编译器）**，不能只报相对输入代码的加速比。

