# 现有知识图谱建模体检报告（自动审计）

> 数据源：`C:\Users\MSI\Desktop\开题\开题\analyze_OB\20251114_153700`｜生成脚本：`tools/kg_adequacy_audit.py`｜审计日期：2026-11-12

## 一句话结论

**当前 KG 不是「建模错了」，而是「建成了知识清单，还不是可执行的知识库」。** 它已经能回答「某算子有哪些优化策略、大致针对什么硬件」，但**不能**回答「这个策略在当前硬件/当前代码条件下是否适用、参数取多少、怎么落地、收益证据是什么、和哪些策略冲突」——而这四件事恰好是 B3（完整 KG）相对 B2（扁平 RAG）唯一可能产生增量的地方。**所以：先做 2 天 Schema v1.5 补丁，再开实验，不要重做 v2，也不要直接开实验。**

## 1. 规模

| 指标 | 数值 |
|---|---:|
| 分析文件 | 4 |
| 算子 | axpy、gemm、hemv、trmm |
| 策略条目 | 206 |
| 关系类型 | 4 |
| 关系条数 | 805 |
| 计算流程（pattern）条目 | 90 |
| 对齐簇 | hardware_feature=64, optimization_strategy=45, tunable_parameter=13 |

### 关系类型分布

| 关系 | 条数 | 实际带有的字段 |
|---|---:|---|
| OPTIMIZES_PATTERN | 326 | head, relation_id, relation_type, tail |
| TARGETS | 206 | head, relation_id, relation_type, tail |
| IS_ILLUSTRATED_BY | 206 | head, relation_id, relation_type, tail |
| HAS_PARAMETER | 67 | head, relation_id, relation_type, tail |

> 只有 4 类关系有实例（声明了 12 类），且边属性只有 `relation_type/id/head/tail`——**没有条件、模板、证据、权重**。这是「图结构无法产生约束操作」的直接证据。

## 2. 策略字段覆盖率（n=206）

| 字段 | 覆盖数 | 覆盖率 | 对 B3 的作用 |
|---|---:|---:|---|
| 有 description.strategy_rationale | 206 | 100.0% | 可解释性 |
| 有 description.implementation_pattern | 206 | 100.0% | 实施步骤 |
| 有 performance_impact（自由文本） | 206 | 100.0% | 收益排序（但不可核查） |
| 有 trade_offs | 206 | 100.0% | 冲突/代价 |
| 有 applicability_conditions | 206 | 100.0% | 可执行条件过滤（B3 核心） |
| 有 tunable_parameters（非空） | 58 | 28.2% | 参数推荐（B3 核心） |
| 有 target_hardware_feature | 206 | 100.0% | 硬件约束（B3 核心） |
| 有 code_example.snippet | 206 | 100.0% | 实施模板的原料 |
| 有 related_patterns | 203 | 98.5% | 检索锚点 |
| 有数值型参数（value_in_code） | 58 | 28.2% | 参数值可从知识推出 |

> 关键：**performance_impact 覆盖率可能很高，但它是 LLM 写的形容词（无 baseline/尺寸/平台/协议），不能当「收益证据」排序**；`applicability_conditions` 是自由中文，无法做谓词判断；`tunable_parameters` 即使有，也缺「由哪个硬件参数推导」的字段。

### 硬件知识是否可计算

- `target_hardware_feature_name` 取值 Top 10：分支预测器(34)、寄存器(19)、SIMD(17)、Cache(17)、地址生成单元(15)、指令流水线(13)、寄存器文件(11)、内存子系统(7)、浮点运算单元(6)、内存带宽(6)
- 含**数值**（KB/MB/bit/寄存器数/周期等）的策略描述：**1 / 206（0.5%）**
- 结论：硬件知识基本是「Cache / SIMD / 寄存器」这类**名词**，没有 `VLEN=256`、`L1d=32KB`、`LMUL≤8` 这种可参与判断的数值。

### 参数名 Top 15

| 参数名 | 出现次数 |
|---|---:|
| unroll_factor | 13 |
| block_size | 10 |
| alignment_boundary | 7 |
| 展开因子 | 6 |
| inc_x | 3 |
| inc_y | 3 |
| block_size_m | 2 |
| block_size_n | 2 |
| vector_register_width | 2 |
| accumulator_count | 2 |
| inc_x2 | 1 |
| inc_y2 | 1 |
| inc_x 和 inc_y | 1 |
| blocking_factor_n | 1 |
| block_columns | 1 |

## 3. 十个 competency questions（能答/部分/不能）

| # | 问题 | 现状 |
|---|---|---|
| CQ1 给定算子，列出它用到的优化策略 | 能（OPTIMIZES_PATTERN + 策略实体） |
| CQ2 给定策略，说明它优化的是哪类计算流程 | 能（OPTIMIZES_PATTERN） |
| CQ3 给定策略，说明它依赖哪些硬件特性 | 部分（只有硬件名词，无参数值） |
| CQ4 给定硬件 profile（x86 AVX2 / RVV VLEN=256），过滤掉不适用的策略 | 不能（无 ISA/参数/硬条件谓词） |
| CQ5 给定用户代码特征（dtype/stride/loop_depth/size），判断策略是否适用 | 不能（applicability_conditions 是自由文本） |
| CQ6 给定策略，给出可调参数及其取值/范围 | 部分（约 28% 有参数，缺「由硬件推导」字段） |
| CQ7 给定策略，给出可执行实施模板（骨架 + 槽位） | 不能（只有自然语言 implementation_pattern + 代码片段） |
| CQ8 给定策略，给出可核查的性能收益证据 | 不能（performance_impact 是形容词，无 baseline/协议） |
| CQ9 判断两条策略是否冲突/可组合 | 不能（CONFLICTS_WITH / COMPOSES_WITH 实例为 0） |
| CQ10 任意一条知识可追溯到源文件行号/版本 | 部分（只有 source_file 文件名，无行号/commit） |

## 4. 判决与建议（先补 3 个缺口，再实验）

**建模方向没有错**（实体类型、三层策略、计算流程锚点都是对的），但当前版本是 v1「知识清单」；要支撑 B3 必须补 3 个会在实验里直接变成混杂因素的缺口：

1. **硬件可计算**（0.5 天）：为 x86 与 K1 各写一份 `HardwareProfile` YAML（ISA、VLEN/LMUL、L1/L2、核数、频率、峰值/带宽），策略边加 `hardware_profile_id` + `parameter_derivation`；不需要全量，先覆盖 Top 20 策略。

2. **条件可执行**（0.5–1 天）：把 Top 20 策略的 `applicability_conditions` 改写成 3–6 个谓词（`dtype in {s,d,c,z}`、`stride_x == 1`、`loop_depth >= 2`、`size_regime in {small,medium,large}`、`aligned == true`、`tail_divisible` 等）；其余策略保留自由文本、标 `condition_status=text`。

3. **模板 + 证据 + 溯源**（1 天）：Top 20 策略各写一个 `ImplementationTemplate`（骨架 + 槽位 + 包含头），并从语料 `_log.txt`/编译命令抽 `SourceArtifact`（file + line + commit/版本）；性能证据先留「待测」字段，实验后回填（不要用 LLM 形容词）。

**做完这 3 个补丁，冻结为 Schema v1.5，开实验。** 实验过程中只允许「回填证据」，不再改 Schema；如果 pilot 失败且能归因到 Schema，再开 v2。

**时间盒**：2 天（Day 1 硬件 + 条件；Day 2 模板 + 溯源 + 冻结）；这 2 天不写代码框架、不跑 LLM 实验。
