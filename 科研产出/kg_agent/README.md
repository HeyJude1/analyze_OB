# kg_agent v1.5+：多架构可执行知识图谱 + 检索 Agent 工作流

## 论文实验快照

`data_thesis/` 含四种算子族的隔离图谱和实际建图源码副本。构建使用本机已有的
`../cx_operator_tool/operator_extract/openblas_newtest1`（相对开题目录）和
`../openblas_src`（本地 v0.3.27，`ce3f668`）。验证与检索只需保留项目内的快照及 schema。

```powershell
cd 科研产出\kg_agent\src
python -m kgagent.cli build --offline --split thesis
python -m kgagent.cli validate --data-dir ../data_thesis
$features = '{"dtype":"d","precision":"double","loop_depth":2,"reduction":true,"stride_x":1}'
python -m kgagent.cli retrieve --data-dir ../data_thesis --strict --op conv1d --source-op dot --profile k1_rvv256 --code-features $features
```

重建仅使用 AXPY、DOT、GEMV、GEMM 的源码。隔离先于静态抽取、规则差分和抽象派生，且排除旧 LLM 分析和 QEMU 日志。
`manifest.json` 中的 `split_audit` 记录留出源码同字节哈希检查和共享源码。DSYMM 的 C、RVV 两份抽取文件以及一个
AVX2 配置条目与 GEMM 共享，单列为近迁移候选。字节哈希检查覆盖已登记文件，语义重复仍需独立核查。

严格检索接收任务契约中的显式特征。缺少硬条件事实时跳过相关卡片，必要时返回空策略包。留出任务可显式指定来源算子，
沿该来源的计算流程与 `GENERALIZES` 边召回抽象策略，并隐藏来源算子的示例骨架。x86 lane 数和 RVV 的 LMUL、
VLMAX 候选由目标 profile 推导。条件中涉及具体 ABI、数据依赖与布局的部分需要在实现与测试阶段落实。

严格模式在精排后处理替代卡和已登记冲突，按组合边进行拓扑排序。策略包中的 `plan.status`
为 `candidate_only`、`no_eligible_strategy` 或 `precedence_cycle`。通过 `--independent`
获得相同召回、硬条件过滤、评分和候选参数的独立策略卡对照。传统示范调用仍会使用默认特征。
历史 QEMU 数值可在原图查证。检索和生成包的收益仅接收匹配目标平台、编译器、工作负载与基线的真机策略级实测。
现有隔离图谱仅包含源码观察。

旧版说明中的统计属于原始参考图谱。

> 严格按 `SPEC.md` 接口契约实现的 pipeline 侧（writer：`src/**`、`schema/**`、`tests/**`、`data/**`、`config.yaml`、`README.md`）。
> **纯 Python 标准库**（json / argparse / re / pathlib / hashlib / unittest / csv / math / copy），
> 不依赖 networkx / jsonschema / pymilvus / requests；**离线 build 不联网**；Windows 路径可直接跑。
>
> v1.5+（task-11）把知识图谱从「generic + RISC-V RVV」扩展为**多架构**：x86 AVX2 / AVX512、ARM NEON / SVE、RISC-V RVV。
> SPEC 的多架构契约章节由 Lead 维护；本 README §4 是实现侧的多架构说明与实测数据。

---

## 1. 一分钟跑通（Windows / PowerShell）

```powershell
cd 科研产出\kg_agent\src

python -m kgagent.cli build --offline          # S0→S6（含多架构层），生成 data/nodes.jsonl / edges.jsonl / manifest.json
python -m kgagent.cli validate                 # schema + 不变量，退出码 0=通过
python -m kgagent.cli retrieve --op dgemm --profile x86_avx2
python -m kgagent.cli retrieve --op dgemm --profile k1_rvv256
python -m kgagent.cli stats --matrix           # 节点/边 + arch×operator 覆盖矩阵 + 跨架构共有策略表
python -m kgagent.cli export-viz               # 只写 viz/graph_data.js + viz/traces.js
```

一键（任意目录）：`python 科研产出\kg_agent\run_demo.py` —— 依次 build → validate → 3 个 profile 检索 → stats --matrix → export-viz。

测试（在 `kg_agent/` 下，69 个用例，含 8 个多架构用例 + 26 个 MPSA 用例）：

```powershell
cd 科研产出\kg_agent
python -m unittest discover -v
```

> `python -m kgagent.cli` 要求当前目录（或 PYTHONPATH）包含 `kg_agent/src`，所以上面先 `cd src`。
> `unittest` 通过 `tests/__init__.py` 自动把 `src/` 注入 `sys.path`。

**数据源**（`config.yaml` 可改，或设 `KG_WORKSPACE_ROOT`）：
- 4 个算子的分析 JSON：`开题/analyze_OB/.../analysis_results/`；
- 12 个算子的 RVV/generic 语料：`../cx_operator_tool/operator_extract/openblas_newtest1/`；
- **多架构源码**：`../openblas_src`（OpenBLAS 0.3.27，git `ce3f668`）的 `kernel/{x86_64,arm64,riscv64,generic}`；
- QEMU 性能汇总：`科研产出/data/riscv_log_summary.csv`。

---

## 2. 流水线（S0→S6，全部可离线重跑）

| 段 | 模块 | 输入 | 输出 | LLM |
|---|---|---|---|---|
| S0 | `corpus.py` | 用户语料 + `_command.txt` + `_log.txt` + CSV | `data/corpus_manifest.jsonl`（cx 语料工件的 sha256/行数/-march） | 0 |
| S0b | `multiarch.py` | `config.multiarch.arch_sources`（glob）+ `openblas_src` | **自动发现 + 复制**到 `data/corpus_multiarch/<arch>/<op>/`，`data/multiarch_manifest.jsonl`（sha256/行号/ISA/profile） | 0 |
| S1 | `static.py` | generic/RVV C kernel + 多架构 C/汇编 | `data/characteristics.jsonl`（893 条静态特征；汇编抽 mnemonics/向量宽度/FMA/prefetch/寄存器分块线索） | 0 |
| S2 | `diff.py` + `multiarch.py` | 同算子 generic vs RVV；同架构 generic vs optimized | `data/diffs.jsonl`（51 条：12 条 RVV 差分 + 39 条多架构记录；C 可用→`method=diff`，汇编-only→`method=static` 特征存在） | 0 |
| S3 | `extract.py` | 分析 JSON + S1 + S2 + 多架构 | `data/strategies_raw.jsonl`（262 条策略：analysis/static/diff/multiarch/cross_arch/abstract） | 可选 |
| S4 | `normalize.py` | S3 + 词表 | 闭集校验、ID 稳定化、`arch_scope`/`arch_evidence`/`arch_parameters`、参数合并、模板实例化、GENERALIZES/COMPOSES/CONFLICTS | 0 |
| S5 | `evidence.py` | `riscv_log_summary.csv` + 12 个 `*_log.txt` | 88 条 `PerformanceEvidence`（qemu_only，含负例） | 0 |
| S6 | `validate.py` | 全部 JSONL | `data/nodes.jsonl` / `edges.jsonl` / `manifest.json`（版本四元组 + 112 个输入 sha256）+ `data/arch_coverage.json` | 0 |

**分层原则**：所有结构事实（dtype、循环深度、SEW/LMUL、参数值、性能数字、行号、硬件数值、汇编 mnemonics）都由规则或日志产生；
LLM（`--llm`）只允许改写策略名与解释文本，且禁止出现收益数字（`validate` 会拒绝）。
`--llm` 模式带 `temperature=0` + JSON 输出 + `sha256(model|prompt_version|prompt)` 全量缓存；offline 只读缓存、未命中直接报错（绝不偷偷联网）。

---

## 3. 数据契约与当前快照

`data/nodes.jsonl`：一行一个节点，必需 `id/type/name/properties/provenance`；12 类实体闭集。
`data/edges.jsonl`：一行一条边，`head/type/tail/properties/provenance`；14 类关系闭集；`(head,rel,tail)` 唯一。
`data/traces/<trace_id>.json`：7 步推理 trace（文件名把 `:` 换成 `_`，兼容 Windows）。

**当前快照规模**：`862 nodes / 3656 edges`
- 14 `Operator`、76 `ComputationPattern`、17 `CodeCharacteristic`、262 `OptimizationStrategy`、262 `ImplementationTemplate`、
  14 `TunableParameter`、6 `ISACapability`、6 `Microarchitecture`、6 `HardwareProfile`、88 `PerformanceEvidence`、111 `SourceArtifact`；
- `manifest.json` 记录 `{schema_version, extractor_version, prompt_version, model_version}`、112 个输入文件 sha256、按类型统计；
- 同一份输入两次 build 字节级一致（幂等，有测试）。

多架构策略在属性里额外带：`arch_scope`、`arch`、`arch_family`、`arch_evidence[]`（每个架构的文件+行号+method）、
`arch_parameters{}`（parameterized 策略的各架构参数）、`evidence_status`（`qemu_operator_evidence` / `source_only`）。

---

## 4. 多架构扩展（x86 / ARM / RISC-V）

### 4.1 覆盖范围（实测 arch × operator 矩阵）

单元格 = 该架构实现的源文件数（`-` = 该架构没有对应实现，如实留空）：

| arch | asum | axpby | axpy | copy | dot | gemm | gemv | hemv | nrm2 | rot | scal | swap | symm | symv | trmm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| x86_avx2 | 2 | - | 2 | 1 | 2 | 2 | 3 | - | 1 | 2 | 2 | 1 | 1 | 2 | 1 |
| x86_avx512 | 1 | - | 1 | - | 1 | 3 | - | - | - | 1 | 1 | - | 1 | 1 | - |
| arm_neon | 2 | - | 2 | 1 | 3 | 2 | 2 | - | 1 | 1 | 1 | 1 | - | - | 1 |
| arm_sve | - | - | - | 1 | 1 | 3 | - | - | - | - | - | - | 1 | - | 1 |
| rvv | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | - | 1 |

- **13 / 14 个算子有 ≥2 个架构的实现**（仅 hemv 只有 RVV，如实保留为单架构）；
- 70 个实现文件 = 用户语料 12 个 RVV kernel + 从 openblas_src 复制的 58 个 x86/ARM/RVV 文件；
- 6 个 `HardwareProfile`：`x86_avx2 / x86_avx512 / arm_neon / arm_sve / k1_rvv256 / qemu-riscv64-vlen256`；
- 6 个 `ISACapability`：AVX2-FMA / AVX512F-FMA / NEON-FMA / SVE-FMA / RVV1.0-vector / RVV1.0-FMA；
- arch_specific 策略按架构：rvv 42 / arm_neon 37 / x86_avx2 29 / x86_avx512 16 / arm_sve 15。

### 4.2 自动发现 + 自包含复制（S0b）

`config.yaml` 的 `multiarch.arch_sources` 按架构声明 glob 规则（不是硬编码绝对路径）：

```yaml
multiarch:
  repo: ../openblas_src                 # 相对 workspace_root
  corpus_dir: data/corpus_multiarch
  arch_sources:
    x86_avx2:
      kernel_dir: kernel/x86_64
      isa_node: isa:avx2.fma.f64
      profiles: [x86_avx2]
      auto_microk_globs: ["{kernel_dir}/d{op}_microk_haswell-*.c"]   # 自动补登记 microkernel
      files:
        gemm: [kernel/x86_64/dgemm_kernel_4x8_haswell.S, kernel/x86_64/dgemm_kernel_16x2_haswell.S]
        axpy: [kernel/x86_64/daxpy.c, kernel/x86_64/daxpy_microk_haswell-2.c]
        ...
```

`multiarch.discover()` 展开 glob → 复制到 `data/corpus_multiarch/<arch>/<op>/`（文件名加 sha256 校验），
**图谱只引用复制件**，不依赖 `openblas_src` 路径；`data/multiarch_manifest.jsonl` 记录每个文件的
`arch/arch_family/isa/profiles/role(main|microk)/kind(c|asm)/sha256/line_count`。
RVV 的 12 个已有算子复用 cx_operator_tool 语料（避免重复复制/重复抽策略）；仅新算子 `asum/axpby` 从 openblas_src 复制。

### 4.3 C 可差分、汇编用静态特征（S1/S2）

- **C 内核**（如 haswell/skylakex microk、cortexa53 NEON C、symm SVE copy）：抽 intrinsic、内联汇编 mnemonics、
  FMA/prefetch、累加器/寄存器块、循环深度；与同算子 generic 基线做差分 → `method=diff`。
- **`.S` 汇编内核**（如 dgemm_kernel_4x8_haswell.S、dgemm_kernel_sve_v1x8.S、gemv_n.S）：
  抽 **指令 mnemonics 计数**、向量寄存器宽度（xmm/ymm/zmm、v/q、z）、FMA 条数、prefetch 条数、
  目标向量寄存器数（寄存器分块线索）、`.macro` 块数与分支数；
  汇编无法做"generic vs optimized 差分"，因此用**特征存在**标注 `method=static`，并在 `arch_evidence` 里给出文件+行号。
- 检测到的 ISA 如实记录：x86 汇编若是 SSE2/标量基线，会标 `x86-64 baseline (SSE2)`，不硬说成 AVX2。

### 4.4 策略的 `arch_scope` 与跨架构合并（S3/S4）

| arch_scope | 含义 | 数量 |
|---|---|---|
| `arch_agnostic` | generic C/抽象策略，适用所有架构（不绑定具体 ISA） | 93 |
| `arch_specific` | 绑定单一架构（`arch_evidence` 给出源码文件+行号+method），检索时按 profile 过滤 | 139 |
| `parameterized` | **跨架构共有策略**：同一 op/family/action 在 ≥2 架构出现，合并保留各架构参数 | 30 |

跨架构合并示例（`stg:gemm:tile_x`，覆盖 5 个架构）：

```json
{
  "arch_scope": "parameterized",
  "arches": ["arm_neon", "arm_sve", "rvv", "x86_avx2", "x86_avx512"],
  "arch_parameters": {
    "x86_avx2":  {"reg_block_m": 4,  "reg_block_n": 8,  "unroll_factor": 32},
    "x86_avx512":{"reg_block_m": 16, "reg_block_n": 2,  "unroll_factor": 32},
    "arm_neon":  {"reg_block_m": 4,  "reg_block_n": 8},
    "arm_sve":   {"reg_block_m": 1,  "reg_block_n": 8},
    "rvv":       {"reg_block_n": 8,  "k_unroll": 8, "lmul": 2, "sew": 64}
  }
}
```

`GENERALIZES` 边把每个 `arch_specific` 策略指向对应的 `parameterized` 策略，跨架构共有策略表可直接打印：

```powershell
python -m kgagent.cli stats --matrix
# [arch × operator 覆盖矩阵] ...
# [跨架构共有策略] parameterized 策略（30 条，如 stg:gemm:tile_x / stg:gemm:vectorize_x / stg:dot:vectorize_x ...）
```

### 4.5 按架构检索

`retrieve --op dgemm --profile <arch>` 在 4 个过滤步里都会用到新维度：
1. `hardware_filter`：`hardware_profile_ids` 不含目标 profile 直接淘汰（如 x86 查询会淘汰 arm/rvv 策略并给出 reason）；
2. 硬 ISA 要求：`STRATEGY_REQUIRES_ISA.hard=true` 且 profile 不支持 → 淘汰；
3. `conflict_resolve`：同 `(family,action)` 去重时**架构特定实现优先**于通用/参数化实现；
4. 打分 `hw` 项：arch_specific=1.0、parameterized=0.85、arch_agnostic=0.6（其余项不变，见 §5）。

实测（`dgemm`）：
- `x86_avx2` → 选 `stg:gemm:*_x86_avx2`（haswell 汇编/microk）+ 架构无关策略；
- `k1_rvv256` → 选 `stg:gemm:*_d`（RVV 差分）+ 架构无关策略；
- `arm_neon` → 选 `stg:gemm:*_arm_neon`；
- `arm_sve` → 选 `stg:gemm:*_arm_sve`。

### 4.6 证据口径（诚实说明）

QEMU 性能日志只有 RISC-V 版本，因此：
- **只有 rvv / arch_agnostic 策略**绑定 `STRATEGY_EVIDENCED_BY`（QEMU operator 级，confidence=0.35、rerun_needed）；
- x86/ARM 的 `arch_specific` 与 `parameterized` 策略 `evidence_status = "source_only"`：只给源码证据（file+line），
  **不把 QEMU 的 RVV 数字错记到 x86/ARM 策略上**；跨架构收益对比需要后续在对应真机上重跑。

---

## 5. 检索：7 步可解释 trace

```
1 pattern_match      算子 -> ComputationPattern（注册表命中）
2 strategy_recall    沿 PATTERN_OPTIMIZED_BY 召回 + STRATEGY_GENERALIZES 抽象/跨架构扩展（通道 D）
3 condition_filter   conditions 对 code_features 求值；hard 条件不满足即淘汰（reason 逐条写出）
4 hardware_filter    hardware_profile_ids 绑定 + hard STRATEGY_REQUIRES_ISA 对 profile ISA 校验
5 conflict_resolve   (family,action) 同族去重（架构特性优先）+ STRATEGY_CONFLICTS_WITH + 寄存器预算
6 score_and_select   六项加权：match_p/fit_c/hw/ev/sup − transfer_penalty，取 top_k
7 pack               StrategyPack(pack-v1)：策略+模板+参数+证据+溯源+prompt_preview
```

每步都有 `name/description/candidates/selected`，被淘汰项都有 `dropped[].reason`（可视化的推理回放数据源）。
示例：`retrieve --op dgemm --profile x86_avx2` 会淘汰 15 条 complex-only 分析策略（dtype 硬条件）、
16 条 arm/rvv/avx512 策略（硬件 profile/ISA 过滤）、11 条同族重复/冲突/寄存器预算策略，
最终返回 4 条 x86 AVX2 具体策略 + 2 条架构无关策略。

---

## 6. 怎么加一个算子

1. 把语料目录（含 `<op>_k.c`、`<op>_k_rvv.c`、`_command.txt`、`*_log.txt`）放进
   `../cx_operator_tool/operator_extract/openblas_newtest1/`；
2. 在 `config.yaml` 的 `corpus.operators` 加一行：

   ```yaml
   myop: {dir: 99_myop_op, dtype: d, blas_level: 1, op_class: vector, structural: none, reduction: false,
          flops_formula: "2*N", bytes_formula: "(2*N)*sizeof(T)"}
   ```
3. 重跑 `build --offline`：S0 自动登记工件、S1 抽特征、S2 做差分、S3/S4 产出策略与边、S6 校验。
4. 若该算子也有三层分析 JSON，把 `{op}_analysis.json` 放进 `analysis_results/`，并把 op 加进 `corpus.analysis_operators`。
5. **要加多架构实现**：在 `multiarch.operators` 注册算子元数据（没有 cx 语料时），并在
   `multiarch.arch_sources.<arch>.files.<op>` 写 glob（或依赖 `auto_globs`）。新架构需要同步在
   `schema/hardware_profiles.yaml` 加 profile + `schema/vocab.yaml` 的 `arch_names`。

**当前 14 个算子**：gemm / trmm / hemv / axpy / gemv / dot / scal / copy / nrm2 / symm / swap / rot（cx 语料，前 4 个有分析 JSON）
+ asum / axpby（多架构层新增）。

---

## 7. 怎么接 LLM（可选）

```yaml
# config.yaml
llm:
  backend: api            # 默认 offline
  model: qwen-plus
  api_base: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  api_key_env: DASHSCOPE_API_KEY
```

```powershell
$env:DASHSCOPE_API_KEY = "..."
python -m kgagent.cli build --llm
```

- 缓存键 `sha256(model + prompt_version + prompt)` 落盘 `data/llm_cache/`；同输入重跑命中缓存，不再计费；
- 调用失败/无 key/超时 → 该条降级为规则文本（不会中断 build）；
- LLM 只改 `name/description/rationale`，条件/参数/模板/行号/数字仍来自规则与日志（`validate` 强制）。

---

## 8. 验收对照

### SPEC §6（原单架构验收，仍全部通过）

| 验收项 | 结果 |
|---|---|
| `build --offline` 从零跑通，生成非空 nodes/edges/manifest | ✅ 862 nodes / 3656 edges；manifest 含版本四元组 + 112 个输入 sha256 |
| `validate` 通过（id 唯一、边端存在、类型闭集、condition_status 非空、性能数字只在 Evidence） | ✅ 0 error / 0 warning |
| ≥20 条策略：≥2 可执行条件 + 1 template + provenance(file+line) + hardware 绑定 | ✅ 262 / 262 条满足（`acceptance_count`） |
| `retrieve --op dgemm --profile x86_avx2` 输出 ≥7 step 且含 dropped 原因 | ✅ 7 步；第 3/4/5 步分别 dropped 15/16/11 条，均带 reason |
| `graph.html` 双击可开、5 项交互（viz engineer 负责） | pipeline 产 `viz/graph_data.js`（862 节点 / 3656 边）+ `viz/traces.js`（6 条 trace 回放数据） |
| `tests/` ≥6 个测试通过 | ✅ 43 个用例（schema 11 / retrieve 9 / pipeline 10 / graph 5 / multiarch 8） |

### task-11 多架构验收

| 验收项 | 结果 |
|---|---|
| config 增加 `arch_sources`（12 算子 × 5 架构映射）+ 脚本自动发现（glob）+ 复制到 `data/corpus_multiarch/<arch>/<op>/` 并记录 sha256/行号 | ✅ `multiarch.discover/materialize`，58 个文件、70 个实现文件、全部复制件 sha256 校验通过 |
| S0/S1 扩展 arch 维度；C 与汇编都抽特征（汇编：mnemonics/向量宽度/FMA/prefetch/寄存器分块线索） | ✅ `static.parse_arch_impl`，893 条特征记录 |
| S2 同架构 C 差分（`method=diff`）；汇编-only 特征存在（`method=static`） | ✅ 39 条多架构差分（C diff 18 / 汇编 static 21） |
| S3/S4 策略加 `arch_scope` + `arch_evidence[]`；跨架构合并保留各 arch 参数 | ✅ 262/262 有 arch_scope；139 arch_specific + 30 parameterized（含 arch_parameters） |
| 统计：arch×op 覆盖矩阵 / 各 arch 策略数 / 跨架构共有策略数；retrieve 按 arch 过滤；validate 0 error；≥6 新测试 | ✅ `stats --matrix`（矩阵见 §4.1，跨架构 30 条）；4 个 profile 检索通过；validate 0 error；8 个新测试 |
| 至少 8 个算子有 ≥2 架构实现；≥20 条策略有 arch_scope；规模不回退 | ✅ 13 个算子多架构；262 条（≥20）；862 nodes / 3656 edges（>527/2218） |
| 更新 README / re-export viz | ✅ 本文档多架构章节；`graph_data.js` / `traces.js` 已重新导出 |

---

## 9. 设计取舍与已知限制（诚实清单）

1. **语料在 workspace 外**：cx 语料 provenance 用 `../cx_operator_tool/...` 相对路径；多架构文件已复制进
   `data/corpus_multiarch/`（自包含），`manifest.inputs` 记录 sha256；
2. **x86/ARM 没有本架构性能日志**：只有源码证据（`evidence_status=source_only`），QEMU 数字只用于 RVV/架构无关策略；
   跨架构收益对比需要后续在对应真机重跑；
3. **汇编无法做语义差分**：汇编内核用静态特征存在标注（`method=static`、`status=inferred`），
   不做"generic vs optimized"归因；部分 x86 汇编只检测到 SSE2/标量基线，名称会如实标注；
4. **SVE 向量长度**：SVE 是可变向量，`hardware_profiles.yaml` 按实现 VL=256 记录并在 note 里说明；
5. **分析 JSON 的策略质量参差**：来自 v1 LLM 抽取（status=raw），provenance 指向分析 JSON 行号（源 `.clean.c` 不在工作区）；
6. **条件谓词仍有限**：8 类谓词可执行，复杂自由文本条件落在 `description`；
7. **viz 边界**：本仓库 pipeline 只写 `viz/graph_data.js`、`viz/traces.js`；`viz/graph.html` 与 `assets/` 归 viz engineer；
8. **SPEC 多架构章节**：SPEC.md 归 Lead 维护，本实现遵循现有契约并在此 README 记录多架构扩展；如需冻结，
   请 Lead 在 SPEC 中补 §多架构（arch_scope/arch_sources/覆盖矩阵字段）。

---

## 10. MPSA：B3/B4 多视角策略分析（E09 实验用）

`src/kgagent/mpsa.py` 把检索结果（7 步 trace 的 StrategyPack）接到「生成 C 内核」这一步，提供两种实验条件：

| 条件 | 流程 | LLM 调用次数 |
|---|---|---|
| **B3** | 把 top-k StrategyPack 直接拼进生成 prompt，一次生成完整 C 函数 | 1 |
| **B4** | 先对 top-`analysis_k` 个策略逐条做「适用性分析」（严格 JSON：applies/evidence/risks/parameters/conflicts），再按 4 标准（supportive / complementarity / accuracy / completeness）综合出报告，最后用「pack + 逐条分析 + 综合报告」生成 | `analysis_k + 2` |

5 个 prompt 模板在 `prompts/`：`analysis.md` / `synthesis.md` / `generate_b3.md` / `generate_b4.md` / `repair.md`（`{{KEY}}` 占位符，缺参数直接报错）。

### 10.1 怎么跑

```powershell
cd 科研产出\kg_agent\src

# 离线自测（不联网、不需要 key；mock 是确定性输出）
python -m kgagent.cli mpsa --op daxpy --profile k1_rvv256 `
  --code ..\..\10_第一周执行包\OpOptBench\tasks\daxpy\input\generic.c `
  --mode b3 --provider mock --out runs\b3_daxpy

# B4（逐条适用性分析 + 4 标准综合）
python -m kgagent.cli mpsa --op daxpy --profile k1_rvv256 `
  --code ..\..\10_第一周执行包\OpOptBench\tasks\daxpy\input\generic.c `
  --mode b4 --provider mock --out runs\b4_daxpy --top-k 6 --analysis-k 3

# 真实 API（key 只从环境变量读，绝不写进仓库）
$env:DASHSCOPE_API_KEY = "..."
python -m kgagent.cli mpsa --op daxpy --profile k1_rvv256 `
  --code ...\generic.c --mode b4 --provider dashscope --out runs\real_b4

# 修复回路（生成的代码编译失败/结果错误时回灌错误做 1 轮修复，见 §10.6）
python -m kgagent.cli mpsa-repair --op daxpy --profile k1_rvv256 `
  --code runs\real_b3\generated.c --error runs\real_b3\harness\build.stderr.log `
  --mode b3 --provider mock --out runs\repair_b3
```

输出：`<out>/mpsa_result.json`（主产物）+ `<out>/generated.c`（清洗后的代码，便于 harness 直接编译）。
`mpsa_result.json` 字段：`op / op_canonical / profile / mode / provider / model / retrieved / analysis / synthesis / prompt / code / code_meta / usage / elapsed_s / created_at / artifacts`；
`usage` 含 `calls + prompt/completion/total_tokens + by_stage(analysis/synthesis/generate)`（mock 的 token 为 0、`calls` 记真实调用次数）。

### 10.2 LLM provider 与安全

- `--provider dashscope`：DashScope OpenAI 兼容端点 `https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`，默认模型 `qwen-plus-2025-04-28`（`--model` 或 `config.yaml` 的 `mpsa.model` 可覆盖）；
- key 只从环境变量 `DASHSCOPE_API_KEY`（`config.yaml: mpsa.api_key_env`）读取，代码里没有 key，也不写进任何输出文件；
- `--provider mock`：不联网、无随机性（同输入两次产物逐字节一致，有测试），用于 E09 链路自测；`--mock-code FILE` 可指定 mock 返回的代码；
- temperature=0、分析/综合两步用 `response_format=json_object`；B4 逐条分析串行调用（顺序稳定、便于复现）。

### 10.3 代码清洗与编译

`clean_code()`：去 markdown 围栏 / 去函数外解释文字 / 只保留目标 `_k` 函数（及前置 `#include`/`#define`/`typedef`）/ 只有函数体时用任务签名包裹 / 模型把函数改名时重命名回任务名 / 括号配平；
若签名用到 `BLASLONG` 且代码未定义，文件头补一行 `typedef long long BLASLONG;`，因此 **generated.c 是可独立编译的翻译单元**，编译时 driver 与 generated.c 当两个 TU 分别编译（如 `gcc -O2 driver/test_daxpy.c generated.c`）。
任务签名优先从 `--code` 的 generic.c 提取，缺省时用内置 daxpy/ddot 签名。mock provider 对 daxpy/ddot 生成与基线语义一致的正确标量实现（n<=0 / da==0 / 任意步长 / 尾部），其它算子回退 `--code` 原函数，因此 mock 产物可直接进 QEMU 编译链路。

### 10.4 退出码

`0` 成功；`2` 参数/清洗/其它运行错误；`3` 缺少 `DASHSCOPE_API_KEY`；`4` LLM 调用或 JSON 解析失败；`5` 检索失败（未知算子/profile、图数据为空/无策略；`mpsa-repair` 不检索，不会出现 5）。失败时 stderr 打 `[error] <异常类型>: ...`。

### 10.5 测试与验收

- `tests/test_mpsa.py` 新增 26 个用例：mock B3/B4 合法 JSON、mock 确定性、ddot 签名识别、`--mock-code` 覆盖、围栏/函数体清洗、缺 key 清晰报错、未知算子、CLI 退出码，以及修复回路（mock repair、内联/文件错误输入、`--mock-code` 覆盖、假 DashScope client、CLI 端到端）等；`python -m unittest discover` 共 **69 tests 全过**（原 43 + 新 26）；
- mock 下 `b3` / `b4` 各端到端跑过（产物 `mpsa_result.json` + `generated.c`）；生成的 daxpy 已用 gcc 编译并用 OpOptBench driver 跑出 `maxdiff=0`；
- 修复回路：用真实 API 失败产物（幻觉 intrinsic 编译错误 / ddot 只取 lane0 的 wrong_answer）做输入，mock 修复产物经 riscv64 交叉编译 + QEMU 验证（daxpy `maxdiff=0`、ddot `1.18e-11`）；
- 真实 DashScope API 由 Lead 跑（本模块的测试与 mock 链路不联网）。

### 10.6 修复回路：编译/运行错误回灌（`mpsa-repair`）

真实 API 冒烟暴露了两类失败：① 幻觉 RVV intrinsic 导致编译失败（如 `__riscv_vmvinf_f64m8`、`__riscv_vlseg1e64_v_f64m8`、`__riscv_vse64`、`__riscv_vfredusum_vs_f64m8_f64`）；② 归约写错导致 wrong_answer（ddot 只取 lane0）。`mpsa.py` 的 `repair_code(...)` 把「原代码 + 错误 + 签名 + profile」回灌给 LLM 做 **1 轮**修复：

```powershell
python -m kgagent.cli mpsa-repair --op daxpy --profile k1_rvv256 `
  --code <cell>\generated.c `
  --error <cell>\harness\build.stderr.log `      # wrong_answer 时传 harness\qemu.stdout.log
  --mode b3 --provider dashscope --out <cell>\repair_r1 `
  [--model qwen-plus-2025-04-28] [--signature "..."] [--mock-code FILE]
```

- `--code` / `--error` 均支持 **文件路径或内联文本**（已存在文件→读；像路径但不存在→报清晰错误；否则按文本）；`--code` 传待修复的原代码，`--error` 传编译器 stderr 或 QEMU 输出（`maxdiff=.../reldiff=...` 会原样进 prompt）；
- 签名优先级：`--signature` > 从 `--code` 提取 > 内置 daxpy/ddot 签名；修复行为与 `--mode` 无关（mode 只记录来源条件）；
- `prompts/repair.md`：要求只输出完整修正 C 函数；**不确定的 intrinsic 一律退回标量实现**；给出 GCC 13 `riscv_vector.h` 实测可编译的 intrinsic 白名单（vsetvl / vle64 / vlse64 / vse64 / vfmacc / vfredusum m8→m1 等）和已证伪名字黑名单；并提示 wrong_answer 先查归约（所有 lane 求和，不能只取 lane0）；
- 输出 `<out>/repair_result.json`（`code / usage / code_meta / input_error / error_source / calls / fix_summary / prompt / elapsed_s / artifacts`）+ `<out>/repaired.c`；`usage` 结构与 mpsa 一致（`by_stage.repair`；mock token=0、calls=1）；
- `--provider mock`：daxpy/ddot 返回与基线语义一致的正确标量实现（可编译可跑），其它算子原样返回；`--mock-code FILE` 可指定 mock"修复后代码"，供实验侧离线联调。

---

## 11. 文件地图

```
kg_agent/
  config.yaml                 # 路径/语料/多架构 arch_sources/证据/检索权重/LLM
  schema/entities.yaml        # 12 类实体
  schema/relations.yaml       # 14 类关系（含定义域/值域/边属性）
  schema/condition_vocab.yaml # 10 个受控谓词 + char_type 映射
  schema/hardware_profiles.yaml # 6 个硬件 profile（x86/ARM/RVV，数值唯一来源）
  schema/vocab.yaml           # family/action/level/stage/status/arch_scope/arch_names 等闭集
  schema/node.schema.json     # node JSON Schema（draft-07 子集，schema.py 内置校验器）
  schema/edge.schema.json
  src/kgagent/                # cli/corpus/static/diff/multiarch/llm/extract/normalize/evidence/
                              # validate/graph/retrieve/viz_export/mpsa + yamlmini/config/schema
  prompts/                    # MPSA 5 个 prompt 模板：analysis/synthesis/generate_b3/generate_b4/repair
  data/
    corpus_manifest.jsonl     # 111 个源工件（cx 语料 + 多架构复制件）
    multiarch_manifest.jsonl  # 58 个多架构实现文件（arch/isa/profiles/sha256/行号）
    characteristics.jsonl     # 893 条 S1 静态特征（C + 汇编）
    diffs.jsonl               # 51 条差分（12 RVV + 39 多架构）
    strategies_raw.jsonl      # 262 条策略原始记录
    nodes.jsonl / edges.jsonl / manifest.json
    arch_coverage.json        # arch×operator 覆盖矩阵 + 跨架构共有策略表
    traces/                   # 6 条检索 trace（x86_avx2 / k1_rvv256 / arm_neon / arm_sve）
    corpus_multiarch/<arch>/<op>/  # 复制后的多架构源文件（自包含）
  tests/                      # 69 个 unittest（schema/retrieve/pipeline/graph/multiarch/mpsa）
  run_demo.py                 # 一键 build→validate→检索 3 profile→stats --matrix→export-viz
```
