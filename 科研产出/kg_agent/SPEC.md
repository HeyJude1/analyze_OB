# kg_agent 构建规范（SPEC v0.1）

> 2026-11-12｜Lead｜目的：把 v1 的抽取/建图代码升级为 **v1.5 可执行知识 + Agent 工作流 + 可视化/推理/检索回放**。
> 本文件是**接口契约**：pipeline 与 viz 两个实现必须严格按此产出/消费数据，互不等待。
> 根目录：`C:\Users\MSI\Desktop\开题\科研产出\kg_agent\`

---

## 1. 目录结构（必须按此创建）

```text
kg_agent/
  README.md
  SPEC.md                  # 本文件
  config.yaml              # 路径、LLM backend、corpus、hardware profiles
  schema/
    entities.yaml          # 12 类实体定义
    relations.yaml         # 14 类关系定义
    condition_vocab.yaml   # 谓词闭集（8 类）
    hardware_profiles.yaml # x86_avx2 / x86_avx512 / k1_rvv256
    node.schema.json       # JSON Schema（jsonschema 可校验）
    edge.schema.json
  data/
    corpus_manifest.jsonl  # S0
    characteristics.jsonl  # S1
    diffs.jsonl            # S2
    strategies_raw.jsonl   # S3
    nodes.jsonl            # S6 最终节点
    edges.jsonl            # S6 最终边
    manifest.json          # 版本四元组 + 输入 sha256 + 统计
    traces/                # 每次检索的 trace（含推理步骤）
      <trace_id>.json
  src/kgagent/
    __init__.py
    cli.py                 # 唯一命令行入口：python -m kgagent.cli <cmd>
    corpus.py              # S0 CorpusAgent
    static.py              # S1 StaticFeatureAgent
    diff.py                # S2 DiffAgent
    llm.py                 # LLMClient（API + cache + offline）
    extract.py             # S3 SemanticAgent
    normalize.py           # S4 AlignAgent
    evidence.py            # S5 EvidenceAgent
    validate.py            # S6 ValidatorAgent
    graph.py               # NetworkX 图加载/查询/路径
    retrieve.py            # 条件过滤 + 打分 + 冲突消解 + trace
    viz_export.py          # 生成 viz/graph_data.js + traces.js
  viz/
    graph.html             # 自包含可视化（本地 vis-network.min.js）
    assets/vis-network.min.js
    graph_data.js          # 由 viz_export.py 生成
    traces.js              # 由 viz_export.py 生成
  tests/
    test_schema.py
    test_retrieve.py
    test_pipeline.py
  run_demo.py              # 一键：build → retrieve → export → 打开说明
```

## 2. 数据契约

### 2.1 节点（`data/nodes.jsonl`，一行一个 JSON）

```json
{
  "id": "stg:gemm:unroll_pointer",
  "type": "OptimizationStrategy",
  "name": "循环展开与指针预计算",
  "properties": {
    "family": "循环展开",
    "level": "code",
    "action": "unroll",
    "description": "...",
    "conditions": [
      {"pred": "loop_depth", "op": "ge", "value": 2, "hard": true}
    ],
    "hardware_profile_ids": ["x86_avx2", "k1_rvv256"],
    "parameter_derivation": {"unroll_factor": "min(4, vector_regs/4)"},
    "template_id": "tpl:gemm:unroll_pointer",
    "status": "verified",
    "confidence": 0.8
  },
  "provenance": {
    "source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k.c",
    "line_start": 19,
    "line_end": 120,
    "commit": "OpenBLAS 0.3.27.dev",
    "extractor_version": "kgagent-v1.5"
  }
}
```

**必需字段**：`id/type/name/properties/provenance`。`provenance.source_file` 必填；行号可 `null` 但要 `line_confidence: approx`。
**ID 规则**：`{type_prefix}:{slug}`，slug 由 name 规范化（小写、空格→下划线、去标点），**发布后不可变**。

### 2.2 边（`data/edges.jsonl`）

```json
{
  "id": "e:pat:gemm_mm_stg:gemm:unroll_pointer",
  "type": "PATTERN_OPTIMIZED_BY",
  "head": "pat:gemm:mm_microkernel",
  "tail": "stg:gemm:unroll_pointer",
  "properties": {"support": 3, "is_primary": true,
                 "condition": {"pred": "dtype", "op": "in", "value": ["s","d"]}},
  "provenance": {"source_file": "...", "line_start": 19, "line_end": 120},
  "confidence": 0.8
}
```

### 2.3 事务/轨迹（`data/traces/<trace_id>.json`）

```json
{
  "trace_id": "trace:20261112T180000:dgemm:x86_avx2",
  "query": {
    "operator": "dgemm",
    "hardware_profile_id": "x86_avx2",
    "code_features": {"dtype": "d", "loop_depth": 3, "stride_x": 1,
                      "size_regime": "large", "aligned": true},
    "top_k": 6
  },
  "steps": [
    {"step": 1, "name": "pattern_match", "description": "算子计算流程匹配",
     "input": {"operator": "dgemm"}, "candidates": ["pat:gemm:mm_microkernel"],
     "selected": ["pat:gemm:mm_microkernel"], "reason": "算子注册表命中"},
    {"step": 2, "name": "strategy_recall", "description": "沿 PATTERN_OPTIMIZED_BY 召回策略",
     "candidates": ["stg:a", "stg:b"], "selected": ["stg:a", "stg:b"]},
    {"step": 3, "name": "condition_filter", "description": "硬条件过滤",
     "candidates": ["stg:a","stg:b","stg:c"], "selected": ["stg:a","stg:c"],
     "dropped": [{"id": "stg:b", "reason": "dtype=single 不满足 dtype=double"}],
     "reason": "谓词求值"},
    {"step": 4, "name": "hardware_filter", "description": "ISA/微架构能力过滤",
     "dropped": [{"id": "stg:d", "reason": "requires AVX512, profile 仅 AVX2"}]},
    {"step": 5, "name": "conflict_resolve", "description": "冲突消解",
     "dropped": [{"id": "stg:e", "reason": "与 stg:a 在寄存器预算上冲突"}]},
    {"step": 6, "name": "score_and_select", "description": "打分排序 + top_k",
     "scores": {"stg:a": 0.81, "stg:c": 0.66}, "selected": ["stg:a", "stg:c"]},
    {"step": 7, "name": "pack", "description": "生成 StrategyPack（注入 prompt）",
     "result": {"strategies": ["stg:a","stg:c"], "templates": ["tpl:..."]}}
  ],
  "result": {
    "strategies": ["stg:a", "stg:c"],
    "pack": {"schema_version": "pack-v1", "strategies": [{...}]},
    "prompt_preview": "B3 prompt 的前 500 字"
  },
  "created_at": "2026-11-12T18:00:00+08:00"
}
```

**硬要求**：每个 step 必须有 `name/description/candidates/selected`，被淘汰的必须有 `dropped[].reason`——这是可视化"推理过程"的数据源。

## 3. 实体/关系（v1.5，固定）

**实体（12）**：`Operator`、`ComputationPattern`、`CodeCharacteristic`、`OptimizationStrategy`、`ImplementationTemplate`、`TunableParameter`、`ISACapability`、`Microarchitecture`、`HardwareProfile`、`PerformanceEvidence`、`SourceArtifact`、`FailureCase`。

**关系（14）**：
`OPERATOR_HAS_PATTERN`、`PATTERN_HAS_CHARACTERISTIC`、`PATTERN_OPTIMIZED_BY`、`STRATEGY_HAS_PARAMETER`、`STRATEGY_HAS_TEMPLATE`、`STRATEGY_APPLIES_WHEN`、`STRATEGY_REQUIRES_ISA`、`STRATEGY_TARGETS_UARCH`、`STRATEGY_EVIDENCED_BY`、`EVIDENCE_ON`、`EVIDENCE_SOURCE`、`STRATEGY_COMPOSES_WITH`、`STRATEGY_CONFLICTS_WITH`、`STRATEGY_GENERALIZES`。

**条件谓词（8）**：`dtype`、`precision`、`stride_x`、`stride_y`、`loop_depth`、`size_regime`、`aligned`、`tail_divisible`、`structural`、`reduction`（以 `condition_vocab.yaml` 为准）。

## 4. CLI（必须实现）

```bash
python -m kgagent.cli build --offline           # S0→S6，全部可离线（不许联网）
python -m kgagent.cli build --llm               # S3/S4 调 LLM（有 cache，失败降级 offline）
python -m kgagent.cli validate                  # schema + 不变量，退出码非 0 表示失败
python -m kgagent.cli retrieve --op dgemm --profile x86_avx2 --top-k 6 --trace-out data/traces/
python -m kgagent.cli export-viz                # 生成 viz/graph_data.js 和 traces.js
python -m kgagent.cli stats                     # 节点/边/覆盖率/条件可执行率
```

**离线 build 的数据来源**（必须支持，不依赖 API）：
- 已有 4 个算子的分析 JSON：`开题/analyze_OB/20251114_153700/analysis_results/*.json`（gemm/trmm/hemv/axpy）；
- 其余 8 个算子：用 S1 静态特征 + S2 差分规则（不需要 LLM）产出基础策略；
- 性能证据：`openblas_newtest1/*_log.txt`（QEMU，只填 `evidence.status="qemu_only"`，不进性能排序）与 `data/riscv_log_summary.csv`。

## 5. 可视化（`viz/graph.html`）

自包含单页（本地 `assets/vis-network.min.js`，不依赖 CDN），必须支持：

1. **图探索**：按实体类型着色/过滤；节点大小=度数；点击节点显示属性、来源文件/行号、关联证据。
2. **路径跳转**：输入 A、B 两个节点，BFS/最短路高亮，面板列出路径上的关系（带关系属性）。
3. **推理链**：点击策略 → 展示"适用条件 → 硬件约束 → 参数推导 → 证据 → 来源"的链式卡片（数据来自节点 properties + 边）。
4. **检索回放**：选择算子 + 硬件 profile → 逐 step 播放 trace：高亮候选/选中/淘汰节点，右侧显示每步 `description/reason/dropped`，支持"上一步/下一步/自动播放/重置"。
5. **搜索**：按名称/类型过滤节点。

`viz_export.py` 生成：
- `graph_data.js`：`window.KG_DATA = {nodes:[...], edges:[...], stats:{...}};`
- `traces.js`：`window.KG_TRACES = {trace_id: {...}, ...};`

## 6. 验收标准（pipeline + viz 共同）

- [ ] `python -m kgagent.cli build --offline` 从零跑通，生成非空 `nodes.jsonl/edges.jsonl`，`manifest.json` 有版本四元组与统计。
- [ ] `validate` 通过：所有节点 id 唯一、边端存在、类型在闭集、策略条件 `condition_status` 非空、性能数字只出现在 PerformanceEvidence。
- [ ] 至少 20 条策略具备：≥2 个可执行条件、1 个 template、1 个 provenance（file+line）、1 个 hardware_profile 绑定。
- [ ] `retrieve --op dgemm --profile x86_avx2` 输出 trace，至少有 7 个 step，含 dropped 原因。
- [ ] `graph.html` 双击可打开（file:// 即可），完成上述 5 项交互；不联网也能用。
- [ ] `tests/` 至少 6 个测试通过。

## 7. 分工与写权限（避免冲突）

| 角色 | 只写 | 只读 |
|---|---|---|
| pipeline engineer | `kg_agent/src/**`、`kg_agent/schema/**`、`kg_agent/tests/**`、`kg_agent/data/**`、`kg_agent/config.yaml`、`kg_agent/README.md` | `kg_agent/SPEC.md`、`viz/` |
| viz engineer | `kg_agent/viz/**` | `kg_agent/SPEC.md`、`kg_agent/data/*.jsonl`、`schema/*` |
| Lead | `kg_agent/SPEC.md`、集成与验收 | 全部 |

> 两个工程师共用 `data/` 的**契约**，但 viz 只读 data、pipeline 只写 data；viz 可用 Lead 提供的 `sample_data` 先开工（见 §2 样例），不必等 pipeline 完成。

---

## 8. v1.5 多架构补充（2026-11-12，已实现）

> 背景：用户问"输入是否只限一个架构"；原 KG 只有 generic + RISC-V RVV。现已扩展为 **x86_AVX2/AVX512 + ARM_NEON/SVE + RISC-V RVV + generic**。

### 8.1 数据契约增量

- implementation / `SourceArtifact` 增加：`arch`（`generic|x86_avx2|x86_avx512|arm_neon|arm_sve|rvv`）、`isa`、`uarch`、`source_lang`（`c|asm`）、`method`（`diff|static`）；
- `OptimizationStrategy` 增加：`arch_scope`（`arch_agnostic|parameterized|arch_specific`）、`arch_evidence[]`、`arch_parameters{}`（按 arch 保留参数，如 `vector_width` / `lmul` / `unroll_factor`）；
- 新增关系：`STRATEGY_IMPLEMENTED_ON`（策略 → 架构实现/证据）；原有 14 类关系不变；
- 新增数据文件：`data/multiarch_manifest.jsonl`、`data/arch_coverage.json`（arch×op 矩阵、跨架构共有策略、ops_by_arch、strategies_by_arch）。

### 8.2 schema 闭集增量

- `arch` 枚举：`generic, x86_avx2, x86_avx512, arm_neon, arm_sve, rvv`；
- `arch_scope` 枚举：`arch_agnostic, parameterized, arch_specific`；
- 硬件 Profile：`x86_avx2`、`x86_avx512`、`arm_neon`（128bit）、`arm_sve`（256bit）、`k1_rvv256`、`qemu-riscv64-vlen256`。

### 8.3 CLI 增量

```bash
python -m kgagent.cli stats --matrix     # arch×op 覆盖矩阵 + 跨架构共有策略
python -m kgagent.cli retrieve --op dgemm --profile arm_sve   # 新 profile 可用
```

### 8.4 验收（已达成）

- ≥8 算子 ≥2 架构（实际 13/14）；
- 262 策略带 `arch_scope`（93 arch_agnostic / 30 parameterized / 139 arch_specific）；
- 30 条跨架构共有策略（带各架构参数）；
- validate 0 error；43 测试通过；两次 build 字节级幂等。

### 8.5 可视化性能契约

- `viz/graph.html`：默认性能模式（只渲染核心实体 + 标签按需 + 关 physics），保留 5 项交互，`?selftest=1` 18/18；
- `viz/graph_fast.html`：Sigma.js WebGL 版，默认算子 2 跳子图（~60 节点），「加载全图」862 节点/3656 边；`?selftest=1` 12/12；本地 `fast_assets/`，不联网。
