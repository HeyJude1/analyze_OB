# Schema v1.5 补丁清单（2 天，先补再实验）

> 2026-11-12｜Lead｜依据：`kg_adequacy_audit.md`（对 206 条策略、805 条关系的实测审计）
> **结论：不要重做 Schema v2，也不要直接开实验。** 用 2 天把 3 个"会在实验里变成混杂因素"的缺口补上，冻结为 v1.5，再跑 E1–E7。

---

## 0. 为什么不是"重做"，也不是"直接实验"

| 做法 | 问题 |
|---|---|
| 直接开实验 | 当前 KG 的硬件是名词、条件是自由文本、实施是段落、收益是形容词、冲突关系 0 条 → **B3 与 B2 的差别只剩"文本组织方式"**，实验一定测不出结构增量，还无法归因 |
| 重做 Schema v2 | 11 实体/15 关系/12 条不变量 + 差分抽取 + 对齐重跑，至少 3–4 周，12 周排期直接崩 |
| **v1.5 补丁（本文件）** | 只动 3 样：**硬件可计算、条件可执行、实施可注入 + 证据可核查**；2 天完成，Top 20 策略手工，其余保留旧字段 |

**判断标准**：补完后再跑一遍 `kg_adequacy_audit` 的 10 个 competency questions，**至少 7 个变成"能答"**（当前是 2 能 + 3 部分 + 5 不能），才允许开 Pilot。

---

## 1. 补丁一：硬件 Profile（0.5 天）

**问题**：206 条策略里只有 1 条硬件描述含数值；`TARGETS` 指向"Cache/SIMD/寄存器"这类名词，无法过滤。
**做法**：新增 `hardware_profiles.yaml`（两份就够：x86 与 K1），策略边加 `hardware_profile_ids` 与 `parameter_derivation`。

```yaml
# kg/hardware_profiles.yaml（v1.5）
x86_avx2:
  vendor: Intel
  uarch: "Haswell/Skylake 类"
  isa: ["AVX2", "FMA", "SSE4.2"]
  vector_bits: 256
  fp64_lanes: 4
  vector_regs: 16
  l1d_kb: 32
  l2_kb: 256
  l3_mb: 8            # 按实际机器改
  cacheline_b: 64
  cores: 8
  freq_ghz: 3.0       # 实测锁频后填
  peak_gflops_fp64: 0 # 实验第 1 天用微基准实测后填，不填厂商标称
  stream_bw_gbps: 0   # 同上

x86_avx512:
  isa: ["AVX512F", "AVX512DQ", "FMA"]
  vector_bits: 512
  fp64_lanes: 8
  vector_regs: 32
  # 其余同 x86 机器实测

k1_rvv256:
  vendor: SpacemiT
  uarch: "K1 (BPI-F3)"
  isa: ["RVV 1.0", "Zba", "Zbb", "Zbc", "Zbs"]
  vector_bits: 256
  fp64_lanes: 4        # VLMAX(e64,m1)=4
  lmul_max: 8
  vector_regs: 32
  l1d_kb: 32
  l2_kb: 512
  l3_mb: 0
  cacheline_b: 64
  cores: 8
  freq_ghz: 1.6        # 锁频后填
  peak_gflops_fp64: 0
  stream_bw_gbps: 0
```

**策略侧只加 3 个字段**（不新建实体，直接挂在策略 YAML 上）：

```yaml
strategy_id: stg:gemm:unroll_pointer
hardware_profile_ids: [x86_avx2, x86_avx512]
parameter_derivation:
  unroll_factor: "min(4, L1d_kb / (2 * dtype_bytes * vector_bits/8))"
  register_block_n: "vector_regs / (2 * lmul)"
evidence_status: pending   # pending | measured | literature
```

---

## 2. 补丁二：可执行条件（0.5–1 天）

**问题**：`applicability_conditions` 是自由中文（例："内层循环体简单且无复杂控制流，循环次数较大…"），检索时无法判断真假。
**做法**：定义 8 个受控谓词，Top 20 策略改写成谓词列表；其余标 `condition_status: text`（检索时降权，不硬过滤）。

```yaml
# kg/condition_vocab.yaml（v1.5）
predicates:
  dtype:        {type: enum, values: [s, d, c, z]}
  precision:    {type: enum, values: [single, double]}
  stride_x:     {type: int,  op: [eq, ne, gt]}
  stride_y:     {type: int,  op: [eq, ne, gt]}
  loop_depth:   {type: int,  op: [ge, le, eq]}
  size_regime:  {type: enum, values: [small, medium, large]}   # 按算子给阈值
  aligned:      {type: bool}
  tail_divisible: {type: bool}     # n % (vector_bits/8/dtype_bytes) == 0
  structural:   {type: enum, values: [symmetric, triangular, hermitian, banded, none]}
  reduction:    {type: bool}       # 是否含跨迭代归约
```

**策略改写示例**（真实策略"循环展开与指针预计算"）：

```yaml
strategy_id: stg:gemm:unroll_pointer
conditions:
  - {pred: loop_depth, op: ge, value: 2, hard: true}
  - {pred: dtype, op: in, value: [s, d], hard: true}
  - {pred: size_regime, op: in, value: [medium, large], hard: false, weight: 0.5}
condition_status: executable
```

**验收**：Top 20 策略每条至少 2 个谓词；跑 `condition_status: executable` 的比例 ≥ Top20 的 100%、全量的 ≥10%（其余标 text 即可）。

---

## 3. 补丁三：实施模板 + 证据 + 溯源（1 天）

**问题**：`implementation_pattern` 是段落、`code_example` 是片段、`performance_impact` 是形容词、来源只有文件名。
**做法**：Top 20 策略各补一个 `template`（骨架 + 槽位）、一个 `evidence`（先 `pending`，实验后回填）、一个 `source`（文件 + 行号）。

```json
{
  "strategy_id": "stg:gemm:unroll_pointer",
  "canonical_name": "循环展开与指针预计算",
  "family": "循环展开",
  "level": "code",
  "action": "unroll",
  "conditions": [
    {"pred": "loop_depth", "op": "ge", "value": 2, "hard": true},
    {"pred": "dtype", "op": "in", "value": ["s", "d"], "hard": true}
  ],
  "hardware_profile_ids": ["x86_avx2", "x86_avx512", "k1_rvv256"],
  "parameter_derivation": {
    "unroll_factor": "min(4, vector_regs / 4)"
  },
  "template": {
    "language": "c",
    "skeleton": "for (i = 0; i + {{unroll_factor}} <= n; i += {{unroll_factor}}) {\n  {{load_and_op}}\n}\nfor (; i < n; ++i) { {{tail_op}} }",
    "slots": [
      {"name": "unroll_factor", "type": "int", "range": [2, 8], "derived_from": "vector_regs"},
      {"name": "load_and_op", "type": "code", "from": "strategy.implementation_pattern"}
    ],
    "check_hints": ["确保 tail 循环处理 n % unroll_factor", "避免指针别名导致编译器不敢向量化"]
  },
  "evidence": {
    "status": "pending",
    "metric": "speedup_vs_C1",
    "value": null,
    "platform": "x86_avx2",
    "baseline": "gcc -O3 -march=native",
    "protocol": "5 warmup + 20 次中位数，见 08b",
    "measured_at": null
  },
  "source": {
    "file": "openblas_newtest1/16_dgemm_op5/dgemm_k.c",
    "line_start": 0,
    "line_end": 0,
    "commit_or_version": "OpenBLAS 0.3.27.dev",
    "extractor_version": "kg-v1.5"
  },
  "condition_status": "executable",
  "confidence": 0.8
}
```

**注意**：
- `performance_impact`/`trade_offs` 保留为"解释文本"，**不得参与排序**；排序只能用 `evidence.value`（未测时用 `evidence_status: pending` 降权）。
- `source.line_start/line_end` 可以先由 grep 定位到函数定义行；拿不到精确行就标 `line_confidence: approx`，但必须有。
- Top 20 怎么选：从 206 条里按"跨算子出现次数 + 是否在 S1 的 6 个算子中 + 覆盖 L1/L2/L3"挑，先覆盖 daxpy/dgemv/dgemm/dtrmm 的高频策略（消歧后 45 个簇里的前 20）。

---

## 4. 两天的执行顺序

| 半天 | 任务 | 产出 | 验收 |
|---|---|---|---|
| D1 上午 | 写 `hardware_profiles.yaml`；在板上/x86 上测 `freq/peak/stream_bw` 回填 | 2 份 profile | 每份 ≥12 个字段 |
| D1 下午 | 定义 `condition_vocab.yaml`；把 Top 20 策略的 `applicability_conditions` 改写成谓词 | Top20 策略 YAML | 每条 ≥2 谓词 |
| D2 上午 | 给 Top 20 写 `template`（骨架 + 槽位 + check_hints） | Top20 模板 | 20/20 有 skeleton + ≥1 slot |
| D2 下午 | 给 Top 20 补 `source`（文件+行号）；跑一遍 adequacy audit；冻结 v1.5；写 `schema_version: v1.5` | 冻结记录 | 10 个 CQ 中 ≥7 个"能答" |

---

## 5. 冻结规则（写进实验日志）

1. **v1.5 冻结后，实验期间只允许"回填 evidence"**（数值、平台、日期），不允许新增实体/关系/条件类型。
2. 如果 Pilot（W5）失败，先回答"是知识缺失、检索错、还是生成能力不足"三选一；**只有确认是知识缺失**，才允许开 v2 补丁，且只补缺口。
3. 论文里要如实写："Schema 经历 v1（知识清单）→ v1.5（可执行补丁）→ 实验冻结"，这本身是工程贡献的一部分，不是黑历史。

---

## 6. 不要做的 5 件事

- ❌ 不要为 206 条策略全部手工写模板（写 Top 20 即可，其余标 `template_status: none`，检索时降权）。
- ❌ 不要重新跑抽取（现有 206 条够用；重跑只会引入新噪声）。
- ❌ 不要引入 Neo4j（文件/SQLite + NetworkX 足够）。
- ❌ 不要把 `performance_impact` 当收益排序依据（它是 LLM 形容词）。
- ❌ 不要新增"知识涌现/推理"模块（12 周内无法验证，答辩会被问穿）。
