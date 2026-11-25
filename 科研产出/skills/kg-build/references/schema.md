# Schema：实体、关系、策略卡

两套口径都要兼容：`analyze_OB1` 的**语义 schema**（7 实体 / 12 关系）与 `kg_agent` 的**可执行 schema**（12 实体 / 14 关系）。图以可执行字段为主键，语义字段做注解。

## 实体（默认 7 类）

| 类型 | 含义 | 必填 | 来源 |
|---|---|---|---|
| `OptimizationStrategy` | 一条可执行的优化策略 | name, level, action, conditions, parameters, template_id, provenance | 分析 + 抽取 |
| `SourcePattern` | 被优化的计算流程/阶段 | name, stage, operator | 分析 |
| `OptimizationPrinciple` | 为什么有效的原理 | name, rationale | 分析（LLM 只写解释） |
| `TunableParameter` | 可调参数 | name, range, unit, value_in_source | 分析 + 静态特征 |
| `CodeCharacteristic` | 代码静态特征 | char_type, value, extraction_method | 静态分析（不由 LLM 编） |
| `ArchitectureCapability` | 硬件/ISA 能力 | isa, feature, element_widths | 配置/硬件手册 |
| `CodeExample` | 代码片段 | snippet, language, provenance | 源码（必须带行号） |

可选扩展（`kg_agent` 口径）：`Operator`、`ComputationPattern`、`ImplementationTemplate`、`ISACapability`、`Microarchitecture`、`HardwareProfile`、`PerformanceEvidence`、`SourceArtifact`、`FailureCase`。

## 关系（默认 12 类）

| 类型 | 头 → 尾 | 说明 |
|---|---|---|
| `OPTIMIZES_PATTERN` | Strategy → Pattern | 这条策略作用于哪个流程 |
| `TARGETS` | Strategy → ArchitectureCapability | 面向哪个硬件能力 |
| `APPLIES_WHEN` | Strategy → CodeCharacteristic | 适用条件（可执行谓词） |
| `REQUIRES` | Strategy → ArchitectureCapability | 硬性依赖 |
| `HAS_PARAMETER` | Strategy → TunableParameter | 参数（边属性带 value_in_instance） |
| `IS_ILLUSTRATED_BY` | Strategy → CodeExample | 代码示例 |
| `GENERALIZES` | Strategy → Strategy | 抽象/同族 |
| `COMPOSES_WITH` | Strategy → Strategy | 可组合（标明顺序） |
| `INSTANCE_OF` | Strategy → Strategy | 具体实例指向抽象策略 |
| `CONFLICTS_WITH` | Strategy → Strategy | 互斥/资源冲突（必须对称） |
| `ANALOGOUS_TO` | Strategy → Strategy | 跨架构类比 |
| `HAS_CHARACTERISTIC` | Pattern → CodeCharacteristic | 流程的静态属性 |

## 策略卡（最重要）

```json
{
  "id": "stg:axpy:vectorize_x86_avx2",
  "type": "OptimizationStrategy",
  "name": "x86 AVX2 向量化内核",
  "properties": {
    "level": "code",
    "action": "vectorize",
    "conditions": [
      {"pred": "dtype", "op": "in", "value": ["d"], "hard": true},
      {"pred": "stride_x", "op": "eq", "value": 1, "hard": true},
      {"pred": "size_regime", "op": "in", "value": ["medium", "large"], "hard": false}
    ],
    "parameters": [
      {"name": "vector_width", "value": 256, "range": [128, 256, 512], "unit": "bits"}
    ],
    "template_id": "tpl:axpy:vectorize_x86_avx2",
    "arch_scope": "arch_specific",
    "performance_claim": null,
    "evidence_ids": ["ev:axpy:geomean"]
  },
  "provenance": {"source_file": "daxpy.c", "line_start": 46, "line_end": 213, "sha256": "…"},
  "confidence": 0.8,
  "status": "verified"
}
```

## 条件谓词闭集（10 个）

`dtype`、`precision`、`stride_x`、`stride_y`、`loop_depth`、`size_regime`、`aligned`、`tail_divisible`、`reduction`、`structural`（扩展：`isa`、`vector_bits`、`vector_regs`）。

规则：
- 每条策略 ≥2 条 `hard: true` 条件；
- 值域用枚举/区间，不用自然语言；
- 谓词必须是**对"代码特征"可求值**的，不是对"意图"。

## 质量规则（与门对应）

| 字段 | 规则 | 门 |
|---|---|---|
| 实体类型 | 必须在闭集 | A |
| id | 唯一；建议保留可读前缀（`stg:`/`pat:`/`par:`） | A |
| 关系端点 | 必须存在 | A |
| provenance | `source_file` + `line_start/line_end`；配置型实体豁免行号 | B |
| conditions | ≥2 硬条件、谓词闭集 | C |
| parameters | `value ∈ range`；单位一致 | C |
| name | 同类型同名必须合并/别名 | D |
| performance_claim | 有声明必须有 `evidence_ids`，否则 `claim_status=unverified` | E |

## 命名规范（避免歧义）

- 来源不要编码进后缀：用 `provenance.source_channel`（`analysis|static|diff|multiarch|cross_arch`），不要用 `_s/_d/_a`；
- 层级用 `level`（`algorithm|code|instruction`），不要塞进名字；
- 同一策略的跨架构变体用 `arch_scope` + `arch` 字段，不要造新 id 前缀。
