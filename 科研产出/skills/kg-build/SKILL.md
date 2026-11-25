---
name: kg-build
description: 用 LLM 子代理从源码构建知识图谱：三级代码分析 → 实体/关系抽取 → 实体消歧与合并 → 组装建图 → 校验门 → 单文件网页导出。Use when the user asks to 建图、抽取实体与关系、实体消歧/合并、知识图谱可视化, or to run/repair the OpenBLAS optimization-strategy KG pipeline (analyze_OB1 / kg_agent).
---

# KG Build：子代理驱动的知识图谱构建

目标：把"源码里的优化经验"变成**可追溯、可检索、可展示**的图。
一句话原则：**LLM 负责语义，规则负责事实，门负责质量。**

## 什么时候用

- 用户要"从代码/文档抽实体和关系，建一张图"；
- 用户要"把已有分析结果（JSON）变成图并可视化"；
- 用户要"修/扩/校验"已有图（消歧、合并、补溯源、查重复）。

不适用：纯文档问答（那是 RAG）、只画示意图（那是画图）。

## 两条入口（先确认走哪条）

| 入口 | 前提 | 起点 | 适合 |
|---|---|---|---|
| **A. 离线**（推荐先跑通） | 已有 `analysis_results/*.json`（如 `analyze_OB1/20251114_153700/analysis_results`） | 直接进 `extractor` | 无 API key、要复现旧结果 |
| **B. 在线** | `DASHSCOPE_API_KEY`（qwen-plus）+ 源码目录 | 先跑 `analyzer` | 新算子/新语料 |

两条入口之后共用 `extractor → resolver → builder → verifier → visualizer`。

## 六个角色（子代理）

角色卡在 `agents/`。**派发方式**：读对应 `.md` 的全文，作为子代理 prompt 的第一段，再附上输入路径与本轮具体任务。

| 角色 | 文件 | 输入 | 输出 |
|---|---|---|---|
| 分析师 | `agents/analyzer.md` | 一个源码文件 | `work/analysis/<op>/<file>.json` |
| 抽取器 | `agents/extractor.md` | analysis JSON | `work/entities/*.jsonl`、`work/relations/*.jsonl` |
| 消歧器 | `agents/resolver.md` | 实体/关系 JSONL | `work/merges.json`（alias + 合并决策） |
| 建图器 | `agents/builder.md` | 三者 | `out/kg.json` + `out/quality.json` |
| 校验员 | `agents/verifier.md` | `out/kg.json` + 源码 | `out/verify.json`（抽样判定 + precision） |
| 可视化 | `agents/visualizer.md` | `out/kg.json` | `out/graph.html` |

串行执行；`analyzer` 可按文件并行（见 `references/orchestration.md`）。

## 流程与五道门

```text
analyze → extract → resolve → build → verify → export
                     ↑          ↑        ↑
                  门 A/B     门 A–D   门 E（人工抽样）
```

| 门 | 查什么 | 不过怎么办 |
|---|---|---|
| A schema | 类型闭集、id 唯一、关系端点存在 | 退回上一步修，不允许"先放着" |
| B 溯源 | 每个实体有 `source_file` + `line_start/line_end` | 退回 extractor 补行号；补不了就降级为 `status=abstract` |
| C 可执行 | 策略条件 ≥2 条硬谓词、谓词在闭集、参数在声明域 | 退回 builder 修条件/参数 |
| D 消歧 | 同类型同名实体必须合并或标别名 | 退回 resolver |
| E 证据 | 性能声明必须指向证据节点，否则标 `claim_status=unverified` | 降级，不删 |

**门不过就不导出**。先写 `out/quality.json`，报告里写清楚哪条卡住、为什么。

## 命令

```bash
# 校验（门 A–D，退出码 0/2/1 = 通过/仅警告/失败）
python scripts/validate_kg.py --kg out/kg.json [--strict] [--json out/validate.json]

# 网页（单文件、离线可开）
python scripts/export_web.py --kg out/kg.json --out out/graph.html [--title "..."]

# 也接受现有格式（kg_agent）
python scripts/export_web.py --nodes data/nodes.jsonl --edges data/edges.jsonl --out out/graph.html
```

## 质量红线（违反就别交付）

1. **没有行号的实体不算知识**——只有文件名、没有行号，等于不可核。
2. **条件不写成谓词就不算可执行**——"适合大矩阵"不是条件；`size_regime in [large]` 才是。
3. **性能数字只能来自日志/实测**，不能来自 LLM；LLM 说"显著提升"的，标 `claim_status=unverified`。
4. **同名必合并**——合并要留 alias，不能直接覆盖。
5. **图必须可重建**：保存 `run.json`（模型、prompt 版本、输入 sha256、时间）。

## 失败与回退

- 没有 API key：只走入口 A；缺 analysis JSON 就明确告诉用户"需要先跑 analyze_OB1 或提供 JSON"。
- analyzer 输出 JSON 解析失败：重试 ≤3 次，仍失败则记录 `status=failed`，跳过该文件并在报告里写明覆盖率。
- resolver 合并冲突（同一实体进两个组）：保留置信度高的一组，另一组降级为 `alias`，不丢数据。
- verifier 抽检 precision < 0.8：**停止导出**，把错例回灌给 extractor 重跑一轮（最多 2 轮）。

## 与现有两个项目的关系

- `analyze_OB1`（LLM 语义图）：提供 `analysis_results` 与"7 实体/12 关系"语义 schema；本 skill 的入口 A 直接吃它。
- `kg_agent`（规则可执行图）：提供闭集谓词、行号溯源、证据与校验思路；本 skill 的 `validate_kg.py` 兼容它的 `nodes.jsonl/edges.jsonl`。
- 不要整图合并两者：以可执行图为主键，语义图只做注解（见 `科研产出/26_analyze_OB1_AI建图评估.md`）。

## 交付物

```text
out/kg.json           # 图（entities + relations + run）
out/quality.json      # 门 A–E 报告 + 统计
out/verify.json       # 抽样校验（可选，但建议有）
out/graph.html        # 单文件网页
work/                 # 中间产物（analysis/entities/relations/merges）
```

细节：`references/schema.md`（实体/关系/策略卡）、`references/contracts.md`（文件与 JSON 契约、门）、`references/orchestration.md`（子代理派发与并行）。
