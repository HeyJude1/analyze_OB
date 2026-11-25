# Contracts：文件布局、JSON 契约、门与回退

## 目录

```text
work/                          # 中间产物，可删可重跑
  analysis/<op>/<file>.json    # analyzer 输出（逐文件）
  entities/<op>.jsonl          # extractor 输出（逐算子）
  relations/<op>.jsonl
  merges.json                  # resolver 输出
out/
  kg.json                      # 最终图
  quality.json                 # 门报告 + 统计
  verify.json                  # 抽样校验（可选）
  graph.html                   # 单文件网页
  run.json                     # 运行清单（模型/版本/输入 sha256）
```

## kg.json

```json
{
  "schema_version": "kg-build/v1",
  "run": {
    "run_id": "20261116T0230",
    "created_at": "2026-11-16T02:30:00+08:00",
    "model": "qwen-plus-2025-09-11",
    "prompt_version": "kg-build/v1",
    "source_root": "…/openblas_src",
    "inputs": [{"path": "kernel/x86_64/daxpy.c", "sha256": "…"}]
  },
  "entities": [ … ],
  "relations": [ … ],
  "stats": {"entities": 0, "relations": 0}
}
```

## 实体

| 字段 | 必填 | 说明 |
|---|---|---|
| `id` | 是 | 全局唯一，建议 `type:op:action[:arch]` |
| `type` | 是 | 闭集（见 schema.md） |
| `name` | 是 | 人类可读；同类型内唯一（否则门 D） |
| `properties` | 否 | 结构化字段（conditions/parameters/level/action…） |
| `provenance` | 是* | `source_file`、`line_start`、`line_end`、`sha256`；配置型实体豁免 |
| `confidence` | 否 | 0–1；LLM 产出建议 ≤0.8，规则产出可高 |
| `status` | 是 | `raw` / `verified` / `rejected` / `abstract` / `failed` |

## 关系

`id`、`type`（闭集）、`head`、`tail`（实体 id）、`properties`（如 `value_in_instance`、`scope`、`role`）、`provenance`（可与头实体共享）。

## 门（build 前必跑）

| 门 | 脚本检查 | 失败处理 |
|---|---|---|
| A schema | 类型闭集、id 唯一、端点存在 | 停止导出，修数据 |
| B 溯源 | 行号覆盖（配置型豁免） | 退回 extractor；补不了降级 `abstract` |
| C 可执行 | ≥2 硬条件、谓词闭集、参数在域 | 退回 builder |
| D 消歧 | 同类型同名（归一化后） | 退回 resolver |
| E 证据 | 性能声明 → `evidence_ids` | 标 `claim_status=unverified`，不删 |

命令：

```bash
python scripts/validate_kg.py --kg out/kg.json --json out/quality.json
# 0=通过 2=仅警告 1=失败
```

## 合并记录（merges.json）

```json
{
  "round": 1,
  "merges": [
    {
      "keep": "stg:gemm:tile_x86_avx2",
      "merge": ["stg:gemm:tile_d"],
      "reason": "同算子同动作同层；参数差异已记入 arch_parameters",
      "evidence": ["name_similarity=0.93", "param_overlap=4/5"],
      "aliases": ["gemm 4x8 分块内核"]
    }
  ],
  "rejected": [{"pair": ["a", "b"], "reason": "跨架构但参数不可迁移"}]
}
```

规则：跨 `type` 不合并；跨 `op` 不合并；有冲突边（`CONFLICTS_WITH`）的两条合并前必须先解冲突；合并后保留所有 alias 与来源。

## 抽样校验（verify.json）

```json
{
  "sample_size": 50,
  "strategy": "分层抽：每类实体≥5，每族策略≥3",
  "labeling": "对照 source_file 的 line_start–line_end 逐条判定",
  "results": {"correct": 42, "wrong": 5, "unsupported": 3, "precision": 0.84},
  "errors": [{"entity": "…", "type": "wrong", "why": "条件写的是 stride=1，源码 46–60 行明确处理 stride>1"}],
  "verdict": "pass"
}
```

判定口径：
- `correct`：源码行能直接支持实体/关系的内容；
- `wrong`：源码行与内容矛盾；
- `unsupported`：源码行无法支持（不是错，但不算已验证）。

**precision < 0.8 → 停止导出，回灌错例重跑 extractor（最多 2 轮）。**

## 网页契约

`export_web.py` 输出单文件 HTML（内嵌 vis-network 与数据），必须显示：
实体/关系计数、溯源覆盖率、孤立节点、重名组；左侧类型/关系筛选与搜索；右侧详情（属性、`source_file:line`、邻居跳转）。

大图（>5000 节点）先降属性长度（`--max-prop-chars 200`）或改用项目自带的 sigma 渲染器（`kg_agent/viz/graph_fast.html`）。

## 失败与覆盖率

- 单文件失败：记录 `status=failed`，继续其他文件；报告写"覆盖率 = 成功文件/总文件"。
- 抽取为空：不要静默通过，报 `empty_extraction` 并附该文件的 schema 提示。
- 任何门失败：`out/quality.json` 里必须有 `errors[]`，且 `graph.html` 不生成（或生成时顶部红条标注"未通过门"）。
