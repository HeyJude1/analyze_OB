# Orchestration：怎么把六个角色派出去

## 四条原则

1. **角色卡 + 路径，不塞正文**：子代理 prompt = 角色卡全文 + 输入/输出路径 + 本轮约束。源码让子代理自己读，避免主代理上下文爆掉。
2. **产物落盘，不靠对话传递**：每个角色写固定路径（见 `contracts.md`），下一步只读文件。
3. **门在阶段之间**：`build` 前必须 `validate_kg.py` 通过；`export` 前必须 `verify` 达标（或明确降级）。
4. **重试有上限**：单文件解析失败 ≤3 次；verifier 不过 ≤2 轮回灌。

## 模式 1：串行派发（默认，最稳）

```text
主代理：
  读 agents/analyzer.md → 派给 subagent（附 file 路径、op、schema 摘要）
  → 等产物 → 校验 JSON → 下一个文件
  → 全部完成后读 agents/extractor.md → 派发（附 analysis 目录）
  → resolver → builder → verifier → visualizer
```

适合：文件少（<10）、要边跑边看、需要人盯着改 prompt。

## 模式 2：workflow 扇出（多文件并行）

`workflow` 脚本**没有文件系统权限**，所以路径必须通过 `args` 传入，读写都由子代理完成。

```js
// args = { skillDir, files: [{op, path}], outDir, schemaHint }
const files = args.files || [];
const results = await parallel(files.map(f => () => agent(
  `你是 kg-build 的 analyzer 子代理。\n` +
  `先读 ${args.skillDir}/agents/analyzer.md 并严格照做。\n` +
  `算子: ${f.op}\n源文件: ${f.path}\n输出到: ${args.outDir}/analysis/${f.op}/\n` +
  `schema 摘要: ${args.schemaHint}\n` +
  `只输出 JSON 结果，字段 out_path/strategies/patterns/errors。`,
  {
    label: `${f.op}/${f.path.split(/[\\/]/).pop()}`,
    phase: 'analyze',
    schema: {
      type: 'object',
      properties: {
        ok: { type: 'boolean' },
        op: { type: 'string' },
        file: { type: 'string' },
        out_path: { type: 'string' },
        strategies: { type: 'integer' },
        patterns: { type: 'integer' },
        errors: { type: 'array', items: { type: 'string' } }
      },
      required: ['ok', 'op', 'file', 'out_path'],
      additionalProperties: true
    }
  }
)));
return results.filter(Boolean);
```

拿到结果后，主代理再串行派 `extractor`（按算子聚合）、`resolver`（全局）、`builder`、`verifier`、`visualizer`。

## 模式 3：离线入口（无 API key）

跳过 `analyzer`，直接把已有 `analysis_results/*.json` 喂给 `extractor`：

```text
读 agents/extractor.md → 派发（输入：analyze_OB1/20251114_153700/analysis_results）
```

注意：旧 JSON 的 `applicability_conditions` 是散文，extractor 必须**转成谓词**；转不了的标 `condition_status=text` 并计入门 C 的降级清单。

## 成本与时长（估算，非承诺）

| 阶段 | LLM 调用 | 说明 |
|---|---|---|
| analyzer | 1 次/文件（3 层可合 1 次或分 3 次） | 长文件先切函数，单次 ≤4k tokens 输出 |
| extractor | 1 次/算子（批量）或规则 | 有规则映射时可零 LLM |
| resolver | 0–1 次/簇 | 先规则/嵌入，再 LLM 判边界 |
| builder | 0 | 纯脚本 |
| verifier | 0–1 次/样本批 | 关键是读源码，不是让 LLM 凭空判 |
| visualizer | 0 | 脚本 |

## 上下文预算

- 角色卡控制在 ~1k tokens；
- 每次派发只带：角色卡 + 输入路径 + 输出路径 + 本轮 3 条以内约束；
- 中间产物用 JSONL，主代理只读汇总（`head`/计数），不整文件读入。

## 质量回环

```text
verifier.precision < 0.8
  → 从 verify.json 取 wrong/unsupported 样本
  → 作为反例附给 extractor（"这些是上次抽错的，避免同类错误"）
  → 重跑 extractor（仅问题算子）
  → 重新 build + verify
  → 仍 < 0.8：停止，写报告（不硬交付）
```
