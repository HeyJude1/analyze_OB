# kg-build

子代理驱动的知识图谱构建 skill：**分析 → 抽取 → 消歧 → 建图 → 校验 → 网页**。

## 快速开始

```bash
# 1) 校验图（门 A–D）
python scripts/validate_kg.py --kg demo/mini_kg.json --strict

# 2) 导出网页（离线单文件）
python scripts/export_web.py --kg demo/mini_kg.json --out demo/mini_graph.html --title "axpy 策略图谱"

# 3) 也支持现有 kg_agent 格式
python scripts/export_web.py --nodes <data>/nodes.jsonl --edges <data>/edges.jsonl --out out/graph.html
```

## 安装为 DSH skill

```powershell
# 从本目录执行；写到用户级 skills 目录（需要写权限）
powershell -ExecutionPolicy Bypass -File install.ps1
# 或手动：把本目录复制到 %USERPROFILE%\.agents\skills\kg-build
```

安装后**新开一个会话**，在可用技能列表里应出现 `kg-build`。

## 目录

```text
SKILL.md                     # 入口：何时用、流程、门、命令
references/schema.md         # 7 实体 / 12 关系 / 策略卡 / 谓词闭集
references/contracts.md      # 文件布局、kg.json 契约、门与回退
references/orchestration.md  # 子代理派发（串行 / workflow 扇出 / 离线入口）
agents/*.md                  # 六个角色卡（作为子代理 prompt）
scripts/validate_kg.py       # 门 A–D 检查
scripts/export_web.py        # 单文件 HTML 导出
scripts/orchestrate.workflow.js  # workflow 扇出示例
demo/mini_kg.json            # 契约示例（可直接跑）
```

## 设计原则

1. **LLM 语义、规则事实、门质量**：LLM 不写性能数字，事实来自源码行号。
2. **没有行号不算知识**：门 B 强制 `source_file + line_start/line_end`。
3. **条件必须是谓词**：门 C 强制 ≥2 条硬条件 + 闭集谓词。
4. **消歧留证据**：合并必须写 `merges.json`，保留 alias 与理由。
5. **门不过不导出**：先出质量报告，再决定是否交付。

## 已知边界

- `analyzer` 需要 `DASHSCOPE_API_KEY`（qwen-plus）；没有 key 走离线入口（吃已有 analysis JSON）。
- 旧 `analyze_OB1` 的 conditions 是散文，`extractor` 转谓词时会有损耗，报告里要看 `condition_status=text` 占比。
- 大图渲染用 vis-network，>5000 节点建议改用 `kg_agent/viz/graph_fast.html`（sigma）或先截断属性。
