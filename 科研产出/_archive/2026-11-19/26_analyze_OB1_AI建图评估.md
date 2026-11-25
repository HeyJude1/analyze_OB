# 26 analyze_OB1 评估：这是一条完整的 AI 建图流水线

> 2026-11-16｜结论：**是，而且它比你现在 `kg_agent` 里的"AI 部分"更彻底**——LLM 不只是起名，而是从代码里抽实体、抽关系、聚类、命名、合并、建图、检索，全链 AI。
> 但它属于"**生成型**"而不是"**验证型**"：没有任何一层对抽取结果做核对。

---

## 0. 一句话判断

| 问题 | 回答 |
|---|---|
| 它算 AI 建图吗？ | **算**。LLM 抽取 + 向量嵌入 + 聚类消解 + LLM 合并 + 图遍历 + 混合检索，是 2024–2025 年标准的 LLM-KG 流水线 |
| 它和你现在的 `kg_agent` 什么关系？ | **它是 `kg_agent` 的上游**：`analysis_results/20251114_153700` 就是 `config.yaml` 里 `paths.analysis_results` 指向的目录，61 条 `_a` 策略全部来自这里 |
| 它有什么问题？ | 无行号溯源、无可用谓词、效果判断是 LLM 散文、无校验层、本地副本缺 prompts/config 不可复现 |
| 它有什么价值？ | 语义/因果/示例知识、多轮实体消解纪律、4 阶段模式检测检索——这些是规则图给不了的；**而且它是一个天然的 baseline** |

---

## 1. 证据：它到底用了哪些 AI

```
模型        qwen-plus-2025-09-11（DashScope OpenAI 兼容接口），temperature=0.1，max_tokens=4000/4096
嵌入        DashScope text-embedding-v3，1024 维，归一化
向量库      Milvus（localhost:19530，database=code_op），COSINE，FLAT/IVF_FLAT
编排        LangGraph：algorithm → code → instruction → finalize（顺序链）
解析        LangChain StructuredOutputParser + 手写 JSON 兜底 + 最多 3 次重试
合并        LLM 逐对融合实体字段（_merge_entity_content_with_llm），多轮（默认 3 轮）
检索        LangChain AgentExecutor + tools（图遍历/向量检索/模式检测）
```

**关键点**：这不是"调一次 GPT 抽三元组"，而是**七个阶段、三次 LLM 介入、两轮实体消解**的工程化流水线。

## 2. 七段流程

```text
① 三级分析（workflow.py，LangGraph）
   源码全文 → LLM 分别产出 算法层 / 代码层 / 指令层 优化策略
   每条策略字段：optimization_name, level, description{rationale, implementation_pattern,
                 performance_impact, trade_offs}, applicability_conditions,
                 tunable_parameters, related_patterns, target_hardware_feature, code_example
        ↓
② 实体/关系抽取（extractor.py）
   7 类实体：optimization_principle, code_characteristic, source_pattern,
            architecture_capability, optimization_strategy, tunable_parameter, code_example
   12–13 类关系：OPTIMIZES_PATTERN / TARGETS / APPLIES_WHEN / REQUIRES / HAS_PARAMETER /
                IS_ILLUSTRATED_BY / GENERALIZES / COMPOSES_WITH / INSTANCE_OF /
                CONFLICTS_WITH / ANALOGOUS_TO / HAS_CHARACTERISTIC
   每个实体带 embedding + source_file + code_snippet
        ↓
③ 实体聚类（cluster.py）
   同类型实体两两余弦相似；贪心选簇心；一个实体只属于一个簇
        ↓
④ 簇内精炼（refine.py，LLM）
   对每个簇做语义分组 + 命名 + is_primary；单实体簇跳过（省 API）
        ↓
⑤ 多轮合并（merger.py + alignment.py，LLM）
   合并实体字段（LLM 融合）、重定向关系边、累积 strategy 上下文；
   跑 3 轮，每轮备份；导出 pattern 频次 CSV
        ↓
⑥ 图引擎（graph.py）
   从 Milvus relation 集合建内存邻接表：BFS/DFS、路径搜索、子图、共现排序、冲突检查
        ↓
⑦ 检索（retrieval.py）
   四阶段模式检测 + 向量相似 + 图上多跳 + 泛化搜索 + Agent 工具调用
```

## 3. 这一版实际跑出了什么（20251114_153700）

| 指标 | 数值 |
|---|---|
| 算子 | 4 个（gemm / trmm / hemv / axpy），20 条文件级分析记录 |
| LLM 抽出的策略 | **206 条**（算法层 72 / 代码层 80 / 指令层 54） |
| 计算流程 | 90 条 |
| 聚类结果 | optimization_strategy 45 簇、hardware_feature 64 簇、tunable_parameter 13 簇 |
| 簇内合并 | 只有 3 组 `similar_groups` + 1 组（**消解很保守**） |
| 关系三元组 | **805 条**，4 类：OPTIMIZES_PATTERN 326 / TARGETS 206 / IS_ILLUSTRATED_BY 206 / HAS_PARAMETER 67 |
| 关系涉及实体 | 766 个 uid |

**注意最后两行**：schema 声明了 7 类实体、12–13 类关系，但**关系里只出现 5 类实体（模式/策略/硬件/代码示例/参数），聚类与合并阶段只覆盖了其中 3 类；12–13 类关系只落地 4 类**。声明宽度 ≠ 实现宽度，答辩时别按 schema 说规模。

另外，206 条是**文件级**记录（20 个文件各抽 10 条左右，含跨文件重复）；`kg_agent` 读入后经规范化只保留了 **61 条**。所以"206 → 61"不是丢数据，是去重+闭集化的结果。

## 4. 它和现在的 `kg_agent` 是什么关系

这是本次评估最重要的一条：

```text
analyze_OB1（LLM 语义图，206 条策略，无行号/无谓词）
        ↓ 被 kg_agent 的 S3 当"分析结果"读入
kg_agent 的 analysis 通道（61 条，ID 以 _a 结尾，condition_status 多为 text）
        ↓ 与 static/diff（66）、multiarch（99）、cross_arch（30）、abstract（6）合并
现在的图（262 条）
```

所以：

- 你现在这 262 条里，**23% 是这条 AI 流水线的产物**；
- 上一轮查到的 **C5 六组重复，全部发生在 `_a`（AI）× `_s`（规则）的交界**——不是巧合，是两条通道对同一件事的两种说法；
- `_a` 策略缺行号、条件多为文本，也是因为它的上游就只到"文件 + 代码片段"，没有行号。

**一句话**：你不是"要不要用 AI 建图"，你**已经用了**；问题是用了之后没有验收。

## 5. 六个可测的问题（拿这些去问它）

| # | 问题 | 证据 |
|---|---|---|
| 1 | **没有行号溯源** | 实体字段只有 `source_file`（文件名）+ `code_snippet`，没有 `start_line/end_line`；无法回答"这条策略在哪一行" |
| 2 | **条件不可执行** | `applicability_conditions` 是自然语言（"当循环步长为固定值且需多次访问复数元素时…"），不是谓词；检索时无法过滤 |
| 3 | **效果判断是 LLM 散文** | 20/20 条分析记录都含"显著减少指令周期数"这类判断，**没有一条给可核对的数字**；`performance_impact` 全是定性描述 |
| 4 | **没有校验层** | refine/merge 都是 LLM 自评；没有金标准、没有抽检、没有交叉验证；消解只找到 3 组相似，保守到可疑 |
| 5 | **本地副本不可复现** | `prompt_loader` 期望 `analyze_OB1/prompts/`，`workflow` 读 `config/.env` 和 `kg_config.json`——**这几样都不在本地副本里**（原工程在服务器 `/home/dgc/mjs/project/analyze_OB`） |
| 6 | **schema 落地率低** | 7 类实体在关系里只出现 5 类、聚类只覆盖 3 类；12–13 类关系只落地 4 类；`APPLIES_WHEN / REQUIRES / CONFLICTS_WITH` 等关键约束关系没有出现在 805 条里 |

第 6 条特别值得注意：**它恰恰缺了"适用条件"和"冲突"这两类关系**——而这正是你现在这张规则图的核心。换句话说，两版是互补的，不是替代的。

## 6. 它比你现在的图强在哪（别一棍子打死）

| 能力 | analyze_OB1 | kg_agent |
|---|---|---|
| 语义词条（为什么这样优化、trade-off、代码示例） | ✅ 强 | ❌ 弱（字段化） |
| 因果/原理层（optimization_principle） | ✅ 有 | ❌ 没有 |
| 实体消解纪律（聚类 + LLM 精炼 + 多轮合并） | ✅ 有流程 | ⚠️ 只有模板化 ID |
| 条件可执行 | ❌ 文本 | ✅ 谓词 |
| 行号溯源 | ❌ 文件级 | ✅ 行级 |
| 证据 | ❌ LLM 散文 | ⚠️ 有节点但全是 QEMU 算子级 |
| 检索 | 向量 + 图 + Agent | 类型化约束检索 |

**结论**：它是"知识层"，你是"执行层"。理想形态是**语义层提供 rationale/示例，执行层提供谓词/模板/证据**——但合并时必须以执行层为主键，语义层只做注解。

## 7. 怎么用它（三个选项，推荐 A+B）

**A. 当 baseline（最划算）**
把"LLM 语义图 + 向量检索"作为对照方法，和"规则图 + 约束检索"比。这正好回答审稿人最想问的："你为什么不直接用 LLM 建图？"——你有实测答案。

**B. 当语义注解层（低成本）**
把 61 条 `_a` 策略的 `rationale / trade_offs / code_example` 挂到对应规则策略上（按算子 + 动作对齐），图的结构不变，只加注解字段。这样语义知识不丢，也不污染可执行性。

**C. 当验证实验（最费工，但最像论文）**
抽 50 条它的三元组，对照源码人工判定对/错/无法判定，报 precision。这一条能直接支撑"AI 建图需要校验"的论点——和 GraphJudge 那篇对话。

**不建议**：把它整张图合并进 `kg_agent`。它的实体没有谓词、没有行号，合并进来只会制造更多 C5 类重复。

## 8. 对论文叙事的影响（这是加分项）

你现在有一个很顺的演进故事，而且**每一步都有数字**：

```text
v1（analyze_OB1）：LLM 端到端建图 → 206 条策略 / 805 条关系
    问题：无行号、条件不可执行、效果判断不可核、无校验
        ↓
v2（kg_agent）：差分 + 规则 + 模板 → 262 条策略 / 3656 条边
    解决：可执行谓词、行级溯源、证据节点、版本化
    遗留：23% 的 v1 策略仍无行号；证据全是 QEMU 算子级
        ↓
v3（本课题）：语义注解 + 校验门 + 自维护
```

写进第 3 章就是一句很硬的话：

> "早期版本采用 LLM 端到端抽取（206 条策略、805 条关系），但发现其产物无法回答'这条策略在哪一行、在什么条件下成立'；因此本文改为差分驱动的可执行知识表示，并保留 LLM 用于语义注解与解释。"

这比"我提出了一个知识图谱"可信得多，因为它承认了失败并给出了替代方案。

## 9. 答辩问答

**Q："你这个图和 LLM 自动建的图有什么区别？"**
> "我做过 LLM 端到端那版（206 条策略、805 条关系），它能给出 rationale 和代码示例，但三个硬伤：没有行号、条件是不可执行的散文、效果判断没有数据。所以现在这版把事实交给差分和规则，LLM 只做语义注解；代价是覆盖面窄，收益是可审计。"

**Q："那 LLM 建的那版就废了？"**
> "没有。它作为 baseline 保留——'LLM 语义图 + 向量检索' vs '规则图 + 约束检索'，正好回答'为什么不能只用 LLM 建图'。"

**Q："你怎么知道 LLM 建的那版错了多少？"**
> "我准备抽 50 条三元组对照源码人工判定，报 precision；这也是我论文里'校验层必要性'的证据。"
