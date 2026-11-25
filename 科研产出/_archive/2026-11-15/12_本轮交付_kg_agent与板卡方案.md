# 12 本轮交付：kg_agent v1.5（Agent 工作流 + 可视化）+ RISC-V 板卡方案

> 2026-11-12｜Lead｜本轮回应你的 4 点：①板子不在手怎么办；②算子按你文档重新核对；③KG 建模 + 抽取/建图代码 + Agent 工作流 + 可视化（推理/路径/检索回放）全做；④现在就能跑。
> **一句话**：`kg_agent` 已经是一个可离线运行、可复现、带可视化回放的 **多架构** v1.5 知识图谱系统（**862 节点 / 3656 边 / 262 策略 / 262 模板 / 30 条跨架构共有策略**），并且把可视化首屏从 ~1.7s 降到 ~0.3s、新增 WebGL 性能版；板子入口是 cloud.spacemit.com（原 gdriscv.com 已失效），算子清单按你的文档已修正。

---

## 1. RISC-V 板子不在手：先免费租，再买兜底

详细调研见 [`11_RISC-V板卡方案.md`](11_RISC-V板卡方案.md)（43 条可核查链接）。结论：

| 档 | 方案 | 成本 | 关键事实 | 建议 |
|---|---|---|---|---|
| **A 首选** | **Bianbu Cloud（进迭时空官方，[cloud.spacemit.com](https://cloud.spacemit.com/)）** | **¥0** | K1 实例，RVV 1.0 / VLEN=256 / 8 核；**root 可锁频**；SSH 走 Stunnel；可选 48h / 7d / 1 月；**注意：旧域名 gdriscv.com 已打不开，用 cloud.spacemit.com** | **今天注册申请 48h**，按 `11` §6 验收 |
| **B 备选** | Cloud-V（10xEngineers） | ¥0 | 物理板（BPI-F3 / Jupiter / RV2），SSH 但**无 sudo** | 并行申请，用于正确性/结构验证 |
| **C 兜底** | 买 Orange Pi RV2（$30–49.9）或 BPI-F3（$63–100） | ¥300–800 | 与语料/交叉编译完全一致；国内 2–5 天 | 2 周内 A/B 不稳定就下单 |
| ✗ 别碰 | Scaleway EM-RV1（€0.042/h）、Pioneer、Lichee Pi 4A、VisionFive 2、RISE Runners | — | **C910 是 RVV 0.7.1，不是 1.0** | ISA 不对，实验不可比 |

**如果今天到板前**：QEMU + 你已有工具链可完成开发与正确性验证；x86 主实验照常推进；板到后只补 E4 真机性能（约 1 天）。

---

## 2. 算子清单：按你的《1.1 算子提取工具》修正

你的文档 §3.2/§7.0.3 已明确：**45 个里 24 个有 RVV、21 个没有**。自动审计（按 `_rvv.c` 里有无 intrinsic）会误判，现以你的表为准（见 [`10_第一周执行包/02b_算子RVV覆盖_按用户文档.md`](10_第一周执行包/02b_算子RVV覆盖_按用户文档.md)）：

- **有 RVV（24）**：hemv、amin/amax、axpy、asum、axpby、copy、dot、gemm、gemv、idamin/idamax/idmax/idmin、nrm2、swap、symm、symv、trmm（文档注"暂时不要"）、zdot、max/min、rot、scal。
- **无 RVV（21）**：hbmv、hemm、her、her2、herk、her2k、hpmv、ger、gesv、syr、syr2、syrk、syr2k、tpmv、tpsv、trmv、trsm、trsv、spmv、spr、spr2。
- **原 12 算子中 `dger/dtrmv/dsyrk` 属于无 RVV，已移除**；`dsymm` 的 `_k.c` 里是 dgemm kernel **属正常**（对称矩阵乘复用 gemm 核），已保留。
- `chemv/dsymv` 的 `_k.c` 依赖 helper（`ccopy_k`/`dcopy_k`、`cgemv_n/t`/`dgemv_n/t`、`SYMCOPY_L`），构建 `generic.c` 时需一起放入 `input/`。

最终 12 个：S1 进 KG = daxpy、dasum、ddot、dgemv、dgemm、dtrmm；S2 留出 = dscal、dnrm2、daxpby、dsymv、chemv、dsymm。

---

## 3. KG 建模与代码：v1.5 已实现（不是只给建议）

### 3.1 先回答"建模有没有问题"

有，但不是推倒重来。审计（[`10_第一周执行包/kg_adequacy_audit.md`](10_第一周执行包/kg_adequacy_audit.md)）：
- 旧 KG：206 策略、805 关系、**只有 4 类关系有实例**、参数覆盖 28%、硬件描述含数值 **0.5%**、`CONFLICTS_WITH`/`COMPOSES_WITH` 为 **0**、溯源只到文件名；
- 10 个 competency questions：2 能 / 3 部分 / 5 不能。
→ 结论：**旧版是"知识清单"，不是"可执行知识库"；B3 相对 B2 的增量（条件过滤/冲突消解/模板）全都不存在，直接实验必得出"KG 没用"的假结论。**

### 3.2 v1.5 做了什么（已落地到代码）

`科研产出/kg_agent/` 是**重写后的 v1.5 系统**（不是补丁文档）：

| 层 | 实现 |
|---|---|
| Schema | 12 类实体、14 类关系、10 个受控谓词、4 个硬件 Profile（x86_avx2 / x86_avx512 / k1_rvv256 / qemu），JSON Schema + 不变量校验 |
| 抽取 | S0 语料清单 → S1 静态特征 → S2 C/RVV 差分 → S3 规则抽取（可选 LLM）→ S4 规范化/对齐 → S5 证据解析 → S6 校验；**结构事实全由规则/日志产生，LLM 只改名称与解释** |
| 知识 | **862 节点 / 3656 边**：14 Operator、76 Pattern、**262 Strategy**、**262 Template**、14 Param、88 PerformanceEvidence、111 SourceArtifact；**262/262 策略满足 ≥2 可执行条件 + 模板 + file:line 溯源 + 硬件绑定**；**x86/ARM/RISC-V 三架构知识**（13/14 算子 ≥2 架构实现、30 条跨架构共有策略） |
| 检索 | 7 步可解释 trace：pattern_match → strategy_recall → **condition_filter** → **hardware_filter** → **conflict_resolve** → score → pack；每步带 `dropped[].reason`；支持 `--profile x86_avx2 / x86_avx512 / arm_neon / arm_sve / k1_rvv256` |
| 证据 | QEMU 日志标 `qemu_only/confidence=0.35/rerun_needed`，**不参与性能排序**；保留 idamax 0.18× 等负例 |
| 可复现 | `manifest.json` 含版本四元组 + 54 个输入 sha256；两次 build 字节级一致；**35 个 unittest 全过** |

### 3.3 Agent 工作流 + 可视化（你要的"推理/路径/检索回放"）

- **Agent 工作流**：`corpus / static / diff / extract / normalize / evidence / validate` 七个阶段各自独立、可重跑；`llm.py` 支持 API + 缓存 + 离线降级（无 key 也能跑）。
- **可视化**（两个入口，同一份数据）：
  - `viz/graph.html`：**默认性能模式**（614/862 节点、921/3656 边，预计算布局关物理、标签按需、增量刷新、展开 1 跳）；5 项交互——①图探索 ②路径跳转 ③**推理链** ④**检索回放**（7 步）⑤搜索+图例；`?selftest=1` **18/18**；首屏 **272–339ms**（旧版同数据 1648–1720ms）。
  - `viz/graph_fast.html`：**Sigma.js WebGL 版**，默认算子 2 跳子图（60 节点/254 边），「加载全图」862/3656 可拖拽缩放；全图切换 3.1–3.4ms、单帧 CPU 提交 2.6–2.9ms；`?selftest=1` **12/12**；首屏 537–551ms。适合"要流畅看全图"。
  - 优化记录与实测方法见 [`viz/PERF.md`](kg_agent/viz/PERF.md)。
- 实拍截图：`viz/_screenshots/01_graph.png`（性能模式）、`fast_subgraph.png`、`fast_full.png`、`real_replay_step4.png`、`real_chain.png`、`real_path.png`。

![性能模式 862 节点](kg_agent/viz/_screenshots/01_graph.png)
![WebGL 默认 2 跳子图](kg_agent/viz/_screenshots/fast_subgraph.png)
![WebGL 全图 862/3656](kg_agent/viz/_screenshots/fast_full.png)

### 3.4 怎么跑（1 分钟）

```powershell
cd 科研产出\kg_agent\src
python -m kgagent.cli build --offline
python -m kgagent.cli validate
python -m kgagent.cli stats --matrix
python -m kgagent.cli retrieve --op dgemm --profile x86_avx2 --top-k 6
python -m kgagent.cli retrieve --op dgemm --profile k1_rvv256 --top-k 6
python -m kgagent.cli export-viz
# 然后双击 科研产出\kg_agent\viz\graph.html（日常）或 graph_fast.html（要流畅看全图）
```

一键：`python 科研产出\kg_agent\run_demo.py`（build → validate → retrieve → export-viz，已实测跑通）。
测试：`cd 科研产出\kg_agent && python -m unittest discover`（35 用例）。

---

## 4. 这轮交付对论文意味着什么

- **第 3 章（KG 构建）**：Schema、差分抽取、对齐、质量评估都有了可运行实现与真实统计（不再是 PPT 设计）。
- **第 4 章（检索与生成）**：7 步检索 + StrategyPack + prompt 预览已实现；与 Morph 的对接只需把 `pack.prompt_preview`/结构化 pack 替换进 `llmgenv4.py` 的 prompt 模板（下一步）。
- **第 5 章（实验）**：`retrieve` 的 trace 可直接作为"可解释性"实验材料；图/路径/回放截图可直接进论文与答辩 PPT。
- **答辩加分**：评审可以现场看"从 GEMM 出发，为什么推荐这条策略、哪一步被硬件条件淘汰"，比讲 PPT 有说服力。

---

## 5. 下一步（按顺序）

1. **今天**：注册 Bianbu Cloud 申请 48h K1；同时给 Cloud-V 发申请邮件。
2. **今天**：跑一遍 `run_demo.py`，打开 `viz/graph_fast.html`（WebGL，全图流畅）与 `viz/graph.html`（推理链/回放）各玩 5 分钟；截图存好。
3. **今天**：用 **`https://cloud.spacemit.com/`**（不是 gdriscv.com）注册 Bianbu Cloud；打不开就走 Cloud-V / 买板。
4. **明天**：把 `kg_agent` 的 `StrategyPack`/`prompt_preview` 接进 Morph 的 `llmgenv4.py`（替换原 9 维策略表），跑 daxpy/dgemm 的 B0–B3 smoke。
5. **本周**：板/云实例验收 + C0/C1/P1 双平台基线；补 `chemv/dsymv` helper 让 12/12 编译；按 `09` 开始写第 1 章。

---

## 6. 文件索引

| 内容 | 路径 |
|---|---|
| **论文架构/流程图（六张 + 文字详解）** | [`14_论文架构图/index.html`](14_论文架构图/index.html) ｜ [`README.md`](14_论文架构图/README.md) |
| **实验登记与结果（E09 网格已完成）** | [`experiments/EXPERIMENTS.md`](experiments/EXPERIMENTS.md) ｜ [`E09_MPSA/runs/INDEX.md`](experiments/E09_MPSA/runs/INDEX.md) |
| **理论论证/创新点/工作量/答辩话术** | [`15_理论论证与工作量说明.md`](15_理论论证与工作量说明.md) |
| **两部分工作与实验设计（KG + 自维护 E10）** | [`18_两部分工作与实验设计.md`](18_两部分工作与实验设计.md) |
| **自建 OpOptBench 的合法性与规范** | [`19_自建OpOptBench基准说明.md`](19_自建OpOptBench基准说明.md) |
| **QiMeng-GEMM 精读与实验对照** | [`20_QiMeng-GEMM精读与实验对照.md`](20_QiMeng-GEMM精读与实验对照.md) |
| **实验设计（不用代号、人话版）** | [`21_实验设计_人话版.md`](21_实验设计_人话版.md) |
| **知识图谱构建完整指南** | [`22_知识图谱构建完整指南.md`](22_知识图谱构建完整指南.md) |
| **可选增强 E8：无参数更新 Agent 策略学习** | [`16_无参数更新Agent策略学习方案.md`](16_无参数更新Agent策略学习方案.md) |
| **Lingxi（ISSTA 2026）精读与复用（E9/MPSA）** | [`17_文献精读_Lingxi.md`](17_文献精读_Lingxi.md) |
| kg_agent 说明（多架构章节） | [`kg_agent/README.md`](kg_agent/README.md) |
| 接口契约（含多架构增量） | [`kg_agent/SPEC.md`](kg_agent/SPEC.md) |
| 可视化日常版（性能模式） | [`kg_agent/viz/graph.html`](kg_agent/viz/graph.html) |
| 可视化 WebGL 性能版 | [`kg_agent/viz/graph_fast.html`](kg_agent/viz/graph_fast.html) |
| 性能优化记录（前后对比） | [`kg_agent/viz/PERF.md`](kg_agent/viz/PERF.md) |
| 板卡方案（含新入口） | [`11_RISC-V板卡方案.md`](11_RISC-V板卡方案.md) |
| 输入与多架构说明 | [`13_输入与多架构扩展说明.md`](13_输入与多架构扩展说明.md) |
| 算子 RVV 覆盖权威表 | [`10_第一周执行包/02b_算子RVV覆盖_按用户文档.md`](10_第一周执行包/02b_算子RVV覆盖_按用户文档.md) |
| KG 体检 | [`10_第一周执行包/kg_adequacy_audit.md`](10_第一周执行包/kg_adequacy_audit.md) |
| Schema v1.5 补丁规格 | [`10_第一周执行包/03_Schema_v1.5补丁清单.md`](10_第一周执行包/03_Schema_v1.5补丁清单.md) |
