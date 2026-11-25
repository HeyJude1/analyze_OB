# 16 无参数更新的 Agent 策略学习：能不能加、怎么加、值不值得

> 2026-11-12｜Lead｜回应："生成这块能不能加 agent 强化学习（不改模型参数）来凑工作量、提升生成能力"
> 结论先说：**方向可以加，而且和你的系统天然衔接；但不能叫"凑工作量"，要叫"生成闭环的反馈学习"；也不能笼统叫"RL"，要精确叫"非参数化策略学习（contextual bandit + 经验记忆）"。**
> 建议做成 **E8（可选增强实验）**：8–10 人日、约 1,200 行代码、2.5–3.5k 字、2–3 张图；放在第 4 章 §4.7 + 第 5 章 E8；不挤占 E3/E4 主实验。

---

## 1. 先澄清概念：什么叫"不改模型参数的 agent 强化学习"

不更新 LLM 权重时，文献里通常是下面四类之一（都是 2025–2026 的热点，但**名字不能混**）：

| 类型 | 学什么 | 代表工作 | 和"RL"的关系 |
|---|---|---|---|
| **In-context RL / 算法蒸馏** | 把"试错轨迹"放进上下文，让冻结 LLM 在上下文里"学会"策略 | Algorithm Distillation（arXiv:2210.14215）、Vintix（ICML 2025） | 是 RL 的"上下文版"，不改权重 |
| **Contextual Bandit 策略选择** | 学一个"在什么状态下选哪个动作"的轻量策略（线性/表格），LLM 冻结 | **CB-Orchestrator**（arXiv:2610.13064，ICML 2026 Workshop）、OMCE（arXiv:2611.01747） | 是 RL 的"无状态/单步"特例，可证明 regret 界 |
| **经验记忆 / 技能库** | 把成功/失败经验写成可检索的"技能/教训"，推理时注入 | **ExpeL**（arXiv:2308.10144）、**Reflexion**（arXiv:2303.11366）、CER（arXiv:2506.06698, ACL 2025）、Voyager（arXiv:2305.16291）、Agent Workflow Memory（arXiv:2409.07429） | 文献称"verbal RL / training-free self-improvement" |
| **非参数化 PPO / 技能进化** | 用 PPO 的"门控/选择"机制管理技能池，但不更新 LLM | **Skill-Pro**（arXiv:2602.01869，ICML 2026 spotlight，"Non-Parametric PPO"） | 借 RL 的优化框架，不更新模型参数 |
| **文本梯度 / 程序优化** | 用反馈改 prompt/程序，而不是改权重 | TextGrad（arXiv:2406.07496）、DSPy（arXiv:2310.03714）、OPRO（arXiv:2309.03409）、RLPrompt（arXiv:2205.12548，训练小策略网络） | 属于"优化"而非严格 RL |

> **你的系统里最自然的是第 2 类 + 第 3 类**：
> - **bandit** 决定"这一次检索多少条、用哪种 prompt、失败后怎么修"；
> - **经验记忆** 把 `FailureCase` 变成可检索的"教训"，下次注入。
> 这样既不改模型参数，又能说清楚"学习"发生在哪里。

---

## 2. 为什么它和你的系统天然衔接（不是硬加）

你现在的闭环已经有"反馈"的原料，只是**没有学习器**：

```text
检索（7 步）→ StrategyPack → Prompt（B0–B3）→ LLM 生成
   ↑                                                ↓
   └── FailureCase ← 失败原因 ← 编译/正确性/性能验证 ←┘
```

- `FailureCase` 已经在图谱 Schema 里（第 12 类实体），但只被"回写"，没有被"用来选动作"；
- 7 步 trace 每一步都有候选/选中/淘汰，天然是**状态特征**；
- 每次生成都有 CR/FCR/S_C1/成本，天然是**奖励信号**。

所以加一个"策略选择器"不是另起炉灶，而是**把已有的反馈闭环补上最后一环**。

---

## 3. E8 设计：反馈驱动的非参数化策略学习

### 3.1 形式化（一句话）

把"每次生成前选什么动作"建模成 **contextual bandit**：
`state = (算子, 层级, dtype, 硬件 profile, 代码特征, 检索统计, 历史失败类型)`，
`action = (检索条数 k, prompt 变体, 策略族偏好, 失败修复动作)`，
`reward = f(正确性, S_C1, 成本)`，策略用 LinUCB/Thompson 学习，LLM 权重不动。

### 3.2 状态、动作、奖励（可直接实现）

**状态特征（约 20 维）**：
- 算子 one-hot（12）、层级（3）、dtype（s/d/c/z）、硬件 profile（x86/rvv）；
- 代码特征：loop_depth、stride_x、size_regime、aligned、tail_divisible；
- 检索统计：候选数、淘汰数、top-1 分数、策略族直方图（vectorize/tile/unroll/prefetch/tail）；
- 历史：第几次尝试、上一次失败类型（compile/numeric/boundary/timeout）。

**动作空间（先 8 个，别一上来做 50 个）**：
| 动作 | 含义 |
|---|---|
| A1 | 检索 k=3（少而精） |
| A2 | 检索 k=6（默认） |
| A3 | 检索 k=9（多召回） |
| A4 | prompt 变体：策略优先 vs 模板优先 |
| A5 | 策略族偏好：vectorize 优先 |
| A6 | 策略族偏好：tile/blocking 优先 |
| A7 | 失败修复：带编译错误重新生成（等预算 1 轮） |
| A8 | 失败修复：重新检索（过滤上次失败族） |

**奖励（预注册，别事后改）**：
```text
r = 1.0   正确（D1–D4 全过）且 S_C1 ≥ 1.0
    0.7   正确但 S_C1 < 1.0
    0.4   编译通过但功能不正确
    0.0   编译失败
另扣成本：r -= λ * (token / 预算)   λ=0.05（可选）
```

### 3.3 两种"学习"实现（建议都做，互为消融）

**① Contextual Bandit（LinUCB，推荐主实现）**
- 每个动作维护 `A_a`、`b_a`，`θ_a = A_a^{-1} b_a`；
- 选动作：`argmax_a [θ_a^T x + α sqrt(x^T A_a^{-1} x)]`；
- 只用标准库实现（矩阵求逆 20×20，numpy 都不用；或允许 numpy）；
- 优点：可解释、样本效率高、能画 learning curve / regret。

**② 经验记忆（in-context，训练-free）**
- 每条经验：`{state摘要, action, reward, 失败类型, 一句教训}`；
- 新任务检索 top-3 相似经验（算子/架构/特征相似度），拼到 prompt 的"历史经验"块；
- 只存"验证过"的成功/失败，不存 LLM 自评；
- 消融：无记忆 / 全量记忆 / 相似检索记忆。

**③ 可选（不建议当主线）**：文本梯度/OPRO 优化 prompt 模板——可作为 E8 的对照，但工作量翻倍，12 周内不划算。

### 3.4 实验设计（E8）

| 项 | 设计 |
|---|---|
| **训练/测试划分** | S1 的 6 个算子做训练（learn），S2 的 6 个算子只测试（transfer）——防止背答案 |
| **规模** | 8 个任务 × 4 episodes × 3 seeds = 96 episodes/方法；6 个方法 ≈ 576 episodes |
| **方法** | M0 静态 KG（固定 k=6）；M1 随机动作；M2 bandit-only；M3 memory-only；M4 bandit+memory（提出）；M5 oracle（每任务最优固定动作，上界） |
| **主指标** | 成功率、平均 S_C1、累计奖励、regret vs oracle、样本效率（到 oracle 80% 所需 episode 数） |
| **次指标** | 迁移到 S2 的成功率、token 成本、动作分布可解释性 |
| **统计** | 3 seeds，均值±95% CI；learning curve 图；bandit 的 regret 曲线 |
| **成本控制** | 先做"快速代理"：用 E2 的推荐质量（R@5/SAR）当 reward 学策略（不跑代码）；再把学到的策略在 6 个算子上跑真实生成验证（E3 子集）——两阶段，省 2/3 机时 |
| **时间** | 8–10 人日（实现 4、实验 3、分析写作 3） |

### 3.5 预期结论（写成三种都行）

- **正向**：bandit+memory 的累计成功率/平均 S_C1 高于静态 KG，且迁移到 S2 仍有效 → "反馈闭环有效"；
- **部分正向**：bandit 只在"失败修复"动作上有效，memory 贡献小 → 报告哪个机制有用；
- **负向**：学习不如静态 → 说明"动作空间设计/奖励信号"是瓶颈，这本身是可写的边界结论（避免硬吹）。

---

## 4. 工作量与论文位置（"加厚"到底加多少）

| 项 | 数量 |
|---|---|
| 新增代码 | `agent_policy.py`（LinUCB + memory + 特征）+ 实验脚本 + 测试 ≈ **1,000–1,500 行** |
| 新增实验 | E8：6 方法 × 96 episodes × 3 seeds + 迁移测试 |
| 新增图表 | 2–3 张：learning curve、regret、动作分布/案例 |
| 新增字数 | 第 4 章 §4.7 约 2,500–3,500 字；第 5 章 E8 约 1,500 字 |
| 新增人日 | **8–10 人日**（总工作量从 119 → 约 128–130 人日） |
| 论文位置 | 第 4 章 §4.7"基于执行反馈的策略选择学习"；第 5 章 §5.x E8；第 1 章贡献可加"闭环反馈"一条（放第 5 条，不挤占前 4 条） |
| 创新点 | **不主张为核心创新**（CB-Orchestrator/Skill-Pro/OMCE 已做过类似），写成"系统闭环增强 + 在算子优化任务上的实证"；答辩时主动说"这是把非参数化学习用于本任务，不是提出新 RL 算法" |

**为什么值得加**：
1. 它把 `FailureCase` 用起来，让"知识图谱 + 反馈"的故事闭环；
2. 提供"学习曲线/regret"这类**很直观的工作量证据**；
3. 与热点（agentic RL、experience memory）对齐，答辩时能接住"为什么不用 RL"的问题；
4. 成本可控（两阶段 + 小动作空间），不挤占主实验。

---

## 5. 风险与边界（必须先想清楚）

| 风险 | 说明 | 对策 |
|---|---|---|
| **"这不是 RL"** | 不改参数、bandit 很小，容易被说"就是调参/启发式" | 精确命名"contextual bandit / 非参数化策略学习"；引用 CB-Orchestrator、Skill-Pro、CER、ExpeL；不称"RL 算法创新" |
| **抢主实验时间** | 8–10 人日从 12 周里挤 | 放在 E3/E4 之后（W9–W10）；若 G2 未过，直接砍掉 E8，写"设计 + 小规模验证" |
| **动作空间太大** | 12 算子 × 8 动作 × 6 方法，样本不够 | 先 8 动作、8 任务；动作分层（先选 k，再选修复）；用先验剪枝 |
| **奖励噪声** | 性能方差大，bandit 学不动 | 用"成功率 + 正确性"为主奖励，S_C1 作次奖励；多次重复取中位数 |
| **和 KG 混淆** | 提升到底来自 KG 还是 learning | 固定 KG 与 LLM，只变"动作选择"；M0 静态 KG 是严格对照 |
| **记忆污染** | 错误经验被反复注入 | 只存验证过的；记忆加"过期/置信度"；消融验证 |

---

## 6. 实施计划（如果决定加）

**第 1 步（1 天）**：写 `agent_policy.py`
- `StateEncoder`：算子/硬件/特征 → 向量；
- `LinUCB`：`select(x)`、`update(x, a, r)`；
- `ExperienceMemory`：`add(record)`、`retrieve(x, k=3)`；
- 单测：动作选择、更新后偏好变化、记忆检索。

**第 2 步（2 天）**：接进实验闭环
- 在 `retrieve` 生成 StrategyPack 前加"策略选择器"；
- 生成/验证后写 reward 到 `episodes.jsonl`；
- 支持 `--policy static|bandit|memory|both`。

**第 3 步（2 天）**：快速代理实验
- 用 E2 的推荐指标当 reward（不跑代码），先跑 96 episodes × 3 seeds，看 learning curve 是否上扬；
- 若代理都不学，直接停，不浪费机时。

**第 4 步（3 天）**：真实生成验证
- 取 6 个算子 × 3 尺寸，跑 bandit+memory vs 静态；
- 画 learning curve / regret / 迁移表。

**第 5 步（2 天）**：写作
- §4.7 方法 + E8 实验 + 2–3 图；答辩话术："这是非参数化策略学习，不是改模型；它把 FailureCase 用起来，和 KG 是互补的。"

---

## 7. 相关文献（已核实，可直接进参考文献）

| 工作 | 出处/ID | 与 E8 的关系 |
|---|---|---|
| Algorithm Distillation（In-context RL） | Laskin et al., arXiv:2210.14215 | 不改参数的 in-context RL 源头 |
| Reflexion | Shinn et al., arXiv:2303.11366 | verbal RL，反馈写进上下文 |
| ExpeL | Zhao et al., arXiv:2308.10144 | 经验学习，无参数更新 |
| CER（Contextual Experience Replay） | Liu et al., arXiv:2506.06698（ACL 2025） | training-free 经验回放 |
| CB-Orchestrator | Sun et al., arXiv:2610.13064（ICML 2026 Workshop） | contextual bandit 选 workflow，和 E8 最接近 |
| Skill-Pro（Non-Parametric PPO） | Mi et al., arXiv:2602.01869（ICML 2026 spotlight） | 非参数化 PPO 管理技能库 |
| OMCE | Zhao et al., arXiv:2611.01747 | bandit 元控制经验使用 |
| RLPrompt | Deng et al., arXiv:2205.12548 | RL 优化 prompt（小策略网络） |
| TextGrad | Yuksekgonul et al., arXiv:2406.07496 | 文本梯度优化 |
| DSPy | Khattab et al., arXiv:2310.03714 | 程序化 prompt 优化 |
| OPRO | Yang et al., arXiv:2309.03409（ICLR 2024） | LLM 作为优化器 |
| Voyager | Wang et al., arXiv:2305.16291 | 技能库/终身学习 |
| Agent Workflow Memory | Wang et al., arXiv:2409.07429 | 工作流记忆 |

> 引用纪律：只引读过的；方法名/ID 已用 arXiv API 核对；不要把这些写成"我们提出"。

---

## 8. 一句话回答"值不值得加"

**值得，但要按"E8：非参数化反馈学习"来加，不是按"RL 大模块"来加。**
- 它能把 `FailureCase` 闭环用起来，给论文加 8–10 人日、2–3 张图、3–4k 字，且和已有系统天然衔接；
- 它**不能**当作核心创新点（已有 CB-Orchestrator/Skill-Pro/OMCE/CER），只能写成"系统闭环增强 + 实证"；
- 必须放在 E3/E4 之后、有时间才做；G2 不过就砍掉，写成"设计 + 小规模验证/未来工作"，不影响毕业。
