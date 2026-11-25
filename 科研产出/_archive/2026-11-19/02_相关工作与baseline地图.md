# 02 相关工作与 baseline 地图

> 任务：task-1（文献侦察）｜作者：lit-scout｜完成日期：2026-11-12｜联网检索截止：2026-11-12
> 检索工具：web_search（多轮，中英文）+ 对关键论文逐篇 web_fetch 核实 arXiv abs 页 / 会议页 / DOI 页
> 引文清单见 `科研产出/refs/lit_scout_refs.md`；文中 `[Rxx]` 对应清单编号

## 一句话结论

**"没有前人积累"是误判，"没有 baseline"更是误判**：2025—2026 年已经形成了至少 6 条成熟的相关研究线，其中 **Xe-Forge、KernelBlaster、QiMeng-Kernel、KEET 四篇是"最近邻"级别**（它们已经分别在做"硬件约束知识库 + 分阶段优化"、"可检索的持久化优化知识库"、"策略与实现解耦"、"性能剖面→知识→优化"）；但**同时检索也证实了一个真实的空白**：**没有人把"优化策略"本身当作可审计、可对齐、带适用条件与失效条件的结构化实体来建模，并以"同语料扁平文本 RAG vs 图结构检索"的严格消融来证明图结构的增量**。用户真正的风险不是"没人做过"，而是"已经有人做得比开题报告的方案更简洁、更快出结果，而我们的方案还停留在 6 模块大框架上"。

三条直接回答：

1. **"HPC 算子优化知识图谱"这个精确命题：检索未找到同名/同义的正式论文**（检索式与负结果见 §1）。但**"知识库/本体 + 性能优化"这个更宽的命题已被做过多轮**：黑盒算法侧有 OPTION 本体 + KG embedding 做性能预测（EvoApplications 2023）[R12]；HPC 系统侧有 ODA 统一本体（GraphSys'25）[R13]、HPC 运行时本体（SC23 D-HPC）[R14]；算子生成侧有 Xe-Forge 的 curated knowledge base [R1] 与 KernelBlaster 的 Persistent CUDA Knowledge Base [R2]。**"知识图谱"这个词没人用在算子优化上，但"结构化优化知识"已经有人在做——这决定了我们必须用"表示形式 + 可审计性 + 可迁移性"来定位，而不是用"首次把 KG 用到 HPC"来定位。**
2. **Baseline 不但有，而且是分层的**：性能参照（OpenBLAS/cuBLAS/PyTorch/torch.compile/TVM-Ansor）、方法学参照（零样本 / few-shot / 扁平文本 RAG / 不对齐 KG / 完整 KG）、系统参照（**6 个已确认开源可直接复现**：KernelBench、KernelBlaster、Xe-Forge、GEAK、TritonBench、AutoTriton；PRAGMA/MARCO 无公开仓库）。**最关键的一个 baseline 是"扁平文本 RAG"**——它同时回答了"这是不是就是 RAG"这个最致命的审稿人质疑。
3. **开题报告的文献综述有一个硬伤：停在 2025 年底，而 2026 年出现了 4 篇直接改变定位的工作**（KernelBlaster 2602.14293、Xe-Forge 2605.26118、KEET 2605.04467、KernelBench-Verified 2607.16241）。答辩前必须补进 Related Work，否则"研究空白"的表述会被当场推翻。

---

## 1. 检索方法与证据（含负结果）

### 1.1 实际使用的检索式（节选，均为 2026-11-12 实际执行）

| # | 检索式（英文/中文原文） | 主要命中 | 结论 |
|---|---|---|---|
| S1 | `kernel optimization knowledge graph LLM generated` | Xe-Forge、KernelBench-Verified、Leeroopedia、HPC ODA-KG | 无"算子优化 KG"同名论文 |
| S2 | `performance knowledge base LLM code optimization GPU kernel` | KernelBlaster、Leeroopedia、KEET | 有"知识库"，无"图" |
| S3 | `HPC operator optimization knowledge graph` | ODA 本体、B-YOS、HPC KG 问答 | 有 HPC+KG，不是算子优化 |
| S4 | `"knowledge graph" GPU kernel optimization LLM` | Reliable Graph-RAG for Codebases、Codebase-Memory、KernelBlaster | 无直接工作 |
| S5 | `"kernel optimization" "knowledge graph"`（精确短语） | 未命中同主题论文 | **负结果成立** |
| S6 | `"optimization knowledge graph" LLM code` | 未命中同主题论文 | **负结果成立** |
| S7 | `ontology performance optimization GPU kernel knowledge representation` | P-MoVE（SCW'24）、OPTION 相关 | 有性能本体，非策略本体 |
| S8 | `算子优化策略库 知识库 大模型 专利` | 中国专利 CN119396387B（"第一算子优化策略库"） | **工业界已有"算子优化策略库"概念**（细节待核实） |
| S9 | `知识图谱 高性能计算 优化 大模型 算子` | 中文无直接命中；命中"kg-tools 昇腾/鲲鹏知识图谱工具集"（工程仓库） | 中文文献线基本空白（仅代理证据） |
| S10 | `Using Knowledge Graphs for Performance Prediction...` | EvoApplications 2023 [R12] | **KG+性能预测的最强方法学先例** |

### 1.2 负结果的正确读法（必须写进论文的"检索局限"）

- **没找到 ≠ 不存在**。本次检索覆盖 arXiv、会议官网、DBLP、HAL、Google Patents 的网页级索引，**不覆盖** IEEE Xplore/ACM DL 全文检索、中文知网/万方、以及各类 workshop 的非公开 proceedings。
- **术语差异会造成系统性漏检**：最近邻工作没有一个用"knowledge graph"这个词来命名自己的优化知识库——Xe-Forge 叫 "curated knowledge base"、KernelBlaster 叫 "Persistent CUDA Knowledge Base"、Leeroopedia 叫 "knowledge wiki"。**用 "knowledge graph" 去检索必然漏掉它们，反过来，评委用 "knowledge base" 去检索也一定能找到它们。这就是"研究空白"表述最危险的地方。**
- 中文文献：知网/万方本次**未能实际检索**（工具与登录限制），只给出建议检索式（§8），**不要在没有检索的情况下写"国内尚无相关研究"**。

### 1.3 一组"证据性的反例"

- **KernelBench-Verified [R5]（2026-07）** 给整个赛道泼了冷水：在修正评测协议（TF32 基线 + 四分布隐藏测试 + 内存效率指标）后，**最强的 GPT-5.5 几何平均加速比只有 0.88×，而标准协议下报的是 1.43×**；**没有任何模型能稳定超过 PyTorch**；最好的模型生成的 kernel 中有 **28% 反而增加了峰值显存**。→ 这对用户是**好消息**：说明"赛道还没被解决"，仍有空间；也是**警告**：我们报的 speedup 必须用可辩护的协议，否则审稿人一句话就能否掉。
- **Reliable Graph-RAG for Codebases [R10]（2026-01）** 直接对比了"AST 派生的确定性图 RAG" vs "LLM 抽取的 KG RAG" vs "纯向量 RAG"：LLM 抽取的 KG 在 Shopizer 上**漏掉 377 个文件**、索引慢、成本高；确定性 AST 图在正确性和多跳 grounding 上最好。→ 这对用户的**两阶段 LLM 抽取 + 对齐**路线是明确的警告：**LLM 抽取的知识图谱会有覆盖率与成本问题，必须报告这些指标**（见 §5 指标）。

---

## 2. Q1：是否已有"HPC 算子优化知识图谱 / 性能知识库"工作？

### 2.1 最近邻 8 篇（按接近程度排序）

| # | 工作（年份） | 它已经做了什么 | 与本工作的真实差异 | 判定 |
|---|---|---|---|---|
| 1 | **Xe-Forge**（arXiv:2605.26118，2026-04）[R1] | 面向 Intel GPU 的多阶段 Triton kernel 优化流水线：最多 9 个优化阶段（算法重构、算子融合、block pointer 现代化、GPU 特定调优、开放式发现），每阶段由 CoVeR 智能体做"生成-在真机验证-失败迭代"；**内置一个 curated knowledge base，编码 Intel GPU 约束（2 的幂 warp 数、GRF 模式、SLM 容量）**；97 个 KernelBench L2 kernel + FlashAttention，几何平均 1.17× vs PyTorch eager，67% kernel 提升，9 个超过 5×（最高 82×）；**已开源：IntelLabs/Xe-Forge（Apache-2.0）** | 知识库是**人工整理的静态硬件约束**，按阶段硬编码进流水线；**不是图结构、不可检索排序、没有实体对齐、没有"策略间组合/冲突"建模、未覆盖策略的适用条件与收益证据**；只做 Intel 一个架构的 Triton kernel，不碰 OpenBLAS 源码 | **最近邻（最强）** |
| 2 | **KernelBlaster**（arXiv:2602.14293，2026-02）[R2] | 记忆增强的上下文强化学习（MAIC-RL）：让 LLM 智能体把每轮探索经验**累积进一个可检索的 Persistent CUDA Knowledge Base**，用 profile 引导的"文本梯度"做跨任务 CUDA 优化搜索；KernelBench L1/L2/L3 几何平均 1.43×/2.50×/1.50×；**已开源（NVlabs/KernelBlaster，Apache-2.0），含 test harness + 可复现评测流水线** | 知识是**经验条目（检索式记忆）**，不是本体化实体-关系；**无实体对齐/知识融合**；不区分"硬件特征""优化策略""适用条件"等语义类型，无法做约束过滤与组合推理；面向生成式探索，不面向"从已有代码库反向抽取知识" | **最近邻（第二强）** |
| 3 | **QiMeng-Kernel / MTMC**（arXiv:2511.20100，AAAI 2026）[R6] | 把"优化策略"与"实现代码"**显式解耦**：宏观层用 RL 训练的轻量 LLM 输出含"优化类型 + 代码作用区域"的语义动作，微观层用通用 LLM 分步实现；KernelBench L1-2 近 100%、L3 70%，比通用/微调 SOTA 高 50%+，比 LLM 最高 7.3×，比 PyTorch Eager 2.2×；TritonBench 59.64% | 知识藏在**RL 策略参数**里，不是可读、可审计、可编辑的图谱；无跨架构、无 OpenBLAS 源码级知识抽取；无"策略为什么适用/何时失效"的显式表示 | **最近邻 + 强 baseline** |
| 4 | **KEET**（arXiv:2605.04467，2026-05）[R3] | 用 LLM 智能体解读 Nsight Compute 剖面，生成有数据依据的性能瓶颈解释与优化建议；**把解释作为上下文喂回下游任务，可提升 LLM 代码优化质量** | 输入是**运行时 profile**，不是源码/库；只做"解释"，**不沉淀为可复用的知识资产**（无图谱、无检索、无跨项目复用） | **最近邻（上游数据源互补）** |
| 5 | **QiMeng-TensorOp**（arXiv:2505.06302，2025-05）[R7] | 一行 prompt 生成张量算子，**自动利用硬件特性与硬件原语**（RISC-V/ARM/GPU），自动调参；比原生 LLM 最高 1291×；**达到 RISC-V 上 OpenBLAS 的 251%、NVIDIA 上 cuBLAS 的 124%**，开发成本降 200× | 知识以**隐式参数知识 + 自动调参**形式存在；无外部知识库/图谱；不做"策略知识的结构化与复用" | **最近邻（且是最强性能上界）** |
| 6 | **Leeroopedia**（在线知识库，2026，Leeroo AI）[R11] | 用 Semantic MediaWiki 建成的 ML/数据工程知识库：**27,667 个页面 = 1,118 Workflows + 7,697 Principles + 16,832 Implementations + 1,245 Heuristics + 909 Environments**，覆盖 vLLM、DeepSpeed、FlashAttention、Triton 等 1000+ 框架；提供 MCP 让 agent 检索 | **这是"结构化优化知识库"最直接的工业先例**，但：不是图数据库/本体验证过的 KG；无实体对齐算法；不针对 HPC/BLAS 源码；**没有回答"结构化是否真的带来下游增益"**（无评估协议） | **强先例（必须引用并区分）** |
| 7 | **ChipNeMo**（NVIDIA，2023）[R15] | 领域自适应 LLM + **RAG 检索内部工程文档**用于芯片设计（含 EDA 脚本生成、bug 分析） | 面向芯片设计文档，不是算子性能优化；知识是文档片段检索，不是结构化图谱 | **可借鉴（RAG 范式）** |
| 8 | **MARCO**（arXiv:2505.03906，2025）[R8] | 多智能体 + **实时 web 检索优化技巧**注入生成；LeetCode 75 上比 Claude 3.5 Sonnet 单独跑低 14.6% 运行时，**web 检索组件贡献 30.9% 提升** | **评测集是 LeetCode 75，不是 HPC kernel/算子**（这是它最致命的弱点，可作弱 baseline 的论据）；知识是网页片段，无结构 | **弱 baseline + 消融范本** |

### 2.2 判断

- **"前人没有做过结构化优化知识"→ 假。** 至少有 Xe-Forge 的 curated KB、KernelBlaster 的 persistent KB、Leeroopedia 的四层知识 wiki、OPTION 本体四条独立证据线。
- **"前人没有做过**算子**层面的优化**知识图谱**"→ 目前检索层面成立，但**证据强度只到"未检出"，不到"不存在"**，且这个空白**很窄**——它窄到只能支撑"表示与评估方法"的贡献，撑不起"填补空白"。
- **"前人没有做过跨架构策略知识迁移"→ 检索层面成立**，这是三条里最实的一条（见 §3.7）。
- **补充一条旁证**：HPC 社区已在公开场合把"用知识图谱策展智能体工作流与运维经验"列为议题（Gainaru, TPC26 lightning talk, 2026-06 [R42]，同一 BoF 议题覆盖 performance modeling / tuning / autonomous HPC workflow orchestration），说明"KG + HPC 优化"在社区层面已被提出，只是还没有落到算子策略层面。该条**是 talk 不是论文**，只能作研究动态引用。

---

## 3. 七个 cluster 梳理

### 表 2：cluster × 代表工作 × 对我们的用法

| Cluster | 代表工作（编号见 refs） | 一句话判断 | 我们怎么用 |
|---|---|---|---|
| **① LLM/Agent 生成高性能算子** | KernelBench [R4]、QiMeng-GEMM [R16]、QiMeng-TensorOp [R7]、QiMeng-Kernel [R6]、GEAK [R9]、PRAGMA [R17]、TritonBench [R18]、AutoTriton [R19] | 已是红海；性能天花板很高（OpenBLAS 的 251%、cuBLAS 的 124%）；主流做法是"训练/RL/多智能体 + profile 反馈"，**知识的结构化不是主线** | 定位为**应用场景与 baseline 来源**，不要在这里拼性能；我们的 delta 必须落在"知识表示与检索"，不是"生成质量" |
| **② KG/RAG for code** | KG4Py [R20]、CodeKGC [R21]、Carta et al. 零样本 KG 构造 [R22]、CodexGraph [R23]、Reliable Graph-RAG [R10]、Codebase-Memory（待核实）[R24] | 技术栈成熟：代码图/知识图 + 检索已被反复验证；**但几乎所有工作都在做"代码结构"图（AST/调用/继承），没人做"优化策略"图** | 借它的**方法**（图 schema、多跳检索、AST vs LLM 抽取的取舍），占它的**空白**（策略知识而非代码结构） |
| **③ 知识驱动编译/自动调优** | Ansor [R25]、MLGO [R26]、CompilerGym [R27]、CLTune [R28]、Knowledge-Guided 编译器自动调优（arXiv:2510.14292，待核实）[R29]、RISC-V 张量程序概率优化 [R30] | 传统自动调优的本质就是"搜索 + 成本模型 + 历史 tunig 日志"，**它就是"性能知识库"的祖师爷**（TenSet 等 tuning log 数据集） | 作**非 LLM 的 baseline**（Ansor/MetaSchedule 同算子对比）；同时借"成本模型"思想设计策略打分函数 |
| **④ 性能知识库 / 性能建模 / roofline** | Roofline [R31]、OPTION 本体 + KG embedding 性能预测（EvoApplications 2023）[R12]、HPC ODA 统一本体（GraphSys'25）[R13]、HPC 运行时本体（SC23 D-HPC）[R14]、P-MoVE（SCW'24）[R32]、**WebGPU 调优配置数据库（GPU Goldmines）[R40]**、Compiler World Models（arXiv:2606.09312，待核实）[R33] | **这是最能支撑"方法合法性"的一簇**：已证明"本体 + KG + 嵌入"可用于性能预测；也已有 HPC 本体的验证方法论（competency questions）；[R40] 还提供了"把设备×shape×配置沉淀成数据库再用"的近期先例 | ①直接引用 OPTION/EvoApplications 证明"KG 用于性能知识是成立的"；②**照搬 GraphSys'25 的 36 条 competency questions 方法来验证我们的 Schema**（正对开题"难点一：Schema 设计"）；③roofline 作性能指标的归一化依据；④[R40] 的配置数据库是我们"策略×硬件×适用条件"知识库的最小原型，可作 Related Work 的"已有雏形"引用 |
| **⑤ 多智能体代码优化** | MARCO [R8]、PRAGMA [R17]、QiMeng-Kernel（宏观/微观双LLM）[R6]、GEAK（Reflexion 式循环）[R9]、KEET（多智能体解读）[R3] | 多智能体在这个任务上已是**默认配置**，不构成创新 | 开题的"多层次 Agent 解析"应**降级为工程实现**，只在方法节描述，不列为创新点 |
| **⑥ Benchmark / dataset / 评测协议** | KernelBench [R4]、KernelBench-Verified [R5]、TritonBench [R18]、HPCTransCompile/HPCTransEval [R34]、DRB-ML（随 LM4HPC）[R35]、DataRaceBench [R36]、Correctness Illusion（arXiv:2606.20128，待核实）[R37] | 基准齐全但**评测协议本身在快速演化**（KernelBench-Verified 已证明旧协议虚高 0.55×） | ①选 KernelBench + 自建 OpenBLAS 算子集；②**评测协议必须显式声明并照 KernelBench-Verified 修正**；③正确性用"多分布隐藏测试"防 reward hacking |
| **⑦ 跨架构知识迁移（Lead 追加）** | QiMeng-TensorOp（RISC-V/ARM/GPU 硬件原语）[R7]、GEAK（AMD MI300X/MI250）[R9]、Xe-Forge（Intel Arc）[R1]、HPCTransCompile（CUDA→CPU 转译）[R34]、HPC-Coder-v2（低资源并行语言）[R38]、LLM-Vectorizer [R39] | 每条线各做**一个**架构的适配；**没有工作把"同一算子的策略在不同架构上的适用/失效"当作知识来建模** | **这是用户最独特、最可辩护的 delta**：用户已有 x86/ARM/RISC-V 的 OpenBLAS 语料，天然具备"同算子 × 多架构"的对照数据，这是绝大多数 LLM 组拿不到的资源 |

### 3.1 对 cluster ⑦ 的展开（因为它是用户真正的差异化资产）

用户手上已有的资源是 **OpenBLAS 同一 kernel 在 x86(AVX2/AVX512)、ARM(NEON/SVE)、RISC-V 三个架构上的不同实现**。现有工作里：

- QiMeng 系列用"硬件原语"做跨架构，但知识是隐式的（模型参数里）；
- GEAK / Xe-Forge 各绑一个厂商（AMD / Intel）；
- HPCTransCompile 做**代码转译**（CUDA→CPU），不做**策略适用性**建模。

**没有人回答："同一条优化策略（如 unroll+prefetch 组合）在 ARM 上收益 12%、在 RISC-V 上为负、在 AVX512 上被编译器优化覆盖"，这种"策略 × 架构 × 适用条件"的三元知识没有被任何工作表示和利用过。** 这比"KG+LLM"值钱得多，也是把用户手上的跨架构语料从"数据"变成"科学贡献"的唯一路径。建议把这条提到创新点第一位（见 §6 改写）。

---

## 4. 对开题报告 3 个创新点的逐条判定

### 表 3：创新点判定

| 创新点（原文摘要） | 最近邻工作 | 判定 | 理由 |
|---|---|---|---|
| **①** KG+LLM 融合 + 多层次 Agent（算法/代码/指令三维）挖掘隐性优化知识 | QiMeng-Kernel（宏观/微观解耦）[R6]、Xe-Forge（9 阶段 + 硬件知识库）[R1]、KernelBlaster（经验知识库）[R2]、CodeKGC / Carta（LLM 建 KG）[R21][R22]、QiMeng-TensorOp（硬件原语知识）[R7] | **需改写** | "多层次 Agent"与"KG+LLM 融合"两项都已不是新东西（2025-2026 已是标配）。真正没被做的只有：**从"实现代码"反向抽取"带适用条件与收益证据的策略知识"**。必须把创新点从"融合"改成"表示 + 可审计性" |
| **②** 两阶段实体对齐（向量粗筛 + LLM 精判） | Carta et al. 2023 迭代零样本 LLM prompt 建 KG [R22]、CodeKGC [R21]、以及 2024-2025 大量 "LLM for entity alignment" 工作；开题自己引用的 Carta 就是"簇"思想 | **不成立（作为创新点）** | 向量粗筛 + LLM 精判是 2023-2024 年的通用做法，**没有任何 HPC 特异性**。可作为"面向 HPC 策略实体的对齐实现 + 评测"保留在方法章，并在实验中报告对齐 P/R，**但不能列为创新点**，否则必被问"这和通用实体对齐有什么区别" |
| **③** 结构化 Prompt 模板（计算流程特征 + 硬件约束 + 优化步骤，减少幻觉） | Xe-Forge（每阶段 CoVeR + 硬件约束）[R1]、PRAGMA（profile 注入）[R17]、MARCO（检索技巧注入）[R8]、QiMeng-Kernel（策略→分步实现）[R6] | **需改写** | "结构化模板 + 硬件约束注入"已被 Xe-Forge 做完并且效果好。能立住的只有**字段的来源与完备性**：模板字段必须由图谱查询得到，且包含**适用条件 / 失效条件 / 可调参数区间 / 收益证据**四元组；并用消融证明每个字段的增量 |

**综合判断**：三个创新点里，**①需改写、②不成立、③需改写**。按现稿提交答辩，"填补空白"和"方法创新"两处都会被正面攻击。可行的替代边界见 §6。

---

## 5. Baseline 候选表与可操作方案

### 5.1 三层 baseline 设计（这是回答"没有 baseline"的核心）

**第一层：性能参照（证明"生成的代码够快"）**

| Baseline | 来源 | 开源 | 复现性 | 适配成本 | 对比指标 |
|---|---|---|---|---|---|
| OpenBLAS 参考实现 | 用户已有语料 [R41] | ✅ 是 | 高（同机编译即可） | **极低（0.5 天）** | speedup、达峰值 % |
| cuBLAS（若上 GPU） | NVIDIA | ✅ | 高 | 中（需 GPU） | speedup |
| PyTorch Eager / `torch.compile` | 官方 | ✅ | 高 | 极低 | speedup |
| TVM Ansor / MetaSchedule [R25] | Apache TVM | ✅ | 中（调优耗时长） | **中高（1-2 周，搜索需 CPU 小时）** | speedup、搜索时间 |
| MLGO [R26] / CompilerGym [R27] | Google / Facebook | ✅ | 中 | 高（编译器级，不对齐算子级） | 仅作旁证，不主用 |

**第二层：方法学参照（证明"图结构带来增量"——必须做，否则就是 RAG）**

| Baseline | 构造方式 | 成本 | 它回答的问题 |
|---|---|---|---|
| B0 零样本 LLM | 原代码 + 硬件规格，不检索 | 1 天 | 知识注入有没有用 |
| B1 few-shot | 塞 3-5 个策略示例 | 1 天 | 示例够不够 |
| **B2 扁平文本 RAG** | **同一份抽取结果，全部拍平成文本块，按 top-k 相似度检索** | **3-5 天** | **图结构 vs 平铺文本（最关键）** |
| B3 不对齐 KG | 保留冗余实体，不做合并 | 2 天 | 实体对齐的价值 |
| B4 无硬件约束 / 无适用条件 | 从模板里去掉对应字段 | 1 天 | 模板各字段的增量 |
| B5 完整方案 | 全流程 | — | 主结果 |

**第三层：已有系统参照（证明"不输 SOTA"）**

| 系统 | 来源 | 开源状态（2026-11-12 已逐个核实） | 复现性 | 适配成本 | 用法 |
|---|---|---|---|---|---|
| KernelBlaster [R2] | arXiv:2602.14293 | **✅ 已开源**：https://github.com/NVlabs/KernelBlaster （Apache-2.0） | 中高 | 中高 | **优先复现对象**（含 test harness + repro pipeline） |
| GEAK [R9] | arXiv:2507.23194 | **✅ 已开源**：https://github.com/AMD-AGI/GEAK （Apache-2.0，活跃维护） | 中 | 高（需 AMD GPU） | 跨厂商对照；无 AMD 卡则只引报告值 |
| Xe-Forge [R1] | arXiv:2605.26118 | **✅ 已开源**：https://github.com/IntelLabs/Xe-Forge （Apache-2.0） | 低（需 Intel GPU） | 很高 | 仅引用，不复现（硬件门槛） |
| QiMeng-Kernel [R6] | arXiv:2511.20100 | **⚠️ 仓库已建、代码未发布**：https://github.com/QiMeng-IPRC/QiMeng-Kernel （项目页标 "Code (coming soon)"） | 低（依赖 RL 训练） | 很高 | 仅作 literature baseline，注明不同模型/预算 |
| PRAGMA [R17] | arXiv:2511.06345 | **❌ 仍未找到公开仓库**（GitHub 检索 0 命中，abs/Comments 无代码链接） | 低 | 高（需重跑 KernelBench） | 只引报告值 |
| MARCO [R8] | arXiv:2505.03906 | **❌ 仍未找到公开仓库**（GitHub 检索 0 命中） | 中 | 低（纯 prompt，可按论文自建） | **弱 baseline**：指出其评测为 LeetCode 75，非 HPC kernel；可自行按描述复刻 |
| AutoTriton [R19] | arXiv:2507.05687 | **✅ 已开源**：https://github.com/AI9Stars/AutoTriton （**无 license 标注**，复用需注意） | 中 | 中（需训练/推理算力） | 可选对照（RL 微调路线） |
| KernelBench 官方 harness [R4] | ICML 2025, PMLR 267:47356-47415 | **✅ 已开源**：https://github.com/ScalingIntelligence/KernelBench | 高 | 低 | **评测框架直接用** |
| TritonBench [R18] | arXiv:2502.14752（ACL 2025 Findings） | **✅ 已开源**：https://github.com/thunlp/TritonBench （Apache-2.0） | 高 | 低 | 补充评测集（Triton 场景） |

> **开源状态已核实（2026-11-12）**：上表依据 arXiv API 元数据 + GitHub API 仓库逐个确认（`api.github.com/repos/...`），而非仅读 abs 页。结论：**6 个可直接复现**（KernelBench、KernelBlaster、Xe-Forge、GEAK、TritonBench、AutoTriton），**1 个代码未发布**（QiMeng-Kernel），**2 个无公开仓库**（PRAGMA、MARCO）。论文里必须写清"哪些重跑了、哪些只引报告值"——这是审稿人必查项。

### 5.2 公平比较的硬约束（写进论文的实验设置）

1. **同模型**（如统一 qwen-max / GPT-4o / DeepSeek-V3），不许 baseline 用小模型、我们的方法用大模型。
2. **同知识量与 token 预算**：B2 扁平 RAG 检索到的**字符数必须与图谱检索注入的字符数匹配**（±10%），否则"图更好"只是"注入了更多字"。
3. **同硬件、同编译选项、同线程数、同矩阵规模集合**；报 speedup 时须说明 baseline 计时方式（TF32 开/关），**直接引用 KernelBench-Verified 的结论说明为什么开 TF32 更公平** [R5]。
4. **正确性门槛**：必须用多分布隐藏测试（KernelBench-Verified 的四分布测试思路）防止"硬编码绕过"骗分 [R5]；数值误差阈值写死。
5. **成本必须报**：token 数、端到端时延、检索时延、人工抽取成本，与生成质量同表呈现。

### 5.3 指标（每条给出计算方式）

| 类别 | 指标 | 计算/测量 |
|---|---|---|
| 正确性 | 编译通过率、功能测试通过率、最大相对误差 | 编译成功数/总数；隐藏测试通过数/总数；`max‖y−y_ref‖/‖y_ref‖` |
| 性能 | speedup、几何平均 speedup、达峰值百分比 | `median(t_ref)/median(t_gen)`；跨规模几何平均；`achieved GFLOPS / peak GFLOPS` |
| 知识质量 | 抽取 P/R/F1、对齐 P/R、schema 覆盖率、图连通度 | 人工标注 200-300 条 gold 集；对齐用双向匹配；覆盖率 = 被至少一条策略覆盖的 kernel 比例 |
| 检索 | Recall@k、nDCG@k、策略可适用率 | 人工判定 top-k 中"确实适用"的比例（这是造不出gold时的替代） |
| 成本 | token、时延、搜索时间 | 直接记录 |

### 5.4 成本估算（**估算，非实测**）

- 搭 B0-B5 全套：**3-4 周**（1 人）。
- Ansor baseline 在 CPU 上对 20 个算子搜索：每算子 0.5-4 CPU 小时，总计 **约 1-3 天机时**（估算）。
- 全量 KernelBench 250 题跑 5 个 baseline：**约 100-300 GPU 小时**（估算，取决于是否迭代）。
- **建议先做 CPU 路线（OpenBLAS x86）**：零 GPU 成本，且正好复用用户已有语料。

---

## 6. "前人没有"类表述：哪些能留、哪些必须改

### 表 5：改写清单（可直接替换进开题/论文）

| 原文（开题报告） | 问题 | 建议改写 |
|---|---|---|
| "填补了该领域在结构化知识支撑方面的研究空白" | 直接被 Xe-Forge / KernelBlaster / Leeroopedia 打脸 | "现有 LLM 算子生成工作主要以隐式参数知识（QiMeng 系列）或经验记忆（KernelBlaster）携带优化知识，缺少**可审计、可对齐、可跨架构比较**的结构化策略知识表示。本文提出并评估一种面向 BLAS 算子源码的优化策略知识图谱表示，及其'检索—生成—验证'闭环。" |
| "知识图谱与高性能算子优化的融合创新" | KG+LLM 融合在 2023-2025 已被做烂（综述、零样本 KG 构造、CodeKGC、GraphGen） | 删除"融合"框架，改为具体空白：**"策略知识的适用条件、失效条件、硬件约束与收益证据的四元组建模"** |
| "提出两阶段对齐策略……有效解决图谱冗余问题" | 向量粗筛 + LLM 精判是通用做法，无 HPC 特异性 | 降级到方法章："针对'优化策略'实体的表述多样性，实现粗筛聚类 + LLM 判别 + 计算流程上下文约束的对齐流程，并在 200 条人工标注上报告 P/R" |
| "结构化 Prompt 模板……减少幻觉，为生成的代码性能提供保障" | 与 Xe-Forge 的 CoVeR + 硬件知识库高度重叠 | "模板字段由知识图谱查询生成，包含**实施步骤 / 适用条件 / 失效条件 / 可调参数区间 / 收益证据**；通过逐字段消融量化各字段对正确率与加速比的贡献" |
| "首个 / 首次 / 国内领先" | 无检索证据支撑 | **全部删除**，或改为"据我们所知（检索式见附录），尚未检索到……"并附录检索式与日期 |

**可以保留、且有证据支撑的三条表述**（这是最值得写进开题的）：

1. **"以开源 BLAS 库多架构核心 kernel 为知识源，反向抽取优化策略"**——未检出同类工作（最近邻 Xe-Forge/KernelBlaster 走 GPU/Triton 与经验记忆路线，且不解析开源 BLAS 源码）。
2. **"把优化策略建模为带适用条件、失效条件、硬件约束、可调参数区间、收益证据的可检索实体"**——QiMeng-Kernel 解耦了策略与实现，但策略是 RL 参数；Xe-Forge 有阶段知识但硬编码且不可检索；**没有任何工作把"策略何时失效"作为一等公民**。
3. **"跨架构（x86/ARM/RISC-V）同一算子的策略适用性与冲突建模"**——cluster ⑦ 中无人做过，且用户手上恰好有这份语料。**这条应升为创新点 #1。**

---

## 7. 对用户下一步的可执行建议（按周）

1. **第 1 周（最重要）**：把 `KernelBench` 官方 harness（https://github.com/ScalingIntelligence/KernelBench ）跑通，再跑通 1 个**已开源**方案的子集（**KernelBlaster**：https://github.com/NVlabs/KernelBlaster ，含 test harness + repro pipeline；备选 **GEAK**：https://github.com/AMD-AGI/GEAK ），确认"能跑实验"的闭环。**这一步的价值大于再读 50 篇论文。**
2. **第 1 周**：定评测协议并冻结成文档：TF32 开/关、warmup 次数、重复次数、取中位数、隐藏测试分布、内存指标。依据 KernelBench-Verified [R5]。
3. **第 2-3 周**：从 OpenBLAS 抽 20-30 个 kernel（x86 先做），人工标注 **50-100 条策略实体 gold**（含适用条件/硬件约束/参数区间）。没有 gold 集，图谱质量无法量化，实验无法写。
4. **第 3-4 周**：搭 B2 扁平文本 RAG 对照。**这是保命实验**——只要它做出来，"是不是 RAG"的质疑就被关掉了。
5. **第 4 周**：用 GraphSys'25 的 competency questions 方法 [R13] 验证 Schema，替代开题里"参考 Carta"的空泛说法。
6. **持续**：Related Work 补 4 段（LLM 算子生成 / KG-RAG for code / HPC 本体与 KG / benchmark 与评测协议），并把 2026 年的 4 篇新工作 [R1][R2][R3][R5] 写进去。
7. **中文文献**：用 CNKI 检索式 `(主题=知识图谱) AND (主题=算子 OR 主题=高性能计算 OR 主题=代码优化)`、`(主题=性能优化) AND (主题=知识库)`，检索后**在论文里写明检索日期与命中数**，不要空口说"国内空白"。

---

## 8. 不确定 / 需 Lead 与用户决策的点

1. **代码可得性：已逐个核实完毕（2026-11-12）**。可直接复现 6 个：KernelBench、KernelBlaster、Xe-Forge、GEAK、TritonBench、AutoTriton；QiMeng-Kernel 仓库已建但代码标注 "coming soon"（`language=null`，无代码）；PRAGMA 与 MARCO 未找到公开仓库。→ 结论：**至少有 1 个强 baseline（KernelBlaster）可真实复现**，"没有 baseline"的担忧不成立。详见 §5.1 表。
2. **KernelBench 的 venue 已核实**：**ICML 2025，PMLR 267: 47356-47415**（PMLR 官方页 `proceedings.mlr.press/v267/ouyang25a.html`，OpenReview `yeoN1iQT1x`）。**开题报告把它当普通 arXiv 预印本引用是错的，终稿须改。** TritonBench 的 venue（ACL 2025 Findings）来自 Lead 核实 [R18]，arXiv abs 页本身未标会议。
3. **QiMeng-GEMM 引用信息已核实**：AAAI 2025, 39(21): 22982–22990，DOI 10.1609/aaai.v39i21.34461（AAAI 官方 OJS 页），与开题报告页码一致；作者全名为 Zhou Q, Wen Y, Chen R, Gao K, Xiong W, Li L, Guo Q, Wu Y, Chen Y。
4. **专利先例**：检索命中中国专利 CN119396387B（含"第一算子优化策略库"字样）[R43]，说明工业界已有"算子优化策略库"概念；**专利全文未读**，建议 Lead 派人确认其权利要求范围，以免论文的"策略库"表述与之冲突。
5. **是否上 GPU**：若无稳定 GPU 资源，建议全程 CPU（OpenBLAS x86 + ARM），跨架构故事反而更完整；若上 GPU 则 baseline 复现成本陡增（Xe-Forge 需 Intel GPU、GEAK 需 AMD GPU，二者基本只能引报告值）。
6. **OPTION 本体的复用**：EvoApplications 2023 [R12] 的 OPTION 本体面向黑盒优化算法，与算子优化策略不完全同构；是"借用其本体设计方法论"还是"尝试对齐到 OPTION"需要 Lead 决策（影响 Schema 章写法）。
7. **Leeroopedia 的引用方式**：它是**在线知识库/MCP 包**而非论文，引用时须注明"在线资源 + 访问日期"，不能当论文引。

---

## 附：本文档的核查标记说明

- 标注 **[Rxx]** 的引文均可在 `科研产出/refs/lit_scout_refs.md` 中查到作者、年份、标题、venue、arXiv ID/DOI/URL。
- **开源状态已按仓库逐个核实（2026-11-12）**：核实方式为 arXiv API（作者/Comments/日期）+ GitHub API `api.github.com/repos/<owner>/<repo>`（仓库存在性、license、更新时间、是否含代码），不是只看 abs 页的 "Code" 区块。
- 凡本文写作"待核实"的条目，表示**已确认标题与存在性，但作者列表、卷期页或开源状态未逐项核对**，禁止在论文中直接引用其细节。
- 凡本文写作"检索未检出/负结果"的，表示**在本次检索范围内**未发现，不等于不存在；论文中必须写成"据我们所知"并附检索式。
