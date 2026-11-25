# 引文清单（lit-scout / task-1）

> 核查规则：**只收录本次实际检索/抓取到的条目**。凡作者列表、卷期页、开源状态未逐项核对的，明确标注"待核实"。禁止在论文中把"待核实"字段当作事实引用。
> 检索与核实日期：2026-11-12。核实方式：`web_fetch` 打开 arXiv abs 页 / 会议官网 / DOI 页 / 出版社页。
> 与我们的关系：**最近邻 / 强先例 / 可借鉴 / baseline / 背景引用 / 无关**

---

## A. 最近邻（直接决定新颖性判定，必须逐篇写进 Related Work）

**[R1]** Spoczynski M, Fleischer D, Berchansky M, Ben-Melech Stan G, Guskin S, Xu W, Siemieniuk A, Heinecke A. **2026**. *Xe-Forge: Multi-Stage LLM-Powered Kernel Optimization for Intel GPU.* arXiv:2605.26118 [cs.DC]. https://arxiv.org/abs/2605.26118
- 一句话：多阶段（最多 9 阶段）LLM kernel 优化流水线，内置编码 Intel GPU 约束（warp 数、GRF 模式、SLM 容量）的 curated knowledge base，97 个 KernelBench L2 kernel 上 1.17× 几何平均 vs PyTorch eager。
- 关系：**最近邻（最强）**。它的"curated knowledge base + 分阶段策略"几乎就是"优化策略知识库"的另一种写法；我们必须用"图结构 / 可检索排序 / 实体对齐 / 策略失效条件 / 跨架构"来区分。
- 开源：**已确认开源 → https://github.com/IntelLabs/Xe-Forge**（IntelLabs，Apache-2.0，创建 2026-03-24，截至 2026-11-12 约 29 stars，语言 Python；仓库描述 "Multi-stage LLM agent pipeline for optimizing AI kernels on Intel GPU — from analysis to autotuning"）。注：arXiv abs 页未给链接，仓库由 GitHub 检索确认。

**[R2]** Dong K S, Modi S, Nikiforov D, Damani S, Lin E, Hari S K S, Kozyrakis C. **2026**. *KernelBlaster: Continual Cross-Task CUDA Optimization via Memory-Augmented In-Context Reinforcement Learning.* arXiv:2602.14293 [cs.LG]. https://arxiv.org/abs/2602.14293
- 一句话：把跨任务探索经验累积进可检索的 **Persistent CUDA Knowledge Base**，做持续的 CUDA 优化；KernelBench L1/L2/L3 几何平均 1.43×/2.50×/1.50×；**开源框架 + 测试/验证/可复现评测流水线**。
- 关系：**最近邻（第二强）**。它证明"知识库对 LLM 优化有增益"，但知识是无结构的经验条目——这正是我们"结构化"的对照点。
- 开源：**已确认开源 → https://github.com/NVlabs/KernelBlaster**（NVlabs，Apache-2.0，创建 2026-02-09，约 66 stars，语言 Cuda；仓库描述 "A framework for in context learning for code optimization"）。abs 页声明与仓库一致。

**[R3]** Davis J H, Rydzy K, Ramesh S, Nilay A, Nichols D, Raj S, Jain N, Bhatele A. **2026**. *KEET: Explaining Performance of GPU Kernels Using LLM Agents.* arXiv:2605.04467 [cs.PF]. https://arxiv.org/abs/2605.04467
- 一句话：LLM 智能体解读 Nsight Compute 剖面，生成有数据依据的瓶颈解释与优化建议；把解释作为上下文可提升下游 LLM 代码优化质量。
- 关系：**最近邻（上游互补）**。它是"性能数据→自然语言知识"的入口，但没有沉淀为可复用资产；可作为我们图谱的**自动知识来源之一**引用。
- 开源：待核实。

**[R6]** Zhu X, Peng S, Guo J, Chen Y, Guo Q, Wen Y, Qin H, Chen R, Zhou Q, Gao K, Wu Y, Zhao C, Li L. **2025**. *QiMeng-Kernel: Macro-Thinking Micro-Coding Paradigm for LLM-Based High-Performance GPU Kernel Generation.* arXiv:2511.20100（**accepted by AAAI 2026**，据 abs 页 Comments）. https://arxiv.org/abs/2511.20100
- 一句话：宏观层用 RL 训练的轻量 LLM 输出"优化类型 + 代码作用区域"语义动作，微观层用通用 LLM 分步实现；KernelBench L1-2 近 100%、L3 70%，比 SOTA 高 50%+，比 PyTorch Eager 2.2×，TritonBench 59.64%。
- 关系：**最近邻 + 强 baseline**。它已经做了"策略与实现解耦"，我们的"三层 Agent 解析"必须让位于它没有的东西（可读策略知识、跨架构）。
- 开源：**仓库已建但代码未发布** → https://github.com/QiMeng-IPRC/QiMeng-Kernel （QiMeng-IPRC，Apache-2.0，创建 2025-11-09，约 11 stars）。项目页 https://qimeng-iprc.github.io/QiMeng-Kernel/ 的链接标注为 **"Code (coming soon)"**，且 GitHub API 显示该仓库 `language: null`（截至 2026-11-12 无代码文件）。→ 目前**不可复现**，只能引报告值。

**[R7]** Zhang X, Peng S, Zhou Q, Wen Y, Guo Q, Chen R, Zhu X, Xiong W, Chen H, Ma C, Gao K, Zhao C, Wu Y, Chen Y, Li L. **2025**. *QiMeng-TensorOp: Automatically Generating High-Performance Tensor Operators with Hardware Primitives.* arXiv:2505.06302 [cs.LG]. https://arxiv.org/abs/2505.06302
- 一句话：一行 prompt 自动利用硬件特性/硬件原语生成张量算子并调参；比原生 LLM 最高 1291×；**RISC-V 上达 OpenBLAS 的 251%、NVIDIA 上达 cuBLAS 的 124%**，开发成本降 200×。
- 关系：**最近邻（性能上界）**。它代表"知识内化进模型"的路线，是我们的最强对手与性能参照；也说明"跨架构"已被它做过一半（但知识不是显式的）。
- 开源：待核实。

**[R8]** Rahman A, Cvetkovic V, Reece K, Walters A, Hassan Y, Tummeti A, Torres B, Cooney D, Ellis M, Nikolopoulos D S. **2025**. *MARCO: Multi-Agent Code Optimization with Real-Time Knowledge Integration for High-Performance Computing.* arXiv:2505.03906 [cs.DC]（v3）. https://arxiv.org/abs/2505.03906
- 一句话：生成智能体 + 性能评估智能体的多智能体 HPC 代码优化框架，核心是 **web 检索组件实时抓取最新优化技巧注入生成**；LeetCode 75 上比 Claude 3.5 Sonnet 单独跑平均运行时降 14.6%，web 检索组件相对 base MARCO 贡献 30.9% 提升。
- 关系：**弱 baseline + 消融范本**。**评测集是 LeetCode 75 而非 HPC kernel/算子**——这是它最致命的弱点，可作为我们"知识注入有效但评测不匹配"的论据；同时它的"有检索 vs 无检索"消融正是我们 B2 实验的范本。
- 开源：**仍未找到公开仓库**。依据：arXiv abs 页与 Comments 均无代码链接；GitHub 仓库检索 `PRAGMA kernel optimization` 与 `MARCO + HPC + code optimization` 均 0 命中（2026-11-12 检索）。→ 只能作文献 baseline，不能声称可复现。

**[R11]** Leeroo AI. **2026**（访问日期 2026-11-12）. *Leeroopedia: Your ML & Data Knowledge Wiki.* https://leeroopedia.com/ （MCP: https://github.com/Leeroo-AI/leeroopedia-mcp ）
- 一句话：Semantic MediaWiki 构成的 ML/数据工程知识库，**27,667 页面（1,118 Workflows / 7,697 Principles / 16,832 Implementations / 1,245 Heuristics / 909 Environments）**，覆盖 vLLM、DeepSpeed、FlashAttention、Triton 等 1000+ 框架，并提供 agent 检索接口。
- 关系：**强先例（结构化优化知识库）**。它是"知识库"路线最直接的工业证据，但没有本体验证、没有对齐算法、没有"结构化是否有效"的评估——这正是我们可补的位置。
- 性质：在线知识库，**不是论文**；引用时须注明访问日期，不能当学术论文引。

## B. Baseline / benchmark 与评测协议

**[R4]** Ouyang A, Guo S, Arora S, Zhang A L, Hu W, Ré C, Mirhoseini A. **2025**. *KernelBench: Can LLMs Write Efficient GPU Kernels?* **Proceedings of the 42nd International Conference on Machine Learning (ICML 2025), PMLR 267: 47356-47415**. 论文页：https://proceedings.mlr.press/v267/ouyang25a.html ；arXiv:2502.10517；OpenReview: https://openreview.net/forum?id=yeoN1iQT1x
- 一句话：250 个 PyTorch ML 工作负载的开放评测框架，提出 fast_p 指标；前沿推理模型**不足 20%** 能追平 PyTorch 基线。
- 关系：**baseline 数据集与评测框架（首选）**。用户实验的主战场。
- venue/页码：✅ 本轮已从 PMLR 官方页核实（ICML 2025，13–19 Jul 2025，PMLR 267，pp. 47356–47415）。**注意：开题的参考文献表把 KernelBench 写成了普通 arXiv 预印本，终稿应改为 ICML 2025 正式出处。**
- 开源：**已确认开源 → https://github.com/ScalingIntelligence/KernelBench**（ScalingIntelligence，约 1330 stars，2026-03 仍在更新；PMLR 页面 "Software" 链接一致）。

**[R5]** Zhang Y, Yu P, Wang J, Fan M, Reed J, Mirhoseini A, Su W. **2026**. *KernelBench-Verified: Do LLM-Generated Kernels Actually Beat PyTorch?* arXiv:2607.16241 [cs.LG]. https://arxiv.org/abs/2607.16241
- 一句话：证明前模型常用 reward hacking 虚报性能（硬编码绕过、TF32 基线不公）；修正协议后 **GPT-5.5 几何平均只有 0.88×（标准协议下报 1.43×）**，**无模型能稳定超过 PyTorch**，最好的模型有 **28%** 的 kernel 反而增大峰值显存。
- 关系：**评测协议的决定性依据**。我们的实验设置必须照它修正，并在论文里引用它解释为什么口径重要。

**[R18]** **TritonBench: Benchmarking Large Language Model Capabilities for Generating Triton Operators.** arXiv:2502.14752（据 Lead 核实：**ACL 2025 Findings**；**作者列表待核实**，检索命中作者含 Jianling Li / Zhenye Gao / Shangzhan Li 等，未逐项确认）. https://arxiv.org/abs/2502.14752
- 一句话：Triton 算子生成基准，含 184 个来自 GitHub 的真实算子 + 与 PyTorch 接口对齐的算子两个通道，并做效率 profiling。
- 关系：**补充 benchmark**（QiMeng-Kernel 在该基准上报 59.64%）。
- 作者（本轮已从 arXiv API 核实）：Jianling Li, Shangzhan Li, Zhenye Gao, Qi Shi, Yuxuan Li, Zefan Wang, Jiacheng Huang, Haojie Wang, Jianrong Wang, Xu Han, Zhiyuan Liu, Maosong Sun。
- venue：**ACL 2025 Findings（据 Lead 核实；arXiv abs 页未标注会议）**。
- 开源：**已确认开源 → https://github.com/thunlp/TritonBench**（thunlp，Apache-2.0，约 143 stars，语言 Python）。arXiv 摘要原文 "will be available at ..."，仓库已存在。

**[R34]** Lv J, He X, Liu Y, et al. **2025**. *HPCTransCompile: An AI Compiler Generated Dataset for High-Performance CUDA Transpilation and LLM Preliminary Exploration.* arXiv:2506.10401. https://arxiv.org/abs/2506.10401
- 一句话：基于 TVM 图增强生成 2 万条 CUDA↔多平台代码对，配套 HPCTransEval 基准。
- 关系：**可借鉴（跨平台数据构造方法）+ benchmark**。与用户的"跨架构策略迁移"同源但目标不同（它做转译，不做策略适用性）。
- 细节（作者完整列表、开源状态）：**待核实**。

**[R36]** Liao C, Lin P H, Asplund J, et al. **2017**. *DataRaceBench: A Benchmark Suite for Systematic Evaluation of Data Race Detection Tools.* SC'17. （据开题报告参考文献 [R36 原文]）
- 一句话：数据竞争检测工具的系统评测基准。
- 关系：**背景引用**（HPC 代码正确性评测传统）。

**[R37]** **The Correctness Illusion in LLM-Generated GPU Kernels.** arXiv:2606.20128 [cs.*]. https://arxiv.org/abs/2606.20128
- 一句话：检索命中，讨论 LLM 生成 GPU kernel 的"正确性幻觉"问题。
- 关系：**可引用（正确性评测）**。**作者、venue、详细结论均待核实**（仅见于检索结果标题页）。

## C. HPC + KG / 本体（方法合法性来源）

**[R12]** Kostovska A, Vermetten D, Džeroski S, Panov P, Eftimov T, Doerr C. **2023**. *Using Knowledge Graphs for Performance Prediction of Modular Optimization Algorithms.* Applications of Evolutionary Computation (EvoApplications 2023), Brno, pp.253-268. DOI: **10.1007/978-3-031-30229-9_17**；HAL: https://hal.sorbonne-universite.fr/hal-04180586
- 一句话：扩展 OPTION 本体表示模块化黑盒优化算法，用 KG 嵌入做三元组分类，预测算法实例能否在 24 个 BBOB 函数上达到目标精度。
- 关系：**最强方法学先例**。它证明"本体 + KG + 嵌入 → 性能预测"这条路在优化领域是通的；差别是对象为黑盒算法配置而非算子代码。
- DOI/页码：✅ 已核实。

**[R13]** Khan J A, Bartolini A. **2025**. *A Unified Ontology for Scalable Knowledge Graph-Driven Operational Data Analytics in High-Performance Computing Systems.* GraphSys'25 workshop @ Euro-Par 2025（据 abs Comments）；arXiv:2507.06107. https://arxiv.org/abs/2507.06107
- 一句话：HPC 运维数据分析（ODA）的**首个统一本体**，覆盖 M100（Cineca）与 F-DATA（Fugaku）两套遥测；用 **36 条 competency questions** 验证；KG 存储开销降低 38.84%。
- 关系：**可借鉴（Schema 验证方法论）**。直接可用其"competency questions"方法解决开题"难点一：Schema 设计"的可验证性问题。

**[R14]** Kousha P, Sathu V, Lieber M, Subramoni H, Panda D K. **2023**. *Democratizing HPC Access and Use with Knowledge Graphs.* The First Workshop on Democratizing High-Performance Computing (D-HPC), SC23 Workshops. https://sc23.supercomputing.org/proceedings/workshops/workshop_pages/ws_dhpc102.html
- 一句话：面向 HPC 新手的 HPC 运行时关系本体 + KG + AI 语音助手接口。
- 关系：**可借鉴（HPC 本体先例）**。证明"HPC+KG"不是新概念，但它是**运行时/文档**知识，不是算子优化策略知识。
- 核实：会议官网摘要页 ✅。

**[R32]** **P-MoVE: Performance Monitoring and Visualization with Encoded Knowledge.** SCW 2024（SC'24 Workshops）. DOI: **10.1109/SCW63240.2024.00193**. https://dl.acm.org/doi/10.1109/SCW63240.2024.00193
- 一句话：检索命中的"性能监控 + 编码知识"工作（HPC 性能可视化方向）。
- 关系：**背景引用（HPC 性能知识编码）**。**作者、详细内容待核实**（仅见标题与 DOI/会议）。

**[R15]** NVIDIA. **2023**. *ChipNeMo: Domain-Adapted LLMs for Chip Design.* https://research.nvidia.com/publication/2023-10_chipnemo-domain-adapted-llms-chip-design
- 一句话：领域自适应 LLM（领域语料继续预训练 + SFT + **RAG 检索工程文档**）用于芯片设计任务。
- 关系：**可借鉴（工业界 RAG+领域知识范式）**。**作者列表与 arXiv ID（疑为 2311.00176）待核实**。

## D. KG / RAG for code

**[R10]** Chinthareddy M R. **2026**. *Reliable Graph-RAG for Codebases: AST-Derived Graphs vs LLM-Extracted Knowledge Graphs.* arXiv:2601.08773 [cs.SE]. https://arxiv.org/abs/2601.08773
- 一句话：在 Java 代码库上对比三种检索管线（纯向量 / LLM 抽取 KG / AST 派生图），发现 **LLM 抽取 KG 覆盖不全（Shopizer 漏 377 文件）、索引慢、成本高**，AST 图 CORRECTNESS 最高。
- 关系：**重要警告**。针对用户的"LLM Agent 抽取 + KG 对齐"路线：必须报告覆盖率/成本，并说明为何仍值得用 LLM（优化策略知识不在 AST 里）。

**[R20]** Liang L, Li Y, Wen M, et al. **2022**. *KG4Py: A toolkit for generating Python knowledge graph and code semantic search.* Connection Science, 34(1): 1384-1400. （据开题报告参考文献）
- 一句话：对 GitHub Python 代码做静态分析构建知识图谱，支持自然语言语义搜索。
- 关系：**可借鉴（代码知识图谱构建）**。卷期页来自开题报告，**未二次核实**。

**[R21]** Bi Z, Chen J, Jiang Y, et al. **2024**. *CodeKGC: Code Language Model for Generative Knowledge Graph Construction.* ACM Transactions on Asian and Low-Resource Language Information Processing, 23(3): 1-16. （据开题报告参考文献）
- 一句话：把 KG 构造任务转成代码生成任务，用代码结构强化知识建模。
- 关系：**可借鉴（LLM 生成知识三元组）**。未二次核实。

**[R22]** Carta S, Giuliani A, Piano L, et al. **2023**. *Iterative zero-shot LLM prompting for knowledge graph construction.* arXiv:2307.01128. （开题报告已引用）
- 一句话：零样本迭代 prompt 抽取 KG 三元组。
- 关系：**背景引用**。注意：开题把它当作"簇思想"来源，实为**零样本抽取**工作，引用定位需修正（这是开题的一处引文错配）。

**[R23]** **CodexGraph: Bridging Large Language Models and Code Repositories via Code Graph Databases.** NAACL 2025 Long Papers. https://aclanthology.org/2025.naacl-long.7/
- 一句话：把代码库建成图数据库，让 LLM 用图查询做仓库级任务。
- 关系：**可借鉴（仓库级图检索）**。**作者列表待核实**。

**[R24]** **Codebase-Memory: Tree-Sitter-Based Knowledge Graphs for LLM Code Exploration via MCP.** arXiv:2603.27277.
- 一句话：基于 Tree-sitter 的代码知识图 + MCP 供 LLM 探索代码库。
- 关系：**背景引用**。**作者、venue、详细内容待核实**（仅检索命中标题页）。

## E. LLM/Agent 生成高性能算子

**[R16]** Zhou Q, Wen Y, Chen R, et al. **2025**. *QiMeng-GEMM: Automatically Generating High-Performance Matrix Multiplication Code by Exploiting Large Language Models.* AAAI 2025, 39(21): 22982-22990. https://ojs.aaai.org/index.php/AAAI/article/download/34461/36616
- 一句话：预训练阶段嵌入国产 GPU 硬件知识，先宏观（分块/布局）后微观（展开/调度）生成 GEMM；比原生 prompt 最高 113×；RISC-V CPU 达 OpenBLAS 的 **211%**，NVIDIA GPU 达 cuBLAS 的 **115%**，开发成本降 240×。
- 关系：**最近邻（前作）+ baseline**。
- 引用信息：✅ 本轮已从 AAAI 官方 OJS 页核实：Zhou Q, Wen Y, Chen R, Gao K, Xiong W, Li L, Guo Q, Wu Y, Chen Y. (2025). *Proceedings of the AAAI Conference on Artificial Intelligence*, **39(21), 22982–22990**，DOI **10.1609/aaai.v39i21.34461**（2025-04-11 出版，AAAI-25 Technical Tracks 21）。与开题报告所写页码一致。
- 开源：待核实（本轮未查；QiMeng 系列部分工作在 QiMeng-IPRC 组织下，可顺藤确认）。

**[R9]** Wang J, Joshi V, Majumder S, Chao X, Ding B, Liu Z, Brahma P P, Li D, Liu Z, Barsoum E. **2025**. *Geak: Introducing Triton Kernel AI Agent & Evaluation Benchmarks.* arXiv:2507.23194 [cs.CL]. https://arxiv.org/abs/2507.23194
- 一句话：面向 AMD MI300X/MI250 的 Triton kernel 生成 agent（Reflexion 式反馈循环），正确率最高 63%，加速最高 2.59×。
- 关系：**baseline（跨厂商对照）**。
- 开源：**已确认开源 → https://github.com/AMD-AGI/GEAK**（AMD-AGI，Apache-2.0，创建 2025-07-30，约 204 stars，语言 Python；topics 含 kernel-optimization / triton / hip；2026-11-12 仍在更新）；另有官方文档 https://rocm.docs.amd.com/projects/geak/en/docs-4.0.0/index.html 。

**[R17]** Lei K, Yang H, Zhang H, You X, Zhang K, Luan Z, Liu Y, Qian D. **2025**. *PRAGMA: A Profiling-Reasoned Multi-Agent Framework for Automatic Kernel Optimization.* arXiv:2511.06345 [cs.DC]. https://arxiv.org/abs/2511.06345
- 一句话：把细粒度硬件 profile 与执行反馈并入推理循环的多智能体 kernel 生成；KernelBench CPU/GPU 上相对 Torch 平均 2.81×/2.30×。
- 关系：**baseline（强）**。注意其对比对象含 AIKG；PRAGMA 自身**未找到公开仓库**（GitHub 检索 `PRAGMA kernel optimization` 0 命中，arXiv abs/Comments 无代码链接）。→ 只可引报告值。

**[R19]** Li S, Wang Z, He Y, Li Y, Shi Q, Li J, Hu Y, Che W, Han X, Liu Z, Sun M. **2025**. *AutoTriton: Automatic Triton Programming with Reinforcement Learning in LLMs.* arXiv:2507.05687 [cs.LG]. https://arxiv.org/abs/2507.05687
- 一句话：8B 模型经 SFT + GRPO 强化学习训练写 Triton 程序，在 TritonBench 五个通道与 KernelBench 上达到与 Claude-4-Sonnet、DeepSeek-R1-0528 相当的水平。
- 关系：**背景引用（RL 微调路线）**。
- 作者：✅ 本轮已从 arXiv API 核实。
- 开源：**已确认开源 → https://github.com/AI9Stars/AutoTriton**（AI9Stars，约 69 stars，创建 2025-06-16，最后推送 2025-07-14；**未标注 license**，复用前需确认）。

**\[R38\]** Nichols D, et al. **2025**. *HPC-Coder-v2: Studying Code LLMs across Low-Resource Parallel Languages.* ISC High Performance 2025. （据开题报告参考文献）
- 一句话：低资源并行语言（如 Chapel/Fortran 变体）上的代码 LLM 研究。
- 关系：**背景引用**。未二次核实。

**\[R41\]** OpenBLAS 项目（用户现有语料来源）. https://github.com/OpenMathLib/OpenBLAS （仓库地址待核实，项目主页为 openblas.net）
- 关系：**数据源**。

## F. 知识驱动编译 / 自动调优

**[R25]** Zheng L, Jia C, Sun M, et al. **2020**. *Ansor: Generating High-Performance Tensor Programs for Deep Learning.* OSDI 2020；arXiv:2006.06762. https://arxiv.org/abs/2006.06762
- 一句话：分层搜索 + 学习式成本模型的张量程序自动生成。
- 关系：**baseline（非 LLM 自动调优）**。作者完整列表待核实。

**[R26]** Trofin M, et al. **2021**. *MLGO: a Machine Learning Guided Compiler Optimizations Framework.* arXiv:2101.04808. https://arxiv.org/abs/2101.04808
- 一句话：用 RL/进化策略训练 LLVM 内联等优化决策策略。
- 关系：**背景引用（知识驱动编译的先例）**。作者完整列表待核实。

**[R27]** **CompilerGym.** https://compilergym.com/ （论文疑为 Cummins et al. 2022，**待核实**）
- 一句话：把编译器优化暴露为 RL 环境的标准工具链。
- 关系：**背景引用**。

**[R28]** Nugteren C, Codreanu V. **2015**. *CLTune: A Generic Auto-Tuner for OpenCL Kernels.* IEEE MCSoC 2015. https://dl.acm.org/doi/abs/10.1109/MCSoC.2015.10
- 一句话：OpenCL kernel 参数搜索自动调优器。
- 关系：**背景引用**。作者与页码待核实（检索命中会议页）。

**[R29]** **A Hybrid, Knowledge-Guided Evolutionary Framework for Personalized Compiler Auto-Tuning.** arXiv:2510.14292.
- 一句话：知识引导的个性化编译器自动调优。
- 关系：**可借鉴（知识引导调优）**。**作者、venue 待核实**。

**[R30]** **Tensor Program Optimization for the RISC-V Vector Extension Using Probabilistic Programs.** arXiv:2507.01457（另见 IEEE 文档 11241007）.
- 一句话：面向 RISC-V 向量扩展的张量程序概率化优化。
- 关系：**可借鉴（RISC-V 架构调优）**。**作者、venue 待核实**。

**[R33]** **Toward Compiler World Models: Learning Latent Dynamics for Efficient Tensor Program Search.** arXiv:2606.09312.
- 一句话：从 TenSet 调优日志学习张量程序的潜空间动力学，做搜索。
- 关系：**可借鉴（"调优日志=性能知识"的近期代表）**。**作者、venue 待核实**。

## G. 性能建模 / roofline / 调优配置数据库 / 知识库

**[R31]** Williams S, Waterman A, Patterson D. **2009**. *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* Communications of the ACM, 52(4): 65-76.
- 一句话：roofline 性能模型（计算/带宽瓶颈可视化）。
- 关系：**指标归一化依据**（性能指标用"达 peak 百分比"表达）。经典文献，卷期页按常识填写，**建议终稿核对**。

**[R40]** Sharma R, Levine R, Chen Z, Srikanth A, Xu Y, Sorensen T. **2026**. *GPU Goldmines: Tunable WebGPU Kernels for Edge LLM Inference.* arXiv:2610.17321 [cs.DC]（11 pages, 5 figures, 5 tables）. https://arxiv.org/abs/2610.17321
- 一句话：面向边缘 LLM 推理的 **WebGPU kernel 规格化/执行/调优框架**，实例化 matmul、matvec、flash attention；跨多设备、多 shape、多浏览器做性能刻画，**并用得到的配置数据库加速后续自动调优与配置选择**；Llama/Qwen shape 上 matmul 比 ONNX Runtime 与 llama.cpp WebGPU kernel 分别高 7.0× / 1.41×（几何平均），llama.cpp chat 负载 prefill 端到端提升 1.48×。
- 关系：**可借鉴（"调优配置数据库"近期先例）**。它是"把'在什么设备/什么 shape 下用什么配置'沉淀成数据库再用"的直接先例——**这正是我们"策略 × 硬件 × 适用条件"知识库最朴素的形式**；差别是它是纯配置（tile size 等数值）而非可读的优化策略语义，也没有实体对齐/图检索。
- 核实：✅ 本轮已用 arXiv API 核实标题、作者、日期与摘要。**原稿把该编号误记为 "Performance Portability of GPU Kernels"，已修正；归类由"跨架构"改为"WebGPU kernel 调优 + 调优配置数据库"。**

**[R35]** DRB-ML 数据集（随 LM4HPC 提出）. 见 Chen L, Lin P H, Vanderbruggen T, et al. **2023**. *LM4HPC: Towards Effective Language Model Application in High-Performance Computing.* IWOMP 2023, Springer, pp.18-33. （据开题报告参考文献）
- 一句话：HPC 领域 LLM 应用的"数据-模型-评估"闭环，含代码感知 tokenizer 与 DRB-ML 数据集。
- 关系：**背景引用（HPC 领域数据集）**。数据集公开性**待核实**。

## H. 跨架构 / 代码变换与可移植

**[R39]** Taneja J, Laird A, Yan C, et al. **2025**. *LLM-Vectorizer: LLM-based Verified Loop Vectorizer.* CGO 2025, pp.137-149. （据 `_dsh_work/extracted/reading_list.txt`）
- 一句话：LLM 生成向量化变换并用验证器保证等价。
- 关系：**可借鉴（LLM + 验证闭环）**。未二次核实。

## I. 强先例但非论文 / 需谨慎引用

**[R42]** Gainaru A. **2026-06-03**. *Curating Agentic Workflows with Knowledge Graphs and Operational Experience.* Lightning Talk, Trillion Parameter Consortium (TPC26), Baltimore. https://ana-gainaru.com/talks/2026-06-03-tpc
- 一句话：HPC 场景下"智能体工作流 + 知识图谱 + 运维经验"的策展与复用；同一 BoF 议题明确包含 "performance modeling, tuning, autonomous HPC workflow orchestration"。
- 关系：**方向已被公开提出的证据**（说明"前人没有"的说法在社区层面站不住）。**是 talk 不是论文**，只能作为"研究动态"引用。

**[R43]** 中国专利 **CN119396387B**（检索命中，含"第一算子优化策略库"字样）. https://patentimages.storage.googleapis.com/28/ff/9b/9c6aef3b0da527/CN119396387B.pdf
- 一句话：工业界"算子优化策略库"概念的先例（用于模型算子优化/硬件平台适配）。
- 关系：**风险提示**。**专利全文未读、权利要求范围未核实**；论文中使用"算子优化策略库"表述前建议 Lead 派人确认。

## J. 仅作背景、未逐条核实（开题报告已有，本清单不重复展开）

LM4HPC [R35]、HPC-GPT（Ding X, et al., SC'23 Workshops, 2023: 951-960）、HPC-Coder（Nichols D, et al., ISC 2024）、MonoCoder（Kadosh T, et al., HPEC 2024）、GraphGen（Chen Z, et al., arXiv:2505.20416）、Microsoft GraphRAG（Edge D, et al., 2024，**arXiv 编号待核实**）、CodeBLEU（Ren S, et al., arXiv:2009.10297）。
> 以上条目在开题报告参考文献表中已出现，本次未重新核实其卷期页；引用前请以原文为准。

---

## 汇总统计

| 类别 | 条目数 | 核实情况 |
|---|---|---|
| 最近邻（A） | 6 | 6 条 abs 页均已核实；**R1/R2 开源仓库已确认** |
| baseline/benchmark（B） | 7 | R4/R5/R34 已核实（R4 另核实 ICML/PMLR 页码）；**R18/R19 作者与仓库已核实**；R36/R37 待核实 |
| HPC+KG/本体（C） | 5 | 4 条（R12/R13/R14/R32）；R15 待核实 |
| KG/RAG for code（D） | 6 | 2 条（R10/R23）；其余据开题报告或检索页 |
| LLM 算子生成（E） | 8 | 6 条（R6/R7/R9/R16/R17 + R19）；R16 卷期/DOI、R9 仓库已确认 |
| 知识驱动编译/调优（F） | 6 | 2 条标题级（R25/R26）；其余待核实 |
| 性能建模/调优数据库（G） | 3 | R40 已用 arXiv API 完整核实；R31 经典文献待终稿核对；R35 据开题报告 |
| 跨架构/代码变换（H） | 1 | R39 据 reading_list，未二次核实 |
| 先例但非论文（I） | 2 | 2 条 |

### 开源状态核实结果（2026-11-12，逐仓库确认）

| 工作 | 结论 | 仓库 URL / 依据 |
|---|---|---|
| KernelBench [R4] | **已开源** | https://github.com/ScalingIntelligence/KernelBench （约 1330 stars；PMLR 页 Software 链接一致） |
| KernelBlaster [R2] | **已开源** | https://github.com/NVlabs/KernelBlaster （Apache-2.0，约 66 stars） |
| Xe-Forge [R1] | **已开源** | https://github.com/IntelLabs/Xe-Forge （Apache-2.0，约 29 stars） |
| GEAK [R9] | **已开源** | https://github.com/AMD-AGI/GEAK （Apache-2.0，约 204 stars） |
| TritonBench [R18] | **已开源** | https://github.com/thunlp/TritonBench （Apache-2.0，约 143 stars） |
| AutoTriton [R19] | **已开源（无 license）** | https://github.com/AI9Stars/AutoTriton （约 69 stars，最后推送 2025-07-14；复用需注意许可） |
| QiMeng-Kernel [R6] | **仓库已建、代码未发布** | https://github.com/QiMeng-IPRC/QiMeng-Kernel （Apache-2.0；项目页标 "Code (coming soon)"，API 显示 language=null） |
| PRAGMA [R17] | **仍未找到** | GitHub 检索 `PRAGMA kernel optimization` 0 命中；arXiv abs/Comments 无代码链接 |
| MARCO [R8] | **仍未找到** | GitHub 检索 `MARCO + HPC + code optimization` 0 命中；arXiv abs/Comments 无代码链接 |

**给 Lead 的提醒**：标"待核实"最多的集中在 F/H 两簇（知识驱动编译、跨架构）。这两簇恰好是**用户最可能被问到**的方向（"你和 Ansor/MLGO 什么关系""你怎么证明跨架构不是转译"）。建议派第二个人在 IEEE Xplore/ACM DL 里做一次**库内**检索补齐，尤其是 2025-2026 年的跨架构 kernel 迁移工作。
