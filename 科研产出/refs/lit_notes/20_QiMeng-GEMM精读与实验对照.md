# 20 QiMeng-GEMM 精读：它的实验到底"简单"在哪、你能借鉴什么

> 论文：Qirui Zhou, Yuanbo Wen, Ruizhi Chen, Ke Gao, Weiqiang Xiong, Ling Li, Qi Guo, Yanjun Wu, Yunji Chen. **QiMeng-GEMM: Automatically Generating High-Performance Matrix Multiplication Code by Exploiting Large Language Models.** AAAI-25, pp. 22982–22990.
> 本地文件：`D:\Zotero file\storage\4MIZFK9Q\Zhou 等 - 2025 - QiMeng-GEMM ... .pdf`（9 页）
> 2026-11-15｜Lead

---

## 0. 一句话结论

**它"对比实验简单"是真的：baseline 只有 vanilla prompt / CoT prompt / OpenBLAS / cuBLAS，没有和其他 AI 算子优化方法比，也没有 seeds/方差/统计检验；但它的"实验规模"不小——4 个 LLM × 3 个硬件 × 7 个尺寸，加 3 组消融。**
所以：**不要因为它 baseline 少就觉得自己也可以少做**；它的 claim 很窄（"我的 prompt 机制比 vanilla/CoT/库好"），你的 claim 更宽（KG vs 扁平 RAG、跨架构、12 算子），需要对应更完整的对照。

---

## 1. 论文主要工作（一句话版）

- **问题**：LLM 能生成功能正确的 GEMM，但生成不了高性能 GEMM；难点是 LLM 不懂硬件架构、prompt 不能释放优化潜力。
- **方法**：把 GEMM 优化拆成 **5 类原语**（Tiling / Reordering / Vectorization / Layout / Pipeline），为每类写 **meta-prompt**（平台无关描述 + 平台特定 hints）；再用 **auto-tuning** 搜索 meta-prompt 的组合（T/R/V/L/P 的排列），迭代生成代码。
- **卖点**：4 个 LLM、3 个平台、7 个尺寸；相对 vanilla 最高 113.67×（C910 CPU）/ 9.65×（RTX 4070）；达到 OpenBLAS 211%、cuBLAS 115%；开发成本从 24 天降到 10 分钟。

---

## 2. 它的实验清单（我逐项数了）

### 2.1 主实验

| 维度 | 取值 |
|---|---|
| LLM | GPT-4o、Claude 3.5 Sonnet、DeepSeek-Coder-V2、Codestral-22B（4 个） |
| 硬件 | XuanTie C910 RISC-V CPU、RTX 4070、A100（3 个） |
| 尺寸 | CPU：256/512/768/1024/1536/2048/4096；GPU：512/768/1024/2048/4096/8192/16384（7 档） |
| baseline | vanilla prompt、CoT prompt、OpenBLAS 0.3.27、cuBLAS 12.1 |
| 指标 | GFLOPS（CPU）/ TFLOPS（GPU） |
| 规模 | 4 LLM × 3 平台 × 7 尺寸 ≈ **84 个配置**（Table 1） |

### 2.2 消融 / 补充实验

| 实验 | 内容 | 规模 |
|---|---|---|
| 平台特定 hints（Table 3） | 5 类模板，w/o hint vs w/ hint | A100 上 5 组 |
| Auto-tuning 搜索轨迹（Fig. 5） | 25 条搜索轨迹，统计功能正确率、超过 OpenBLAS 的比例 | C910 上 25 条 |
| 开发成本（Table 4） | junior/senior coder vs QiMeng，GPU/CPU 两平台 | 2 平台 × 3 种方式 |

### 2.3 它**没有**做的（也就是"简单"的地方）

1. **没有和其他 AI 算子优化方法比**：没有 KernelBlaster、Xe-Forge、PRAGMA/MARCO、Ansor/TVM 等，baseline 只有 prompt 变体和两个库；
2. **没有 seeds / 方差 / 置信区间 / 显著性检验**：性能表看起来是单次测量；
3. **没有 held-out 数据集**：只做 GEMM 一个算子，没有"训练/测试算子"划分；
4. **没有系统报告功能正确率**：只在搜索轨迹里提了 76% 的 trail 生成正确代码，没有 CR/FCR 表；
5. **没有跨架构知识迁移实验**：它是"每个平台分别调 prompt"，不是"一套知识迁到新平台"；
6. **没有成本-性能的细粒度分析**：只有总开发时间对比。

---

## 3. 它和你现在的方案对比

| 维度 | QiMeng-GEMM（AAAI-25） | 你的方案（当前） | 谁更充分 |
|---|---|---|---|
| 算子数 | 1（GEMM） | 12–14（L1/L2/L3） | 你更广 |
| 硬件 | 3（CPU/2 GPU） | 2 类（x86/RISC-V 真机；ARM 知识） | 它 GPU 多，你架构知识多 |
| LLM 数 | 4 | 1（qwen-plus，建议加到 2） | 它更强 |
| 条件/对照 | vanilla/CoT/2 库 | C0/C1/P1 + B0/B1/B2/B3（+B4） | 你更系统 |
| 规模 | 84 配置 × 单次 | 12 算子 × 6 条件 × 3 尺寸 × 3 seeds（E3/E4）≈ 1000+ 次 | 你更大 |
| 消融 | hints、auto-tuning、成本 | KG 组件消融（B2/B3、去硬件/条件/参数/模板/冲突） | 你更细 |
| 统计 | 无 seeds/检验 | 3 seeds + 配对检验（计划） | 你更严 |
| held-out | 无 | S1/S2 划分 | 你更严 |
| 外部 AI baseline | 无 | 计划里也弱（GPU 方法不可直接比） | 平手/都弱 |
| 成本分析 | 开发时间 24 天→10 分钟 | 计划 token/时延；可补"人类实现时间" | 它更有故事 |
| 可复现 | 未强调 | 每 run 可复现包 | 你更强 |

**结论**：你的实验设计在"系统性"上已经超过它；它强在 **4 个 LLM × 3 平台 × 7 尺寸的规模**，以及 **auto-tuning 搜索曲线**、**开发成本**这两个叙事。你不需要因为它简单而降低标准，反而可以把它的两个亮点补进来。

---

## 4. 值得借鉴的 3 件事

### 4.1 多 LLM 外部效度（最值得补）

它用 4 个 LLM 证明"方法不是只对 GPT-4o 有效"。你现在只有 qwen-plus；建议：
- 至少加 **1 个第二 LLM**（如 qwen-max / DeepSeek / Claude 中的一个），在 **6 个 S1 算子 × B3 vs B2** 上跑；
- 不需要全量 12 算子，只要能说"趋势在两个模型上一致"。

### 4.2 尺寸曲线（你已经有，但要画出来）

它对 7 个尺寸画性能曲线，说明"矩阵越大收益越大"。你有 1024/16384/1048576，建议：
- 画 `S_C1` 随尺寸的曲线（至少 daxpy/dgemm 两个代表算子）；
- 说明"小尺寸受启动/测量噪声影响，大尺寸更能体现策略价值"。

### 4.3 搜索/迭代曲线（你正好有 E09 的 round0→round1）

它的 Fig. 5 是"25 条搜索轨迹 → 性能分布/正确率"。你有 E09 的 repair 轨迹：
- 画"初始生成 → 第 1 轮修复 → final"的 FCR/性能变化；
- 统计"修复触发率、转正率、负例"——这比它的搜索曲线更贴近你的方法。

### 4.4 开发成本对比（可选，但答辩很加分）

它用"junior 24 天 vs 10 分钟"讲成本。你可以做一个**小规模、可辩护**的估计：
- 1 个 dgemm kernel：人类专家实现/调优需要多少天（引用文献 + 访谈/估算）；
- 你的 agent 从检索到生成到通过测试需要多少分钟；
- 明确写"这是估算，不是严格 user study"。

---

## 5. 不要学它的地方（否则会被抓）

1. **不要只跟 vanilla/CoT/库比**：你的 claim 是"结构化 KG vs 扁平 RAG"，所以 B2 是必须的；否则审稿人会问"你怎么知道不是多给文本的功劳"。
2. **不要单次测量**：LLM 生成有随机性，至少 3 seeds；不然性能差异可能只是采样噪声。
3. **不要只在 S1 上报告**：必须有 S2 留出；否则"跨算子泛化"不成立。
4. **不要只报赢的**：E09 已经出现 B4 只有成本没有正确性增量的负结果，照实写，反而可信。
5. **不要声称"全面超过 AI 优化方法"**：它也没做，你也别做；你的对照是"同 LLM/同预算 + 编译器/专家上下界"。

---

## 6. 给你的"最低充分"实验清单（硕士论文版）

| 实验 | 最小规模 | 说明 |
|---|---|---|
| E1 KG 质量 | 6 算子 × 120 条金标 × 2 人 | 已有计划 |
| E2 策略推荐 | 12 算子 × 2 profile × 4 条件 | 已有计划 |
| E3 x86 生成 | 12 算子 × 6 条件 × 3 尺寸 × **3 seeds** | 主表 |
| E4 RISC-V 生成 | 12 算子 × 4 条件 × 2 尺寸 × **3 seeds**（真机） | 主表 |
| E5 跨架构迁移 | 6 算子 × 3 知识源 | 特色 |
| E6 消融 | B2 vs B3 + 去硬件/条件/参数/模板/冲突 | 证明增量 |
| E7 案例/失败 | 3 案例 + 20 失败样本 | 答辩加分 |
| **E8 多 LLM（新增）** | 2 个 LLM × 6 算子 × B2/B3 | 外部效度 |
| E9 MPSA（可选） | 6 算子 × B3/B4 × 3 seeds | 已跑 24 cells |
| E10 自维护（第二部分） | 5 轮 × 3 seeds × 6 组 | pilot 先行 |

> 这套比 QiMeng-GEMM 的"单算子 × 单次"更严；你不需要再加"所有 AI 方法"的对比，只要把 **B2/B3、seeds、S2 留出、消融** 做扎实。

---

## 7. 答辩时怎么提这篇论文

> "QiMeng-GEMM 用 meta-prompt + auto-tuning 在 GEMM 上做到 OpenBLAS 的 211%，但它的 baseline 主要是 vanilla/CoT prompt 和库，没有和其他 AI 优化方法比，也没有 seeds/held-out；这说明了在这个方向'同 LLM/同预算的对照 + 多尺寸 + 多 LLM'比'堆很多外部 SOTA'更现实。我们的实验在此基础上补了 B2/B3、S2 留出和 3 seeds，所以在'知识结构是否有增量'这个问题上更严格。"

**一句话**：它确实"对比简单"，但它的 claim 也窄；你的 claim 更宽，所以你的实验设计已经更系统。**补一个第二 LLM + 尺寸曲线 + 搜索/修复曲线**，就够了，不用焦虑"没把所有 AI 方法都跑一遍"。
