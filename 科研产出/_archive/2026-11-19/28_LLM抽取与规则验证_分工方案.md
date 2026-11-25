# 28 抽取分工：LLM 提议，程序验证

> 2026-11-16｜回应"不用 LLM 抽取工作量太大、规则太死板"的质疑。
> 先认错：我上一轮说"LLM 只参与注解"说得过头了。**LLM 应该参与抽取**——但不能是"抽完直接入库"，而是"**抽完必须过验证**"。

---

## 0. 先纠正一句话

| 我说的（过头） | 更准确的说法 |
|---|---|
| LLM 只参与语义注解，不参与事实生成 | LLM **可以提议**全部字段（策略、条件、参数、行号）；**事实是否入库由程序验证决定** |

区别不在"用不用 LLM"，在"**谁有最终否决权**"。
LLM 负责**广度**（能想到什么），程序负责**精度**（能不能被源码支持）。

---

## 1. 你的 prompt 缺三样东西（这是工作量大的真正原因）

我看了 `开题/prompts/` 下 12 个 YAML，全部 **没有**出现 `line_start`、`line_end`、`evidence`、`代码引用` 中的任何一个。

具体看 `analysis/analysis_wrapper.yaml`：

```yaml
template: |
  {system_prompt}
  请分析以下{algorithm}算子的源代码，识别{level}层的优化策略：
  {source_content}
  ...
```

两个致命细节：

1. **`source_content` 是多个文件拼起来的，没有行号**。模型即使想给行号也给不了——它看不到行号。
2. **没有要求"引用原文"**。模型只能复述，不能举证。

所以输出必然是"散文条件 + 无出处 + 无行号"。**然后你写了 4,251 行 Python 去补**：

```text
extract.py    1382 行
validate.py    912 行
static.py      625 行
normalize.py   520 行
multiarch.py   513 行
diff.py        143 行
evidence.py    156 行
─────────────────────
合计          4251 行
```

**这 4,251 行里，很大一部分不是在"抽知识"，是在"修 LLM 输出"**（补 ID、补条件、补参数域、补溯源）。这不是你规则写得多，是**抽取契约没设计好**，把成本推到了下游。

一句话：**你不是需要更多规则，你是需要改 prompt 的输出契约。**

---

## 2. 三种分工的真实成本

| | 规则抽取（现状 77%） | LLM 全链（analyze_OB1） | **LLM 提议 + 程序验证（建议）** |
|---|---|---|---|
| 开发量 | 4,251 行 + 数月 | 12 个 prompt + ~2 周 | 改 prompt（2 天）+ 验证器（~500 行，3 天） |
| 运行成本 | 0 | tokens × 文件数 | tokens（略高于全链，因要引用原文） |
| 召回 | 低（只认写死的模式） | 高 | 高（LLM 提议） |
| 精度 | 高 | 低（无校验） | **高（验证器兜底）** |
| 新算子 | 要改代码 | 直接跑 | 直接跑 |
| 可审计 | 强 | 弱 | **强（引用 + 行号 + 拒绝理由）** |

**关键**：规则的价值不在"抽取"，在"**验证**"。验证规则是通用不变量（行号存在、引用匹配、谓词与静态特征一致、参数出现在源码里、模板能编译），**不针对具体算子写死**，所以不"死板"。

---

## 3. 新契约：让 LLM 能举证

### 3.1 输入改造

- 不要传整份拼接源码；传**带行号的片段**：

```text
  46 | void daxpy_k(int n, double da, double *x, double *y, int inc_x, int inc_y) {
  47 |   if (inc_x == 1 && inc_y == 1) {
  48 |     for (i = 0; i + 4 <= n; i += 4) {
  49 |       _mm256_storeu_pd(y+i, _mm256_fmadd_pd(da, _mm256_loadu_pd(x+i), _mm256_loadu_pd(y+i)));
```

- 最好给**配对**：`generic.c` 与 `optimized.c` 的 diff（hunk 级），让模型看到"改了什么"。

### 3.2 输出契约（每条策略）

```json
{
  "optimization_name": "AVX2 四路展开 + FMA",
  "level": "instruction",
  "action": "vectorize",
  "evidence_quote": "_mm256_storeu_pd(y+i, _mm256_fmadd_pd(da, _mm256_loadu_pd(x+i), _mm256_loadu_pd(y+i)))",
  "line_start": 48,
  "line_end": 50,
  "conditions": [
    {"pred": "stride_x", "op": "eq", "value": 1, "hard": true,
     "justification_quote": "if (inc_x == 1 && inc_y == 1)"},
    {"pred": "dtype", "op": "in", "value": ["d"], "hard": true,
     "justification_quote": "double da, double *x, double *y"}
  ],
  "parameters": [
    {"name": "vector_width", "value": 256, "value_in_source": "_mm256_*", "unit": "bits"}
  ],
  "confidence": 0.8,
  "cannot_verify": ["size_regime"]
}
```

三条硬约束：
- **没有 `evidence_quote` 的策略不收**；
- **每条硬条件必须有 `justification_quote`**；
- **禁止写性能数字**（`performance_impact` 改成定性机制描述）。

### 3.3 验证器查什么（5 类，全部可自动化）

| 检查 | 方法 | 拒绝原因 |
|---|---|---|
| 引用真实 | `evidence_quote` 归一化空白后，必须出现在 `[line_start, line_end]` 内 | `quote_not_found` |
| 行号真实 | 文件存在、行号在范围内、区间内含 quote | `line_out_of_range` |
| 谓词一致 | 条件 vs 静态特征（dtype/stride/loop_depth/ISA…）；不一致即拒 | `predicate_conflict` |
| 参数有据 | `value_in_source` 出现在源码行，或能由 ISA 配置推导 | `param_unsourced` |
| ISA 兼容 | `REQUIRES` 的 ISA ⊆ 目标硬件 profile | `isa_mismatch` |
| 模板可编译 | `gcc -fsyntax-only -m<arch>`（有模板时） | `template_compile_fail` |

**这五类检查 ~500 行**，一次写、所有算子复用。拒绝的进 `rejected.jsonl`（带原因），这本身是第二部分的训练素材。

---

## 4. 为什么这样能同时解决"工作量大"和"太死板"

- **广度交给 LLM**：新模式、新算子、新架构，不用改代码；没见过也能提。
- **精度交给程序**：引用、行号、谓词、参数、编译，五道自动检查；LLM 编不了。
- **人只看中间**：自动拒掉一部分，剩下的抽样看；人工从"逐条核 206 条"降到"抽 30 条"。
- **规则不再是"内容规则"，而是"约束规则"**：不写"AXPY 要展开 4 次"，只写"你引用的行必须真的在文件里"。**这类规则对任何算子都成立，所以不死板。**

---

## 5. 什么仍然验证不了（诚实清单）

| 类型 | 例子 | 处理 |
|---|---|---|
| 语义因果 | "这样写减少 cache miss" | 标 `claim_status=unverified`，只进注解层，不参与过滤 |
| 性能收益 | "提速 1.8×" | 必须来自证据节点（日志/实测），否则删掉数字只留机制 |
| 抽象原则 | "数据级并行适用于规则访问" | 可保留为 `OptimizationPrinciple`，但不作为选择依据 |

---

## 6. 本周就能做的 A/B（半天跑完，能写进论文）

对 `axpy` 一个算子：

- **A 组**：用现有 `algorithm_optimizer.yaml` 跑（无行号、无引用）；
- **B 组**：用新契约跑（带行号片段 + 引用 + 谓词）；
- 两组都过验证器，统计：候选数、通过率、拒绝原因分布；人工抽 20 条算 precision。

预期：A 组通过率极低（拿不出引用），B 组通过率 50–70%。**这个对比就是"可验证抽取"的实验证据**，也直接回答审稿人"LLM 抽取不可信怎么办"。

---

## 7. 论文里怎么写

第 3 章的小节可以叫：

> **可验证的 LLM 提议抽取**：LLM 负责提出策略候选与条件假设，并强制引用源码行；程序验证器检查引用真实性、行号范围、谓词一致性与参数来源，只有通过验证的候选进入知识图谱。未通过者进入拒绝集，作为后续抽取的反例。

贡献句：

> 与端到端 LLM 建图（GraphRAG/analyze_OB1）不同，本文不信任 LLM 的"事实"，只信任它"提出的假设"；与纯规则抽取不同，本文不让规则承担发现，只让规则承担验证。

**这句话同时回应了你的两个担心：工作量（LLM 提议）与死板（规则只做不变量）。**

---

## 附录：一个能跑的例子

`skills/kg-build/demo/verify_demo.py` 用真实文件 `daxpy_microk_haswell-2.c`（AVX2 内联汇编微内核）演示了 5 张卡、6 条规则的判定：

```text
C1 AVX2 四路展开 + FMA        PASS
C2 循环控制优化（引文漂移）     FAIL  R1: 引文不在 48–50 行内
C3 对齐加载（对齐判断错误）     FAIL  R4: vmovups 是未对齐，却声称 aligned=true
C4 八路展开（单位算错）         FAIL  R6: addq $16 对应展开 4，卡片却写 8
C5 单精度向量化（精度错误）     FAIL  R3: 源码是 double(pd)，卡片却没写 dtype=d
```

运行：

```bash
python 科研产出/skills/kg-build/demo/verify_demo.py
```

这 6 条规则里没有一个字提到 AXPY/GEMM——它们只检查"你声称的和源码是否一致"，所以换任何算子都能用。
