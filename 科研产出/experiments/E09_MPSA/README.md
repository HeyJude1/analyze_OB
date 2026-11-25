# E09_MPSA：多视角策略分析（MPSA）消融 —— B3 vs B4

> 状态：🟡 冒烟链路（mock）已通；真实 API 待 Lead 跑；性能留 K1 真机
> 负责人：exp-harness-engineer（框架/冒烟）、Lead（真实 API）
> 开始：2026-11-12
> 历史依据：[Lingxi 精读](../../refs/lit_notes/17_文献精读_Lingxi.md)、[旧评估协议](../../_archive/2026-11-19/04_实验设计与评估协议.md)、[旧 RISC-V 协议](../../_archive/2026-11-19/08b_RISC-V真机实验协议.md)。

---

## 1. 背景与科学问题

Lingxi（ISSTA 2026）的 "parallel knowledge-guided analysis scaling" 表明：把检索到的多条程序性知识
**并行做适用性分析、再综合**，比把所有知识直接塞进一个 prompt 更容易被模型真正使用（去掉 scaling 有知识时
−3.10%，无知识时只 +0.31%）。本项目把"知识"换成算子优化 KG 的 **StrategyPack（策略卡）**，检验：

> **RQ（E09）**：在相同 LLM、相同 token 预算口径下，B4（多视角分析 + 综合后再生成）相对 B3（策略包直注入生成）
> 是否提高生成代码的**知识合规率 KCS** 与**功能正确率 FCR**？成本增加多少？

## 2. 条件定义

| 条件 | 定义 | 知识注入方式 | 调用次数（k=3 时） |
|---|---|---|---|
| **B3** | 单次生成（对照） | 检索到的 top-k StrategyPack 拼接成一个生成 prompt，直接生成 C 函数 | 1（generate） |
| **B4** | MPSA（处理组） | 对每个 StrategyPack 单独做"适用性分析"（applies/evidence/risks/parameters/conflicts）→ 按 4 标准（supportive/complementarity/accuracy/completeness）综合成分析报告 → 报告+策略包注入生成 prompt → 生成 C 函数 | 3 分析 + 1 综合 + 1 生成 = 5 |

- 同 LLM、同 profile（`k1_rvv256`）、同任务卡、同 top-k（默认 6）、同 seed；
- **成本必须计入**：B4 的分析/综合 token 与耗时都记入 `usage`（不追求"B4 更便宜"，只要求如实可比）；
- 实现：`kg_agent` 的 `mpsa` 子命令（B3/B4 两种 mode），本目录脚本只负责调度与评测。

**修复回路（可选，本次 E09 用 1 轮）**：B3/B4 的**初始生成**若 `compile_failed`（错误输入 = `build.stderr`）
或 `wrong_answer`（错误输入 = `qemu.stdout` 的 `maxdiff/reldiff`），调用 `kg_agent mpsa-repair` 修一次，
修复后重新编译/QEMU。报告必须**同时给 FCR_initial 与 FCR_final**，并把 repair token 计入成本；
每个 cell 保留 `generated_initial.c`（初版）与 `generated.c`（最终版）。

## 3. 假设与预注册判据

| 编号 | 假设 | 冒烟阶段（本目录先验） | 正式阶段（E09 完整版） |
|---|---|---|---|
| **H1** | B4 的知识合规率 KCS 高于 B3 | 链路能算出 KCS（非 null），方向待真实 API 数据 | 配对比较（同 op/seed），B4 − B3 的 KCS 均值差 > 0 |
| **H2** | B4 不降低功能正确率（FCR ≥ B3） | mock 下两者都 = 100%（仅链路） | FCR(B4) ≥ FCR(B3)，且 ≤ −5pp 视为失败信号 |
| **H3** | 成本代价可量化：B4 token ≈ B3 的 2–3 倍 | 记录 `usage.total_tokens` 与 calls | 报 token/时延/每任务成本，不设"更便宜"目标 |

判定纪律：**先看 H2（正确性）再看 H1（机制）**；若 FCR 掉太多，KCS 的提升不作为正结果。

## 4. 指标定义

| 指标 | 定义 | 来源字段 |
|---|---|---|
| **FCR** | 成功完成"编译通过 + 数值正确"的运行单元占比：`FCR = #{correct}/#cells` | `results.jsonl` 的 `compile_ok & correct` |
| **KCS**（启发式 v0） | `KCS = 生成代码中出现 strong 实现痕迹的已检索策略数 / 其中可静态检查的策略数`；weak 痕迹（如 `if (n<=0)` 这类）不计分子；`reorder/split/fuse/scalarize/reduce_register_pressure` 无稳定痕迹 → 不计入分母；没有可检查策略时记 `null` | `results.jsonl` 的 `kcs` / `kcs_detail`；实现见 `../harness/kcs.py` |
| **成本** | `usage.total_tokens`、`usage.calls`、`mpsa_elapsed_s`、harness 编译耗时 | `results.jsonl` |
| **FCR_initial / FCR_final** | 修复回路开启时的修复前/后功能正确率（必须分开报） | `status_initial` / `status` |
| **repair 成本** | `repair_tokens`、`repair_usage`、`repair_rounds`、逐轮明细 `repairs[]` | `results.jsonl` |
| **正确性辅助** | `maxdiff`、`reldiff`、`qemu_exit`（driver 退出码 0=正确） | `results.jsonl` |
| **性能（本实验不报）** | S_comp / EGC / fast_1 | 仅 K1 真机，见 `08b` §2–§3 |

> **红线**：QEMU 上的 `best=` 毫秒数**不是性能**，只说明程序能跑完。E09 的 RISC-V 侧结论目前只到"编译 + 正确性"。
> KCS 是**启发式**：只证明"关键词/指令/编译指示出现过"，不证明策略被正确使用；论文里必须标注 heuristic 并给人工抽检校准。

## 5. 任务与数据（L1 向量算子）

| op | 语义 | 接口 | 容差（driver 判定） | 来源 |
|---|---|---|---|---|
| `daxpy` | `y := da*x + y` | `int daxpy_k(BLASLONG n, ..., double da, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, ...)` | `reldiff < 1e-12`（driver exit 0）；harness 复核 `abs maxdiff ≤ 1e-10` | OpOptBench `tasks/daxpy`（generic.c 原样复制；driver 增加 reldiff 以兼容 RVV FMA 末位舍入） |
| `ddot` | `dot := Σ x[i]*y[i]` | `double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)` | `reldiff < 1e-9`（参考值用 long double）；harness 复核 `abs maxdiff ≤ 1e-5` | OpOptBench `tasks/ddot`（原任务无 driver，E09 新增） |

- 规模：默认取 task.yaml 的 `sizes` 中位数 = **16384**；`run_smoke.py --sizes 1024,16384` 可扩；
- 硬件 profile：`k1_rvv256`（建模目标 K1：VLEN=256、fp64 lanes=4、L1d 32KB/L2 512KB）；
- 评测环境：QEMU 用户态（正确性前置），编译口径固定
  `-O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -static`（与 `08b` C1 一致）。

## 6. 执行步骤

### 6.1 mock 冒烟（不联网，验证"生成→编译→QEMU→记录"链路；**不代表方法效果**）

```powershell
cd C:\Users\MSI\Desktop\开题
$env:PYTHONIOENCODING = "utf-8"
python experiments\E09_MPSA\run_smoke.py --provider mock --ops daxpy --modes b3 --seed 1
# 全矩阵（2 算子 × 2 条件 × 1 seed）：
python experiments\E09_MPSA\run_smoke.py --provider mock --ops daxpy,ddot --modes b3,b4 --seed 1
```

### 6.2 真实 API（Lead 执行；需 DashScope key，**不要写进仓库**）

```powershell
$env:DASHSCOPE_API_KEY = "sk-..."          # 只在本会话设置
python experiments\E09_MPSA\run_smoke.py --provider dashscope --ops daxpy,ddot --modes b3,b4 --seed 1
# 成本可控：先单算子单条件 1 次，确认 token 与链路正常后再跑全矩阵
python experiments\E09_MPSA\run_smoke.py --provider dashscope --ops daxpy --modes b3 --seed 1
```

### 6.3 扩展（正式实验）

```powershell
python experiments\E09_MPSA\run_smoke.py --provider dashscope --ops daxpy,ddot --modes b3,b4 --seed 1 --sizes 1024,16384 --top-k 6
# 多 seed：--seed 1,2,3（会展开成多行，run_id 带 seeds 标签）
```

### 6.4 性能（另起，不在本脚本）

K1 真机上按 `08b` §2.5 计时协议跑同一对 B3/B4 生成代码（同 warmup/中位数/锁频），
产物登记到 `../EXPERIMENTS.md` 的 K1 真机条目（不混入 QEMU 记录）。

### 6.5 修复回路（`--repair-rounds`，可选；本次实验 1 轮）

mock 自测（用真实 `mpsa-repair` CLI 的 mock provider，构造初始失败，验证两条分支）：

```powershell
python experiments\E09_MPSA\run_smoke.py --provider mock --ops daxpy --modes b3 --seed 71 --repair-rounds 1 `
    --mock-code experiments\E09_MPSA\selftest\broken_compile_daxpy.c --run-id selftest_repair_compile --no-summary
python experiments\E09_MPSA\run_smoke.py --provider mock --ops ddot --modes b3 --seed 72 --repair-rounds 1 `
    --mock-code experiments\E09_MPSA\selftest\wrong_answer_ddot.c --run-id selftest_repair_wrong --no-summary
```

真实 API（需 `DASHSCOPE_API_KEY`）：

```powershell
python experiments\E09_MPSA\run_smoke.py --provider dashscope --ops daxpy,ddot --modes b3,b4 --seed 1 `
    --repair-rounds 1 --run-id real_repair1_all --tag repair
```

输出：每 cell 的 `generated_initial.c` / `generated.c` / `repair_r1/`（`repair_result.json`）/ `harness_r1/`；
`results.jsonl` 含 `status_initial`、`repair_rounds`、`repair_tokens`、`repairs[]`；run md 按
**initial → 修复轮次 → final** 展开；summary 自动区有 FCR_initial/FCR_final/修复转正/repair_rounds 列。

## 7. 输出文件

| 文件 | 内容 |
|---|---|
| `results.jsonl` | 逐运行单元结果（append-only，字段见 `../README.md` §1.3） |
| `runs/<run_id>.md` | 本次 run 报告：配置/命令/LLM 输出摘要/编译/正确性/耗时/token/KCS |
| `runs/<run_id>/<op>_<cond>/` | `mpsa_result.json`、`generated_initial.c`（初版）、`generated.c`（最终版） |
| `runs/<run_id>/<op>_<cond>/harness/` | 初始实现：`build.sh`、`run.sh`、`build.*.log`、`qemu.*.log`、`result.json`（可复查） |
| `runs/<run_id>/<op>_<cond>/repair_r<k>/` | 第 k 轮修复：`repair_result.json`、`generated.c`、`repair_stdout/stderr.log` |
| `runs/<run_id>/<op>_<cond>/harness_r<k>/` | 修复后重编译/QEMU：日志与 `result.json` |
| `summary.md` | 结论与自动汇总（`AUTO-SUMMARY` 区间） |

## 8. 失败分类（记录到 `results.jsonl.status`）

| status | 含义 | 排查 |
|---|---|---|
| `mpsa_failed` | mpsa 子命令失败：无 key / API 报错 / 检索失败 / 参数错误 | 看 `runs/<run_id>/<op>_<cond>/mpsa_stderr.log`；退出码 3=缺 key、4=LLM/JSON、5=检索、2=其它 |
| `no_code` | `mpsa_result.json` 里 `code` 为空 | 看该目录 `mpsa_result.json` 的 `warnings` |
| `compile_failed` | 交叉编译失败 | `harness/build.stderr.log` |
| `qemu_timeout` | QEMU 运行超时 | `harness/qemu.stdout.log`、`run.sh` |
| `wrong_answer` | 编译过但数值不对（driver exit≠0 或 maxdiff 超容差） | `harness/qemu.stdout.log` 的 maxdiff/reldiff |
| `harness_error` | harness 自身异常（WSL 不可用、脚本写入失败等） | 看 `results.jsonl.error` 与 run 报告 |
| `ok` | 编译 + 正确（**不含性能含义**） | — |

> 修复回路开启时：`status_initial` = 修复前状态，`status` = 最终状态；`repair_rounds>0` 表示该 cell 进入过修复回路，
> `repair_backend` 记录用的是真 CLI（`cli`）还是本地兜底（`stub`，仅自测链路）。

## 9. 已知限制

1. mock 的 `code` 是按 generic.c 签名生成的标量参考实现（确定性），只验证链路，**不能**用来比较 B3/B4；
   也因此 mock 下 KCS 常为 0（标量代码没有向量指令痕迹），这是预期行为，不代表方法失败；
2. QEMU 计时不可作性能结论；RISC-V 侧只报"编译率/正确率"；
3. KCS 为启发式 v0，`uncheckable` 策略不计分母，可能高估/低估；正式分析需人工抽检 ≥20 条校准；
4. token 成本以 provider 返回的 usage 为准；mock/stub 为 0；
5. 本实验的 `tasks/` 只含 daxpy/ddot 两个算子（冒烟/试点），正式 E09 可扩展更多 L1/L2 算子（复制 OpOptBench 任务卡即可）；
6. 修复回路本次只做 **1 轮**；`repair` 的输入是真实编译/QEMU 错误（不含人工提示），mock/stub 修复不代表模型修复能力；
7. `selftest_*` 开头的 run 是框架自测（mock + 构造失败），**不进正式统计**。

## 10. 参考

- `../../17_文献精读_Lingxi.md` §3 复用 C / 复用 E（KCS 指标）
- `../../04_实验设计与评估协议.md` §5.3（条件定义 B0–B13）、§6（指标/容差）
- `../../08b_RISC-V真机实验协议.md` §2.2/§2.5/§5.1（编译口径、计时协议、QEMU 红线）
