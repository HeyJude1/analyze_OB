# E09_MPSA 摘要

> 人写区：结论、关键数字、图、下一步。`AUTO-SUMMARY` 区间由 `run_smoke.py` 根据 `results.jsonl` 刷新，不要手改。

## 结论（2026-11-12 网格冒烟完成）

**结论：24-cell 网格（daxpy/ddot × B3/B4 × seed 1/2/3 × n=1024/16384）已跑完；初始 FCR 66.7%（16/24）→ 最终 95.8%（23/24）。QEMU 只验正确性，性能待 K1 真机。**

> 本次网格的**可复现包**（输入算子/硬件/检索 trace/每轮迭代/最终算子）：[`artifacts/real_grid_seed_sizes/repro/README.md`](artifacts/real_grid_seed_sizes/repro/README.md)。每个 cell 的 `final/` 可 `bash build.sh && bash run.sh` 自行复核。

- **最终网格 run**：`real_grid_seed_sizes`（2026-11-12 21:12–21:24，dashscope `qwen-plus-2025-04-28`，repair=1 轮）
  - **总体**：24 cells，初始 16/24（66.7%）→ 修复后 23/24（95.8%）；gen tokens 355,819 + repair tokens 22,860；用时 796s。
  - **B3**：12 cells，初始 8/12（66.7%）→ 最终 **12/12（100%）**；触发 repair 4 cells，4 个全部转正；gen 106,976（8,915/cell）+ repair 11,706。
  - **B4**：12 cells，初始 8/12（66.7%）→ 最终 **11/12（91.7%）**；触发 repair 4 cells，3 个转正；gen 248,843（20,737/cell）+ repair 11,154；**B4/B3 gen token ≈2.33×**。
  - **唯一失败**：`ddot × B4 × seed1 × n=1024`：初始 wrong_answer → 1 轮 repair 后变成 compile_failed（修复把代码改坏）；保留在 `results.jsonl` 与报告里。
  - **修复回路**：8 cells 触发、7 个转正（compile_failed 与 wrong_answer 都修过）；repair 对 FCR 的贡献是 **+29.2 个百分点**（66.7→95.8），但每次 repair 约 2.5–3.5k tokens。
  - **B4 的正确性增量在当前样本不显著**：初始 FCR 与 B3 相同（8/12），最终还低 1 个 cell；同时 token 成本 2.33×。这是需要如实写进论文的负向/中性信号，不能硬吹 B4。

| 条件 | cells | 初始正确 | 最终正确 | FCR_initial | FCR_final | repair 触发 | 转正 | gen tokens/cell | repair tokens |
|---|---|---|---|---|---|---|---|---|---|
| B3 | 12 | 8 | 12 | 66.7% | **100%** | 4 | 4 | 8,915 | 11,706 |
| B4 | 12 | 8 | 11 | 66.7% | **91.7%** | 4 | 3 | 20,737 | 11,154 |
| 合计 | 24 | 16 | 23 | 66.7% | **95.8%** | 8 | 7 | 14,826 | 22,860 |
- **此前 4-cell run**：`real_repair1_all` 4/4（100%），用于验证链路；网格数据才是当前主结果。
- **口径**：QEMU 的 `best_ms` 仅存档，禁止进性能表；KCS 是启发式 v0（修复后标量代码 KCS=0），人工抽检后再下结论。

**下一步（按顺序）**：
1. 扩到 dgemm/dgemv（需 pack + driver）后重跑网格；
2. 板子到位后 K1 真机测 `S_C0/S_C1/EGC`（QEMU 数据不进性能表）；
3. 对唯一失败 cell 做 E7 案例分析（wrong_answer→repair 变 compile_failed 的边界）；
4. 人工抽检 KCS、校准启发式；结果稳定后再写第 5 章。

## 修复回路（task-4，`--repair-rounds`）

- **实现/自测**：`run_smoke.py --repair-rounds N`：`compile_failed` 用 `harness/build.stderr.log`、
  `wrong_answer` 用 `harness/qemu.stdout.log`（含 maxdiff/reldiff）调 `kg_agent mpsa-repair`，修复后重新编译/QEMU；
  `generated_initial.c` 保留初版、`generated.c` 写最终版。已用**真实 mpsa-repair CLI（mock provider）**闭环验证两条分支：
  `selftest_repair_compile`（initial `compile_failed` → final `ok`）、`selftest_repair_wrong`（initial `wrong_answer` → final `ok`）。
- **真实 run `real_repair1_all`（2026-11-12 20:57，4 cells，单 seed）**：初始 FCR=**100%（4/4）**，
  最终 FCR=**100%（4/4）**，**修复触发 0 次**、repair tokens=0。→ 本次真实 API 的初始生成全部直接正确，
  修复回路**未被触发**；真实修复效果需在后续出现失败时观察。
- **历史对照（非严格配对，仅记录）**：`real_smoke1–3` 共 10 cells，初始仅 5 ok（50%，含 4 个幻觉 intrinsic 编译失败 + 1 个归约错）；
  mpsa-engineer 修复 prompt 幻觉名后 `real_repair1_all` 4/4 成功。**单 seed 小样本，不作统计结论**；
  说明写进论文时须注明 prompt/CLI 版本（`mpsa_impl` 已记录）。
- 口径：`status_initial` 与 `status` 分开记；`repair_rounds>0` 才算真实触发修复。

## 证据口径

- `ok` 只表示"编译通过 + 数值正确"；QEMU 的 `best_ms` 不进性能表。
- KCS 是启发式 v0（实现痕迹静态检查），需人工抽检校准。
- **mock 生成的是标量参考实现，代码里没有向量指令 → KCS≈0 属预期**，不是方法失败；KCS 的对比要等真实 API 数据（首轮真实 KCS B3/B4 均 0.25，样本太小，不作结论）。

<!-- AUTO-SUMMARY:BEGIN -->
共 **46** 条运行单元；最近 run：`real_grid_seed_sizes`（2026-11-12T21:24:52+08:00）。

| provider / mpsa / repair | 条件 | cells | FCR_initial | FCR_final | 编译通过率(final) | 修复触发 | 修复转正 | repair_rounds | 平均 gen tokens/cell | 平均 repair tokens/cell | KCS 均值 (n) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dashscope / cli / cli | b3 | 4 | 0.0% | 100.0% | 100.0% | 4 | 4 | 4 | 8938.8 | 2926.5 | 0.00 (4) |
| dashscope / cli / cli | b4 | 4 | 0.0% | 75.0% | 75.0% | 4 | 3 | 4 | 20954.5 | 2788.5 | 0.06 (4) |
| dashscope / cli / — | b3 | 15 | 86.7% | 86.7% | 93.3% | 0 | 0 | 0 | 8926.9 | 0.0 | 0.25 (15) |
| dashscope / cli / — | b4 | 15 | 80.0% | 80.0% | 80.0% | 0 | 0 | 0 | 20621.7 | 0.0 | 0.25 (15) |
| mock / cli / cli | b3 | 2 | 0.0% | 100.0% | 100.0% | 2 | 2 | 2 | 0.0 | 0.0 | 0.00 (2) |
| mock / cli / — | b3 | 4 | 100.0% | 100.0% | 100.0% | 0 | 0 | 0 | 0.0 | 0.0 | 0.00 (4) |
| mock / cli / — | b4 | 2 | 100.0% | 100.0% | 100.0% | 0 | 0 | 0 | 0.0 | 0.0 | 0.00 (2) |

> FCR_initial = 修复前 correct 比例；FCR_final = 最终 correct 比例；修复转正 = 初始失败且最终正确。
> `ok` 仅表示编译+正确，**不含性能**。mock/stub 行只验证链路，不代表方法效果。
> 每个 run 的详细报告在 `runs/<YYYY-MM-DD_HHMM>__<run_id>__<STATUS>.md`（见 `runs/INDEX.md`）；逐 cell 流水在 `results.jsonl`。
<!-- AUTO-SUMMARY:END -->

## 下一步

1. **扩算子**：补 dgemm/dgemv（pack + driver）后重跑网格；
2. **上真机**：板子到位后 K1 按 `08b` 测 `S_C0/S_C1/EGC`（QEMU 不进性能表）；
3. **E7 案例**：分析唯一失败 `ddot×B4×seed1×1024`（wrong_answer→repair 后 compile_failed）；
4. **KCS 校准**：人工抽检启发式，确认“知识合规率”能不能作为指标；
5. 结果稳定后再写第 5 章，不要拿 24-cell 冒烟当正式结论。
