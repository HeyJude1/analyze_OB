# EXPERIMENTS：实验总登记表

> 现行基准与实验优先级见 [实验与验收](../03_实验与验收.md)。旧编号和 E09 保留为过程记录。

| 现行项目 | 状态 | 工件 |
| --- | --- | --- |
| 初始算子基准 | 五个任务含独立卷积留出 | [算子清单](../算子输入/manifest.json) |
| 隔离建图与严格查询 | 四个建图算子，快照和边界校验已落地 | [图谱快照](../kg_agent/data_thesis/manifest.json) |
| 四组 Prompt 配对 | 五个任务、两个画像、四组条件、三次重复的输入可直接准备 | [实验工具](thesis/README.md) |
| 九算子四组功能对照 | 单次重复全矩阵 72/72，编译 35→44，正确 18→19 | [实验设置与结论](thesis/functional_2026-11-19/README.md) · [逐单元汇总](thesis/functional_2026-11-19/结果汇总.md) |
| 双平台性能主实验 | 待测；正式运行前冻结实际画像与工具链 | [现行计时口径](../03_实验与验收.md) |

> 每个实验一行；新增实验先在此登记，再在 `EXX_name/README.md` 写协议。
> 状态：⬜ 未开始 ｜ 🟡 进行中 ｜ ✅ 已完成 ｜ ⛔ 阻塞 ｜ ❌ 放弃
> 最近更新：2026-11-12

| ID | 名称 | 状态 | 开始日期 | 负责人 | 目录 | 结果文件 | 备注 |
|---|---|---|---|---|---|---|---|
| E09_MPSA | MPSA 消融：B3（KG 策略直注入）vs B4（多视角策略分析+综合后生成），L1 算子 daxpy/ddot，RVV 256-bit/QEMU 正确性 | ✅ 网格冒烟完成（24 cells：初始 66.7% → 最终 95.8%；B3 100% / B4 91.7%；待 K1 真机性能） | 2026-11-12 | exp-harness-engineer（框架）/ Lead（真实 API） | `E09_MPSA/` | `E09_MPSA/results.jsonl`、`E09_MPSA/runs/INDEX.md`、`E09_MPSA/artifacts/<run_id>/repro/` | 主结果 run `real_grid_seed_sizes`（24 cells，daxpy/ddot × B3/B4 × seed1/2/3 × 1024/16384）：初始 16/24 → 最终 23/24；repair 触发 8、转正 7；gen tokens 355,819 + repair 22,860；B4 gen tokens ≈2.33× B3；唯一失败 ddot×B4×seed1×1024。**每个 run 均生成可复现包**（输入/硬件/检索 trace/每轮迭代/最终代码+bin）；QEMU 只作正确性前置，不作性能结论；详见 `E09_MPSA/summary.md` |

## 规划中（未建目录，不占正式编号）

| 候选 | 说明 | 依赖 |
|---|---|---|
| E01–E08 | [旧实验矩阵](../_archive/2026-11-19/04_实验设计与评估协议.md) §7 | 历史规划 |
| E10 | [知识自维护旧设计](../_archive/2026-11-19/02b_第二部分_知识自维护.md) §3 | 历史规划 |
| K1 真机性能（R2–R5） | [旧 K1 协议](../_archive/2026-11-19/08b_RISC-V真机实验协议.md) §6 | 历史规划 |

## 变更记录

- 2026-11-12：建立本表；登记 E09_MPSA；mock 冒烟全矩阵（daxpy/ddot × b3/b4 × seed1）5 个运行单元全部"编译+正确"，框架与 QEMU harness 验证通过。
- 2026-11-12（晚）：Lead 用真实 DashScope API 首轮试跑 E09（daxpy/ddot × b3/b4），出现 4 个 compile_failed（LLM 幻觉 RVV intrinsic）与 1 个 wrong_answer（ddot 归约只取 lane0）——是**实验可观察信号**，非框架故障；框架已能区分并记录失败类型。`run_smoke.py` 已增加 `mpsa_impl` 溯源字段。
- 2026-11-12（夜）：修 prompt 归约示例 + 加 `mpsa-repair`；最终 run `real_repair1_all` 真实 API **4/4 cells 编译+正确（FCR=100%）**；B3 8.8k vs B4 20.8k tokens/cell；repair 触发 0。E09 状态改为"冒烟完成"，下一步扩 seeds/规模与 K1 真机性能。
- 2026-11-12（更晚）：E09 网格 run `real_grid_seed_sizes`（24 cells：daxpy/ddot × B3/B4 × seed1/2/3 × 1024/16384）：初始 66.7% → 最终 **95.8%**；repair 触发 8、转正 7；B3 最终 100%，B4 最终 91.7%；gen tokens 355,819 + repair 22,860；B4/B3 token ≈2.33×。已用 `organize_runs.py` 整理：`runs/` 一 run 一个 md，产物移到 `artifacts/<run_id>/`，新增 `runs/INDEX.md`。
- 2026-11-12（夜，task-4）：`run_smoke.py --repair-rounds` 集成完成（compile_failed→build.stderr、wrong_answer→qemu.stdout→mpsa-repair→重编译/QEMU）；用真实 mpsa-repair CLI（mock）验证两条分支均"初始失败→修复→ok"；`real_repair1_all` 初始全对故真实 repair 触发 0。selftest_* run 仅框架自测，不进正式统计。
