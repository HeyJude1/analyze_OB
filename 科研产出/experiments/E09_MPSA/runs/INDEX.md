# E09 runs 索引

> 命名规则：`YYYY-MM-DD_HHMM__<run_id>__<状态>.md`；每个 run **一个 md 文件**。
> 状态：`OK` 全部编译+正确；`MIXED` 部分失败；`FAIL` 全部失败；`MOCK` 离线链路；`SELFTEST` 框架自测。
> 原始产物（generated.c / harness / mpsa_result.json）在 `../artifacts/<run_id>/`。

> **最新正式 run**：`real_grid_seed_sizes`（MIXED，cells=24，ok=23，2026-11-12T21:12:26）

## 正式 run（按时间从新到旧）

| 时间 | run_id | provider | cells | ok | 状态 | 报告文件 | 产物 | 复现包 |
|---|---|---|---|---|---|---|---|---|
| 2026-11-12T21:12:26 | `real_grid_seed_sizes` | dashscope | 24 | 23 | **MIXED** | [2026-11-12_2112__real_grid_seed_sizes__MIXED.md](2026-11-12_2112__real_grid_seed_sizes__MIXED.md) | `artifacts/real_grid_seed_sizes` | [repro](../artifacts/real_grid_seed_sizes/repro/README.md) |
| 2026-11-12T20:57:55 | `real_repair1_all` | dashscope | 4 | 4 | **OK** | [2026-11-12_2057__real_repair1_all__OK.md](2026-11-12_2057__real_repair1_all__OK.md) | `artifacts/real_repair1_all` | [repro](../artifacts/real_repair1_all/repro/README.md) |
| 2026-11-12T20:48:16 | `real_smoke3_all` | dashscope | 4 | 1 | **MIXED** | [2026-11-12_2048__real_smoke3_all__MIXED.md](2026-11-12_2048__real_smoke3_all__MIXED.md) | `artifacts/real_smoke3_all` | [repro](../artifacts/real_smoke3_all/repro/README.md) |
| 2026-11-12T20:47:01 | `real_smoke2_ddot` | dashscope | 2 | 2 | **OK** | [2026-11-12_2047__real_smoke2_ddot__OK.md](2026-11-12_2047__real_smoke2_ddot__OK.md) | `artifacts/real_smoke2_ddot` | [repro](../artifacts/real_smoke2_ddot/repro/README.md) |
| 2026-11-12T20:45:28 | `real_smoke1_ddot` | dashscope | 2 | 0 | **FAIL** | [2026-11-12_2045__real_smoke1_ddot__FAIL.md](2026-11-12_2045__real_smoke1_ddot__FAIL.md) | `artifacts/real_smoke1_ddot` | [repro](../artifacts/real_smoke1_ddot/repro/README.md) |
| 2026-11-12T20:44:20 | `real_smoke1_daxpy` | dashscope | 2 | 2 | **OK** | [2026-11-12_2044__real_smoke1_daxpy__OK.md](2026-11-12_2044__real_smoke1_daxpy__OK.md) | `artifacts/real_smoke1_daxpy` | [repro](../artifacts/real_smoke1_daxpy/repro/README.md) |

## Mock 链路（不花 token，不代表方法效果）

| 时间 | run_id | cells | ok | 状态 | 报告文件 |
|---|---|---|---|---|---|
| 2026-11-12T21:38:29 | `test_auto_repro` | 1 | 1 | MOCK | [2026-11-12_2138__test_auto_repro__MOCK.md](_mock/2026-11-12_2138__test_auto_repro__MOCK.md) |
| 2026-11-12T21:27:28 | `test_auto_org` | 1 | 1 | MOCK | [2026-11-12_2127__test_auto_org__MOCK.md](_mock/2026-11-12_2127__test_auto_org__MOCK.md) |
| 2026-11-12T20:44:16 | `smoke_seed1_20261112_204416` | 4 | 4 | MOCK | [2026-11-12_2044__smoke_seed1_20261112_204416__MOCK.md](_mock/2026-11-12_2044__smoke_seed1_20261112_204416__MOCK.md) |
| 2026-11-12T20:44:13 | `smoke_seed1_20261112_204413` | 1 | 1 | MOCK | [2026-11-12_2044__smoke_seed1_20261112_204413__MOCK.md](_mock/2026-11-12_2044__smoke_seed1_20261112_204413__MOCK.md) |

## 框架自测（repair 回路/无修复回归）

| 时间 | run_id | cells | ok | 状态 | 报告文件 |
|---|---|---|---|---|---|
| 2026-11-12T20:57:32 | `selftest_norepair` | 1 | 1 | SELFTEST | [2026-11-12_2057__selftest_norepair__SELFTEST.md](_selftest/2026-11-12_2057__selftest_norepair__SELFTEST.md) |
| 2026-11-12T20:57:07 | `selftest_repair_wrong` | 1 | 1 | SELFTEST | [2026-11-12_2057__selftest_repair_wrong__SELFTEST.md](_selftest/2026-11-12_2057__selftest_repair_wrong__SELFTEST.md) |
| 2026-11-12T20:57:02 | `selftest_repair_compile` | 1 | 1 | SELFTEST | [2026-11-12_2057__selftest_repair_compile__SELFTEST.md](_selftest/2026-11-12_2057__selftest_repair_compile__SELFTEST.md) |

---
*最后整理：2026-11-12T21:39:16*
