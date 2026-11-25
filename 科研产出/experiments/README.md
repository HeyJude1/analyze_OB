# experiments：实验目录规范与执行框架

> 主论文采用 [2026-11-19 实验协议](../03_实验与验收.md) 与 [同预算比较准备工具](thesis/README.md)。以下 v1.0 规范用于重放历史实验。

> v1.0｜2026-11-12｜维护：exp-harness-engineer
> 目的：让"每个实验怎么建、每次 run 怎么跑、结果放哪、怎么追踪"固定下来，后续换人也能接着跑。
> 口径依据：`../04_实验设计与评估协议.md`（指标/统计）、`../08b_RISC-V真机实验协议.md`（RISC-V 计时与红线）。

---

## 0. 一句话

**每个实验一个文件夹**（`EXX_name/`）；**每次运行一个 run id**，整理后对应 **`runs/<YYYY-MM-DD_HHMM>__<run_id>__<STATUS>.md` 一个报告文件**，原始产物在 `artifacts/<run_id>/`；`runs/INDEX.md` 是总索引（最新/OK/MIXED/MOCK/SELFTEST 一眼可见）。
**每个（算子×条件×seed）单元一行**追加到 `results.jsonl`；结论写进 `summary.md`；总表登记在 `EXPERIMENTS.md`。

```
experiments/
├── README.md                # 本文件：规范与框架说明
├── EXPERIMENTS.md           # 实验总登记表（id/名称/状态/日期/负责人/结果文件）
├── new_experiment.py        # 一键建实验目录（带模板）
├── harness/                 # 与具体实验无关的执行基础设施
│   ├── qemu.py              #   WSL 交叉编译 + QEMU 运行 + 正确性解析（compile_run）
│   ├── kcs.py               #   知识合规率启发式 v0（静态检查生成代码是否用上策略）
│   ├── mock_mpsa.py         #   mpsa CLI 未就绪时的本地 mock 兜底（仅 mock provider）
│   └── repro_package.py     #   可复现包：输入/硬件/检索链/每轮迭代/最终算子
├── organize_runs.py         # 整理器：runs/ 一 run 一 md + artifacts/ + INDEX.md
└── E09_MPSA/                # 一个实验（MPSA: B3 vs B4）
    ├── README.md            #   实验协议：假设/条件/指标/步骤/限制
    ├── summary.md           #   结论汇总（AUTO-SUMMARY 区由脚本自动更新）
    ├── results.jsonl        #   结果流水（append-only，一行一个 cell）
    ├── runs/                #   **每次运行一个 .md 报告**（整理后）；mock/selftest 在子目录
    │   ├── INDEX.md         #     总索引：时间/run_id/provider/cells/ok/状态/报告/产物
    │   └── 2026-11-12_2112__real_grid_seed_sizes__MIXED.md
    ├── artifacts/           #   原始产物（mpsa_result / generated.c / harness / 日志）
    │   └── <run_id>/
    ├── tasks/               #   OpOptBench 风格任务卡（可复制）
    │   └── <op>/{task.yaml, input/generic.c, driver/test_<op>.c}
    └── run_smoke.py         #   跑一个实验的脚本（调用 kg_agent mpsa + harness）
```

---

## 1. 文件契约

### 1.1 实验目录（`EXX_name/`）

用 `python experiments/new_experiment.py --id E10 --name 名称` 创建，必须包含：

| 文件/目录 | 作用 | 谁写 |
|---|---|---|
| `README.md` | 实验协议（假设、条件、指标、步骤、限制） | 人 |
| `results.jsonl` | 结果流水，**只追加、不覆盖**；一行一个运行单元 | 脚本 |
| `summary.md` | 结论/表格；`<!-- AUTO-SUMMARY:BEGIN/END -->` 之间由脚本刷新 | 人 + 脚本 |
| `runs/` | 每次运行一个 `.md` 报告（整理后）；`INDEX.md` 总索引；mock/selftest 分目录 | 脚本 + organize_runs.py |
| `artifacts/` | 原始产物（每 run 一个目录） | 脚本 |
| 执行脚本（如 `run_smoke.py`） | 定义如何产生一次 run | 实验负责人 |

### 1.2 一次 run 的命名

`<run_id> = <标签>_seed<seed>_<YYYYmmdd_HHMMSS>`，例如 `smoke_seed1_20261112_204500`。
一个 run id 对应：

- `runs/<YYYY-MM-DD_HHMM>__<run_id>__<STATUS>.md` —— **人读，一个 run 只对应这一个 md**：配置、命令、LLM 输出摘要、编译结果、正确性、耗时、token、修复轮次；`STATUS ∈ {OK, MIXED, FAIL, MOCK, SELFTEST}`；
- `artifacts/<run_id>/<op>_<condition>[/_n<size>]/` —— **机读**：`mpsa_result.json`、`generated_initial.c`（初版）、
  `generated.c`（最终）、`build.sh`、`run.sh`、`build.*.log`、`qemu.*.log`、`result.json`，以及修复回路的
  `repair_r<k>/`（`repair_result.json` + 修复后代码）与 `harness_r<k>/`（重编译/QEMU 日志）。

跑完 `run_smoke.py` 会自动调用 `organize_runs.py`：重命名报告、移动产物、刷新 `runs/INDEX.md`。手动整理：
`python experiments/organize_runs.py --experiment experiments/E09_MPSA`。

### 1.3 `results.jsonl` 字段（append-only）

一行一个 JSON（UTF-8），**同一 (run_id, op, condition, seed, size) 不可重复**；重跑请换 run_id。`paths` 字段是写入时路径（历史 `runs/<run_id>/`）；整理后产物统一在 `artifacts/<run_id>/`。

| 字段 | 说明 |
|---|---|
| `ts` | ISO 时间戳（本地时区） |
| `experiment` / `run_id` | 实验 id / run id |
| `op` / `condition` / `seed` / `size` / `reps` | 运行单元标识 |
| `provider` / `model` / `profile` | LLM provider、模型名、硬件 profile（如 `k1_rvv256`） |
| `mpsa_backend` / `mpsa_exit` / `mpsa_elapsed_s` | `cli`=真 CLI；`stub`=本地兜底（仅 mock 链路自检，不代表方法效果） |
| `mpsa_impl` | 生成所用 `kg_agent` 版本 + `mpsa.py` / `cli.py` sha256（防止实验中途升级 CLI 导致结果不可追溯；2026-11-12 起记录，早期记录无此字段） |
| `status_initial` / `compile_ok_initial` / `correct_initial` / `maxdiff_initial` / `qemu_exit_initial` | **修复前**（初始生成）的结果快照；无修复字段的老记录按 `status` 理解 |
| `repair_backend` / `repair_rounds` / `repair_tokens` / `repair_usage` / `repair_out_dir` / `repairs` | **修复回路**（`--repair-rounds`）：`cli`=真 mpsa-repair、`stub`=本地兜底；尝试轮数、修复 token 与逐轮明细（错误类型/输入日志/repair exit/修复后 status） |
| `usage` | token 计数（`total_tokens` 等）；mock/stub 为 0 |
| `retrieved_count` / `retrieved_ids` / `analysis_count` | 检索到策略数、策略 id、b4 分析次数 |
| `status` | `ok` / `compile_failed` / `wrong_answer` / `qemu_timeout` / `mpsa_failed` / `no_code` / `harness_error` |
| `compile_ok` / `correct` / `maxdiff` / `reldiff` / `qemu_exit` / `timeout` | 编译与正确性 |
| `best_ms_qemu_only` / `qemu_only` | QEMU 上的 best 时间；**仅存档，禁止当性能结论** |
| `kcs` / `kcs_detail` | 知识合规率启发式 v0（无知识/不可检查时为 `null`） |
| `generated_code_sha256` / `generated_code_bytes` | 生成代码指纹 |
| `paths` / `error` / `notes` | 相对路径、失败原因、备注 |

`status` 判定顺序：`mpsa_failed`（没拿到 `mpsa_result.json`）→ `no_code`（code 为空）→
`compile_failed` → `qemu_timeout` → `wrong_answer`（退出码非 0 或 maxdiff 超容差）→ `ok`。

> **口径红线**：`ok` 只表示"编译通过 + 数值正确"，**不含性能含义**。
> 性能（S_comp / EGC / fast_1）只能在 K1 真机上按 `08b` 协议测；QEMU 计时只作"程序能跑完"的证据。

### 1.4 `summary.md` 结构

```markdown
# EXX 摘要
...
<!-- AUTO-SUMMARY:BEGIN -->
（由脚本根据 results.jsonl 自动生成，勿手改此区间）
<!-- AUTO-SUMMARY:END -->
...
（人写的结论、图、下一步）
```

### 1.5 `EXPERIMENTS.md` 总表

每新增一个实验追加一行：`ID | 名称 | 状态 | 开始日期 | 负责人 | 目录 | 结果文件 | 备注`。
状态用：⬜ 未开始 / 🟡 进行中 / ✅ 已完成 / ⛔ 阻塞 / ❌ 放弃。

---

## 2. 新增实验 / 新增 run

```powershell
# 1) 建目录（自动生成 README/summary/results.jsonl/runs/ 模板）
python experiments\new_experiment.py --id E10 --name my_experiment --owner 你的名字

# 2) 在 EXPERIMENTS.md 登记一行

# 3) 复制一份执行脚本模板（以 E09_MPSA/run_smoke.py 为参考），改 TASKS/条件/指标
# 4) 跑 run，检查 runs/INDEX.md（最新正式 run）与 results.jsonl
# 5) 把结论写进 summary.md 手写区

# 已有实验的冒烟：
python experiments\E09_MPSA\run_smoke.py --provider mock --ops daxpy --modes b3 --seed 1
```

### 2.1 修复回路（`--repair-rounds N`，可选）

初始生成 → 编译/QEMU 失败时，把**真实错误反馈**给 LLM 修一次（默认 0=关闭；E09 本次用 1）：

| 初始失败 `status` | 传给 repair 的错误输入 |
|---|---|
| `compile_failed` | `<cell>/harness/build.stderr.log` |
| `wrong_answer` | `<cell>/harness/qemu.stdout.log`（含 `maxdiff=` / `reldiff=`） |

调用 `kg_agent mpsa-repair` → 读 `repair_result.json.code` 写回 `<cell>/generated.c` → 重新编译/QEMU 到
`<cell>/harness_r1/`。**初版代码保存为 `generated_initial.c`，不覆盖**。

- 记录字段：`status_initial` / `compile_ok_initial` / `correct_initial`、`repair_backend`（cli|stub）、
  `repair_rounds`、`repair_tokens`、`repair_usage`、`repair_out_dir`、`repairs[]`（逐轮明细）；
- `--repair-backend auto|cli|stub`：`auto` 在 mock 且 CLI 缺 `mpsa-repair` 时退回本地
  `harness/mock_repair.py`（仅链路自检，记录 `repair_backend=stub`）；真实 API 必须用 CLI；
- 自测（用 CLI mock 构造失败，验证两条分支）：
  ```powershell
  python experiments\E09_MPSA\run_smoke.py --provider mock --ops daxpy --modes b3 --seed 71 --repair-rounds 1 `
      --mock-code experiments\E09_MPSA\selftest\broken_compile_daxpy.c --run-id selftest_repair_compile --no-summary
  python experiments\E09_MPSA\run_smoke.py --provider mock --ops ddot --modes b3 --seed 72 --repair-rounds 1 `
      --mock-code experiments\E09_MPSA\selftest\wrong_answer_ddot.c --run-id selftest_repair_wrong --no-summary
  ```

---

## 2.5 可复现包（`harness/repro_package.py`）

每次正式 run 自动生成 `artifacts/<run_id>/repro/`，**先留原始数据、最后才是人读 md**：

```text
repro/
├── README.md                 # 怎么复现（检索链 + 编译 + QEMU）
├── manifest.json             # run 信息 + 每个 cell 的输入/检索/final 映射
├── env.txt                   # gcc/qemu/OS 版本
├── inputs/
│   ├── <op>/generic.c        # 输入算子（原样）
│   ├── <op>/task.yaml
│   ├── <op>/driver/test_<op>.c
│   ├── hardware/hardware_profiles.yaml   # 硬件 profile
│   └── retrieval/<op>_<profile>_k<top_k>_{trace,pack}.json  # 7 步检索链
└── cells/<op>_<mode>_n<size>_seed<seed>/
    ├── query.json / strategies.json / prompts/{analysis,synthesis,generate}.txt
    ├── round0/               # 初始生成：generated.c + build.sh/run.sh + result.json + 日志
    ├── round1..N/            # 每一轮修复：同上 + repair_result.json
    └── final/                # 最终代码 + build.sh/run.sh + bin + expected.txt
```

复现：`cd artifacts/<run_id>/repro/cells/<cell>/final && bash build.sh && bash run.sh`。
`run_smoke.py` 跑完会自动调用；手动补：`python experiments/harness/repro_package.py --experiment experiments/E09_MPSA --run-id <run_id>`。

## 3. QEMU harness（`harness/qemu.py`）

```python
import sys; sys.path.insert(0, "experiments")
from harness.qemu import compile_run

res = compile_run(
    "experiments/E09_MPSA/tasks/daxpy",   # 任务目录（含 task.yaml）
    "artifacts/<run_id>/daxpy_b3/generated.c",  # 待测实现
    "artifacts/<run_id>/daxpy_b3/harness",      # 产物目录
    n=16384, reps=10,
)
res["compile_ok"], res["correct"], res["maxdiff"], res["qemu_exit"], res["timeout"]
```

行为：

1. 生成 `build.sh` / `run.sh`（**UTF-8 无 BOM + LF**，否则 WSL 下中文路径与 `set -e` 会坏）；
2. `wsl.exe -d Ubuntu -e bash <script>` 调 `riscv64-linux-gnu-gcc`（flags 固定为
   `-O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -static`）交叉编译 `driver + generated.c`；
3. `qemu-riscv64 -cpu rv64,v=true,vlen=256,zba=true,zbb=true,zbc=true,zbs=true ./bin <n> <reps>` 运行；
4. 解析 driver 的 `maxdiff=` / `reldiff=` / `best=` 与退出码，返回
   `compile_ok / correct / maxdiff / qemu_exit / stdout / stderr / timeout`（另有诊断字段）；
5. gcc / QEMU 的 stdout、stderr 分文件落盘（`build.stdout.log` 等），Windows 侧按 UTF-8 读，
   避免 WSL 自身 UTF-16 警告与子进程日志混流。

**约定**：driver 必须打印 `maxdiff=`，退出码 0=正确、非 0=错误。driver 参考实现可直接抄
`E09_MPSA/tasks/daxpy/driver/test_daxpy.c`。

---

## 4. 环境依赖与排错

| 依赖 | 说明 |
|---|---|
| Windows Python ≥3.10 | 纯标准库，无需 pip install |
| WSL Ubuntu（默认 `Ubuntu`，可用 `EXP_WSL_DISTRO` 覆盖） | 已装 `riscv64-linux-gnu-gcc`、`qemu-riscv64`、`timeout` |
| `kg_agent` | 实验脚本通过 `科研产出/kg_agent/src` 调 `python -m kgagent.cli`；只读，不要改 |

常见现象：

- `wsl.exe` 的 stderr 会打 `localhost 代理配置…NAT 模式…` 警告——**忽略**，stdout 才是程序输出；
- QEMU 的 `vector version is not specified, use the default value v1.0` 是警告，不是错误；
- 编译失败先看 `artifacts/<run_id>/<op>_<cond>/build.stderr.log`；超时看 `qemu.*.log`；注意 `results.jsonl.paths` 里是当时写入的路径，整理后以 `artifacts/<run_id>/` 为准。
- 中文路径可以正常用，前提是脚本为 UTF-8（无 BOM）+ LF——脚本由 Python 生成时显式指定编码。

---

## 5. 相关文档

- [历次实验设计与评估协议](../_archive/2026-11-19/04_实验设计与评估协议.md)；
- [历史 K1 真机实验协议](../_archive/2026-11-19/08b_RISC-V真机实验协议.md)；
- `../17_文献精读_Lingxi.md`：MPSA（B4/E9）的来源与复用方案；
- `E09_MPSA/README.md`：第一个按本规范落地的实验。
