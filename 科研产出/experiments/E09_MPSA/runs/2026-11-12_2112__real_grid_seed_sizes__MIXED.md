# E09 MPSA run `real_grid_seed_sizes`

> 生成时间：2026-11-12T21:12:26+08:00 ｜ 实验：E09_MPSA ｜ provider：`dashscope` ｜ profile：`k1_rvv256`

## 1. 配置

| 项 | 值 |
|---|---|
| run_id | `real_grid_seed_sizes` |
| provider / model | `dashscope` / `qwen-plus-2025-04-28` |
| 条件 | b3（B3 · StrategyPack 直注入 → 单次生成）, b4（B4 · 多视角分析 + 综合 → 生成（MPSA）） |
| 算子 | daxpy, ddot |
| seeds | 1, 2, 3 |
| sizes / reps | 1024, 16384 / 10 |
| mpsa backend | cli |
| mpsa impl | kgagent `1.5.0` / mpsa.py sha256 `48badce182d4fb6e` / cli.py `8e6a3b596ac5c44e` |
| repair | rounds=1 / backend=auto |
| WSL / CC | `Ubuntu` / `riscv64-linux-gnu-gcc` |
| CFLAGS | `-O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -static` |
| QEMU | `qemu-riscv64` `-cpu rv64,v=true,vlen=256,zba=true,zbb=true,zbc=true,zbs=true` |

## 2. 命令

mpsa（示例，实际每 cell 的完整命令见 `results.jsonl.mpsa_cmd`）：

```powershell
C:\Users\MSI\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe -m kgagent.cli mpsa --op daxpy --profile k1_rvv256 --code C:\Users\MSI\Desktop\开题\科研产出\experiments\E09_MPSA\tasks\daxpy\input\generic.c --mode b3 --provider dashscope --out C:\Users\MSI\Desktop\开题\科研产出\experiments\E09_MPSA\../artifacts/real_grid_seed_sizes/daxpy_b3_n1024 --top-k 6
```

编译 + QEMU（`experiments/harness/qemu.py` 生成，产物在 `harness/`）：

```bash
riscv64-linux-gnu-gcc -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -static ... -o bin
qemu-riscv64 -cpu rv64,v=true,vlen=256,zba=true,zbb=true,zbc=true,zbs=true ./bin 1024 10
```

## 3. 结果总览

| op | 条件 | size | initial | repair_rounds | compile_ok | correct | maxdiff | qemu_exit | gen_tokens | repair_tokens | mpsa_s | KCS | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| daxpy | b3 | 1024 | `compile_failed` | 1 | ✅ | ✅ | 0.00e+00 | 0 | 7167 | 3458 | 12.704 | 0.00 | `ok` |
| daxpy | b3 | 16384 | `ok` | 0 | ✅ | ✅ | 0.00e+00 | 0 | 7167 | 0 | 12.884 | 0.25 | `ok` |
| daxpy | b3 | 1024 | `compile_failed` | 1 | ✅ | ✅ | 0.00e+00 | 0 | 7188 | 2759 | 13.701 | 0.00 | `ok` |
| daxpy | b3 | 16384 | `ok` | 0 | ✅ | ✅ | 0.00e+00 | 0 | 7225 | 0 | 13.981 | 0.25 | `ok` |
| daxpy | b3 | 1024 | `ok` | 0 | ✅ | ✅ | 0.00e+00 | 0 | 7167 | 0 | 12.839 | 0.25 | `ok` |
| daxpy | b3 | 16384 | `ok` | 0 | ✅ | ✅ | 0.00e+00 | 0 | 7156 | 0 | 13.302 | 0.25 | `ok` |
| daxpy | b4 | 1024 | `compile_failed` | 1 | ✅ | ✅ | 0.00e+00 | 0 | 19485 | 2946 | 52.047 | 0.00 | `ok` |
| daxpy | b4 | 16384 | `ok` | 0 | ✅ | ✅ | 0.00e+00 | 0 | 18692 | 0 | 44.469 | 0.25 | `ok` |
| daxpy | b4 | 1024 | `compile_failed` | 1 | ✅ | ✅ | 0.00e+00 | 0 | 19178 | 2915 | 47.522 | 0.00 | `ok` |
| daxpy | b4 | 16384 | `ok` | 0 | ✅ | ✅ | 0.00e+00 | 0 | 19618 | 0 | 52.862 | 0.25 | `ok` |
| daxpy | b4 | 1024 | `ok` | 0 | ✅ | ✅ | 0.00e+00 | 0 | 19401 | 0 | 50.234 | 0.25 | `ok` |
| daxpy | b4 | 16384 | `ok` | 0 | ✅ | ✅ | 0.00e+00 | 0 | 19519 | 0 | 55.162 | 0.25 | `ok` |
| ddot | b3 | 1024 | `ok` | 0 | ✅ | ✅ | 1.14e-13 | 0 | 10963 | 0 | 16.513 | 0.25 | `ok` |
| ddot | b3 | 16384 | `ok` | 0 | ✅ | ✅ | 0.00e+00 | 0 | 10536 | 0 | 9.983 | 0.25 | `ok` |
| ddot | b3 | 1024 | `ok` | 0 | ✅ | ✅ | 1.14e-13 | 0 | 10292 | 0 | 7.031 | 0.25 | `ok` |
| ddot | b3 | 16384 | `compile_failed` | 1 | ✅ | ✅ | 1.18e-11 | 0 | 10733 | 2836 | 11.394 | 0.00 | `ok` |
| ddot | b3 | 1024 | `ok` | 0 | ✅ | ✅ | 1.14e-13 | 0 | 10715 | 0 | 14.233 | 0.25 | `ok` |
| ddot | b3 | 16384 | `compile_failed` | 1 | ✅ | ✅ | 1.18e-11 | 0 | 10667 | 2653 | 10.592 | 0.00 | `ok` |
| ddot | b4 | 1024 | `wrong_answer` | 1 | ❌ | ❌ | — | None | 22662 | 2529 | 54.424 | 0.25 | `compile_failed` |
| ddot | b4 | 16384 | `ok` | 0 | ✅ | ✅ | 0.00e+00 | 0 | 21757 | 0 | 43.975 | 0.25 | `ok` |
| ddot | b4 | 1024 | `ok` | 0 | ✅ | ✅ | 1.14e-13 | 0 | 22159 | 0 | 47.324 | 0.25 | `ok` |
| ddot | b4 | 16384 | `ok` | 0 | ✅ | ✅ | 0.00e+00 | 0 | 21757 | 0 | 18.554 | 0.25 | `ok` |
| ddot | b4 | 1024 | `compile_failed` | 1 | ✅ | ✅ | 5.68e-13 | 0 | 22493 | 2764 | 52.154 | 0.00 | `ok` |
| ddot | b4 | 16384 | `ok` | 0 | ✅ | ✅ | 0.00e+00 | 0 | 22122 | 0 | 47.092 | 0.25 | `ok` |

汇总：**cells=24，ok=23，FCR_final=95.8%（initial 66.7%）**，编译通过率 final 95.8%（initial 70.8%），总 gen tokens=355819（LLM calls=72），KCS 均值 0.18（n=24）
修复回路：**8 cells 触发修复，修复后转正 7，修复 tokens=22860**（初始失败类型见第 4 节）。

> `ok`/FCR 只表示“编译通过 + 数值正确”，**不含性能含义**；QEMU 的 `best=` 毫秒数不得进性能表（见 `08b` §5.1）。

## 4. 逐 cell 详情

### daxpy × b3（seed=1，n=1024）

- **initial**：status=`compile_failed`；compile_ok=❌，correct=❌，maxdiff=—，qemu_exit=—，kcs=0.25
- **修复回路**：backend=`cli`，执行 1 轮
  - r1：input=`compile_failed`，error=build_stderr（`../artifacts/real_grid_seed_sizes/daxpy_b3_n1024/harness/build.stderr.log`）→ repair exit=0、5.30s、tokens=3458 → status=`ok` correct=✅；fix_summary changed=True chars 1694→704
- **status（final）**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=12.70s，model=`qwen-plus-2025-04-28`
- **检索**：4 条策略 `stg:axpy:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:axpy:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:axpy:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:axpy:specialize_s` 任意步长 inc 累加访问
- **B3 分析**：无（策略包直注入）
- **prompt**：generate=17536 chars，analysis=0 chars，synthesis=0 chars
- **生成代码**：704 bytes，sha256=`2d35df4f1fcaaec6`，存档 `../artifacts/real_grid_seed_sizes/daxpy_b3_n1024/generated.c`
- **编译（final）**：✅ ok（3.07s）；正确性：✅，maxdiff=0.000e+00，reldiff=0.000e+00，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.0063 ms（qemu_only=True）
- **token**：total=7167（prompt=6502 / completion=665 / calls=1）
- **KCS（启发式 v0）**：0.00（matched 0 / checkable 4；retrieved 4；weak-only 2；uncheckable 0）

  最终代码前 20 行（全文见 `../artifacts/real_grid_seed_sizes/daxpy_b3_n1024/generated.c`；初版见 `../artifacts/real_grid_seed_sizes/daxpy_b3_n1024/generated_initial.c`）：

  ```c
  typedef long long BLASLONG;
  
  int daxpy_k(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, double da, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG dummy2)
  {
      if (n <= 0 || da == 0.0) return 0;
  
      BLASLONG i;
      if (inc_x == 1 && inc_y == 1) {
          for (i = 0; i < n; i++) {
              y[i] += da * x[i];
          }
      } else if (inc_y == 1) {
          BLASLONG ix = 0;
          for (i = 0; i < n; i++) {
              y[i] += da * x[ix];
              ix += inc_x;
          }
      } else {
          BLASLONG ix = 0, iy = 0;
          for (i = 0; i < n; i++) {
  ```

### daxpy × b3（seed=1，n=16384）

- **initial**：status=`ok`；compile_ok=✅，correct=✅，maxdiff=0.000e+00，qemu_exit=0，kcs=0.25
- **修复回路**：未触发（初始 `ok`，无需/不可修复）
- **status（final）**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=12.88s，model=`qwen-plus-2025-04-28`
- **检索**：4 条策略 `stg:axpy:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:axpy:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:axpy:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:axpy:specialize_s` 任意步长 inc 累加访问
- **B3 分析**：无（策略包直注入）
- **prompt**：generate=17536 chars，analysis=0 chars，synthesis=0 chars
- **生成代码**：1694 bytes，sha256=`c8f0fe5280089529`，存档 `../artifacts/real_grid_seed_sizes/daxpy_b3_n16384/generated.c`
- **编译（final）**：✅ ok（0.51s）；正确性：✅，maxdiff=0.000e+00，reldiff=0.000e+00，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.2656 ms（qemu_only=True）
- **token**：total=7167（prompt=6502 / completion=665 / calls=1）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 4；weak-only 3；uncheckable 0）

  最终代码前 20 行（全文见 `../artifacts/real_grid_seed_sizes/daxpy_b3_n16384/generated.c`；初版见 `../artifacts/real_grid_seed_sizes/daxpy_b3_n16384/generated_initial.c`）：

  ```c
  #include <riscv_vector.h>
  typedef long long BLASLONG;
  int daxpy_k(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, double da, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG dummy2)
  {
      if (n <= 0 || da == 0.0) return 0;
  
      if (inc_x == 1 && inc_y == 1) {
          BLASLONG i = 0;
          size_t vl;
  
          while (i < n) {
              vl = __riscv_vsetvl_e64m8(n - i);
  
              vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
              vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
              vy = __riscv_vfmacc_vf_f64m8(vy, da, vx, vl);
              __riscv_vse64_v_f64m8(y + i, vy, vl);
  
              i += vl;
          }
  ```

### daxpy × b3（seed=2，n=1024）

- **initial**：status=`compile_failed`；compile_ok=❌，correct=❌，maxdiff=—，qemu_exit=—，kcs=0.25
- **修复回路**：backend=`cli`，执行 1 轮
  - r1：input=`compile_failed`，error=build_stderr（`../artifacts/real_grid_seed_sizes/daxpy_b3_n1024/harness/build.stderr.log`）→ repair exit=0、5.19s、tokens=2759 → status=`ok` correct=✅；fix_summary changed=True chars 1745→721
- **status（final）**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=13.70s，model=`qwen-plus-2025-04-28`
- **检索**：4 条策略 `stg:axpy:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:axpy:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:axpy:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:axpy:specialize_s` 任意步长 inc 累加访问
- **B3 分析**：无（策略包直注入）
- **prompt**：generate=17536 chars，analysis=0 chars，synthesis=0 chars
- **生成代码**：721 bytes，sha256=`62ad86462bf6e029`，存档 `../artifacts/real_grid_seed_sizes/daxpy_b3_n1024/generated.c`
- **编译（final）**：✅ ok（0.21s）；正确性：✅，maxdiff=0.000e+00，reldiff=0.000e+00，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.0063 ms（qemu_only=True）
- **token**：total=7188（prompt=6502 / completion=686 / calls=1）
- **KCS（启发式 v0）**：0.00（matched 0 / checkable 4；retrieved 4；weak-only 2；uncheckable 0）

  最终代码前 20 行（全文见 `../artifacts/real_grid_seed_sizes/daxpy_b3_n1024/generated.c`；初版见 `../artifacts/real_grid_seed_sizes/daxpy_b3_n1024/generated_initial.c`）：

  ```c
  typedef long long BLASLONG;
  
  int daxpy_k(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, double da, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG dummy2)
  {
      if (n <= 0 || da == 0.0) return 0;
  
      BLASLONG i;
      if (inc_x == 1 && inc_y == 1) {
          for (i = 0; i < n; i++) {
              y[i] += da * x[i];
          }
      } else if (inc_y == 1) {
          BLASLONG ix = 0;
          for (i = 0; i < n; i++) {
              y[i] += da * x[ix];
              ix += inc_x;
          }
      } else {
          BLASLONG ix = 0;
          BLASLONG iy = 0;
  ```

### daxpy × b3（seed=2，n=16384）

- **initial**：status=`ok`；compile_ok=✅，correct=✅，maxdiff=0.000e+00，qemu_exit=0，kcs=0.25
- **修复回路**：未触发（初始 `ok`，无需/不可修复）
- **status（final）**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=13.98s，model=`qwen-plus-2025-04-28`
- **检索**：4 条策略 `stg:axpy:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:axpy:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:axpy:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:axpy:specialize_s` 任意步长 inc 累加访问
- **B3 分析**：无（策略包直注入）
- **prompt**：generate=17536 chars，analysis=0 chars，synthesis=0 chars
- **生成代码**：1893 bytes，sha256=`0491c04dba02aea6`，存档 `../artifacts/real_grid_seed_sizes/daxpy_b3_n16384/generated.c`
- **编译（final）**：✅ ok（0.49s）；正确性：✅，maxdiff=0.000e+00，reldiff=0.000e+00，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.2588 ms（qemu_only=True）
- **token**：total=7225（prompt=6502 / completion=723 / calls=1）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 4；weak-only 3；uncheckable 0）

  最终代码前 20 行（全文见 `../artifacts/real_grid_seed_sizes/daxpy_b3_n16384/generated.c`；初版见 `../artifacts/real_grid_seed_sizes/daxpy_b3_n16384/generated_initial.c`）：

  ```c
  #include <riscv_vector.h>
  typedef long long BLASLONG;
  int daxpy_k(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, double da, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG dummy2)
  {
      if (n <= 0 || da == 0.0) return 0;
  
      if (inc_x == 1 && inc_y == 1) {
          BLASLONG i = 0;
          size_t vl;
  
          // 向量长度为 64 位双精度浮点数，LMUL=8，即最多 8 个寄存器组
          while (i < n) {
              vl = __riscv_vsetvl_e64m8(n - i);
  
              vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
              vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
              vy = __riscv_vfmacc_vf_f64m8(vy, da, vx, vl);
              __riscv_vse64_v_f64m8(y + i, vy, vl);
  
              i += vl;
  ```

### daxpy × b3（seed=3，n=1024）

- **initial**：status=`ok`；compile_ok=✅，correct=✅，maxdiff=0.000e+00，qemu_exit=0，kcs=0.25
- **修复回路**：未触发（初始 `ok`，无需/不可修复）
- **status（final）**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=12.84s，model=`qwen-plus-2025-04-28`
- **检索**：4 条策略 `stg:axpy:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:axpy:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:axpy:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:axpy:specialize_s` 任意步长 inc 累加访问
- **B3 分析**：无（策略包直注入）
- **prompt**：generate=17536 chars，analysis=0 chars，synthesis=0 chars
- **生成代码**：1694 bytes，sha256=`65317082979d91e6`，存档 `../artifacts/real_grid_seed_sizes/daxpy_b3_n1024/generated.c`
- **编译（final）**：✅ ok（0.46s）；正确性：✅，maxdiff=0.000e+00，reldiff=0.000e+00，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.0168 ms（qemu_only=True）
- **token**：total=7167（prompt=6502 / completion=665 / calls=1）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 4；weak-only 3；uncheckable 0）

  最终代码前 20 行（全文见 `../artifacts/real_grid_seed_sizes/daxpy_b3_n1024/generated.c`；初版见 `../artifacts/real_grid_seed_sizes/daxpy_b3_n1024/generated_initial.c`）：

  ```c
  #include <riscv_vector.h>
  typedef long long BLASLONG;
  int daxpy_k(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, double da, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG dummy2)
  {
      if (n <= 0 || da == 0.0) return 0;
  
      if (inc_x == 1 && inc_y == 1) {
          BLASLONG i = 0;
          size_t vl;
  
          while (i < n) {
              vl = __riscv_vsetvl_e64m8(n - i);
  
              vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
              vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
              vy = __riscv_vfmacc_vf_f64m8(vy, da, vx, vl);
              __riscv_vse64_v_f64m8(y + i, vy, vl);
  
              i += vl;
          }
  ```

### daxpy × b3（seed=3，n=16384）

- **initial**：status=`ok`；compile_ok=✅，correct=✅，maxdiff=0.000e+00，qemu_exit=0，kcs=0.25
- **修复回路**：未触发（初始 `ok`，无需/不可修复）
- **status（final）**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=13.30s，model=`qwen-plus-2025-04-28`
- **检索**：4 条策略 `stg:axpy:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:axpy:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:axpy:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:axpy:specialize_s` 任意步长 inc 累加访问
- **B3 分析**：无（策略包直注入）
- **prompt**：generate=17536 chars，analysis=0 chars，synthesis=0 chars
- **生成代码**：1686 bytes，sha256=`a5f20fe8d51498c6`，存档 `../artifacts/real_grid_seed_sizes/daxpy_b3_n16384/generated.c`
- **编译（final）**：✅ ok（0.47s）；正确性：✅，maxdiff=0.000e+00，reldiff=0.000e+00，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.2544 ms（qemu_only=True）
- **token**：total=7156（prompt=6502 / completion=654 / calls=1）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 4；weak-only 3；uncheckable 0）

  最终代码前 20 行（全文见 `../artifacts/real_grid_seed_sizes/daxpy_b3_n16384/generated.c`；初版见 `../artifacts/real_grid_seed_sizes/daxpy_b3_n16384/generated_initial.c`）：

  ```c
  typedef long long BLASLONG;
  
  #include <riscv_vector.h>
  int daxpy_k(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, double da, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG dummy2)
  {
      if (n <= 0 || da == 0.0) return 0;
  
      if (inc_x == 1 && inc_y == 1) {
          BLASLONG i = 0;
          size_t vl;
          while (i < n) {
              vl = __riscv_vsetvl_e64m8(n - i);
              vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
              vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
              vy = __riscv_vfmacc_vf_f64m8(vy, da, vx, vl);
              __riscv_vse64_v_f64m8(y + i, vy, vl);
              i += vl;
          }
      } else if (inc_y == 1) {
          BLASLONG i = 0;
  ```

### daxpy × b4（seed=1，n=1024）

- **initial**：status=`compile_failed`；compile_ok=❌，correct=❌，maxdiff=—，qemu_exit=—，kcs=0.25
- **修复回路**：backend=`cli`，执行 1 轮
  - r1：input=`compile_failed`，error=build_stderr（`../artifacts/real_grid_seed_sizes/daxpy_b4_n1024/harness/build.stderr.log`）→ repair exit=0、4.92s、tokens=2946 → status=`ok` correct=✅；fix_summary changed=True chars 1787→748
- **status（final）**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=52.05s，model=`qwen-plus-2025-04-28`
- **检索**：4 条策略 `stg:axpy:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:axpy:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:axpy:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:axpy:specialize_s` 任意步长 inc 累加访问
- **B4 分析**：3 条（applies=3）；综合 criteria={"supportive": 0.9, "complementarity": 0.85, "accuracy": 0.7, "completeness": 0.8}
  - overall：三策略组合在主循环向量化、尾部处理、寄存器调优方面具备良好一致性与互补性，适用于当前特征代码，但需补充边界处理与别名防护以确保完整性和正确性
- **prompt**：generate=21631 chars，analysis=16810 chars，synthesis=3958 chars
- **生成代码**：748 bytes，sha256=`8c56d82d7f8ace73`，存档 `../artifacts/real_grid_seed_sizes/daxpy_b4_n1024/generated.c`
- **编译（final）**：✅ ok（1.72s）；正确性：✅，maxdiff=0.000e+00，reldiff=0.000e+00，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.0059 ms（qemu_only=True）
- **token**：total=19485（prompt=17115 / completion=2370 / calls=5）
- **KCS（启发式 v0）**：0.00（matched 0 / checkable 4；retrieved 4；weak-only 2；uncheckable 0）

  最终代码前 20 行（全文见 `../artifacts/real_grid_seed_sizes/daxpy_b4_n1024/generated.c`；初版见 `../artifacts/real_grid_seed_sizes/daxpy_b4_n1024/generated_initial.c`）：

  ```c
  typedef long long BLASLONG;
  
  int daxpy_k(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, double da, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG dummy2)
  {
      if (n <= 0) return 0;
      if (da == 0.0) return 0;
  
      if (inc_x == 1 && inc_y == 1) {
          for (BLASLONG i = 0; i < n; i++) {
              y[i] += da * x[i];
          }
      } else if (inc_y == 1) {
          BLASLONG ix = 0;
          for (BLASLONG i = 0; i < n; i++) {
              y[i] += da * x[ix];
              ix += inc_x;
          }
      } else {
          BLASLONG ix = 0;
          BLASLONG iy = 0;
  ```

### daxpy × b4（seed=1，n=16384）

- **initial**：status=`ok`；compile_ok=✅，correct=✅，maxdiff=0.000e+00，qemu_exit=0，kcs=0.25
- **修复回路**：未触发（初始 `ok`，无需/不可修复）
- **status（final）**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=44.47s，model=`qwen-plus-2025-04-28`
- **检索**：4 条策略 `stg:axpy:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:axpy:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:axpy:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:axpy:specialize_s` 任意步长 inc 累加访问
- **B4 分析**：3 条（applies=3）；综合 criteria={"supportive": 0.8, "complementarity": 0.9, "accuracy": 0.7, "completeness": 0.75}
  - overall：三策略组合具备良好互补性与参数一致性，主循环向量化结合尾部动态处理，适合单位步长大规模 double 精度 daxpy 实现
- **prompt**：generate=21068 chars，analysis=16810 chars，synthesis=3651 chars
- **生成代码**：1603 bytes，sha256=`d224512327847199`，存档 `../artifacts/real_grid_seed_sizes/daxpy_b4_n16384/generated.c`
- **编译（final）**：✅ ok（2.08s）；正确性：✅，maxdiff=0.000e+00，reldiff=0.000e+00，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.2244 ms（qemu_only=True）
- **token**：total=18692（prompt=16667 / completion=2025 / calls=5）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 4；weak-only 3；uncheckable 0）

  最终代码前 20 行（全文见 `../artifacts/real_grid_seed_sizes/daxpy_b4_n16384/generated.c`；初版见 `../artifacts/real_grid_seed_sizes/daxpy_b4_n16384/generated_initial.c`）：

  ```c
  typedef long long BLASLONG;
  
  #include <riscv_vector.h>
  int daxpy_k(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, double da, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG dummy2)
  {
      if (n <= 0 || da == 0.0) return 0;
  
      if (inc_x == 1 && inc_y == 1) {
          BLASLONG i = 0;
          while (i < n) {
              size_t vl = __riscv_vsetvl_e64m8(n - i);
              vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
              vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
              vfloat64m8_t vres = __riscv_vfmacc_vf_f64m8(vy, da, vx, vl);
              __riscv_vse64_v_f64m8(y + i, vres, vl);
              i += vl;
          }
      } else if (inc_y == 1) {
          BLASLONG i = 0;
          while (i < n) {
  ```

### daxpy × b4（seed=2，n=1024）

- **initial**：status=`compile_failed`；compile_ok=❌，correct=❌，maxdiff=—，qemu_exit=—，kcs=0.25
- **修复回路**：backend=`cli`，执行 1 轮
  - r1：input=`compile_failed`，error=build_stderr（`../artifacts/real_grid_seed_sizes/daxpy_b4_n1024/harness/build.stderr.log`）→ repair exit=0、5.29s、tokens=2915 → status=`ok` correct=✅；fix_summary changed=True chars 1690→731
- **status（final）**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=47.52s，model=`qwen-plus-2025-04-28`
- **检索**：4 条策略 `stg:axpy:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:axpy:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:axpy:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:axpy:specialize_s` 任意步长 inc 累加访问
- **B4 分析**：3 条（applies=3）；综合 criteria={"supportive": 0.8, "complementarity": 0.9, "accuracy": 0.7, "completeness": 0.8}
  - overall：建议组合向量化与尾部处理策略，结合寄存器分组优化，注意尾部数值一致性与别名处理。
- **prompt**：generate=21363 chars，analysis=16810 chars，synthesis=3803 chars
- **生成代码**：731 bytes，sha256=`be3c68bcc17d4f4d`，存档 `../artifacts/real_grid_seed_sizes/daxpy_b4_n1024/generated.c`
- **编译（final）**：✅ ok（1.71s）；正确性：✅，maxdiff=0.000e+00，reldiff=0.000e+00，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.0059 ms（qemu_only=True）
- **token**：total=19178（prompt=16943 / completion=2235 / calls=5）
- **KCS（启发式 v0）**：0.00（matched 0 / checkable 4；retrieved 4；weak-only 2；uncheckable 0）

  最终代码前 20 行（全文见 `../artifacts/real_grid_seed_sizes/daxpy_b4_n1024/generated.c`；初版见 `../artifacts/real_grid_seed_sizes/daxpy_b4_n1024/generated_initial.c`）：

  ```c
  typedef long long BLASLONG;
  
  int daxpy_k(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, double da, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG dummy2)
  {
      if (n <= 0) return 0;
      if (da == 0.0) return 0;
  
      if (inc_x == 1 && inc_y == 1) {
          for (BLASLONG i = 0; i < n; i++) {
              y[i] += da * x[i];
          }
      } else if (inc_y == 1) {
          BLASLONG ix = 0;
          for (BLASLONG i = 0; i < n; i++) {
              y[i] += da * x[ix];
              ix += inc_x;
          }
      } else {
          BLASLONG ix = 0, iy = 0;
          for (BLASLONG i = 0; i < n; i++) {
  ```

### daxpy × b4（seed=2，n=16384）

- **initial**：status=`ok`；compile_ok=✅，correct=✅，maxdiff=0.000e+00，qemu_exit=0，kcs=0.25
- **修复回路**：未触发（初始 `ok`，无需/不可修复）
- **status（final）**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=52.86s，model=`qwen-plus-2025-04-28`
- **检索**：4 条策略 `stg:axpy:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:axpy:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:axpy:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:axpy:specialize_s` 任意步长 inc 累加访问
- **B4 分析**：3 条（applies=3）；综合 criteria={"supportive": 0.85, "complementarity": 0.9, "accuracy": 0.7, "completeness": 0.8}
  - overall：三条策略可组合使用，具备良好支持性和互补性，数值一致性风险可控，覆盖主循环和尾部处理，但需补充通用步长和边界处理逻辑。
- **prompt**：generate=21738 chars，analysis=16810 chars，synthesis=4050 chars
- **生成代码**：1803 bytes，sha256=`100167d2924d7064`，存档 `../artifacts/real_grid_seed_sizes/daxpy_b4_n16384/generated.c`
- **编译（final）**：✅ ok（2.10s）；正确性：✅，maxdiff=0.000e+00，reldiff=0.000e+00，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.2339 ms（qemu_only=True）
- **token**：total=19618（prompt=17194 / completion=2424 / calls=5）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 4；weak-only 3；uncheckable 0）

  最终代码前 20 行（全文见 `../artifacts/real_grid_seed_sizes/daxpy_b4_n16384/generated.c`；初版见 `../artifacts/real_grid_seed_sizes/daxpy_b4_n16384/generated_initial.c`）：

  ```c
  typedef long long BLASLONG;
  
  #include <riscv_vector.h>
  int daxpy_k(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, double da, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG dummy2)
  {
      if (n <= 0) return 0;
      if (da == 0.0) return 0;
  
      if (inc_x == 1 && inc_y == 1) {
          BLASLONG i = 0;
          const size_t vlmax = __riscv_vsetvlmax_e64m8();
          vfloat64m8_t vda = __riscv_vfmv_v_f_f64m8(da, vlmax);
  
          for (; i + vlmax <= n; i += vlmax) {
              size_t vl = __riscv_vsetvl_e64m8(n - i);
              vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
              vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
              vy = __riscv_vfmacc_vv_f64m8(vy, vda, vx, vl);
              __riscv_vse64_v_f64m8(y + i, vy, vl);
          }
  ```

### daxpy × b4（seed=3，n=1024）

- **initial**：status=`ok`；compile_ok=✅，correct=✅，maxdiff=0.000e+00，qemu_exit=0，kcs=0.25
- **修复回路**：未触发（初始 `ok`，无需/不可修复）
- **status（final）**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=50.23s，model=`qwen-plus-2025-04-28`
- **检索**：4 条策略 `stg:axpy:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:axpy:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:axpy:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:axpy:specialize_s` 任意步长 inc 累加访问
- **B4 分析**：3 条（applies=3）；综合 criteria={"supportive": 0.9, "complementarity": 0.95, "accuracy": 0.75, "completeness": 0.8}
  - overall：三策略组合具备高可行性，可覆盖主循环向量化、尾部处理、寄存器调优，适合 double 型、大尺寸、单位步长场景，需注意边界与别名处理。
- **prompt**：generate=21701 chars，analysis=16810 chars，synthesis=4155 chars
- **生成代码**：1713 bytes，sha256=`7e8aa40ea32962a8`，存档 `../artifacts/real_grid_seed_sizes/daxpy_b4_n1024/generated.c`
- **编译（final）**：✅ ok（2.21s）；正确性：✅，maxdiff=0.000e+00，reldiff=0.000e+00，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.0142 ms（qemu_only=True）
- **token**：total=19401（prompt=17091 / completion=2310 / calls=5）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 4；weak-only 3；uncheckable 0）

  最终代码前 20 行（全文见 `../artifacts/real_grid_seed_sizes/daxpy_b4_n1024/generated.c`；初版见 `../artifacts/real_grid_seed_sizes/daxpy_b4_n1024/generated_initial.c`）：

  ```c
  typedef long long BLASLONG;
  
  #include <riscv_vector.h>
  int daxpy_k(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, double da, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG dummy2)
  {
      if (n <= 0) return 0;
      if (da == 0.0) return 0;
  
      if (inc_x == 1 && inc_y == 1) {
          BLASLONG i = 0;
          size_t vl;
  
          while (i < n) {
              vl = __riscv_vsetvl_e64m8(n - i);
              vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
              vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
              vy = __riscv_vfmacc_vf_f64m8(vy, da, vx, vl);
              __riscv_vse64_v_f64m8(y + i, vy, vl);
              i += vl;
          }
  ```

### daxpy × b4（seed=3，n=16384）

- **initial**：status=`ok`；compile_ok=✅，correct=✅，maxdiff=0.000e+00，qemu_exit=0，kcs=0.25
- **修复回路**：未触发（初始 `ok`，无需/不可修复）
- **status（final）**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=55.16s，model=`qwen-plus-2025-04-28`
- **检索**：4 条策略 `stg:axpy:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:axpy:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:axpy:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:axpy:specialize_s` 任意步长 inc 累加访问
- **B4 分析**：3 条（applies=3）；综合 criteria={"supportive": 0.85, "complementarity": 0.9, "accuracy": 0.75, "completeness": 0.8}
  - overall：三条策略在参数一致、阶段互补的前提下可组合使用，形成针对 large size_regime、double 精度、RISC-V Vector Extension 的完整 daxpy 实现，但需注意尾部处理与边界分支的显式实现。
- **prompt**：generate=21731 chars，analysis=16810 chars，synthesis=3935 chars
- **生成代码**：1824 bytes，sha256=`1ca51d47cf39db72`，存档 `../artifacts/real_grid_seed_sizes/daxpy_b4_n16384/generated.c`
- **编译（final）**：✅ ok（2.12s）；正确性：✅，maxdiff=0.000e+00，reldiff=0.000e+00，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.2251 ms（qemu_only=True）
- **token**：total=19519（prompt=17116 / completion=2403 / calls=5）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 4；weak-only 3；uncheckable 0）

  最终代码前 20 行（全文见 `../artifacts/real_grid_seed_sizes/daxpy_b4_n16384/generated.c`；初版见 `../artifacts/real_grid_seed_sizes/daxpy_b4_n16384/generated_initial.c`）：

  ```c
  #include <riscv_vector.h>
  typedef long long BLASLONG;
  int daxpy_k(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, double da, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG dummy2)
  {
      if (n <= 0) return 0;
      if (da == 0.0) return 0;
  
      if (inc_x == 1 && inc_y == 1) {
          BLASLONG i = 0;
          size_t vl;
  
          while (i < n) {
              vl = __riscv_vsetvl_e64m8(n - i);
  
              vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
              vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
              vy = __riscv_vfmacc_vf_f64m8(vy, da, vx, vl);
              __riscv_vse64_v_f64m8(y + i, vy, vl);
  
              i += vl;
  ```

### ddot × b3（seed=1，n=1024）

- **initial**：status=`ok`；compile_ok=✅，correct=✅，maxdiff=1.137e-13，qemu_exit=0，kcs=0.25
- **修复回路**：未触发（初始 `ok`，无需/不可修复）
- **status（final）**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=16.51s，model=`qwen-plus-2025-04-28`
- **检索**：6 条策略 `stg:dot:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:dot:split_d` 向量归约累加链；`stg:dot:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:dot:tile_d` 向量寄存器输出分块；`stg:dot:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:dot:reduce_register_pressure_s` 标量归约累加链
- **B3 分析**：无（策略包直注入）
- **prompt**：generate=27842 chars，analysis=0 chars，synthesis=0 chars
- **生成代码**：2856 bytes，sha256=`cdd5d82f1216a243`，存档 `../artifacts/real_grid_seed_sizes/ddot_b3_n1024/generated.c`
- **编译（final）**：✅ ok（2.08s）；正确性：✅，maxdiff=1.137e-13，reldiff=2.234e-16，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.0110 ms（qemu_only=True）
- **token**：total=10963（prompt=9884 / completion=1079 / calls=1）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 6；weak-only 2；uncheckable 2）

  最终代码前 20 行（全文见 `../artifacts/real_grid_seed_sizes/ddot_b3_n1024/generated.c`；初版见 `../artifacts/real_grid_seed_sizes/ddot_b3_n1024/generated_initial.c`）：

  ```c
  typedef long long BLASLONG;
  
  #include <riscv_vector.h>
  double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
  {
      double dot = 0.0;
      if (n <= 0) return dot;
  
      size_t vlmax = __riscv_vsetvlmax_e64m8();
      vfloat64m8_t vsum = __riscv_vfmv_v_f_f64m8(0.0, vlmax);
  
      if (inc_x == 1 && inc_y == 1) {
          /* 单位步长快速路径：vle64 + vfmvvp */
          BLASLONG i = 0;
          for (; i < n; i += vlmax) {
              size_t vl = __riscv_vsetvl_e64m8(n - i);
              vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
              vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
              vsum = __riscv_vfmacc_vv_f64m8(vsum, vx, vy, vl);
          }
  ```

### ddot × b3（seed=1，n=16384）

- **initial**：status=`ok`；compile_ok=✅，correct=✅，maxdiff=0.000e+00，qemu_exit=0，kcs=0.25
- **修复回路**：未触发（初始 `ok`，无需/不可修复）
- **status（final）**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=9.98s，model=`qwen-plus-2025-04-28`
- **检索**：6 条策略 `stg:dot:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:dot:split_d` 向量归约累加链；`stg:dot:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:dot:tile_d` 向量寄存器输出分块；`stg:dot:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:dot:reduce_register_pressure_s` 标量归约累加链
- **B3 分析**：无（策略包直注入）
- **prompt**：generate=27842 chars，analysis=0 chars，synthesis=0 chars
- **生成代码**：1595 bytes，sha256=`b415172adca75cfa`，存档 `../artifacts/real_grid_seed_sizes/ddot_b3_n16384/generated.c`
- **编译（final）**：✅ ok（0.47s）；正确性：✅，maxdiff=0.000e+00，reldiff=0.000e+00，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.1726 ms（qemu_only=True）
- **token**：total=10536（prompt=9884 / completion=652 / calls=1）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 6；weak-only 2；uncheckable 2）

  最终代码前 20 行（全文见 `../artifacts/real_grid_seed_sizes/ddot_b3_n16384/generated.c`；初版见 `../artifacts/real_grid_seed_sizes/ddot_b3_n16384/generated_initial.c`）：

  ```c
  typedef long long BLASLONG;
  
  #include <riscv_vector.h>
  double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
  {
      double dot = 0.0;
      if (n <= 0) return dot;
  
      size_t vlmax = __riscv_vsetvlmax_e64m8();
      vfloat64m8_t vsum = __riscv_vfmv_v_f_f64m8(0.0, vlmax);
  
      if (inc_x == 1 && inc_y == 1) {
          BLASLONG i = 0;
          for (; i < n; ) {
              size_t vl = __riscv_vsetvl_e64m8(n - i);
              vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
              vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
              vsum = __riscv_vfmacc_vv_f64m8(vsum, vx, vy, vl);
              i += vl;
          }
  ```

### ddot × b3（seed=2，n=1024）

- **initial**：status=`ok`；compile_ok=✅，correct=✅，maxdiff=1.137e-13，qemu_exit=0，kcs=0.25
- **修复回路**：未触发（初始 `ok`，无需/不可修复）
- **status（final）**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=7.03s，model=`qwen-plus-2025-04-28`
- **检索**：6 条策略 `stg:dot:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:dot:split_d` 向量归约累加链；`stg:dot:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:dot:tile_d` 向量寄存器输出分块；`stg:dot:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:dot:reduce_register_pressure_s` 标量归约累加链
- **B3 分析**：无（策略包直注入）
- **prompt**：generate=27842 chars，analysis=0 chars，synthesis=0 chars
- **生成代码**：1017 bytes，sha256=`3c995e7c80c2d70b`，存档 `../artifacts/real_grid_seed_sizes/ddot_b3_n1024/generated.c`
- **编译（final）**：✅ ok（0.50s）；正确性：✅，maxdiff=1.137e-13，reldiff=2.234e-16，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.0134 ms（qemu_only=True）
- **token**：total=10292（prompt=9884 / completion=408 / calls=1）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 6；weak-only 2；uncheckable 2）

  最终代码前 20 行（全文见 `../artifacts/real_grid_seed_sizes/ddot_b3_n1024/generated.c`；初版见 `../artifacts/real_grid_seed_sizes/ddot_b3_n1024/generated_initial.c`）：

  ```c
  typedef long long BLASLONG;
  
  #include <riscv_vector.h>
  double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
  {
      double dot = 0.0;
  
      if (n <= 0) return dot;
  
      if (inc_x == 1 && inc_y == 1) {
          BLASLONG i = 0;
          size_t vlmax = __riscv_vsetvlmax_e64m8();
          vfloat64m8_t vsum = __riscv_vfmv_v_f_f64m8(0.0, vlmax);
  
          for (; i < n; i += vlmax) {
              size_t vl = __riscv_vsetvl_e64m8(n - i);
              vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
              vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
              vsum = __riscv_vfmacc_vv_f64m8(vsum, vx, vy, vl);
          }
  ```

### ddot × b3（seed=2，n=16384）

- **initial**：status=`compile_failed`；compile_ok=❌，correct=❌，maxdiff=—，qemu_exit=—，kcs=0.25
- **修复回路**：backend=`cli`，执行 1 轮
  - r1：input=`compile_failed`，error=build_stderr（`../artifacts/real_grid_seed_sizes/ddot_b3_n16384/harness/build.stderr.log`）→ repair exit=0、3.56s、tokens=2836 → status=`ok` correct=✅；fix_summary changed=True chars 2065→545
- **status（final）**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=11.39s，model=`qwen-plus-2025-04-28`
- **检索**：6 条策略 `stg:dot:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:dot:split_d` 向量归约累加链；`stg:dot:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:dot:tile_d` 向量寄存器输出分块；`stg:dot:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:dot:reduce_register_pressure_s` 标量归约累加链
- **B3 分析**：无（策略包直注入）
- **prompt**：generate=27842 chars，analysis=0 chars，synthesis=0 chars
- **生成代码**：545 bytes，sha256=`81c23c9cf1a2ef88`，存档 `../artifacts/real_grid_seed_sizes/ddot_b3_n16384/generated.c`
- **编译（final）**：✅ ok（0.18s）；正确性：✅，maxdiff=1.182e-11，reldiff=1.444e-15，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.1001 ms（qemu_only=True）
- **token**：total=10733（prompt=9884 / completion=849 / calls=1）
- **KCS（启发式 v0）**：0.00（matched 0 / checkable 4；retrieved 6；weak-only 1；uncheckable 2）

  最终代码前 20 行（全文见 `../artifacts/real_grid_seed_sizes/ddot_b3_n16384/generated.c`；初版见 `../artifacts/real_grid_seed_sizes/ddot_b3_n16384/generated_initial.c`）：

  ```c
  typedef long long BLASLONG;
  
  double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
  {
      if (n <= 0) return 0.0;
  
      double dot = 0.0;
  
      if (inc_x == 1 && inc_y == 1) {
          for (BLASLONG i = 0; i < n; i++) {
              dot += x[i] * y[i];
          }
      } else if (inc_y == 1) {
          for (BLASLONG i = 0; i < n; i++) {
              dot += x[i * inc_x] * y[i];
          }
      } else {
          for (BLASLONG i = 0; i < n; i++) {
              dot += x[i * inc_x] * y[i * inc_y];
          }
  ```

### ddot × b3（seed=3，n=1024）

- **initial**：status=`ok`；compile_ok=✅，correct=✅，maxdiff=1.137e-13，qemu_exit=0，kcs=0.25
- **修复回路**：未触发（初始 `ok`，无需/不可修复）
- **status（final）**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=14.23s，model=`qwen-plus-2025-04-28`
- **检索**：6 条策略 `stg:dot:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:dot:split_d` 向量归约累加链；`stg:dot:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:dot:tile_d` 向量寄存器输出分块；`stg:dot:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:dot:reduce_register_pressure_s` 标量归约累加链
- **B3 分析**：无（策略包直注入）
- **prompt**：generate=27842 chars，analysis=0 chars，synthesis=0 chars
- **生成代码**：2178 bytes，sha256=`c5616f65a25d07a5`，存档 `../artifacts/real_grid_seed_sizes/ddot_b3_n1024/generated.c`
- **编译（final）**：✅ ok（0.48s）；正确性：✅，maxdiff=1.137e-13，reldiff=2.234e-16，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.0107 ms（qemu_only=True）
- **token**：total=10715（prompt=9884 / completion=831 / calls=1）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 6；weak-only 2；uncheckable 2）

  最终代码前 20 行（全文见 `../artifacts/real_grid_seed_sizes/ddot_b3_n1024/generated.c`；初版见 `../artifacts/real_grid_seed_sizes/ddot_b3_n1024/generated_initial.c`）：

  ```c
  typedef long long BLASLONG;
  
  #include <riscv_vector.h>
  double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
  {
      if (n <= 0) return 0.0;
  
      double dot = 0.0;
      size_t vlmax = __riscv_vsetvlmax_e64m8();
      vfloat64m8_t vsum = __riscv_vfmv_v_f_f64m8(0.0, vlmax);
  
      if (inc_x == 1 && inc_y == 1) {
          /* 单位步长快速路径：vle64 + vfmvvp */
          BLASLONG i = 0;
          for (; i < n; i += vlmax) {
              size_t vl = __riscv_vsetvl_e64m8(n - i);
              vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
              vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
              vsum = __riscv_vfmacc_vv_f64m8(vsum, vx, vy, vl);
          }
  ```

### ddot × b3（seed=3，n=16384）

- **initial**：status=`compile_failed`；compile_ok=❌，correct=❌，maxdiff=—，qemu_exit=—，kcs=0.25
- **修复回路**：backend=`cli`，执行 1 轮
  - r1：input=`compile_failed`，error=build_stderr（`../artifacts/real_grid_seed_sizes/ddot_b3_n16384/harness/build.stderr.log`）→ repair exit=0、3.43s、tokens=2653 → status=`ok` correct=✅；fix_summary changed=True chars 1879→534
- **status（final）**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=10.59s，model=`qwen-plus-2025-04-28`
- **检索**：6 条策略 `stg:dot:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:dot:split_d` 向量归约累加链；`stg:dot:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:dot:tile_d` 向量寄存器输出分块；`stg:dot:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:dot:reduce_register_pressure_s` 标量归约累加链
- **B3 分析**：无（策略包直注入）
- **prompt**：generate=27842 chars，analysis=0 chars，synthesis=0 chars
- **生成代码**：534 bytes，sha256=`4aa16274c7534686`，存档 `../artifacts/real_grid_seed_sizes/ddot_b3_n16384/generated.c`
- **编译（final）**：✅ ok（0.17s）；正确性：✅，maxdiff=1.182e-11，reldiff=1.444e-15，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.0972 ms（qemu_only=True）
- **token**：total=10667（prompt=9884 / completion=783 / calls=1）
- **KCS（启发式 v0）**：0.00（matched 0 / checkable 4；retrieved 6；weak-only 1；uncheckable 2）

  最终代码前 20 行（全文见 `../artifacts/real_grid_seed_sizes/ddot_b3_n16384/generated.c`；初版见 `../artifacts/real_grid_seed_sizes/ddot_b3_n16384/generated_initial.c`）：

  ```c
  typedef long long BLASLONG;
  
  double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
  {
      if (n <= 0) return 0.0;
  
      double dot = 0.0;
      BLASLONG i;
  
      if (inc_x == 1 && inc_y == 1) {
          for (i = 0; i < n; i++) {
              dot += x[i] * y[i];
          }
      } else if (inc_y == 1) {
          for (i = 0; i < n; i++) {
              dot += x[i * inc_x] * y[i];
          }
      } else {
          for (i = 0; i < n; i++) {
              dot += x[i * inc_x] * y[i * inc_y];
  ```

### ddot × b4（seed=1，n=1024）

- **initial**：status=`wrong_answer`；compile_ok=✅，correct=❌，maxdiff=5.090e+02，qemu_exit=2，kcs=0.25
- **修复回路**：backend=`cli`，执行 1 轮
  - r1：input=`wrong_answer`，error=qemu_stdout（`../artifacts/real_grid_seed_sizes/ddot_b4_n1024/harness/qemu.stdout.log`）→ repair exit=0、10.82s、tokens=2529 → status=`compile_failed` correct=❌；fix_summary changed=True chars 1415→1389；error：compile_failed： 24 | vl = __riscv_vsetvl_e64m8(n - i); | | ^~~~~~~~~~~~~~~~~~~~ | /mnt/c/Users/MSI/Desktop/开题/科研产出/experiments/E09_MPSA/../artifacts/real_grid_seed_sizes/ddot_b4_n1024/harness_r1/generated.c:2 …
- **status（final）**：`compile_failed`；error：compile_failed： 24 | vl = __riscv_vsetvl_e64m8(n - i); | | ^~~~~~~~~~~~~~~~~~~~ | /mnt/c/Users/MSI/Desktop/开题/科研产出/experiments/E09_MPSA/../artifacts/real_grid_seed_sizes/ddot_b4_n1024/harness_r1/generated.c:25:13: error: unknown type name ‘vfloat64m8_t’ | 25 | vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i,  …
- **mpsa**：backend=`cli` exit=0 elapsed=54.42s，model=`qwen-plus-2025-04-28`
- **检索**：6 条策略 `stg:dot:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:dot:split_d` 向量归约累加链；`stg:dot:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:dot:tile_d` 向量寄存器输出分块；`stg:dot:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:dot:reduce_register_pressure_s` 标量归约累加链
- **B4 分析**：3 条（applies=3）；综合 criteria={"supportive": 0.9, "complementarity": 0.85, "accuracy": 0.7, "completeness": 0.8}
  - overall：建议组合使用 specialize_d、split_d 和 tail_handle_d 策略，注意尾部处理与主循环边界一致性，控制寄存器压力，确保数值一致性。
- **prompt**：generate=32245 chars，analysis=16061 chars，synthesis=4254 chars
- **生成代码**：1389 bytes，sha256=`9d861177715bb279`，存档 `../artifacts/real_grid_seed_sizes/ddot_b4_n1024/generated.c`
- **编译（final）**：❌ failed（2.11s）；正确性：❌，maxdiff=—，reldiff=—，qemu_exit=None，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=— ms（qemu_only=True）
- **token**：total=22662（prompt=20299 / completion=2363 / calls=5）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 6；weak-only 2；uncheckable 2）

  最终代码前 20 行（全文见 `../artifacts/real_grid_seed_sizes/ddot_b4_n1024/generated.c`；初版见 `../artifacts/real_grid_seed_sizes/ddot_b4_n1024/generated_initial.c`）：

  ```c
  typedef long long BLASLONG;
  
  double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
  {
      if (n <= 0) return 0.0;
  
      double dot = 0.0;
  
      if (inc_x == 1 && inc_y == 1) {
          BLASLONG i = 0;
          size_t vlmax = __riscv_vsetvlmax_e64m8();
          vfloat64m8_t vsum = __riscv_vfmv_v_f_f64m8(0.0, vlmax);
  
          for (; i <= n - vlmax; i += vlmax) {
              vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vlmax);
              vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vlmax);
              vsum = __riscv_vfmacc_vv_f64m8(vsum, vx, vy, vlmax);
          }
  
          vfloat64m1_t vred = __riscv_vfredusum_vs_f64m8_f64m1(vsum, __riscv_vfmv_v_f_f64m1(0.0, 1), vlmax);
  ```

### ddot × b4（seed=1，n=16384）

- **initial**：status=`ok`；compile_ok=✅，correct=✅，maxdiff=0.000e+00，qemu_exit=0，kcs=0.25
- **修复回路**：未触发（初始 `ok`，无需/不可修复）
- **status（final）**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=43.98s，model=`qwen-plus-2025-04-28`
- **检索**：6 条策略 `stg:dot:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:dot:split_d` 向量归约累加链；`stg:dot:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:dot:tile_d` 向量寄存器输出分块；`stg:dot:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:dot:reduce_register_pressure_s` 标量归约累加链
- **B4 分析**：3 条（applies=3）；综合 criteria={"supportive": 0.9, "complementarity": 0.85, "accuracy": 0.7, "completeness": 0.8}
  - overall：建议组合策略以提升向量利用率，但需注意尾部处理和归约顺序带来的数值一致性风险，并补充任意步长的标量实现路径。
- **prompt**：generate=31639 chars，analysis=16061 chars，synthesis=3750 chars
- **生成代码**：1098 bytes，sha256=`7ef4d640ba32edfe`，存档 `../artifacts/real_grid_seed_sizes/ddot_b4_n16384/generated.c`
- **编译（final）**：✅ ok（2.12s）；正确性：✅，maxdiff=0.000e+00，reldiff=0.000e+00，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.1746 ms（qemu_only=True）
- **token**：total=21757（prompt=19818 / completion=1939 / calls=5）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 6；weak-only 2；uncheckable 2）

  最终代码前 20 行（全文见 `../artifacts/real_grid_seed_sizes/ddot_b4_n16384/generated.c`；初版见 `../artifacts/real_grid_seed_sizes/ddot_b4_n16384/generated_initial.c`）：

  ```c
  typedef long long BLASLONG;
  
  #include <riscv_vector.h>
  double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
  {
      double dot = 0.0;
  
      if (n <= 0) return dot;
  
      if (inc_x == 1 && inc_y == 1) {
          BLASLONG i = 0;
          size_t vlmax = __riscv_vsetvlmax_e64m8();
          vfloat64m8_t vsum = __riscv_vfmv_v_f_f64m8(0.0, vlmax);
  
          for (; i <= n - vlmax; i += vlmax) {
              size_t vl = __riscv_vsetvl_e64m8(n - i);
              vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
              vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
              vsum = __riscv_vfmacc_vv_f64m8(vsum, vx, vy, vl);
          }
  ```

### ddot × b4（seed=2，n=1024）

- **initial**：status=`ok`；compile_ok=✅，correct=✅，maxdiff=1.137e-13，qemu_exit=0，kcs=0.25
- **修复回路**：未触发（初始 `ok`，无需/不可修复）
- **status（final）**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=47.32s，model=`qwen-plus-2025-04-28`
- **检索**：6 条策略 `stg:dot:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:dot:split_d` 向量归约累加链；`stg:dot:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:dot:tile_d` 向量寄存器输出分块；`stg:dot:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:dot:reduce_register_pressure_s` 标量归约累加链
- **B4 分析**：3 条（applies=3）；综合 criteria={"supportive": 0.7, "complementarity": 0.85, "accuracy": 0.65, "completeness": 0.8}
  - overall：策略组合在向量化、归约拆分和尾部处理上形成互补，主循环优化较为完整，但需注意寄存器压力、归约顺序一致性与边界条件处理
- **prompt**：generate=32054 chars，analysis=16061 chars，synthesis=4006 chars
- **生成代码**：1054 bytes，sha256=`1559b176c76b79c1`，存档 `../artifacts/real_grid_seed_sizes/ddot_b4_n1024/generated.c`
- **编译（final）**：✅ ok（2.30s）；正确性：✅，maxdiff=1.137e-13，reldiff=2.234e-16，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.0105 ms（qemu_only=True）
- **token**：total=22159（prompt=20074 / completion=2085 / calls=5）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 6；weak-only 1；uncheckable 2）

  最终代码前 20 行（全文见 `../artifacts/real_grid_seed_sizes/ddot_b4_n1024/generated.c`；初版见 `../artifacts/real_grid_seed_sizes/ddot_b4_n1024/generated_initial.c`）：

  ```c
  typedef long long BLASLONG;
  
  #include <riscv_vector.h>
  double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
  {
      double dot = 0.0;
  
      if (n <= 0) return dot;
  
      if (inc_x == 1 && inc_y == 1) {
          BLASLONG i = 0;
          size_t vlmax = __riscv_vsetvlmax_e64m8();
          vfloat64m8_t vsum = __riscv_vfmv_v_f_f64m8(0.0, vlmax);
  
          for (; i <= n - vlmax; i += vlmax) {
              vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vlmax);
              vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vlmax);
              vsum = __riscv_vfmacc_vv_f64m8(vsum, vx, vy, vlmax);
          }
  ```

### ddot × b4（seed=2，n=16384）

- **initial**：status=`ok`；compile_ok=✅，correct=✅，maxdiff=0.000e+00，qemu_exit=0，kcs=0.25
- **修复回路**：未触发（初始 `ok`，无需/不可修复）
- **status（final）**：`ok`
- **mpsa**：backend=`cli` exit=4 elapsed=18.55s，model=`qwen-plus-2025-04-28`
- **检索**：6 条策略 `stg:dot:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:dot:split_d` 向量归约累加链；`stg:dot:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:dot:tile_d` 向量寄存器输出分块；`stg:dot:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:dot:reduce_register_pressure_s` 标量归约累加链
- **B4 分析**：3 条（applies=3）；综合 criteria={"supportive": 0.9, "complementarity": 0.85, "accuracy": 0.7, "completeness": 0.8}
  - overall：建议组合策略以提升向量利用率，但需注意尾部处理和归约顺序带来的数值一致性风险，并补充任意步长的标量实现路径。
- **prompt**：generate=31639 chars，analysis=16061 chars，synthesis=3750 chars
- **生成代码**：1098 bytes，sha256=`7ef4d640ba32edfe`，存档 `../artifacts/real_grid_seed_sizes/ddot_b4_n16384/generated.c`
- **编译（final）**：✅ ok（2.19s）；正确性：✅，maxdiff=0.000e+00，reldiff=0.000e+00，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.1746 ms（qemu_only=True）
- **token**：total=21757（prompt=19818 / completion=1939 / calls=5）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 6；weak-only 2；uncheckable 2）

  最终代码前 20 行（全文见 `../artifacts/real_grid_seed_sizes/ddot_b4_n16384/generated.c`；初版见 `../artifacts/real_grid_seed_sizes/ddot_b4_n16384/generated_initial.c`）：

  ```c
  typedef long long BLASLONG;
  
  #include <riscv_vector.h>
  double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
  {
      double dot = 0.0;
  
      if (n <= 0) return dot;
  
      if (inc_x == 1 && inc_y == 1) {
          BLASLONG i = 0;
          size_t vlmax = __riscv_vsetvlmax_e64m8();
          vfloat64m8_t vsum = __riscv_vfmv_v_f_f64m8(0.0, vlmax);
  
          for (; i <= n - vlmax; i += vlmax) {
              size_t vl = __riscv_vsetvl_e64m8(n - i);
              vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
              vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
              vsum = __riscv_vfmacc_vv_f64m8(vsum, vx, vy, vl);
          }
  ```

### ddot × b4（seed=3，n=1024）

- **initial**：status=`compile_failed`；compile_ok=❌，correct=❌，maxdiff=—，qemu_exit=—，kcs=0.25
- **修复回路**：backend=`cli`，执行 1 轮
  - r1：input=`compile_failed`，error=build_stderr（`../artifacts/real_grid_seed_sizes/ddot_b4_n1024/harness/build.stderr.log`）→ repair exit=0、3.29s、tokens=2764 → status=`ok` correct=✅；fix_summary changed=True chars 1991→443
- **status（final）**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=52.15s，model=`qwen-plus-2025-04-28`
- **检索**：6 条策略 `stg:dot:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:dot:split_d` 向量归约累加链；`stg:dot:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:dot:tile_d` 向量寄存器输出分块；`stg:dot:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:dot:reduce_register_pressure_s` 标量归约累加链
- **B4 分析**：3 条（applies=3）；综合 criteria={"supportive": 0.8, "complementarity": 0.9, "accuracy": 0.7, "completeness": 0.8}
  - overall：三策略组合具备良好互补性和中等偏上支持度，适合用于实现高性能 ddot 算子，但需注意寄存器压力、尾部处理和数值一致性风险
- **prompt**：generate=31867 chars，analysis=16061 chars，synthesis=3880 chars
- **生成代码**：443 bytes，sha256=`648cc45210e3082a`，存档 `../artifacts/real_grid_seed_sizes/ddot_b4_n1024/generated.c`
- **编译（final）**：✅ ok（1.82s）；正确性：✅，maxdiff=5.684e-13，reldiff=1.117e-15，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.0061 ms（qemu_only=True）
- **token**：total=22493（prompt=20037 / completion=2456 / calls=5）
- **KCS（启发式 v0）**：0.00（matched 0 / checkable 4；retrieved 6；weak-only 1；uncheckable 2）

  最终代码前 20 行（全文见 `../artifacts/real_grid_seed_sizes/ddot_b4_n1024/generated.c`；初版见 `../artifacts/real_grid_seed_sizes/ddot_b4_n1024/generated_initial.c`）：

  ```c
  typedef long long BLASLONG;
  
  double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
  {
      if (n <= 0) return 0.0;
  
      double dot = 0.0;
  
      if (inc_x == 1 && inc_y == 1) {
          BLASLONG i = 0;
          for (; i < n; ++i) {
              dot += x[i] * y[i];
          }
      } else {
          BLASLONG i = 0;
          for (; i < n; ++i) {
              dot += x[i * inc_x] * y[i * inc_y];
          }
      }
  ```

### ddot × b4（seed=3，n=16384）

- **initial**：status=`ok`；compile_ok=✅，correct=✅，maxdiff=0.000e+00，qemu_exit=0，kcs=0.25
- **修复回路**：未触发（初始 `ok`，无需/不可修复）
- **status（final）**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=47.09s，model=`qwen-plus-2025-04-28`
- **检索**：6 条策略 `stg:dot:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:dot:split_d` 向量归约累加链；`stg:dot:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:dot:tile_d` 向量寄存器输出分块；`stg:dot:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:dot:reduce_register_pressure_s` 标量归约累加链
- **B4 分析**：3 条（applies=3）；综合 criteria={"supportive": 0.9, "complementarity": 0.95, "accuracy": 0.7, "completeness": 0.9}
  - overall：三策略组合具备高互补性与一致性，可形成完整实现流程，适用于 k1_rvv256 上的大规模 double 点积计算。
- **prompt**：generate=31956 chars，analysis=16061 chars，synthesis=4011 chars
- **生成代码**：1185 bytes，sha256=`d9d8ccfba64352e8`，存档 `../artifacts/real_grid_seed_sizes/ddot_b4_n16384/generated.c`
- **编译（final）**：✅ ok（2.21s）；正确性：✅，maxdiff=0.000e+00，reldiff=0.000e+00，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.1702 ms（qemu_only=True）
- **token**：total=22122（prompt=20039 / completion=2083 / calls=5）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 6；weak-only 1；uncheckable 2）

  最终代码前 20 行（全文见 `../artifacts/real_grid_seed_sizes/ddot_b4_n16384/generated.c`；初版见 `../artifacts/real_grid_seed_sizes/ddot_b4_n16384/generated_initial.c`）：

  ```c
  typedef long long BLASLONG;
  
  #include <riscv_vector.h>
  double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
  {
      double dot = 0.0;
  
      if (n <= 0) return dot;
  
      // 主路径：单位步长向量加速
      if (inc_x == 1 && inc_y == 1) {
          BLASLONG i = 0;
          size_t vlmax = __riscv_vsetvlmax_e64m8();
          vfloat64m8_t vsum = __riscv_vfmv_v_f_f64m8(0.0, vlmax);
  
          for (; i <= n - vlmax; i += vlmax) {
              vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vlmax);
              vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vlmax);
              vsum = __riscv_vfmacc_vv_f64m8(vsum, vx, vy, vlmax);
          }
  ```

## 5. 结论与下一步

- 本次为真实 API 运行：可直接进入 B3 vs B4 的配对比较（同 op/seed）；注意 token 成本已计入。
- QEMU 只作正确性前置；性能（S_comp/EGC）须在 K1 真机按 `08b` 协议另测。
- 失败单元按 `status` 分类处理：`compile_failed`/`wrong_answer` 应进入 E7 失败分析。
- **修复回路**：初始 FCR 66.7% → 最终 95.8%；触发 8 cells（8 轮），修复后转正 7，repair tokens=22860。
- 产物目录：`../artifacts/real_grid_seed_sizes/`；流水：`results.jsonl`。
