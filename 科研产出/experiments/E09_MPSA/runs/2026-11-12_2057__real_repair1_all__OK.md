# E09 MPSA run `real_repair1_all`

> 生成时间：2026-11-12T20:57:55+08:00 ｜ 实验：E09_MPSA ｜ provider：`dashscope` ｜ profile：`k1_rvv256`

## 1. 配置

| 项 | 值 |
|---|---|
| run_id | `real_repair1_all` |
| provider / model | `dashscope` / `qwen-plus-2025-04-28` |
| 条件 | b3（B3 · StrategyPack 直注入 → 单次生成）, b4（B4 · 多视角分析 + 综合 → 生成（MPSA）） |
| 算子 | daxpy, ddot |
| seeds | 1 |
| sizes / reps | 16384 / 10 |
| mpsa backend | cli |
| mpsa impl | kgagent `1.5.0` / mpsa.py sha256 `48badce182d4fb6e` / cli.py `8e6a3b596ac5c44e` |
| repair | rounds=1 / backend=auto |
| WSL / CC | `Ubuntu` / `riscv64-linux-gnu-gcc` |
| CFLAGS | `-O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -static` |
| QEMU | `qemu-riscv64` `-cpu rv64,v=true,vlen=256,zba=true,zbb=true,zbc=true,zbs=true` |

## 2. 命令

mpsa（示例，实际每 cell 的完整命令见 `results.jsonl.mpsa_cmd`）：

```powershell
C:\Users\MSI\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe -m kgagent.cli mpsa --op daxpy --profile k1_rvv256 --code C:\Users\MSI\Desktop\开题\科研产出\experiments\E09_MPSA\tasks\daxpy\input\generic.c --mode b3 --provider dashscope --out C:\Users\MSI\Desktop\开题\科研产出\experiments\E09_MPSA\../artifacts/real_repair1_all/daxpy_b3 --top-k 6
```

编译 + QEMU（`experiments/harness/qemu.py` 生成，产物在 `harness/`）：

```bash
riscv64-linux-gnu-gcc -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -static ... -o bin
qemu-riscv64 -cpu rv64,v=true,vlen=256,zba=true,zbb=true,zbc=true,zbs=true ./bin 16384 10
```

## 3. 结果总览

| op | 条件 | size | initial | repair_rounds | compile_ok | correct | maxdiff | qemu_exit | gen_tokens | repair_tokens | mpsa_s | KCS | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| daxpy | b3 | 16384 | `ok` | 0 | ✅ | ✅ | 0.00e+00 | 0 | 7194 | 0 | 14.71 | 0.25 | `ok` |
| daxpy | b4 | 16384 | `ok` | 0 | ✅ | ✅ | 0.00e+00 | 0 | 19414 | 0 | 48.896 | 0.25 | `ok` |
| ddot | b3 | 16384 | `ok` | 0 | ✅ | ✅ | 0.00e+00 | 0 | 10407 | 0 | 10.38 | 0.25 | `ok` |
| ddot | b4 | 16384 | `ok` | 0 | ✅ | ✅ | 0.00e+00 | 0 | 22116 | 0 | 46.709 | 0.25 | `ok` |

汇总：**cells=4，ok=4，FCR_final=100.0%（initial 100.0%）**，编译通过率 final 100.0%（initial 100.0%），总 gen tokens=59131（LLM calls=12），KCS 均值 0.25（n=4）
修复回路：**0 cells 触发修复，修复后转正 0，修复 tokens=0**（初始失败类型见第 4 节）。

> `ok`/FCR 只表示“编译通过 + 数值正确”，**不含性能含义**；QEMU 的 `best=` 毫秒数不得进性能表（见 `08b` §5.1）。

## 4. 逐 cell 详情

### daxpy × b3（seed=1，n=16384）

- **initial**：status=`ok`；compile_ok=✅，correct=✅，maxdiff=0.000e+00，qemu_exit=0，kcs=0.25
- **修复回路**：未触发（初始 `ok`，无需/不可修复）
- **status（final）**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=14.71s，model=`qwen-plus-2025-04-28`
- **检索**：4 条策略 `stg:axpy:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:axpy:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:axpy:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:axpy:specialize_s` 任意步长 inc 累加访问
- **B3 分析**：无（策略包直注入）
- **prompt**：generate=17536 chars，analysis=0 chars，synthesis=0 chars
- **生成代码**：1745 bytes，sha256=`642fe21b38e907b7`，存档 `../artifacts/real_repair1_all/daxpy_b3/generated.c`
- **编译（final）**：✅ ok（2.06s）；正确性：✅，maxdiff=0.000e+00，reldiff=0.000e+00，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.2202 ms（qemu_only=True）
- **token**：total=7194（prompt=6502 / completion=692 / calls=1）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 4；weak-only 3；uncheckable 0）

  最终代码前 20 行（全文见 `../artifacts/real_repair1_all/daxpy_b3/generated.c`；初版见 `../artifacts/real_repair1_all/daxpy_b3/generated_initial.c`）：

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
              vfloat64m8_t vres = __riscv_vfmacc_vf_f64m8(vy, da, vx, vl);
              __riscv_vse64_v_f64m8(y + i, vres, vl);
  
              i += vl;
          }
  ```

### daxpy × b4（seed=1，n=16384）

- **initial**：status=`ok`；compile_ok=✅，correct=✅，maxdiff=0.000e+00，qemu_exit=0，kcs=0.25
- **修复回路**：未触发（初始 `ok`，无需/不可修复）
- **status（final）**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=48.90s，model=`qwen-plus-2025-04-28`
- **检索**：4 条策略 `stg:axpy:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:axpy:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:axpy:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:axpy:specialize_s` 任意步长 inc 累加访问
- **B4 分析**：3 条（applies=3）；综合 criteria={"supportive": 0.85, "complementarity": 0.9, "accuracy": 0.7, "completeness": 0.8}
  - overall：三策略组合具备良好一致性与互补性，可实现完整向量化与尾部处理，但需注意别名处理与边界数值一致性。
- **prompt**：generate=21549 chars，analysis=16810 chars，synthesis=3986 chars
- **生成代码**：1626 bytes，sha256=`01bab333d1714d63`，存档 `../artifacts/real_repair1_all/daxpy_b4/generated.c`
- **编译（final）**：✅ ok（2.09s）；正确性：✅，maxdiff=0.000e+00，reldiff=0.000e+00，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.2205 ms（qemu_only=True）
- **token**：total=19414（prompt=17124 / completion=2290 / calls=5）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 4；weak-only 3；uncheckable 0）

  最终代码前 20 行（全文见 `../artifacts/real_repair1_all/daxpy_b4/generated.c`；初版见 `../artifacts/real_repair1_all/daxpy_b4/generated_initial.c`）：

  ```c
  typedef long long BLASLONG;
  
  #include <riscv_vector.h>
  int daxpy_k(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, double da, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG dummy2)
  {
      if (n <= 0) return 0;
      if (da == 0.0) return 0;
  
      if (inc_x == 1 && inc_y == 1)
      {
          BLASLONG i = 0;
          size_t vl;
  
          while (i < n)
          {
              vl = __riscv_vsetvl_e64m8(n - i);
  
              vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
              vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
              vy = __riscv_vfmacc_vf_f64m8(vy, da, vx, vl);
  ```

### ddot × b3（seed=1，n=16384）

- **initial**：status=`ok`；compile_ok=✅，correct=✅，maxdiff=0.000e+00，qemu_exit=0，kcs=0.25
- **修复回路**：未触发（初始 `ok`，无需/不可修复）
- **status（final）**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=10.38s，model=`qwen-plus-2025-04-28`
- **检索**：6 条策略 `stg:dot:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:dot:split_d` 向量归约累加链；`stg:dot:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:dot:tile_d` 向量寄存器输出分块；`stg:dot:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:dot:reduce_register_pressure_s` 标量归约累加链
- **B3 分析**：无（策略包直注入）
- **prompt**：generate=27842 chars，analysis=0 chars，synthesis=0 chars
- **生成代码**：1153 bytes，sha256=`c73904233fbee8b3`，存档 `../artifacts/real_repair1_all/ddot_b3/generated.c`
- **编译（final）**：✅ ok（0.51s）；正确性：✅，maxdiff=0.000e+00，reldiff=0.000e+00，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.1711 ms（qemu_only=True）
- **token**：total=10407（prompt=9884 / completion=523 / calls=1）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 6；weak-only 2；uncheckable 2）

  最终代码前 20 行（全文见 `../artifacts/real_repair1_all/ddot_b3/generated.c`；初版见 `../artifacts/real_repair1_all/ddot_b3/generated_initial.c`）：

  ```c
  typedef long long BLASLONG;
  
  #include <riscv_vector.h>
  double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
  {
   BLASLONG i=0;
   BLASLONG ix=0,iy=0;
  
   if (n <= 0) return(0.0);
  
   if (inc_x == 1 && inc_y == 1) {
    size_t vlmax = __riscv_vsetvlmax_e64m8();
    vfloat64m8_t vsum = __riscv_vfmv_v_f_f64m8(0.0, vlmax);
    BLASLONG remaining = n;
  
    while(remaining >= vlmax) {
     size_t vl = __riscv_vsetvl_e64m8(vlmax);
     vfloat64m8_t vx = __riscv_vle64_v_f64m8(x, vl);
     vfloat64m8_t vy = __riscv_vle64_v_f64m8(y, vl);
     vsum = __riscv_vfmacc_vv_f64m8(vsum, vx, vy, vl);
  ```

### ddot × b4（seed=1，n=16384）

- **initial**：status=`ok`；compile_ok=✅，correct=✅，maxdiff=0.000e+00，qemu_exit=0，kcs=0.25
- **修复回路**：未触发（初始 `ok`，无需/不可修复）
- **status（final）**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=46.71s，model=`qwen-plus-2025-04-28`
- **检索**：6 条策略 `stg:dot:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:dot:split_d` 向量归约累加链；`stg:dot:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:dot:tile_d` 向量寄存器输出分块；`stg:dot:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:dot:reduce_register_pressure_s` 标量归约累加链
- **B4 分析**：3 条（applies=3）；综合 criteria={"supportive": 0.85, "complementarity": 0.9, "accuracy": 0.7, "completeness": 0.75}
  - overall：三个策略可组合，覆盖主循环向量化与尾部处理，但需注意归约顺序、尾部对齐和别名问题，策略参数需根据硬件调整
- **prompt**：generate=31923 chars，analysis=16061 chars，synthesis=3940 chars
- **生成代码**：1215 bytes，sha256=`7a3c2ffea573e622`，存档 `../artifacts/real_repair1_all/ddot_b4/generated.c`
- **编译（final）**：✅ ok（2.07s）；正确性：✅，maxdiff=0.000e+00，reldiff=0.000e+00，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.1665 ms（qemu_only=True）
- **token**：total=22116（prompt=20011 / completion=2105 / calls=5）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 6；weak-only 1；uncheckable 2）

  最终代码前 20 行（全文见 `../artifacts/real_repair1_all/ddot_b4/generated.c`；初版见 `../artifacts/real_repair1_all/ddot_b4/generated_initial.c`）：

  ```c
  #include <riscv_vector.h>
  typedef long long BLASLONG;
  double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
  {
      if (n <= 0) return 0.0;
  
      double dot = 0.0;
  
      if (inc_x == 1 && inc_y == 1) {
          /* 单位步长快速路径 */
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
- **修复回路**：初始 FCR 100.0% → 最终 100.0%；触发 0 cells（0 轮），修复后转正 0，repair tokens=0。
- 产物目录：`../artifacts/real_repair1_all/`；流水：`results.jsonl`。
