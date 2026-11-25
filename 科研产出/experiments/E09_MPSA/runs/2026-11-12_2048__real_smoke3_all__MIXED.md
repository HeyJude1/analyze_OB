# E09 MPSA run `real_smoke3_all`

> 生成时间：2026-11-12T20:48:16+08:00 ｜ 实验：E09_MPSA ｜ provider：`dashscope` ｜ profile：`k1_rvv256`

## 1. 配置

| 项 | 值 |
|---|---|
| run_id | `real_smoke3_all` |
| provider / model | `dashscope` / `qwen-plus-2025-04-28` |
| 条件 | b3（B3 · StrategyPack 直注入 → 单次生成）, b4（B4 · 多视角分析 + 综合 → 生成（MPSA）） |
| 算子 | daxpy, ddot |
| seeds | 1 |
| sizes / reps | 16384 / 10 |
| mpsa backend | cli |
| WSL / CC | `Ubuntu` / `riscv64-linux-gnu-gcc` |
| CFLAGS | `-O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -static` |
| QEMU | `qemu-riscv64` `-cpu rv64,v=true,vlen=256,zba=true,zbb=true,zbc=true,zbs=true` |

## 2. 命令

mpsa（示例，实际每 cell 的完整命令见 `results.jsonl.mpsa_cmd`）：

```powershell
C:\Users\MSI\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe -m kgagent.cli mpsa --op daxpy --profile k1_rvv256 --code C:\Users\MSI\Desktop\开题\科研产出\experiments\E09_MPSA\tasks\daxpy\input\generic.c --mode b3 --provider dashscope --out C:\Users\MSI\Desktop\开题\科研产出\experiments\E09_MPSA\../artifacts/real_smoke3_all/daxpy_b3 --top-k 6
```

编译 + QEMU（`experiments/harness/qemu.py` 生成，产物在 `harness/`）：

```bash
riscv64-linux-gnu-gcc -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -static ... -o bin
qemu-riscv64 -cpu rv64,v=true,vlen=256,zba=true,zbb=true,zbc=true,zbs=true ./bin 16384 10
```

## 3. 结果总览

| op | 条件 | size | compile_ok | correct | maxdiff | qemu_exit | tokens | mpsa_s | KCS | status |
|---|---|---|---|---|---|---|---|---|---|---|
| daxpy | b3 | 16384 | ❌ | ❌ | — | None | 7016 | 13.215 | 0.25 | `compile_failed` |
| daxpy | b4 | 16384 | ❌ | ❌ | — | None | 18754 | 45.755 | 0.25 | `compile_failed` |
| ddot | b3 | 16384 | ✅ | ✅ | 0.00e+00 | 0 | 10451 | 15.643 | 0.25 | `ok` |
| ddot | b4 | 16384 | ❌ | ❌ | — | None | 21780 | 47.222 | 0.25 | `compile_failed` |

汇总：**cells=4，ok=1，FCR=25.0%**，编译通过率 25.0%，总 tokens=58001（LLM calls=12），KCS 均值 0.25（n=4）

> `ok`/FCR 只表示“编译通过 + 数值正确”，**不含性能含义**；QEMU 的 `best=` 毫秒数不得进性能表（见 `08b` §5.1）。

## 4. 逐 cell 详情

### daxpy × b3（seed=1，n=16384）

- **status**：`compile_failed`；error：compile_failed： | __riscv_vlse64_v_f64m8 | /mnt/c/Users/MSI/Desktop/开题/科研产出/experiments/E09_MPSA/../artifacts/real_smoke3_all/daxpy_b3/harness/generated.c:26:31: error: incompatible types when initializing type ‘vfloat64m8_t’ using type ‘int’ | /mnt/c/Users/MSI/Desktop/开题/科研产出/experiments/E09_MPSA/runs/real …
- **mpsa**：backend=`cli` exit=0 elapsed=13.21s，model=`qwen-plus-2025-04-28`
- **检索**：4 条策略 `stg:axpy:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:axpy:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:axpy:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:axpy:specialize_s` 任意步长 inc 累加访问
- **B3 分析**：无（策略包直注入）
- **prompt**：generate=17305 chars，analysis=0 chars，synthesis=0 chars
- **生成代码**：1676 bytes，sha256=`509e5f7acef99854`，存档 `../artifacts/real_smoke3_all/daxpy_b3/generated.c`
- **编译**：❌ failed（1.66s）；正确性：❌，maxdiff=—，reldiff=—，qemu_exit=None，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=— ms（qemu_only=True）

  ```text
     26 |             vfloat64m8_t vx = __riscv_vlseg1e64_v_f64m8(x + ix, vl);
        |                               ^~~~~~~~~~~~~~~~~~~~~~~~~
        |                               __riscv_vlse64_v_f64m8
  /mnt/c/Users/MSI/Desktop/开题/科研产出/experiments/E09_MPSA/../artifacts/real_smoke3_all/daxpy_b3/harness/generated.c:26:31: error: incompatible types when initializing type ‘vfloat64m8_t’ using type ‘int’
  /mnt/c/Users/MSI/Desktop/开题/科研产出/experiments/E09_MPSA/../artifacts/real_smoke3_all/daxpy_b3/harness/generated.c:41:31: error: incompatible types when initializing type ‘vfloat64m8_t’ using type ‘int’
     41 |             vfloat64m8_t vx = __riscv_vlseg1e64_v_f64m8(x + ix, vl);
        |                               ^~~~~~~~~~~~~~~~~~~~~~~~~
  /mnt/c/Users/MSI/Desktop/开题/科研产出/experiments/E09_MPSA/../artifacts/real_smoke3_all/daxpy_b3/harness/generated.c:42:31: error: incompatible types when initializing type ‘vfloat64m8_t’ using type ‘int’
     42 |             vfloat64m8_t vy = __riscv_vlseg1e64_v_f64m8(y + iy, vl);
        |                               ^~~~~~~~~~~~~~~~~~~~~~~~~
  ```
- **token**：total=7016（prompt=6351 / completion=665 / calls=1）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 4；weak-only 3；uncheckable 0）

  生成代码前 20 行（全文见 `../artifacts/real_smoke3_all/daxpy_b3/generated.c`）：

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
      } else if (inc_y == 1) {
          BLASLONG i = 0;
  ```

### daxpy × b4（seed=1，n=16384）

- **status**：`compile_failed`；error：compile_failed： | int | /usr/lib/gcc-cross/riscv64-linux-gnu/13/include/riscv_vector.h:94:9: note: expected ‘vfloat64m8_t’ but argument is of type ‘int’ | 94 | #pragma riscv intrinsic "vector" | | ^~~~~ | /mnt/c/Users/MSI/Desktop/开题/科研产出/experiments/E09_MPSA/../artifacts/real_smoke3_all/daxpy_b4/harness/gen …
- **mpsa**：backend=`cli` exit=0 elapsed=45.76s，model=`qwen-plus-2025-04-28`
- **检索**：4 条策略 `stg:axpy:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:axpy:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:axpy:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:axpy:specialize_s` 任意步长 inc 累加访问
- **B4 分析**：3 条（applies=3）；综合 criteria={"supportive": 0.9, "complementarity": 0.85, "accuracy": 0.7, "completeness": 0.8}
  - overall：三个策略可组合应用，覆盖主循环向量化、尾部处理和寄存器参数调优，但需特别注意尾部处理和内存别名问题
- **prompt**：generate=21154 chars，analysis=16810 chars，synthesis=3975 chars
- **生成代码**：1260 bytes，sha256=`fdcb088d84eef6ff`，存档 `../artifacts/real_smoke3_all/daxpy_b4/generated.c`
- **编译**：❌ failed（1.75s）；正确性：❌，maxdiff=—，reldiff=—，qemu_exit=None，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=— ms（qemu_only=True）

  ```text
        |                                                                                    ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        |                                                                                    |
        |                                                                                    int
  /usr/lib/gcc-cross/riscv64-linux-gnu/13/include/riscv_vector.h:94:9: note: expected ‘vfloat64m8_t’ but argument is of type ‘int’
     94 | #pragma riscv intrinsic "vector"
        |         ^~~~~
  /mnt/c/Users/MSI/Desktop/开题/科研产出/experiments/E09_MPSA/../artifacts/real_smoke3_all/daxpy_b4/harness/generated.c:26:13: warning: implicit declaration of function ‘__riscv_vsxei64’; did you mean ‘__riscv_vsm_v_b4’? [-Wimplicit-function-declaration]
     26 |             __riscv_vsxei64(y, iy, vacc, vl);
        |             ^~~~~~~~~~~~~~~
        |             __riscv_vsm_v_b4
  ```
- **token**：total=18754（prompt=16769 / completion=1985 / calls=5）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 4；weak-only 3；uncheckable 0）

  生成代码前 20 行（全文见 `../artifacts/real_smoke3_all/daxpy_b4/generated.c`）：

  ```c
  typedef long long BLASLONG;
  
  #include <riscv_vector.h>
  int daxpy_k(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, double da, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG dummy2)
  {
      if (n <= 0) return 0;
      if (da == 0.0) return 0;
  
      BLASLONG i = 0;
      const size_t vlmax = __riscv_vsetvlmax_e64m8();
      vfloat64m8_t vacc;
  
      // Fast path for unit increments
      if (inc_x == 1 && inc_y == 1) {
          for (; i <= n - vlmax; i += vlmax) {
              size_t vl = __riscv_vsetvl_e64m8(n - i);
              vacc = __riscv_vfmacc_vf_f64m8(__riscv_vle64_v_f64m8(y + i, vl), da, __riscv_vle64_v_f64m8(x + i, vl), vl);
              __riscv_vse64(y + i, vacc, vl);
          }
      } else {
  ```

### ddot × b3（seed=1，n=16384）

- **status**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=15.64s，model=`qwen-plus-2025-04-28`
- **检索**：6 条策略 `stg:dot:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:dot:split_d` 向量归约累加链；`stg:dot:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:dot:tile_d` 向量寄存器输出分块；`stg:dot:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:dot:reduce_register_pressure_s` 标量归约累加链
- **B3 分析**：无（策略包直注入）
- **prompt**：generate=27611 chars，analysis=0 chars，synthesis=0 chars
- **生成代码**：1780 bytes，sha256=`0a3d037f33d2c648`，存档 `../artifacts/real_smoke3_all/ddot_b3/generated.c`
- **编译**：✅ ok（2.10s）；正确性：✅，maxdiff=0.000e+00，reldiff=0.000e+00，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.1768 ms（qemu_only=True）
- **token**：total=10451（prompt=9733 / completion=718 / calls=1）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 6；weak-only 2；uncheckable 2）

  生成代码前 20 行（全文见 `../artifacts/real_smoke3_all/ddot_b3/generated.c`）：

  ```c
  #include <riscv_vector.h>
  #include <stddef.h>
  typedef long long BLASLONG;
  double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
  {
      if (n <= 0) return 0.0;
  
      double dot = 0.0;
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

### ddot × b4（seed=1，n=16384）

- **status**：`compile_failed`；error：compile_failed：/mnt/c/Users/MSI/Desktop/开题/科研产出/experiments/E09_MPSA/../artifacts/real_smoke3_all/ddot_b4/harness/generated.c: In function ‘ddot_k’: | /mnt/c/Users/MSI/Desktop/开题/科研产出/experiments/E09_MPSA/../artifacts/real_smoke3_all/ddot_b4/harness/generated.c:23:29: warning: implicit declaration of function ‘__ri …
- **mpsa**：backend=`cli` exit=0 elapsed=47.22s，model=`qwen-plus-2025-04-28`
- **检索**：6 条策略 `stg:dot:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:dot:split_d` 向量归约累加链；`stg:dot:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:dot:tile_d` 向量寄存器输出分块；`stg:dot:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:dot:reduce_register_pressure_s` 标量归约累加链
- **B4 分析**：3 条（applies=3）；综合 criteria={"supportive": 0.8, "complementarity": 0.9, "accuracy": 0.65, "completeness": 0.85}
  - overall：三策略组合后可覆盖主循环向量化、归约链优化与尾部处理，策略参数一致且互补性强，适合k1_rvv256平台下的ddot实现，但需注意数值一致性与边界条件处理。
- **prompt**：generate=31557 chars，analysis=16061 chars，synthesis=3720 chars
- **生成代码**：1241 bytes，sha256=`e9690ecbd6f6e83a`，存档 `../artifacts/real_smoke3_all/ddot_b4/generated.c`
- **编译**：❌ failed（0.81s）；正确性：❌，maxdiff=—，reldiff=—，qemu_exit=None，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=— ms（qemu_only=True）

  ```text
  /mnt/c/Users/MSI/Desktop/开题/科研产出/experiments/E09_MPSA/../artifacts/real_smoke3_all/ddot_b4/harness/generated.c: In function ‘ddot_k’:
  /mnt/c/Users/MSI/Desktop/开题/科研产出/experiments/E09_MPSA/../artifacts/real_smoke3_all/ddot_b4/harness/generated.c:23:29: warning: implicit declaration of function ‘__riscv_vfredusum_vs_f64m8_f64’; did you mean ‘__riscv_vfredusum_vs_f64m8_f64m1’? [-Wimplicit-function-declaration]
     23 |         vfloat64m1_t vred = __riscv_vfredusum_vs_f64m8_f64(vsum, __riscv_vfmv_v_f_f64m1(0.0, 1), vlmax);
        |                             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        |                             __riscv_vfredusum_vs_f64m8_f64m1
  /mnt/c/Users/MSI/Desktop/开题/科研产出/experiments/E09_MPSA/../artifacts/real_smoke3_all/ddot_b4/harness/generated.c:23:29: error: incompatible types when initializing type ‘vfloat64m1_t’ using type ‘int’
  ```
- **token**：total=21780（prompt=19710 / completion=2070 / calls=5）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 6；weak-only 1；uncheckable 2）

  生成代码前 20 行（全文见 `../artifacts/real_smoke3_all/ddot_b4/generated.c`）：

  ```c
  #include <riscv_vector.h>
  typedef long long BLASLONG;
  double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
  {
      if (n <= 0) return 0.0;
  
      double dot = 0.0;
      size_t vlmax = __riscv_vsetvlmax_e64m8();
  
      if (inc_x == 1 && inc_y == 1) {
          /* 单位步长快速路径：vle64 + vdot */
          BLASLONG i = 0;
          vfloat64m8_t vsum = __riscv_vfmv_v_f_f64m8(0.0, vlmax);
  
          for (; i <= n - vlmax; i += vlmax) {
              size_t vl = __riscv_vsetvl_e64m8(vlmax);
              vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
              vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
              vsum = __riscv_vfmacc_vv_f64m8(vsum, vx, vy, vl);
          }
  ```

## 5. 结论与下一步

- 本次为真实 API 运行：可直接进入 B3 vs B4 的配对比较（同 op/seed）；注意 token 成本已计入。
- QEMU 只作正确性前置；性能（S_comp/EGC）须在 K1 真机按 `08b` 协议另测。
- 失败单元按 `status` 分类处理：`compile_failed`/`wrong_answer` 应进入 E7 失败分析。
- 产物目录：`../artifacts/real_smoke3_all/`；流水：`results.jsonl`。
