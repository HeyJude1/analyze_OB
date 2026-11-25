# E09 MPSA run `real_smoke1_daxpy`

> 生成时间：2026-11-12T20:44:20+08:00 ｜ 实验：E09_MPSA ｜ provider：`dashscope` ｜ profile：`k1_rvv256`

## 1. 配置

| 项 | 值 |
|---|---|
| run_id | `real_smoke1_daxpy` |
| provider / model | `dashscope` / `qwen-plus-2025-04-28` |
| 条件 | b3（B3 · StrategyPack 直注入 → 单次生成）, b4（B4 · 多视角分析 + 综合 → 生成（MPSA）） |
| 算子 | daxpy |
| seeds | 1 |
| sizes / reps | 16384 / 10 |
| mpsa backend | cli |
| WSL / CC | `Ubuntu` / `riscv64-linux-gnu-gcc` |
| CFLAGS | `-O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -static` |
| QEMU | `qemu-riscv64` `-cpu rv64,v=true,vlen=256,zba=true,zbb=true,zbc=true,zbs=true` |

## 2. 命令

mpsa（示例，实际每 cell 的完整命令见 `results.jsonl.mpsa_cmd`）：

```powershell
C:\Users\MSI\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe -m kgagent.cli mpsa --op daxpy --profile k1_rvv256 --code C:\Users\MSI\Desktop\开题\科研产出\experiments\E09_MPSA\tasks\daxpy\input\generic.c --mode b3 --provider dashscope --out C:\Users\MSI\Desktop\开题\科研产出\experiments\E09_MPSA\../artifacts/real_smoke1_daxpy/daxpy_b3 --top-k 6
```

编译 + QEMU（`experiments/harness/qemu.py` 生成，产物在 `harness/`）：

```bash
riscv64-linux-gnu-gcc -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -static ... -o bin
qemu-riscv64 -cpu rv64,v=true,vlen=256,zba=true,zbb=true,zbc=true,zbs=true ./bin 16384 10
```

## 3. 结果总览

| op | 条件 | size | compile_ok | correct | maxdiff | qemu_exit | tokens | mpsa_s | KCS | status |
|---|---|---|---|---|---|---|---|---|---|---|
| daxpy | b3 | 16384 | ✅ | ✅ | 0.00e+00 | 0 | 6652 | 12.811 | 0.25 | `ok` |
| daxpy | b4 | 16384 | ✅ | ✅ | 0.00e+00 | 0 | 18726 | 48.473 | 0.25 | `ok` |

汇总：**cells=2，ok=2，FCR=100.0%**，编译通过率 100.0%，总 tokens=25378（LLM calls=6），KCS 均值 0.25（n=2）

> `ok`/FCR 只表示“编译通过 + 数值正确”，**不含性能含义**；QEMU 的 `best=` 毫秒数不得进性能表（见 `08b` §5.1）。

## 4. 逐 cell 详情

### daxpy × b3（seed=1，n=16384）

- **status**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=12.81s，model=`qwen-plus-2025-04-28`
- **检索**：4 条策略 `stg:axpy:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:axpy:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:axpy:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:axpy:specialize_s` 任意步长 inc 累加访问
- **B3 分析**：无（策略包直注入）
- **prompt**：generate=16777 chars，analysis=0 chars，synthesis=0 chars
- **生成代码**：1551 bytes，sha256=`d83e0d4358087fb0`，存档 `../artifacts/real_smoke1_daxpy/daxpy_b3/generated.c`
- **编译**：✅ ok（0.51s）；正确性：✅，maxdiff=0.000e+00，reldiff=0.000e+00，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.2222 ms（qemu_only=True）
- **token**：total=6652（prompt=6031 / completion=621 / calls=1）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 4；weak-only 3；uncheckable 0）

  生成代码前 20 行（全文见 `../artifacts/real_smoke1_daxpy/daxpy_b3/generated.c`）：

  ```c
  #include <riscv_vector.h>
  typedef long long BLASLONG;
  int daxpy_k(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, double da, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG dummy2)
  {
      if (n <= 0 || da == 0.0) return 0;
  
      if (inc_x == 1 && inc_y == 1) {
          BLASLONG i = 0;
          while (i < n) {
              size_t vl = __riscv_vsetvl_e64m8(n - i);
              vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
              vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
              vy = __riscv_vfmacc_vf_f64m8(vy, da, vx, vl);
              __riscv_vse64_v_f64m8(y + i, vy, vl);
              i += vl;
          }
      } else if (inc_y == 1) {
          BLASLONG i = 0;
          while (i < n) {
              size_t vl = __riscv_vsetvl_e64m8(n - i);
  ```

### daxpy × b4（seed=1，n=16384）

- **status**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=48.47s，model=`qwen-plus-2025-04-28`
- **检索**：4 条策略 `stg:axpy:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:axpy:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:axpy:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:axpy:specialize_s` 任意步长 inc 累加访问
- **B4 分析**：3 条（applies=3）；综合 criteria={"supportive": 0.9, "complementarity": 0.85, "accuracy": 0.7, "completeness": 0.8}
  - overall：推荐组合策略可有效覆盖主循环向量化与尾部处理，需注意尾部控制与别名处理以保证数值一致性
- **prompt**：generate=20623 chars，analysis=16810 chars，synthesis=3957 chars
- **生成代码**：1689 bytes，sha256=`a3ce06d1e58c4062`，存档 `../artifacts/real_smoke1_daxpy/daxpy_b4/generated.c`
- **编译**：✅ ok（2.13s）；正确性：✅，maxdiff=0.000e+00，reldiff=0.000e+00，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.3225 ms（qemu_only=True）
- **token**：total=18726（prompt=16518 / completion=2208 / calls=5）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 4；weak-only 3；uncheckable 0）

  生成代码前 20 行（全文见 `../artifacts/real_smoke1_daxpy/daxpy_b4/generated.c`）：

  ```c
  #include <riscv_vector.h>
  typedef long long BLASLONG;
  int daxpy_k(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, double da, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG dummy2)
  {
      if (n <= 0) return 0;
      if (da == 0.0) return 0;
  
      if (inc_x == 1 && inc_y == 1) {
          BLASLONG i = 0;
          const double alpha = da;
          for (; i < n; ) {
              size_t vl = __riscv_vsetvl_e64m8(n - i);
              vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
              vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
              vy = __riscv_vfmacc_vf_f64m8(vy, alpha, vx, vl);
              __riscv_vse64_v_f64m8(y + i, vy, vl);
              i += vl;
          }
      } else if (inc_y == 1) {
          BLASLONG i = 0;
  ```

## 5. 结论与下一步

- 本次为真实 API 运行：可直接进入 B3 vs B4 的配对比较（同 op/seed）；注意 token 成本已计入。
- QEMU 只作正确性前置；性能（S_comp/EGC）须在 K1 真机按 `08b` 协议另测。
- 失败单元按 `status` 分类处理：`compile_failed`/`wrong_answer` 应进入 E7 失败分析。
- 产物目录：`../artifacts/real_smoke1_daxpy/`；流水：`results.jsonl`。
