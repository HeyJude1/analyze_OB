# E09 MPSA run `real_smoke2_ddot`

> 生成时间：2026-11-12T20:47:01+08:00 ｜ 实验：E09_MPSA ｜ provider：`dashscope` ｜ profile：`k1_rvv256`

## 1. 配置

| 项 | 值 |
|---|---|
| run_id | `real_smoke2_ddot` |
| provider / model | `dashscope` / `qwen-plus-2025-04-28` |
| 条件 | b3（B3 · StrategyPack 直注入 → 单次生成）, b4（B4 · 多视角分析 + 综合 → 生成（MPSA）） |
| 算子 | ddot |
| seeds | 1 |
| sizes / reps | 16384 / 10 |
| mpsa backend | cli |
| WSL / CC | `Ubuntu` / `riscv64-linux-gnu-gcc` |
| CFLAGS | `-O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -static` |
| QEMU | `qemu-riscv64` `-cpu rv64,v=true,vlen=256,zba=true,zbb=true,zbc=true,zbs=true` |

## 2. 命令

mpsa（示例，实际每 cell 的完整命令见 `results.jsonl.mpsa_cmd`）：

```powershell
C:\Users\MSI\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe -m kgagent.cli mpsa --op ddot --profile k1_rvv256 --code C:\Users\MSI\Desktop\开题\科研产出\experiments\E09_MPSA\tasks\ddot\input\generic.c --mode b3 --provider dashscope --out C:\Users\MSI\Desktop\开题\科研产出\experiments\E09_MPSA\../artifacts/real_smoke2_ddot/ddot_b3 --top-k 6
```

编译 + QEMU（`experiments/harness/qemu.py` 生成，产物在 `harness/`）：

```bash
riscv64-linux-gnu-gcc -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -static ... -o bin
qemu-riscv64 -cpu rv64,v=true,vlen=256,zba=true,zbb=true,zbc=true,zbs=true ./bin 16384 10
```

## 3. 结果总览

| op | 条件 | size | compile_ok | correct | maxdiff | qemu_exit | tokens | mpsa_s | KCS | status |
|---|---|---|---|---|---|---|---|---|---|---|
| ddot | b3 | 16384 | ✅ | ✅ | 0.00e+00 | 0 | 10791 | 17.773 | 0.25 | `ok` |
| ddot | b4 | 16384 | ✅ | ✅ | 0.00e+00 | 0 | 22069 | 48.286 | 0.25 | `ok` |

汇总：**cells=2，ok=2，FCR=100.0%**，编译通过率 100.0%，总 tokens=32860（LLM calls=6），KCS 均值 0.25（n=2）

> `ok`/FCR 只表示“编译通过 + 数值正确”，**不含性能含义**；QEMU 的 `best=` 毫秒数不得进性能表（见 `08b` §5.1）。

## 4. 逐 cell 详情

### ddot × b3（seed=1，n=16384）

- **status**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=17.77s，model=`qwen-plus-2025-04-28`
- **检索**：6 条策略 `stg:dot:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:dot:split_d` 向量归约累加链；`stg:dot:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:dot:tile_d` 向量寄存器输出分块；`stg:dot:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:dot:reduce_register_pressure_s` 标量归约累加链
- **B3 分析**：无（策略包直注入）
- **prompt**：generate=27611 chars，analysis=0 chars，synthesis=0 chars
- **生成代码**：2783 bytes，sha256=`c1d965c1e815490d`，存档 `../artifacts/real_smoke2_ddot/ddot_b3/generated.c`
- **编译**：✅ ok（2.13s）；正确性：✅，maxdiff=0.000e+00，reldiff=0.000e+00，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.1760 ms（qemu_only=True）
- **token**：total=10791（prompt=9733 / completion=1058 / calls=1）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 6；weak-only 1；uncheckable 2）

  生成代码前 20 行（全文见 `../artifacts/real_smoke2_ddot/ddot_b3/generated.c`）：

  ```c
  #include <riscv_vector.h>
  #include <stddef.h>
  #include <stdio.h>
  #include <stdlib.h>
  typedef long long BLASLONG;
  double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
  {
      BLASLONG i = 0;
      double dot = 0.0;
  
      if (n <= 0) return dot;
  
      size_t vlmax = __riscv_vsetvlmax_e64m8();
      vfloat64m8_t vsum = __riscv_vfmv_v_f_f64m8(0.0, vlmax);
  
      if (inc_x == 1 && inc_y == 1) {
          /* 单位步长快速路径：vle64 + vfmul + vfredusum */
          for (; i <= n - vlmax; i += vlmax) {
              size_t vl = __riscv_vsetvl_e64m8(vlmax);
              vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
  ```

### ddot × b4（seed=1，n=16384）

- **status**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=48.29s，model=`qwen-plus-2025-04-28`
- **检索**：6 条策略 `stg:dot:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:dot:split_d` 向量归约累加链；`stg:dot:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:dot:tile_d` 向量寄存器输出分块；`stg:dot:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:dot:reduce_register_pressure_s` 标量归约累加链
- **B4 分析**：3 条（applies=3）；综合 criteria={"supportive": 0.85, "complementarity": 0.9, "accuracy": 0.7, "completeness": 0.85}
  - overall：三策略组合可形成主循环向量化+归约优化+尾部掩码处理的完整实现，但需注意数值一致性和边界处理细节。
- **prompt**：generate=31704 chars，analysis=16061 chars，synthesis=3888 chars
- **生成代码**：1434 bytes，sha256=`b2b17250401cb0fb`，存档 `../artifacts/real_smoke2_ddot/ddot_b4/generated.c`
- **编译**：✅ ok（2.07s）；正确性：✅，maxdiff=0.000e+00，reldiff=0.000e+00，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.1711 ms（qemu_only=True）
- **token**：total=22069（prompt=19852 / completion=2217 / calls=5）
- **KCS（启发式 v0）**：0.25（matched 1 / checkable 4；retrieved 6；weak-only 2；uncheckable 2）

  生成代码前 20 行（全文见 `../artifacts/real_smoke2_ddot/ddot_b4/generated.c`）：

  ```c
  typedef long long BLASLONG;
  
  #include <riscv_vector.h>
  double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
  {
      if (n <= 0) return 0.0;
  
      double dot = 0.0;
  
      if (inc_x == 1 && inc_y == 1) {
          /* 单位步长快速路径：vle64 + vfmacc + vfredusum */
          size_t vlmax = __riscv_vsetvlmax_e64m8();
          vfloat64m8_t vsum = __riscv_vfmv_v_f_f64m8(0.0, vlmax);
  
          BLASLONG i = 0;
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
- 产物目录：`../artifacts/real_smoke2_ddot/`；流水：`results.jsonl`。
