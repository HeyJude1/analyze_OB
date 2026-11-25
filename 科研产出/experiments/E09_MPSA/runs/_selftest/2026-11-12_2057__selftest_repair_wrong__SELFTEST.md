# E09 MPSA run `selftest_repair_wrong`

> 生成时间：2026-11-12T20:57:07+08:00 ｜ 实验：E09_MPSA ｜ provider：`mock` ｜ profile：`k1_rvv256`
>
> ⚠️ mock 冒烟：结果仅验证链路，不代表方法效果

## 1. 配置

| 项 | 值 |
|---|---|
| run_id | `selftest_repair_wrong` |
| provider / model | `mock` / `mock` |
| 条件 | b3（B3 · StrategyPack 直注入 → 单次生成） |
| 算子 | ddot |
| seeds | 72 |
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
C:\Python313\python.exe -m kgagent.cli mpsa --op ddot --profile k1_rvv256 --code C:\Users\MSI\Desktop\开题\科研产出\experiments\E09_MPSA\tasks\ddot\input\generic.c --mode b3 --provider mock --out C:\Users\MSI\Desktop\开题\科研产出\experiments\E09_MPSA\../artifacts/selftest_repair_wrong/ddot_b3 --top-k 6 --mock-code C:\Users\MSI\Desktop\开题\科研产出\experiments\E09_MPSA\selftest\wrong_answer_ddot.c
```

编译 + QEMU（`experiments/harness/qemu.py` 生成，产物在 `harness/`）：

```bash
riscv64-linux-gnu-gcc -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -static ... -o bin
qemu-riscv64 -cpu rv64,v=true,vlen=256,zba=true,zbb=true,zbc=true,zbs=true ./bin 16384 10
```

## 3. 结果总览

| op | 条件 | size | initial | repair_rounds | compile_ok | correct | maxdiff | qemu_exit | gen_tokens | repair_tokens | mpsa_s | KCS | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ddot | b3 | 16384 | `wrong_answer` | 1 | ✅ | ✅ | 1.18e-11 | 0 | 0 | 0 | 0.107 | 0.00 | `ok` |

汇总：**cells=1，ok=1，FCR_final=100.0%（initial 0.0%）**，编译通过率 final 100.0%（initial 100.0%），总 gen tokens=0（LLM calls=1），KCS 均值 0.00（n=1）
修复回路：**1 cells 触发修复，修复后转正 1，修复 tokens=0**（初始失败类型见第 4 节）。

> `ok`/FCR 只表示“编译通过 + 数值正确”，**不含性能含义**；QEMU 的 `best=` 毫秒数不得进性能表（见 `08b` §5.1）。

## 4. 逐 cell 详情

### ddot × b3（seed=72，n=16384）

- **initial**：status=`wrong_answer`；compile_ok=✅，correct=❌，maxdiff=8.188e+03，qemu_exit=2，kcs=0.00
- **修复回路**：backend=`cli`，执行 1 轮
  - r1：input=`wrong_answer`，error=qemu_stdout（`../artifacts/selftest_repair_wrong/ddot_b3/harness/qemu.stdout.log`）→ repair exit=0、0.06s、tokens=0 → status=`ok` correct=✅
- **status（final）**：`ok`
- **mpsa**：backend=`cli` exit=0 elapsed=0.11s，model=`mock`
- **检索**：6 条策略 `stg:dot:specialize_d` LMUL/SEW 向量寄存器分组调优；`stg:dot:split_d` 向量归约累加链；`stg:dot:tail_handle_d` 动态 vl / 掩码尾部处理；`stg:dot:tile_d` 向量寄存器输出分块；`stg:dot:vectorize_d` RVV 向量化内核（vsetvl + vle/vfmacc/vse）；`stg:dot:reduce_register_pressure_s` 标量归约累加链
- **B3 分析**：无（策略包直注入）
- **prompt**：generate=27611 chars，analysis=0 chars，synthesis=0 chars
- **生成代码**：318 bytes，sha256=`29bb91ddcbffa378`，存档 `../artifacts/selftest_repair_wrong/ddot_b3/generated.c`
- **编译（final）**：✅ ok（0.43s）；正确性：✅，maxdiff=1.182e-11，reldiff=1.444e-15，qemu_exit=0，timeout=False
- **QEMU 计时（仅存档，不是性能）**：best=0.0999 ms（qemu_only=True）
- **token**：total=0（prompt=0 / completion=0 / calls=1）
- **KCS（启发式 v0）**：0.00（matched 0 / checkable 4；retrieved 6；weak-only 0；uncheckable 2）

  最终代码前 20 行（全文见 `../artifacts/selftest_repair_wrong/ddot_b3/generated.c`；初版见 `../artifacts/selftest_repair_wrong/ddot_b3/generated_initial.c`）：

  ```c
  typedef long long BLASLONG;
  
  double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
  {
      BLASLONG i;
      BLASLONG ix = 0;
      BLASLONG iy = 0;
      double acc = 0.0;
  
      for (i = 0; i < n; i++) {
          acc += x[ix] * y[iy];
          ix += inc_x;
          iy += inc_y;
      }
      return acc;
  }
  ```

## 5. 结论与下一步

- 本次为 **mock 冒烟**：mock 代码是确定性标量参考实现，FCR/KCS/token **不代表 B3 vs B4 的方法效果**，仅验证链路与记录格式。
- QEMU 只作正确性前置；性能（S_comp/EGC）须在 K1 真机按 `08b` 协议另测。
- 失败单元按 `status` 分类处理：`compile_failed`/`wrong_answer` 应进入 E7 失败分析。
- **修复回路**：初始 FCR 0.0% → 最终 100.0%；触发 1 cells（1 轮），修复后转正 1，repair tokens=0。
- 产物目录：`../artifacts/selftest_repair_wrong/`；流水：`results.jsonl`。
