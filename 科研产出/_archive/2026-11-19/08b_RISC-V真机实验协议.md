# 08b RISC-V 真机实验协议（SpacemiT K1 / Banana Pi F3 主线，精简执行版）

> v1.0｜2026-11-12｜riscv-experimenter｜依据：[`00_论文总纲与两部分表述.md`](00_论文总纲与两部分表述.md)、`04` §1/§4–§7、[`22_知识图谱构建完整指南.md`](22_知识图谱构建完整指南.md)，以及本机对用户工具链的实测（附录 A）
> 证据分层：**✅已核实**（本机实测/官网）｜**◐待核实**（需板上确认）｜**✗推断**（估算）
> 本文只规定"RISC-V 真机上怎么测、能说什么"；指标定义与统计以 `04` 为准。

---

## 0. 一句话结论

真机把跨架构实验从"只能报正确性"升级为"能报加速比"，而且成本很低：**用户语料里已有 55 MB 的 OpenBLAS RISC-V 预编译静态库（TARGET=RISCV64_ZVL256B，✅已实测可链接），P1 基线不需要编 OpenBLAS**；45 算子 × 3 条件 × 3 尺寸的整套矩阵，交叉编译约 1 h（单算子实测 46 s）、板上跑 15–30 min。唯一必须先做的：**第 1 天 30 分钟确认板子真是 K1 类（VLEN=256、有 cpufreq）**，然后冻结计时协议——现有 `bench.h` 用 `CLOCK_REALTIME`、`benchmark_common.h` 用 `clock()`，都不能直接用于论文数字。

---

## 1. 第 1 天：30 分钟环境确认

### 1.1 必确认 10 项与 K1 期望值

| # | 项 | 命令 | K1 期望 ✅ | 不符怎么办 |
|---|---|---|---|---|
| 1 | SoC/板卡 | `tr -d '\0' < /proc/device-tree/model` | `SpacemiT K1` / `BPI-F3` / `spacemit,x60` | 走 §1.4 判定树 |
| 2 | 核数/簇 | `nproc`；`lscpu` | 8 核，2 簇×4（cpu0–3 / cpu4–7） | 记录实际值 |
| 3 | ISA | `grep -m1 '^isa' /proc/cpuinfo` | 含 `v`、`zba/zbb/zbc/zbs` | 无 `v` → 只能标量档 |
| 4 | **VLEN** | 见 1.3 的 `rvv_vlen_probe` | **256 bit**（VLMAX e64m1=4） | 128 → `zvl128b` 且必须重编 OpenBLAS |
| 5 | 缓存 | `lscpu -C` 或 `/sys/.../cache/index*/size` | L1d 32 KB/核、L2 512 KB/簇、**无 L3** | 取不到用默认值并标◐ |
| 6 | 频率档位 | `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies` | `614400 819200 1000000 1228800 1600000` | 无 sysfs → §3.4 的降级 |
| 7 | 调频策略 | `cat .../cpu0/cpufreq/scaling_governor` | `schedutil`/`performance` | — |
| 8 | 温度 | `cat /sys/class/thermal/thermal_zone*/temp` | 空闲 40–55 ℃，满载 <85 ℃ | 无 thermal → 看频率掉落 |
| 9 | 板载 gcc | `gcc --version` | `13.x`（Bianbu 上常见）◐ | 没有也不影响：全程交叉编译 |
| 10 | OpenBLAS | `ldconfig -p \| grep -i openblas`；`dpkg -l \| grep -i openblas` | 多半**没有** ◐ | 有就作 P1b，没有用预编译 `.a`（§3.1） |

### 1.2 一键确认脚本 `board_probe.sh`（复制到板上即跑）

```bash
#!/bin/bash
# bash board_probe.sh | tee board_report_$(date +%F).txt
set -u
hr(){ printf '\n===== %s =====\n' "$1"; }
hr "型号/环境"; date; uname -a
tr -d '\0' < /proc/device-tree/model 2>/dev/null; echo
cat /etc/os-release 2>/dev/null | head -4
hr "CPU"; nproc; command -v lscpu >/dev/null && lscpu | head -20
grep -m4 -E '^(isa|uarch|mmu)' /proc/cpuinfo
for d in /sys/devices/system/cpu/cpu0/cache/index*; do
  [ -d "$d" ] && echo "L$(cat $d/level) $(cat $d/type) $(cat $d/size) shared=$(cat $d/shared_cpu_list)"
done
hr "VLEN"; [ -x ./rvv_vlen_probe ] && ./rvv_vlen_probe || echo "缺 rvv_vlen_probe（本文件 1.3 提供）"
hr "编译器"; command -v gcc && gcc --version | head -1
command -v gcc >/dev/null && { echo 'int main(){return 0;}' >/tmp/t.c
  gcc -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d /tmp/t.c -o /tmp/t 2>/dev/null \
    && echo "zvl256b 选项: OK" || echo "zvl256b: NO（退 rv64gcv）"; }
hr "频率/调频"; ls /sys/devices/system/cpu/cpu0/cpufreq/ 2>/dev/null || echo "无 cpufreq"
for f in scaling_driver scaling_governor scaling_available_frequencies scaling_cur_freq; do
  p=/sys/devices/system/cpu/cpu0/cpufreq/$f; [ -f "$p" ] && echo "$f = $(cat $p)"
done
command -v cpupower >/dev/null && cpupower frequency-info | head -12 || echo "no cpupower"
hr "温度"; for z in /sys/class/thermal/thermal_zone*; do
  [ -d "$z" ] && echo "$z $(cat $z/type 2>/dev/null) $(cat $z/temp 2>/dev/null)"; done
hr "内存/OpenBLAS"; free -h
ldconfig -p 2>/dev/null | grep -i openblas || echo "ldconfig 无 openblas"
dpkg -l 2>/dev/null | grep -i openblas || echo "dpkg 无 openblas"
ls /usr/lib/*/libopenblas* /usr/local/lib/libopenblas* 2>/dev/null
hr "工具/网络"; for c in taskset time perf tmux python3; do
  printf '%-8s %s\n' $c "$(command -v $c || echo MISSING)"; done
sudo -n true 2>/dev/null && echo "sudo 免密 OK" || echo "sudo 需密码"
ip -br a 2>/dev/null | head -4
hr "判读"; echo "K1: VLEN=256, 8 核, 档位含 1600000；QEMU 特征见 §1.3"
```

### 1.3 VLEN 运行时探测（`/proc/cpuinfo` 看不到 VLEN，必须跑一小段代码）

```c
/* vlen_probe.c —— 交叉编译：$CC -O3 -march=rv64gcv_zvl256b -mabi=lp64d -static vlen_probe.c -o rvv_vlen_probe */
#include <stdio.h>
int main(void) {
    unsigned long vl;
    __asm__ volatile("vsetvli %0, zero, e8, m1, ta, ma" : "=r"(vl)); /* vl = VLEN/8 */
    printf("VLEN=%lu bits  VLMAX(e64,m1)=%lu  VLMAX(e64,m4)=%lu\n", vl*8, vl/8, vl/8*4);
    return 0;
}
```

判读：`VLEN=256` 即 K1 类；若同时跑一个 naive dgemm 256³ **超过 500 ms**，说明在 QEMU/极低频（真机 ✗推断 5–60 ms）。

### 1.4 判定树（只分三支，不铺开）

```text
有 'v' + VLEN=256         → A：按本协议主线（zvl256b）
有 'v' + VLEN=128         → B：改 -march=rv64gcv_zvl128b，OpenBLAS TARGET=RISCV64_ZVL128B 重编；仓库预编译库不能直接用
无 'v'                    → C：RVV 路线不成立；RISC-V 只做正确性回归，性能结论全归 x86（同 04 原方案）
```

---

## 2. 交叉编译 + 部署 + 运行

### 2.1 工具链（注意真名，别照抄脚本）

用户 `_command.txt` 里是 `riscv64-linux-gcc`（服务器软链），**本机这份 Bootlin 工具链真名是 `riscv64-buildroot-linux-gnu-gcc`**（gcc 14.3.0 / glibc 2.41 / binutils 2.43.1 ✅），同目录也有 `riscv64-linux-gcc` 副本，两个都能用。

```bash
# WSL2 / Linux 服务器
export RISCV=/mnt/c/Users/MSI/Desktop/cx_operator_tool/operator_extract/toolchain/riscv-gcc-14.3/bin
export CC=$RISCV/riscv64-buildroot-linux-gnu-gcc
export OBJ=$RISCV/riscv64-buildroot-linux-gnu-objdump
export CORPUS=/mnt/c/Users/MSI/Desktop/cx_operator_tool/operator_extract/openblas_newtest1
```

### 2.2 三个编译口径（C0 / C1 / 候选）

```bash
FLAGS_C0="-O2 -march=rv64imafdc -mabi=lp64d -static"                       # 输入代码原样，无向量
FLAGS_C1="-O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d \
          -ffast-math -static -mrvv-vector-bits=zvl"                       # 编译器自动向量化基线

# 例：dgemm 驱动（沿用既有 _dgemm_dgc.c + 提取的 kernel + 预编译库）
cd $CORPUS/16_dgemm_op5
$CC -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -ffast-math \
    -Wno-error=incompatible-pointer-types -w -DSHOW_TIME -DDOUBLE \
    ../_file/benchmark/_dgemm_dgc.c dgemm_z.c dgemm_nn.c dgemm_nn_rvv.c dgemm_k.c dgemm_k_rvv.c \
    dgemm_beta.c dgemm_otcopy.c dgemm_oncopy.c dgemm_beta_rvv.c dgemm_otcopy_rvv.c dgemm_oncopy_rvv.c \
    ../_file/lib/libopenblas_riscv64_zvl256bp_rvv-r0.3.27.dev.a \
    -I../_file/include -static -lc -lm -lpthread -o /tmp/dgemm_rvv.out
```
✅ **本机实测**：该命令编译通过，耗时 **46 s**，产物 **941 KB** 静态 ELF（未 strip），`objdump -d | grep -c vsetvl` = **56**。

### 2.3 必须做的 RVV 断言（防止 flags 写错无人发现）

```bash
# C1：循环被向量化。✅实测输出 vsetvli / vle64.v / vfmacc.vv
$CC $FLAGS_C1 -c candidate_daxpy.c -o /tmp/c1.o
$OBJ -d /tmp/c1.o | grep -E 'vsetvl|vfmacc|vle64|vse64'
#   e:  0d8577d7  vsetvli a5,a0,e64,m1,ta,ma
#  24:  b22190d7  vfmacc.vv v1,v3,v2
# C0：必须 0 条向量指令。✅实测 0
$CC $FLAGS_C0 -c candidate_daxpy.c -o /tmp/c0.o
echo "C0 vec instr = $($OBJ -d /tmp/c0.o | grep -cE 'vsetvl|vfmacc|vfadd|vle64|vse64')"
# C1 加 -mrvv-vector-bits=zvl 后：GCC 报 "loop vectorized using 32 byte vectors"（=256 bit）✅实测
```

> **别用 `zvl1024b`**：`make_rvv_risc64.sh` 第 34 行 `TARGET=RISCV64_ZVL1024B` 与文件头注释（ZVL256B）、`_command.txt` 真实命令（`zvl256b`）三处不一致；OpenBLAS `TargetList.txt` 里**没有** ZVL1024B（✅已核实），K1 是 256 bit。用 `zvl256b`。

### 2.4 部署与运行

```bash
# 传（宿主）：45 算子 × 3 条件 ≈ 130 MB，千兆 2 分钟
rsync -az ~/rvbench/bin/ user@192.168.1.50:~/rvbench/bin/
# 或 scp -r bin/ user@<ip>:~/rvbench/
# 没网：串口 115200（picocom -b 115200 /dev/ttyUSB0），或拔 SD 卡拷进 rootfs

# 板上：静默 + 绑核（簇0=cpu0-3，簇1=cpu4-7）
sudo systemctl stop bluetooth cups avahi-daemon 2>/dev/null; sudo swapoff -a 2>/dev/null
export OPENBLAS_NUM_THREADS=1            # 预编译库是 pthread 版，不设会默认开 8 线程！
taskset -c 2 ./bin/dgemm_c1 64 128 512
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 taskset -c 0-3 ./bin/dgemm_c1 512   # 4T 口径
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 taskset -c 0-7 ./bin/dgemm_c1 512   # 8T 口径

# 锁频（能锁就锁，论文要写这一句）
sudo cpupower frequency-set -g userspace && sudo cpupower frequency-set -f 1.60GHz
# 无 cpupower 时用 sysfs：
for c in /sys/devices/system/cpu/cpu[0-9]*/cpufreq; do
  echo userspace | sudo tee $c/scaling_governor >/dev/null
  echo 1600000   | sudo tee $c/scaling_setspeed >/dev/null 2>&1 || \
  echo 1600000   | sudo tee $c/scaling_max_freq $c/scaling_min_freq >/dev/null
done
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq   # 期望 1600000

# 可选：perf（K1 的 PMU 驱动可用性◐待核实；不可用就用 /usr/bin/time + CLOCK_MONOTONIC_RAW）
perf stat -e cycles,instructions,task-clock taskset -c 2 ./bin/dgemm_c1 512
```

**降频监控（K1 无风扇，必做）**：后台 `while :; do date +%s; cat /sys/class/thermal/thermal_zone0/temp; cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq; sleep 5; done > thermal.csv`。若温度 >85 ℃ 或频率掉到 1.2288 GHz 以下，该次运行作废，加风扇/降档重测。

### 2.5 C 计时骨架（`clock_gettime(CLOCK_MONOTONIC_RAW)` + 5 warmup + 20 次中位数）

```c
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#define WARMUP 5
#define REPS 20
#define MAXREP 64
#define RSD_LIMIT 3.0

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);       /* 不用 REALTIME、不用 clock() */
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
static void sink(const void *p, size_t n) {        /* 防 -O3 把循环删掉 */
    __asm__ __volatile__("" : : "r"(p), "r"(n) : "memory");
}
static int cmpd(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y);
}
static double median(double *v, int n) {
    double t[MAXREP]; memcpy(t, v, sizeof(double) * n);
    qsort(t, n, sizeof(double), cmpd);
    return (n & 1) ? t[n / 2] : 0.5 * (t[n / 2 - 1] + t[n / 2]);
}
static double rsd(double *v, int n, double med) {
    double s = 0; for (int i = 0; i < n; i++) s += (v[i]-med)*(v[i]-med);
    s = sqrt(s / (n > 1 ? n - 1 : 1));
    return med > 0 ? 100.0 * s / med : 1e9;
}
/* 一次 (op,size,variant) 测量：每轮重填输入 → 跑调用 → 计时，RSD>3% 重测≤3 次 */
double time_one(void (*refill)(void), void (*run)(void), void *out, size_t outb,
                double *out_rsd, int *out_ok) {
    double t[MAXREP]; int reps = REPS;
    for (int attempt = 0; attempt < 3; attempt++) {
        for (int i = 0; i < WARMUP; i++) { refill(); sink(out, outb); run(); }
        for (int i = 0; i < reps; i++) {
            refill(); sink(out, outb);
            double t0 = now_sec(); run(); t[i] = (now_sec() - t0) * 1e3;
        }
        double med = median(t, reps), r = rsd(t, reps, med);
        *out_rsd = r; *out_ok = (r <= RSD_LIMIT);
        if (*out_ok) return med;
    }
    *out_ok = 0; return median(t, reps);            /* 不可靠：标 ok=0，不进主表 */
}
```

使用方式：每个算子写 `refill()`（`memcpy(工作区, pristine, bytes)`）与 `run()`（调用候选实现），P1 的 `run` 换成调用 OpenBLAS 符号（`dgemm_`/`daxpy_`，从静态库链接）。输出 JSONL：`{"op","variant","size","median_ms","rsd_pct","gflops","maxdiff","nrep","ok"}`；正确性每 (op,size) 都做一次（与 P1 逐元素比，容差按 `04` §6.1 自校准：先用"参考 vs 参考"测 η0，容差取 10×η0）。

---

## 3. 基线 C0 / C1 / P1 在板上怎么拿

| 编号 | 定义 | 怎么得到 | 备注 |
|---|---|---|---|
| **C0** | 输入通用 C 原样，`-O2 -march=rv64imafdc`，1T | 用 §2.2 的 `FLAGS_C0` 编同一份源码，`objdump` 断言 0 向量指令 | 定义"没优化"；`S=1.0` 参照 |
| **C1** | 同一份源码 `-O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -ffast-math`，1T | 同上换 `FLAGS_C1`；`-fopt-info-vec` 存档 | **最关键下界**：回答"是不是只赢在 O3/自动向量化" |
| **P1** | OpenBLAS RVV 专家实现（性能上界分母） | **首选**：直接链接仓库预编译库 `libopenblas_riscv64_zvl256bp_rvv-r0.3.27.dev.a`（55 MB，TARGET=ZVL256B，✅实测可链接，**无需编译**）；**次选**：板上 `libopenblas-dev`（版本可能不同，记 `openblas_get_config()`）；**兜底**：源码交叉编译（下） | 同一次运行同时得到 P1（Ref 列）与候选，别拆两套 |
| P1-gen（可选） | OpenBLAS **标量**专家实现 | 链接 `libopenblas_riscv64_generic-r0.3.27.dev.a`（40 MB） | **做策略归因**：P1-gen→P1 的差 = 纯向量化收益；C1→P1-gen = 分块/算法收益 |

**源码交叉编译 OpenBLAS（只在找不到预编译库/需要 VLEN=128 时做）**：

```bash
git clone --depth 1 -b v0.3.27 https://github.com/OpenMathLib/OpenBLAS.git
cd OpenBLAS
make -j$(nproc) TARGET=RISCV64_ZVL256B BINARY=64 CROSS=1 \
  CC=$CC AR=$RISCV/riscv64-buildroot-linux-gnu-ar RANLIB=$RISCV/riscv64-buildroot-linux-gnu-ranlib \
  HOSTCC=gcc NOFORTRAN=1 NO_LAPACK=1 USE_THREAD=1 NUM_THREADS=8 NO_SHARED=1 \
  COMMON_OPT="-O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -ffast-math"
ls -lh libopenblas_riscv64_zvl256b*.a      # 期望 ~50–60 MB 静态库
```
时间估算：**BLAS-only（`NOFORTRAN=1 NO_LAPACK=1`）在 8–16 核 x86 上约 15–40 min，4 核笔记本 40–90 min**（✗推断，未实测）；带 LAPACK（需 gfortran）再加 1–2 h。⚠️ 用户 `make_rvv_risc64.sh` 需改三处：`TARGET` 改 `RISCV64_ZVL256B`、`NO_SHARED=1`（板上要静态）、`TOOLCHAIN_PREFIX` 改 `riscv64-buildroot-linux-gnu-`。

---

## 4. 可测性要点（决定哪些数字能进论文）

- **缓存预算（✅K1 规格）**：L1d 32 KB/核 → L2 512 KB/簇（4 核共享）→ **无 L3** → LPDDR4x 32-bit@2666（理论 10.6 GB/s）。因此"L3 档"在 K1 上不存在，用 **DRAM 档**代替：向量类 n∈{1024, 16384, 1048576}；gemv n∈{256, 1024, 4096}；dgemm n∈{64, 128, 512}。
- **稳定可测**：dgemm/dsyrk/dsymm/dtrmm/dtrsm（n≥128）、dgemv/dsymv（n≥256）、daxpy/ddot/dnrm2/daxpy 类（n≥1024）——向量化收益 1.5–4×，且 RSD 易压到 3% 内。
- **不稳定/别做结论**：`dtrsv/dtpsv/dtrmv`（串行依赖、尺寸小）；`idamax/idmax/idmin`（分支多、QEMU 日志里判据也最弱）；`dgesv`（LAPACK 算法差异大）；复数族 `cher/cher2k/chemm`（无 RVV 专家实现，只能看 C1）。既有日志中 **21 个算子无 RVV 版本**（✅`openblas_cxtest/result/result.txt`），这些只能给 C0/C1 对比。
- **VLEN/LMUL**：f64 的 VLMAX = 4×LMUL（m1=4, m2=8, m4=16）。建议加一个 **LMUL/SEW 微矩阵**（3 配置 × 3 算子 × 3 尺寸，约 20 min），量化"可调参数"这一 KG 实体的收益边界，直接支撑 `04` 的 A4/A5 硬件错配消融。
- **STREAM**：官方源码 https://www.cs.virginia.edu/stream/ ，`gcc -O3 -march=rv64gcv_zvl256b -fopenmp -static stream.c`，数组 ≥8×L2（如 N=2M double），1T/8T 各测；把 triad 的 GB/s 作为 roofline 带宽。实测前不要引用厂商标称值。

---

## 5. 风险与降级

| 风险 | 概率 | 影响 | 触发判据 | 降级方案 |
|---|---|---|---|---|
| 板子不在手 / 坏了 | 中 | 高 | 第 1 天拿不到 `board_report` | 性能结论全部留在 x86（`04` 原方案）；RISC-V 只做 §6 的正确性回归；RQ3 降级为"结构性迁移" |
| 散热降频 | **高**（F3 无风扇） | 中 | 温度 >85 ℃ 或 `scaling_cur_freq` <1.6 GHz | 锁 1.2 或 1.0 GHz 重测，论文注明"锁频档"，并加风扇/竖放；同一条件下比较仍成立 |
| OpenBLAS 编不出来 | 低 | 低 | `make` 失败 | **不影响主线**：用仓库预编译 `.a`（✅已验证）；再不行用 `libopenblas_generic.a` 作 P1-gen |
| VLEN 不是 256 | 中 | 中 | `rvv_vlen_probe` ≠ 256 | 128 → `zvl128b` + 重编 OpenBLAS（TARGET=RISCV64_ZVL128B）；无 `v` → 支 C |
| 计时噪声大（RSD>3%） | 中 | 中 | 重测 3 次仍超 | 提高 reps 到 50、加长单次工作集、换 cpu2 核、关中断；仍不行则该点标 `ok=0` 剔除并写明 |
| QEMU 兜底越界 | 低 | **致命（学术诚信）** | 有人想把 QEMU 数字写进性能表 | 见 §5.1 红线 |

### 5.1 能说 / 不能说（写进论文"实验限制"段）

| 能说 | 不能说 |
|---|---|
| "在 SpacemiT K1（8 核、VLEN=256、1.6 GHz 锁频）上，生成实现相对编译器基线 C1 的加速比 S_comp = X" | "RVV 比 x86 快/慢 N 倍"（跨 ISA 绝对值不可比，只能比**同 ISA 内的相对提升**） |
| "相对 OpenBLAS 的差距闭合度 EGC = Y"（作为分母，不作为"超越"声明） | "超过 OpenBLAS"（能量化的只有 EGC/S_lib，不设"打败"目标） |
| "同一策略在 x86/RVV 上的相对增益分布不同，边界是…"（配对比较 ΔGM） | "VLEN=256 上最优的 LMUL 对所有 RVV 芯片都最优"（只能外推到 K1 类） |
| "QEMU 上通过正确性验证的代码，在真机上 FCR = Z"（QEMU 只作正确性**前置**） | "QEMU 时间 / MFlops"（任何性能数字） |
| "既有 54 个 QEMU 日志显示 Ref/Migration 逐尺寸通过率 99.0%（10375 行中 178 行 fail）" | "既有日志的性能加速比"（QEMU 量级无意义） |

> **既有日志的定位（✅Lead 已核实 44/44 含 qemu；本次审计 54 个日志、10375 条记录、10197 pass / 178 fail，fail 全部来自 `16_dgemm_op0`）**：只当**正确性回归**与**失败案例来源**（`03` §6.5 的 `FailureCase`），不进任何性能表；真机必须按 §2–§3 重跑。

---

## 6. RISC-V 侧实验矩阵（6 行）与机时

| # | 任务 | 条件 | 规模 | 产出 | 预计机时 |
|---|---|---|---|---|---|
| **R0** | 环境基线 + roofline | 1T/8T | STREAM + 峰值微基准 + 2 个尺寸 dgemm | `env_report`、BW/Peak | 0.5 h |
| **R1** | 既有日志审计（0 机时） | — | 54 个 `*_log.txt` | 正确性回归表 + 178 条 fail 清单 | 0.2 h（脚本） |
| **R2** | C0 vs C1 主扫描 | 1T，锁频 1.6 GHz | 45 算子 × 3 尺寸 × 5 warmup+20 次 | `t1_median.csv`、S_c1/c0 | 编译 1 h + 跑 15 min |
| **R3** | P1 参照 | 1T/4T/8T | 同 R2（Ref 列） | S_lib、EGC 分母 | 跑 20 min |
| **R4** | 生成代码（B0/B1/B2/B3） | 1T | S2 抽 12 算子 × 2 尺寸 × 4 条件 | 跨架构主表（RQ2/RQ3） | 编译 20 min + 跑 20 min |
| **R5** | 多线程 + LMUL/SEW 微矩阵 | 4T/8T；LMUL∈{1,2,4} | 8 个 L3/L2 算子；3 算子 × 3 尺寸 | 参数收益边界（A4/A5） | 0.5 h |

**合计：约 3.5–4.5 h（含 3 次重复与调试），可在一个晚上 + 一个半天完成。** 与 x86 侧的唯一口径差异是 `-march` 与频率；其余（同源码、同 warmup/reps/中位数、同容差来源、同 1T/4T 定义）必须一致，这样 `S_comp` 才能跨平台配对比较（这正是 RQ3 的做法）。

---

## 7. 待核实与需决策

1. **◐** 板上 cpufreq 驱动名与 `cpupower` 是否存在（Bianbu vendor 内核 vs mainline：spacemit K1 的 cpufreq-dt 支持 2026-03 才进 mainline）→ §1 脚本第 6 项确认。
2. **◐** K1 的向量峰值（每周期几条 256-bit FMA）官方未公开；用 STREAM + dgemm 实测反推，不引用未公开数字。
3. **◐** 板载 gcc 版本（决定能否板上原生编译；不影响交叉编译路线）。
4. **待用户决策**：是否买 5 V 风扇（~20 元，直接决定能不能稳定锁 1.6 GHz）；板子网络是插网线还是 USB gadget。

---

## 附录 A：本机实测记录（2026-11-12，Windows 11 + WSL2 Ubuntu，`/mnt/c` 上编译）

```text
工具链   riscv64-buildroot-linux-gnu-gcc.br_real (Buildroot 2021.11-18033) 14.3.0 / glibc 2.41 / binutils 2.43.1
RVV 编译 -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -ffast-math -static  → rc=0
objdump  vsetvli a5,a5,e64,m2,ta,ma / vfmacc.vf v2,fa0,v4 / vse64.v  （RVV 指令确实生成）
C1 自动向量化  -fopt-info-vec："loop vectorized using 32 byte vectors"（加 -mrvv-vector-bits=zvl）
                不加该选项："loop vectorized using variable length vectors"
C0     -O2 -march=rv64imafdc → 向量指令数 0
dgemm 驱动  编译 46 s，产物 941 KB 静态 ELF，含 56 处 vsetvl
bench 骨架   daxpy 927 KB（28 处 vsetvl）、dgemm 944 KB（35 处）均 rc=0
日志审计     54 个 *_log.txt / 10375 条记录 / 10197 pass / 178 fail（全在 16_dgemm_op0）
             _command.txt 中每目录 qemu-riscv64 出现 2–14 次；dgemm 200³ Ref=899.6 ms/17.79 MFlops → QEMU
QEMU 坑     WSL2 Ubuntu 缺 libnuma.so.1，自带 qemu-riscv64 无法启动 → sudo apt install libnuma1（或 apt install qemu-user）
```

**引用**（可核查）：K1 规格与板卡 https://wiki.banana-pi.org/Banana_Pi_BPI-F3 、https://raw.githubusercontent.com/bit-brick/bit-brick-docs/refs/heads/main/docs/k1/hardware/processor-brief.mdx ；K1 频率 OPP 与 cpufreq 补丁 https://lore.kernel.org/spacemit/20260308-shadow-deps-v1-0-0ceb5c7c07eb@mailbox.org/T/ ；OpenBLAS 目标列表 https://github.com/OpenMathLib/OpenBLAS/blob/develop/TargetList.txt 与 `Makefile.riscv64`；RISC-V GCC 选项 https://gcc.gnu.org/onlinedocs/gcc/RISC-V-Options.html ；STREAM https://www.cs.virginia.edu/stream/ ；工具链出处 https://toolchains.bootlin.com/ 。
