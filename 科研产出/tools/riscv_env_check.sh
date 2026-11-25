#!/usr/bin/env bash
# RISC-V 真机环境确认脚本（在板子上运行，约 30 秒）
# 用法：scp 到板子后 `bash riscv_env_check.sh`，把输出保存为 riscv_env_<date>.txt
# 目的：确认板卡型号、VLEN、GCC、OpenBLAS、频率与绑核能力，决定实验口径。

set -u
echo "===== 1. 基本信息 ====="
date
uname -a
echo "--- /proc/cpuinfo ---"
cat /proc/cpuinfo | head -60
echo "--- /proc/meminfo ---"
grep -E "MemTotal|MemAvailable" /proc/meminfo

echo
echo "===== 2. 工具链 ====="
which gcc || true
gcc --version 2>&1 | head -3
which g++ || true
which python3 || true
python3 --version 2>&1 || true
which taskset || true
which cpupower || true
which perf || true

echo
echo "===== 3. 编译器 RVV 支持 ====="
echo 'int main(){return 0;}' > /tmp/t.c
if gcc -O2 -march=rv64gcv -mabi=lp64d /tmp/t.c -o /tmp/t 2>/tmp/t.err; then
  echo "OK: gcc 支持 -march=rv64gcv"
else
  echo "FAIL: gcc 不支持 rv64gcv"; cat /tmp/t.err
fi

echo
echo "===== 4. VLEN 检测（读 vlenb CSR，单位 bit）====="
cat > /tmp/vlen.c <<'EOF'
#include <stdio.h>
int main(void) {
    unsigned long vlenb = 0;
    asm volatile("csrr %0, vlenb" : "=r"(vlenb));
    printf("vlenb = %lu bytes, VLEN = %lu bits\n", vlenb, vlenb * 8);
    return 0;
}
EOF
if gcc -O2 -march=rv64gcv -mabi=lp64d /tmp/vlen.c -o /tmp/vlen 2>/tmp/vlen.err; then
  /tmp/vlen || echo "运行失败（可能 CSR 不可读）"
else
  echo "VLEN 检测编译失败（可能不是 RVV 1.0 工具链）"; cat /tmp/vlen.err
fi

echo
echo "===== 5. OpenBLAS 是否已安装 ====="
ldconfig -p 2>/dev/null | grep -i openblas || echo "ldconfig 未找到 openblas（可能未安装）"
find /usr /opt /usr/local -maxdepth 4 -name "libopenblas*" 2>/dev/null | head -10
python3 - <<'EOF' 2>/dev/null || true
try:
    import numpy as np
    print("numpy:", np.__version__)
    np.show_config()
except Exception as e:
    print("numpy 未安装或导入失败:", e)
EOF

echo
echo "===== 6. 频率与核心 ====="
lscpu 2>/dev/null | head -30 || true
for f in /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq \
         /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor \
         /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq; do
  [ -f "$f" ] && echo "$f = $(cat $f)"
done
echo "可用核心: $(nproc)"

echo
echo "===== 7. 一次 OpenBLAS dgemm 冒烟测试（若 OpenBLAS 可用）====="
cat > /tmp/dgemm_bench.c <<'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>
static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
int main(int argc, char** argv) {
    int n = argc > 1 ? atoi(argv[1]) : 256;
    double *A = malloc(sizeof(double)*n*n), *B = malloc(sizeof(double)*n*n), *C = malloc(sizeof(double)*n*n);
    for (int i = 0; i < n*n; i++) { A[i] = 1.0/(i+1); B[i] = 1.0/(i+2); C[i] = 0; }
    // warmup
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n,n,n, 1.0, A,n, B,n, 0.0, C,n);
    int rep = 10; double best = 1e30;
    for (int r = 0; r < rep; r++) {
        double t0 = now_ms();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n,n,n, 1.0, A,n, B,n, 0.0, C,n);
        double dt = now_ms() - t0;
        if (dt < best) best = dt;
    }
    double gf = 2.0*n*n*n / (best*1e6);
    printf("dgemm n=%d: best %.3f ms, %.2f GFLOPS (C[0]=%f)\n", n, best, gf, C[0]);
    return 0;
}
EOF
if gcc -O3 -march=rv64gcv -mabi=lp64d /tmp/dgemm_bench.c -o /tmp/dgemm_bench -lopenblas 2>/tmp/dgemm.err; then
  for n in 128 256 512; do /tmp/dgemm_bench $n; done
else
  echo "dgemm 冒烟测试跳过（未链接到 OpenBLAS）"; head -5 /tmp/dgemm.err
fi

echo
echo "===== 8. 结论提示 ====="
echo "把本输出发回给 Lead；重点看：VLEN、GCC 版本、OpenBLAS 位置、dgemm GFLOPS、CPU 频率。"
echo "若 VLEN=256 且 dgemm n=256 在 5-50 ms 量级，说明是真机；若 100 ms 以上或明显抖动，检查是否 QEMU/降频。"
