#!/usr/bin/env bash
# 自动生成：experiments/harness/qemu.py（Windows→WSL 交叉编译）
# 本文件必须 UTF-8 无 BOM + LF；手改请保持该编码。
# 子进程 stdout/stderr 落盘到 build.stdout.log / build.stderr.log（避免 WSL UTF-16 警告混流）。
set -uo pipefail
cd "/mnt/c/Users/MSI/Desktop/开题/科研产出/experiments/E09_MPSA/runs/real_grid_seed_sizes/daxpy_b4_n1024/harness_r1" || { echo "[build] FAIL: cd 失败"; exit 8; }
echo "[build] cc=riscv64-linux-gnu-gcc"
echo "[build] driver=/mnt/c/Users/MSI/Desktop/开题/科研产出/experiments/E09_MPSA/tasks/daxpy/driver/test_daxpy.c"
echo "[build] generated=/mnt/c/Users/MSI/Desktop/开题/科研产出/experiments/E09_MPSA/runs/real_grid_seed_sizes/daxpy_b4_n1024/harness_r1/generated.c"
timeout 300s riscv64-linux-gnu-gcc -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -static "/mnt/c/Users/MSI/Desktop/开题/科研产出/experiments/E09_MPSA/tasks/daxpy/driver/test_daxpy.c" "/mnt/c/Users/MSI/Desktop/开题/科研产出/experiments/E09_MPSA/runs/real_grid_seed_sizes/daxpy_b4_n1024/harness_r1/generated.c" -lm -o "/mnt/c/Users/MSI/Desktop/开题/科研产出/experiments/E09_MPSA/runs/real_grid_seed_sizes/daxpy_b4_n1024/harness_r1/daxpy_test" > build.stdout.log 2> build.stderr.log
rc=$?
echo "[build] gcc_exit=$rc"
if [ $rc -ne 0 ]; then
  echo "[build] FAIL（见 build.stderr.log）"
  exit $rc
fi
if [ ! -x "/mnt/c/Users/MSI/Desktop/开题/科研产出/experiments/E09_MPSA/runs/real_grid_seed_sizes/daxpy_b4_n1024/harness_r1/daxpy_test" ]; then
  echo "[build] FAIL：未生成可执行文件 /mnt/c/Users/MSI/Desktop/开题/科研产出/experiments/E09_MPSA/runs/real_grid_seed_sizes/daxpy_b4_n1024/harness_r1/daxpy_test"
  exit 9
fi
echo "[build] ok bin=/mnt/c/Users/MSI/Desktop/开题/科研产出/experiments/E09_MPSA/runs/real_grid_seed_sizes/daxpy_b4_n1024/harness_r1/daxpy_test bytes=$(stat -c %s "/mnt/c/Users/MSI/Desktop/开题/科研产出/experiments/E09_MPSA/runs/real_grid_seed_sizes/daxpy_b4_n1024/harness_r1/daxpy_test" 2>/dev/null || echo '?')"
exit 0
