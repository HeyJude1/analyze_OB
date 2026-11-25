#!/usr/bin/env bash
# 自动生成：experiments/harness/qemu.py（Windows→QEMU 用户态运行）
# 本文件必须 UTF-8 无 BOM + LF；手改请保持该编码。
# QEMU 的 stdout/stderr 落盘到 qemu.stdout.log / qemu.stderr.log（harness 直接读文件解析）。
set -uo pipefail
cd "/mnt/c/Users/MSI/Desktop/开题/科研产出/experiments/E09_MPSA/runs/real_grid_seed_sizes/daxpy_b3_n16384/harness" || { echo "[run] FAIL: cd 失败"; exit 8; }
echo "[run] qemu-riscv64 -cpu rv64,v=true,vlen=256,zba=true,zbb=true,zbc=true,zbs=true ./bin 16384 10"
timeout --signal=TERM --kill-after=5s 300s qemu-riscv64 -cpu "rv64,v=true,vlen=256,zba=true,zbb=true,zbc=true,zbs=true" "/mnt/c/Users/MSI/Desktop/开题/科研产出/experiments/E09_MPSA/runs/real_grid_seed_sizes/daxpy_b3_n16384/harness/daxpy_test" 16384 10 > qemu.stdout.log 2> qemu.stderr.log
rc=$?
echo "[run] qemu_exit=$rc"
echo "[run] stdout_bytes=$(stat -c %s qemu.stdout.log 2>/dev/null || echo '?')"
exit $rc
