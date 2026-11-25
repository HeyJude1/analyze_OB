#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
riscv64-linux-gnu-gcc -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -static driver/test_ddot.c generated.c -lm -o bin
echo "[build] ok: $(pwd)/bin"
