#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
qemu-riscv64 -cpu rv64,v=true,vlen=256,zba=true,zbb=true,zbc=true,zbs=true ./bin 1024 10
