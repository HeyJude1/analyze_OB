# 可复现包：real_grid_seed_sizes

> 生成时间：2026-11-12T21:39:05+08:00｜provider=dashscope｜model=qwen-plus-2025-04-28｜profile=k1_rvv256

## 目录

```text
inputs/            输入算子 generic.c、task.yaml、driver、硬件 profile、检索链 trace/pack
cells/<cell>/      query/strategies/prompts + round0(初始) + round1+(修复) + final(最终代码+bin)
manifest.json      本次 run 的全部映射（哪些 cell、状态、输入/检索/最终目录）
commands.txt       每个 cell 的原始 mpsa 命令 + build/run 命令（历史 out 路径已迁到本包）
env.txt            gcc/qemu 版本
```

## 1. 复现检索链

```bash
cd kg_agent/src
python -m kgagent.cli retrieve --op daxpy --profile k1_rvv256 --top-k 6 --trace-out /tmp/traces
```
本包已保存检索结果 `inputs/retrieval/<op>_<profile>_k<top_k>_trace.json` 与 `_pack.json`；硬件 profile 见 `inputs/hardware/<profile>.yaml`；原始生成命令见 `commands.txt`。可视化里回放请用 `kg_agent/viz/graph.html?panel=replay`。

## 2. 复现编译 + QEMU

```bash
cd cells/daxpy_b3_n1024_seed1/final
bash build.sh && bash run.sh
```
每个 round 目录都有独立的 `build.sh`/`run.sh`；`round0` 是初始生成，`round1..` 是修复轮，`final` 是最终结果。
QEMU 只验证正确性，计时不是性能结论。
