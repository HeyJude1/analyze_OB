#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""根据配对审计结果生成 OpOptBench-12 任务清单（md + yaml）。"""
import csv
import json
import os

BASE = r"C:\Users\MSI\Desktop\cx_operator_tool\operator_extract\openblas_newtest1"
AUDIT = r"C:\Users\MSI\Desktop\开题\科研产出\10_第一周执行包\op_pairing_audit.csv"
OUT_DIR = r"C:\Users\MSI\Desktop\开题\科研产出\10_第一周执行包"

LEVEL = {
    "daxpy": "L1", "dasum": "L1", "ddot": "L1", "dscal": "L1", "dnrm2": "L1",
    "daxpby": "L1", "dcopy": "L1", "drot": "L1", "dswap": "L1",
    "damax": "L1", "damin": "L1", "dmax": "L1", "dmin": "L1",
    "idamax": "L1", "idamin": "L1", "idmax": "L1", "idmin": "L1",
    "dgemv": "L2", "dsymv": "L2", "chemv": "L2",
    "dgemm": "L3", "dtrmm": "L3", "dsymm": "L3",
}

SIZES = {
    "L1": [1024, 16384, 1048576],
    "L2": [256, 1024, 4096],
    "L3": [64, 128, 512],
}

# S1 进 KG；S2 只在评测里出现
S1 = ["daxpy", "dasum", "ddot", "dgemv", "dgemm", "dtrmm"]
S2 = ["dscal", "dnrm2", "daxpby", "dsymv", "chemv", "dsymm"]


def main():
    rows = {r["op"]: r for r in csv.DictReader(open(AUDIT, encoding="utf-8-sig"))}
    tasks = []
    for op in S1 + S2:
        r = rows.get(op)
        if not r:
            print("WARN missing", op)
            continue
        level = LEVEL.get(op, "?")
        tasks.append({
            "op": op,
            "level": level,
            "role": "S1_seen" if op in S1 else "S2_unseen",
            "dir": r["dir"],
            "input": r["k_file"],
            "rvv_expert": r["rvv_file"],
            "pair_status": r["pair_status"],
            "log": r["log_file"],
            "log_geomean_qemu": r["log_geo_speedup_ge32"],
            "sizes": SIZES.get(level, []),
            "path": os.path.join(BASE, r["dir"]),
        })

    # YAML
    yaml_lines = ["# OpOptBench-12 任务清单（v0.1，2026-11-12）",
                  "# 由 tools/gen_optbench_list.py 从 op_pairing_audit.csv 生成",
                  "benchmark: OpOptBench-12",
                  "platforms: [x86_64, riscv64_k1_vlen256]",
                  "conditions: [C0, C1, P1, B0, B1, B2, B3]",
                  "tasks:"]
    for t in tasks:
        yaml_lines.append("  - op: %s" % t["op"])
        yaml_lines.append("    level: %s" % t["level"])
        yaml_lines.append("    role: %s" % t["role"])
        yaml_lines.append("    dir: %s" % t["dir"])
        yaml_lines.append("    input: %s" % t["input"])
        yaml_lines.append("    rvv_expert: %s" % t["rvv_expert"])
        yaml_lines.append("    pair_status: %s" % t["pair_status"])
        yaml_lines.append("    sizes: [%s]" % ", ".join(str(s) for s in t["sizes"]))
        yaml_lines.append("    note: %s" % (
            "QEMU 日志 geomean=%s（只作 sanity，不作性能）" % t["log_geomean_qemu"]
            if t["log_geomean_qemu"] else "无日志"))
    open(os.path.join(OUT_DIR, "optbench_tasks.yaml"), "w", encoding="utf-8").write("\n".join(yaml_lines) + "\n")

    # Markdown
    md = []
    md.append("# OpOptBench-12 任务清单（按配对审计修正版）\n")
    md.append("> 2026-11-12｜由 `tools/gen_optbench_list.py` 生成｜依据 `op_pairing_audit.csv` 与 `kg_adequacy_audit.md`\n")
    md.append("## 一句话\n")
    md.append("原计划的 12 个算子里 `dger/dtrmv/dsyrk` 的 `_rvv.c` 没有向量指令，不能当 RVV 知识源；本清单换成审计后**配对正常 + 有 RVV 指令 + 有日志**的算子。**S1（6 个）进 KG，S2（6 个）只在评测出现**。\n")
    md.append("\n## 任务表\n")
    md.append("| # | 算子 | 层级 | 角色 | 目录 | 输入 C | RVV 专家实现 | 尺寸 | QEMU 日志 geomean* | 备注 |")
    md.append("|---|---|---|---|---|---|---|---|---:|---|")
    for i, t in enumerate(tasks, 1):
        note = ""
        if t["op"] == "dsymm":
            note = "**需人工确认**：`_k.c` 里是 dgemm kernel；若确认不可用，换 dcopy/drot"
        if t["pair_status"] != "ok":
            note = (note + "；" if note else "") + "pair_status=%s" % t["pair_status"]
        md.append("| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            i, t["op"], t["level"], "S1 进 KG" if t["role"] == "S1_seen" else "S2 留出",
            t["dir"], t["input"], t["rvv_expert"], "/".join(str(s) for s in t["sizes"]),
            t["log_geomean_qemu"] or "—", note or "—"))
    md.append("")
    md.append("> \\* QEMU 日志的 `Migration/Ref` 几何平均只用于确认「语料可用、函数能跑」，**不是性能数字**，不能进论文性能表。\n")
    md.append("\n## 每个任务要准备的文件\n")
    md.append("```text\nOpOptBench/tasks/<op>/\n  task.yaml          # 签名、语义不变量、尺寸、flops、容差\n  input/generic.c    # 由 <op>_k.c 清洗（去 common.h/BLASLONG 依赖，保留接口）\n  ref/lib_ref.c      # x86: 系统 OpenBLAS；RISC-V: 仓库预编译库或 <op>_k_rvv.c\n  driver/bench.c     # 改造自 _file/benchmark/_<op>_dgc.c，输出 JSONL\n  tests/gen_input.c  # D1-D4 四个隐藏分布\n  tests/tolerance    # 参考 vs 参考自校准\n```\n")
    md.append("\n## S1 / S2 的用途\n")
    md.append("- **S1（6 个）**：它们的优化实现进入 KG，用来验证「知识确实抓到了」；对应 E1 金标与 E2/E3 的 in-domain 结果。\n")
    md.append("- **S2（6 个）**：不进 KG，只在评测出现，是**跨算子泛化**的主结果集；匹配论文第 5 章的「未见算子」表。\n")
    md.append("- dsymm 若人工确认不可用，S2 的 L3 位置用 dcopy/drot 顶替（保持 12 个总数，层级比例略变，论文中如实说明）。\n")
    md.append("\n## 下一步（W1 内完成）\n")
    md.append("1. 人工确认 dsymm 的 `_k.c` 是否可用（看它是否只是复用 dgemm kernel）。\n")
    md.append("2. 写 `clean_generic.py`：把 12 个 `_k.c` 清洗为可独立编译的 `generic.c`（替换 `common.h`、`BLASLONG`、`IFLOAT` 等）。\n")
    md.append("3. 先在 x86 编 C0/C1，再交叉编译到 RISC-V，确认 12/12 能跑通。\n")
    md.append("4. 用 `_<op>_dgc.c` 改造 driver，输出 JSONL；容差按参考 vs 参考自校准。\n")
    open(os.path.join(OUT_DIR, "02_OpOptBench-12任务清单.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("tasks:", len(tasks))
    print("S1:", S1)
    print("S2:", S2)
    print("output:", os.path.join(OUT_DIR, "02_OpOptBench-12任务清单.md"))


if __name__ == "__main__":
    main()
