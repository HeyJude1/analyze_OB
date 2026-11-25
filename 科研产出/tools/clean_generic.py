#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 12 个算子的 *_k.c 清洗成可独立编译的 input/generic.c，并生成 task.yaml 骨架。

清洗规则：
  1. 去掉预处理行标记 `# 29 "xxx.c" 2` 这类行标记；
  2. 把 #include "common.h" 替换为最小 typedef 头（BLASLONG/blasint）；
  3. 保留函数原样（后续可按需重命名做匿名化）；
  4. 生成 task.yaml（op/level/role/sizes/signature 待补）。

用法：python clean_generic.py
输出：10_第一周执行包/OpOptBench/tasks/<op>/input/generic.c + task.yaml
"""
import os
import re

BASE = r"C:\Users\MSI\Desktop\cx_operator_tool\operator_extract\openblas_newtest1"
OUT = r"C:\Users\MSI\Desktop\开题\科研产出\10_第一周执行包\OpOptBench\tasks"

# op -> (dir, level, role, sizes)
TASKS = {
    "daxpy":  ("13_daxpy_op",  "L1", "S1_seen",   [1024, 16384, 1048576]),
    "dasum":  ("11_dasum_op",  "L1", "S1_seen",   [1024, 16384, 1048576]),
    "ddot":   ("15_ddot_op",   "L1", "S1_seen",   [1024, 16384, 1048576]),
    "dgemv":  ("17_dgemv_op",  "L2", "S1_seen",   [256, 1024, 4096]),
    "dgemm":  ("16_dgemm_op5", "L3", "S1_seen",   [64, 128, 512]),
    "dtrmm":  ("37_dtrmm_op1", "L3", "S1_seen",   [64, 128, 512]),
    "dscal":  ("24_dscal_op",  "L1", "S2_unseen", [1024, 16384, 1048576]),
    "dnrm2":  ("22_dnrm2_op",  "L1", "S2_unseen", [1024, 16384, 1048576]),
    "daxpby": ("12_daxpby_op", "L1", "S2_unseen", [1024, 16384, 1048576]),
    "dsymv":  ("30_dsymv_op",  "L2", "S2_unseen", [256, 1024, 4096]),
    "chemv":  ("03_chemv_op",  "L2", "S2_unseen", [256, 1024, 4096]),
    "dsymm":  ("29_dsymm_op1", "L3", "S2_unseen", [64, 128, 512]),
}

MIN_HEADER = """/* 由 tools/clean_generic.py 从 OpenBLAS 提取的 generic.c
 * 来源：{source}
 * 清洗：去掉 common.h 依赖，仅保留最小 typedef；函数体保持原样（未优化参考实现）。
 */
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

typedef long long BLASLONG;
typedef int blasint;
"""

PREPROC_LINE_MARK = re.compile(r'^#\s+\d+\s+".*"\s*\d*\s*$', re.MULTILINE)
INCLUDE_COMMON = re.compile(r'^[ \t]*#\s*include\s*[<"]common\.h[>"]\s*$', re.MULTILINE)


def clean(src_path):
    txt = open(src_path, encoding="utf-8", errors="ignore").read()
    txt = PREPROC_LINE_MARK.sub("", txt)
    txt = INCLUDE_COMMON.sub("", txt)
    # 去掉可能的 #include "common_*.h"
    txt = re.sub(r'^[ \t]*#\s*include\s*[<"]common_[^>"]*[>"]\s*$', "", txt, flags=re.MULTILINE)
    return txt.strip() + "\n"


def main():
    for op, (d, level, role, sizes) in TASKS.items():
        task_dir = os.path.join(OUT, op)
        input_dir = os.path.join(task_dir, "input")
        os.makedirs(input_dir, exist_ok=True)
        src = os.path.join(BASE, d, "%s_k.c" % op)
        if not os.path.exists(src):
            print("MISSING", op, src)
            continue
        body = clean(src)
        header = MIN_HEADER.format(source=os.path.join(d, "%s_k.c" % op))
        out_path = os.path.join(input_dir, "generic.c")
        open(out_path, "w", encoding="utf-8").write(header + "\n" + body)
        # task.yaml
        yaml = """# OpOptBench task（v0.1，自动生成，需人工补 signature/semantics/flops）
op: {op}
level: {level}
role: {role}
source: {src}
sizes: [{sizes}]
input: input/generic.c
rvv_expert: {rvv}
interface: "待补：从 <op>.c 抽取函数签名"
semantics: "待补：BLAS 语义"
invariants: ["待补"]
flops: "待补"
tolerance: "待补：参考 vs 参考自校准"
""".format(op=op, level=level, role=role,
               src=os.path.join(d, "%s_k.c" % op),
               sizes=", ".join(str(s) for s in sizes),
               rvv=os.path.join(d, "%s_k_rvv.c" % op))
        open(os.path.join(task_dir, "task.yaml"), "w", encoding="utf-8").write(yaml)
        print("OK", op, "->", out_path)


if __name__ == "__main__":
    main()
