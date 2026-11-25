#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解析 cx_operator_tool 里 OpenBLAS 算子的逐尺寸 benchmark 日志（*_log.txt）。

输出：
  - 科研产出/data/riscv_log_summary.csv   每个算子一行
  - 科研产出/data/riscv_log_all_sizes.csv 所有 (op, size) 明细
用法：
  python parse_riscv_logs.py [--base <openblas_newtest1 目录>]
"""
import argparse
import csv
import glob
import math
import os
import re
import statistics
import sys

DEFAULT_BASE = r"C:\Users\MSI\Desktop\cx_operator_tool\operator_extract\openblas_newtest1"
OUT_DIR = r"C:\Users\MSI\Desktop\开题\科研产出\data"

LINE_RE = re.compile(
    r"^\s*(\d+):Ref:\s*([0-9.]+)\s*MFlops\s*([0-9.]+)\s*ms\s*\|\s*"
    r"Migration:\s*([0-9.]+)\s*MFlops\s*([0-9.]+)\s*ms\s*\|\s*"
    r"Speedup:\s*([0-9.]+)x\s*=>\s*(\w+)",
    re.MULTILINE,
)
AVG_RE = re.compile(r"Average Speedup:\s*([0-9.]+)x")
MAX_RE = re.compile(r"Max Speedup\s*:\s*([0-9.]+)x")
MIN_RE = re.compile(r"Min Speedup\s*:\s*([0-9.]+)x")


def op_name_from_dir(d):
    m = re.match(r"^(\d+)_([a-z0-9]+)_", d)
    return m.group(2) if m else d


def is_canonical(d):
    # 优先选 01_chbmv_op 这种标准目录，排除 _cxtest/_kertest/_ori/_copy/_op0/_op1/_op5
    return re.match(r"^\d+_[a-z0-9]+_op$", d) is not None


def detect_runner(op_dir):
    """看 _command.txt 是否出现 qemu，用于判断日志可能来自模拟器还是真机。"""
    cmd = os.path.join(op_dir, "_command.txt")
    if not os.path.exists(cmd):
        return "unknown"
    try:
        txt = open(cmd, "r", encoding="utf-8", errors="ignore").read()
    except OSError:
        return "unknown"
    has_qemu = "qemu" in txt.lower()
    has_board = any(k in txt.lower() for k in ["riscv64-linux-gcc", "riscv64-unknown", "riscv64-buildroot"])
    if has_qemu and not has_board:
        return "qemu"
    if has_qemu and has_board:
        # 既编译又用 qemu 跑，仍按 qemu 处理
        return "board_then_qemu"
    if has_board:
        return "cross_compile_unknown_run"
    return "unknown"


def geomean(vals):
    vals = [v for v in vals if v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--min-size", type=int, default=32, help="计算几何平均时过滤小于该尺寸的点")
    args = ap.parse_args()

    base = args.base
    if not os.path.isdir(base):
        print("目录不存在:", base)
        sys.exit(1)

    # 收集每个算子号的候选目录
    candidates = {}
    for d in os.listdir(base):
        full = os.path.join(base, d)
        if not os.path.isdir(full):
            continue
        m = re.match(r"^(\d+)_([a-z0-9]+)_", d)
        if not m:
            continue
        num, op = m.group(1), m.group(2)
        logs = glob.glob(os.path.join(full, "*_log.txt"))
        if not logs:
            continue
        # 同一个目录下可能有多个 log，取最大的
        log = max(logs, key=os.path.getsize)
        candidates.setdefault(num, []).append((d, op, log))

    os.makedirs(OUT_DIR, exist_ok=True)
    summary_rows = []
    detail_rows = []

    for num in sorted(candidates, key=lambda x: int(x)):
        items = candidates[num]
        # 优先标准目录；否则取日志最大的
        canon = [it for it in items if is_canonical(it[0])]
        chosen = canon[0] if canon else max(items, key=lambda it: os.path.getsize(it[2]))
        d, op, log = chosen
        op_dir = os.path.join(base, d)
        txt = open(log, "r", encoding="utf-8", errors="ignore").read()
        rows = []
        for m in LINE_RE.finditer(txt):
            size = int(m.group(1))
            ref_ms = float(m.group(3))
            mig_ms = float(m.group(5))
            speedup = float(m.group(6))
            passed = m.group(7).lower() == "pass"
            rows.append((size, ref_ms, mig_ms, speedup, passed))
        if not rows:
            continue

        # 去重（同一 size 可能因多组参数重复，取第一次）
        seen = set()
        uniq = []
        for r in rows:
            if r[0] in seen:
                continue
            seen.add(r[0])
            uniq.append(r)

        # 过滤小尺寸后统计
        big = [r for r in uniq if r[0] >= args.min_size and r[4]]
        sp_big = [r[3] for r in big]
        pass_rate = sum(1 for r in uniq if r[4]) / len(uniq)
        avg_parsed = AVG_RE.search(txt)
        max_parsed = MAX_RE.search(txt)
        min_parsed = MIN_RE.search(txt)

        # 取 P1 基线：Ref 时间在 size>=min_size 的中位数（ms）
        ref_big = [r[1] for r in uniq if r[0] >= args.min_size]
        summary_rows.append({
            "op": op,
            "dir": d,
            "log_file": os.path.basename(log),
            "runner": detect_runner(op_dir),
            "n_sizes": len(uniq),
            "min_size": min(r[0] for r in uniq),
            "max_size": max(r[0] for r in uniq),
            "pass_rate": round(pass_rate, 4),
            "geomean_speedup_size_ge_%d" % args.min_size: round(geomean(sp_big), 4) if sp_big else "",
            "median_speedup": round(statistics.median(sp_big), 4) if sp_big else "",
            "ref_ms_median_size_ge_%d" % args.min_size: round(statistics.median(ref_big), 4) if ref_big else "",
            "log_reported_avg": float(avg_parsed.group(1)) if avg_parsed else "",
            "log_reported_max": float(max_parsed.group(1)) if max_parsed else "",
            "log_reported_min": float(min_parsed.group(1)) if min_parsed else "",
        })
        for r in uniq:
            detail_rows.append({
                "op": op, "size": r[0], "ref_ms": r[1], "mig_ms": r[2],
                "speedup": r[3], "pass": int(r[4]),
                "ref_mflops": "", "mig_mflops": "",
            })

    if not summary_rows:
        print("没有解析到任何日志")
        sys.exit(1)

    sum_path = os.path.join(OUT_DIR, "riscv_log_summary.csv")
    with open(sum_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    det_path = os.path.join(OUT_DIR, "riscv_log_all_sizes.csv")
    with open(det_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        w.writeheader()
        w.writerows(detail_rows)

    runners = {}
    for s in summary_rows:
        runners[s["runner"]] = runners.get(s["runner"], 0) + 1
    sp_all = [s["geomean_speedup_size_ge_%d" % args.min_size] for s in summary_rows
              if s["geomean_speedup_size_ge_%d" % args.min_size] != ""]
    print("解析算子数:", len(summary_rows))
    print("runner 分布:", runners)
    print("通过率均值:", round(statistics.mean(s["pass_rate"] for s in summary_rows), 4))
    if sp_all:
        print("跨算子 geomean(speedup) 中位数:", round(statistics.median(sp_all), 4),
              " 最小:", round(min(sp_all), 4), " 最大:", round(max(sp_all), 4))
    print("输出:", sum_path)
    print("输出:", det_path)
    print("\n前 10 行预览:")
    for s in summary_rows[:10]:
        print(s["op"], s["runner"], s["geomean_speedup_size_ge_%d" % args.min_size], s["pass_rate"])


if __name__ == "__main__":
    main()
