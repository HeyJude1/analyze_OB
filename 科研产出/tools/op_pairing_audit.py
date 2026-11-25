#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
审计 cx_operator_tool/operator_extract/openblas_newtest1 下 45 个算子的 C/RVV 配对质量。

输出：
  10_第一周执行包/op_pairing_audit.csv
  10_第一周执行包/pairing_audit.md（摘要）
用法：
  python op_pairing_audit.py [--base <openblas_newtest1>] [--out <输出目录>]
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
DEFAULT_OUT = r"C:\Users\MSI\Desktop\开题\科研产出\10_第一周执行包"

FUNC_RE = re.compile(
    r"^[ \t]*(?:static[ \t]+)?(?:inline[ \t]+)?"
    r"(?:void|int|long|double|float|BLASLONG|blasint|IFLOAT|FLOAT|XFLOAT|"
    r"complex[ \t]+float|complex[ \t]+double)[ \t]+"
    r"([A-Za-z_][A-Za-z0-9_]*)[ \t]*\(",
    re.MULTILINE,
)

LOG_LINE_RE = re.compile(
    r"^\s*(\d+):Ref:\s*[0-9.]+\s*MFlops\s*([0-9.]+)\s*ms\s*\|\s*"
    r"Migration:\s*[0-9.]+\s*MFlops\s*([0-9.]+)\s*ms\s*\|\s*"
    r"Speedup:\s*([0-9.]+)x\s*=>\s*(\w+)",
    re.MULTILINE,
)


def read_text(path):
    try:
        return open(path, "r", encoding="utf-8", errors="ignore").read()
    except OSError:
        return ""


def funcs_of(path):
    txt = read_text(path)
    return sorted(set(FUNC_RE.findall(txt)))


def has_rvv(path):
    txt = read_text(path)
    return bool(re.search(r"__riscv_v|vsetvl|vfloat|vfmacc|vle\d+_v", txt))


def parse_log(path, min_size=32):
    txt = read_text(path)
    rows = []
    for m in LOG_LINE_RE.finditer(txt):
        size = int(m.group(1))
        ref_ms, mig_ms, sp = float(m.group(2)), float(m.group(3)), float(m.group(4))
        passed = m.group(5).lower() == "pass"
        rows.append((size, ref_ms, mig_ms, sp, passed))
    # 去重
    seen, uniq = set(), []
    for r in rows:
        if r[0] in seen:
            continue
        seen.add(r[0])
        uniq.append(r)
    big = [r for r in uniq if r[0] >= min_size and r[4] and r[3] > 0]
    geo = None
    if big:
        geo = math.exp(sum(math.log(r[3]) for r in big) / len(big))
    pass_rate = (sum(1 for r in uniq if r[4]) / len(uniq)) if uniq else None
    return {
        "log_n": len(uniq),
        "log_pass_rate": round(pass_rate, 4) if pass_rate is not None else "",
        "log_geo_speedup_ge32": round(geo, 4) if geo else "",
        "log_ref_ms_median_ge32": round(statistics.median([r[1] for r in uniq if r[0] >= min_size]), 4)
        if any(r[0] >= min_size for r in uniq) else "",
    }


def pick_dir(entries, num, base):
    """优先标准 _op 目录；否则取 _op5/_op1/_op0 中文件最多的。"""
    def score(d):
        s = 0
        if re.match(r"^\d+_[a-z0-9]+_op$", d):
            s += 100
        if "_cxtest" in d or "_kertest" in d or "_ori" in d or "copy" in d:
            s -= 50
        s += len(glob.glob(os.path.join(base, d, "*.c")))
        return s
    return max(entries, key=score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    base, out = args.base, args.out
    if not os.path.isdir(base):
        print("目录不存在:", base)
        sys.exit(1)
    os.makedirs(out, exist_ok=True)

    # 按算子编号聚合目录
    by_num = {}
    for d in os.listdir(base):
        full = os.path.join(base, d)
        if not os.path.isdir(full):
            continue
        m = re.match(r"^(\d+)_([a-z0-9]+)_op", d)
        if not m:
            continue
        by_num.setdefault(m.group(1), (m.group(2), []))[1].append(d)

    rows = []
    for num in sorted(by_num, key=int):
        op, dirs = by_num[num]
        d = pick_dir(dirs, num, base)
        op_dir = os.path.join(base, d)

        c_files = [f for f in glob.glob(os.path.join(op_dir, "*.c"))
                   if not os.path.basename(f).startswith("zzz_")]
        k_candidates = [f for f in c_files if re.search(r"_k\.c$", os.path.basename(f))]
        rvv_candidates = [f for f in c_files if re.search(r"_k_rvv\.c$", os.path.basename(f))]
        k_file = k_candidates[0] if k_candidates else ""
        rvv_file = rvv_candidates[0] if rvv_candidates else ""

        k_funcs = funcs_of(k_file) if k_file else []
        rvv_funcs = funcs_of(rvv_file) if rvv_file else []
        # 判断 k 文件是否属于该算子：函数名里是否含 op 或其常见变体（amax/axpy 等）
        token = op.lower()
        # BLAS 复数前缀：zdotu 的 kernel 常叫 zdotc，chbmv 的 kernel 叫 chbmv
        variants = {token, token[1:] if len(token) > 1 else token}
        if token.startswith("z"):
            variants.add("z" + token[1:])
        k_own = any(v in fn.lower() for fn in k_funcs for v in variants)
        rvv_own = any(v in fn.lower() for fn in rvv_funcs for v in variants)

        # 配对判定
        if not k_file:
            pair_status = "no_k_file"
        elif not k_own:
            pair_status = "mispair_k(%s)" % (",".join(k_funcs) or "?")
        elif not rvv_file:
            pair_status = "no_rvv"
        elif not has_rvv(rvv_file):
            pair_status = "rvv_file_no_intrinsics"
        elif not rvv_own:
            pair_status = "mispair_rvv(%s)" % (",".join(rvv_funcs) or "?")
        else:
            pair_status = "ok"

        # log
        logs = glob.glob(os.path.join(op_dir, "*_log.txt"))
        log_path = max(logs, key=os.path.getsize) if logs else ""
        log_info = parse_log(log_path) if log_path else {}

        # command 检查
        cmd_path = os.path.join(op_dir, "_command.txt")
        cmd = read_text(cmd_path)
        cmd_qemu = "qemu" in cmd.lower()
        cmd_zvl = "zvl256b" in cmd.lower()

        rows.append({
            "num": int(num),
            "op": op,
            "dir": d,
            "n_c_files": len(c_files),
            "k_file": os.path.basename(k_file),
            "rvv_file": os.path.basename(rvv_file),
            "k_funcs": ";".join(k_funcs),
            "rvv_funcs": ";".join(rvv_funcs),
            "k_own": int(k_own),
            "rvv_own": int(rvv_own),
            "pair_status": pair_status,
            "rvv_intrinsics": int(has_rvv(rvv_file)) if rvv_file else 0,
            "log_file": os.path.basename(log_path),
            "log_n": log_info.get("log_n", ""),
            "log_pass_rate": log_info.get("log_pass_rate", ""),
            "log_geo_speedup_ge32": log_info.get("log_geo_speedup_ge32", ""),
            "log_ref_ms_median_ge32": log_info.get("log_ref_ms_median_ge32", ""),
            "cmd_has_qemu": int(cmd_qemu),
            "cmd_has_zvl256b": int(cmd_zvl),
        })

    csv_path = os.path.join(out, "op_pairing_audit.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 统计
    total = len(rows)
    status_count = {}
    for r in rows:
        status_count[r["pair_status"]] = status_count.get(r["pair_status"], 0) + 1
    ok = [r for r in rows if r["pair_status"] == "ok"]
    mis = [r for r in rows if r["pair_status"].startswith("mispair")]
    no_rvv = [r for r in rows if r["pair_status"] == "no_rvv"]
    no_intr = [r for r in rows if r["pair_status"] == "rvv_file_no_intrinsics"]
    no_k = [r for r in rows if r["pair_status"] == "no_k_file"]

    md = []
    md.append("# 算子 C/RVV 配对校验报告（自动生成）\n")
    md.append("> 数据源：`%s`｜生成脚本：`tools/op_pairing_audit.py`\n" % base)
    md.append("## 一句话结论\n")
    md.append("共审计 **%d** 个算子：配对正常 **%d**，配对错误 **%d**，有 RVV 文件但**无向量指令** **%d**，无 `_k.c` **%d**。**只有配对正常的算子才能进实验；无向量指令的只能做 C0/C1 对比，不能当 RVV 知识源。**\n" % (
        total, len(ok), len(mis), len(no_intr), len(no_k)))
    md.append("\n## 状态分布\n")
    md.append("| 状态 | 数量 | 含义 |")
    md.append("|---|---:|---|")
    meanings = {
        "ok": "C/RVV 配对正确，可进实验（x86 + RISC-V 双平台）",
        "rvv_file_no_intrinsics": "有 `_k_rvv.c` 但文件里没有 RVV intrinsic/`vsetvl` → 只能做 C0/C1 对比，不能当 RVV 知识源",
        "no_rvv": "没有 `_k_rvv.c` 文件 → 只能做 C0/C1 对比",
        "no_k_file": "没有找到 `_k.c` 文件（目录结构异常）",
    }
    for st, cnt in sorted(status_count.items(), key=lambda x: -x[1]):
        if st.startswith("mispair"):
            meaning = "配对错误：`_k` 或 `_rvv` 文件里的函数名不属于该算子"
        else:
            meaning = meanings.get(st, "—")
        md.append("| %s | %d | %s |" % (st, cnt, meaning))
    md.append("")

    if mis:
        md.append("## 配对错误的算子（必须人工处理）\n")
        md.append("| 算子 | 目录 | k_file 里实际函数 | rvv_file 里实际函数 |")
        md.append("|---|---|---|---|")
        for r in mis:
            md.append("| %s | %s | %s | %s |" % (r["op"], r["dir"], r["k_funcs"], r["rvv_funcs"]))
        md.append("")

    md.append("## 可用算子清单（配对正常 + 有日志）\n")
    md.append("| 算子 | 层级建议 | RVV | 日志通过率 | geomean(Migration/Ref, size≥32) |")
    md.append("|---|---|---|---:|---:|")
    for r in ok:
        md.append("| %s | %s | %s | %s | %s |" % (
            r["op"], "L1/L2/L3 待定", "有" if r["rvv_intrinsics"] else "无",
            r["log_pass_rate"], r["log_geo_speedup_ge32"]))
    md.append("")

    md_path = os.path.join(out, "pairing_audit.md")
    open(md_path, "w", encoding="utf-8").write("\n".join(md))

    print("审计完成：%d 个算子" % total)
    print("状态分布:", status_count)
    print("配对正常:", [r["op"] for r in ok])
    print("配对错误:", [r["op"] for r in mis])
    print("输出:", csv_path)
    print("输出:", md_path)


if __name__ == "__main__":
    main()
