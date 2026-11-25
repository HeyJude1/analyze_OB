#!/usr/bin/env python3
"""verify_demo.py —— 用真实 OpenBLAS 内核演示"规则"到底是什么。

被验证的文件：kg_agent/data/corpus_multiarch/x86_avx2/axpy/daxpy_microk_haswell-2.c
它是一段 AVX2 内联汇编微内核：4 个 ymm 装 y、4 条 vfmadd231pd、addq $16。

脚本里定义 5 张"LLM 提议的策略卡"（1 张对的 + 4 张典型错的），
然后用 6 条规则逐张检查，打印 PASS/FAIL 和原因。

关键：这 6 条规则没有一个字提到 AXPY。它们只检查"你说的和源码是否一致"。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "kg_agent" / "data" / "corpus_multiarch" / "x86_avx2" / "axpy" / "daxpy_microk_haswell-2.c"


def norm(text: str) -> str:
    """把空白折叠成单空格，便于做引用匹配。"""
    return " ".join(str(text).split())


def load_lines(path: Path) -> list:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def window(lines: list, start: int, end: int) -> str:
    if start < 1 or end < start:
        return ""
    return norm(" ".join(lines[start - 1:end]))


# ---------------------------------------------------------------- 被验证的卡片


def cards() -> list:
    return [
        {  # 1. 正确卡
            "id": "C1", "name": "AVX2 四路展开 + FMA 融合乘加",
            "evidence_quote": "vfmadd231pd       (%2,%0,8), %%ymm0  , %%ymm12",
            "line_start": 44, "line_end": 55,
            "conditions": [
                {"pred": "dtype", "op": "in", "value": ["d"], "justification_quote": "vfmadd231pd"},
                {"pred": "vector_bits", "op": "eq", "value": 256, "justification_quote": "%%ymm12"},
                {"pred": "aligned", "op": "eq", "value": False, "justification_quote": "vmovups"},
            ],
            "parameters": [{"name": "unroll_factor", "value": 4, "value_in_source": "addq $16"}],
        },
        {  # 2. 引用漂移：说在 48-50 行，引文其实在 57 行
            "id": "C2", "name": "循环控制优化（引文漂移）",
            "evidence_quote": "addq $16, %0",
            "line_start": 48, "line_end": 50,
            "conditions": [{"pred": "dtype", "op": "in", "value": ["d"], "justification_quote": "vfmadd231pd"}],
            "parameters": [],
        },
        {  # 3. 对齐错：vmovups 是未对齐访问，却声称 aligned=true
            "id": "C3", "name": "对齐加载（对齐判断错误）",
            "evidence_quote": "vmovups                  (%3,%0,8), %%ymm12",
            "line_start": 44, "line_end": 47,
            "conditions": [{"pred": "aligned", "op": "eq", "value": True, "justification_quote": "vmovups"}],
            "parameters": [],
        },
        {  # 4. 单位错：addq $16 是 16 个 double = 4 个 ymm，声称展开 8 次
            "id": "C4", "name": "八路展开（单位算错）",
            "evidence_quote": "vfmadd231pd       (%2,%0,8), %%ymm0  , %%ymm12",
            "line_start": 44, "line_end": 59,
            "conditions": [{"pred": "vector_bits", "op": "eq", "value": 256, "justification_quote": "%%ymm"}],
            "parameters": [{"name": "unroll_factor", "value": 8, "value_in_source": "addq $16"}],
        },
        {  # 5. 精度错：vfmadd231pd 是 double，却声称 single
            "id": "C5", "name": "单精度向量化（精度判断错误）",
            "evidence_quote": "vfmadd231pd       (%2,%0,8), %%ymm0  , %%ymm12",
            "line_start": 48, "line_end": 48,
            "conditions": [{"pred": "dtype", "op": "in", "value": ["s"], "justification_quote": "vfmadd231pd"}],
            "parameters": [],
        },
    ]


# ---------------------------------------------------------------- 六条规则


def r1_quote_in_range(card, lines, ctx) -> str | None:
    """引文必须出现在它声称的行区间内。"""
    q = norm(card.get("evidence_quote", ""))
    if not q:
        return "缺少 evidence_quote"
    if q not in window(lines, card["line_start"], card["line_end"]):
        return "引文不在 %d–%d 行内" % (card["line_start"], card["line_end"])
    return None


def r2_lines_valid(card, lines, ctx) -> str | None:
    s, e = card.get("line_start"), card.get("line_end")
    if not isinstance(s, int) or not isinstance(e, int):
        return "缺行号"
    if not (1 <= s <= e <= len(lines)):
        return "行号越界（文件共 %d 行）" % len(lines)
    return None


def r3_dtype_consistent(card, lines, ctx) -> str | None:
    """算术指令后缀 pd/ps 决定精度：pd=double，ps=single。

    注意只看算术指令（vfmadd/vadd/...），不要看 vmovups——它是搬运算，
    AT&T 下 OpenBLAS 常混用，靠它会误判。
    """
    conds = [c for c in card.get("conditions", []) if c.get("pred") == "dtype"]
    if not conds:
        return None
    text = window(lines, card["line_start"], card["line_end"])
    found = set(re.findall(r"v(?:fmadd|fmsub|add|mul|sub|div)\d*p([sd])\b", text))
    for c in conds:
        vals = c.get("value") if isinstance(c.get("value"), list) else [c.get("value")]
        if "d" in found and "d" not in vals:
            return "源码是 double(pd)，卡片却没写 dtype=d"
        if "s" in found and "s" not in vals:
            return "源码是 single(ps)，卡片却没写 dtype=s"
    return None


def r4_alignment_consistent(card, lines, ctx) -> str | None:
    """vmovups=未对齐；vmovapd/vmovaps=对齐。"""
    conds = [c for c in card.get("conditions", []) if c.get("pred") == "aligned"]
    if not conds:
        return None
    text = window(lines, card["line_start"], card["line_end"])
    claims_aligned = bool(conds[0].get("value"))
    if "vmovups" in text and claims_aligned:
        return "vmovups 是未对齐访问，却声称 aligned=true"
    if ("vmovapd" in text or "vmovaps" in text) and not claims_aligned:
        return "vmovapd/aps 是对齐访问，却声称 aligned=false"
    return None


def r5_vector_width(card, lines, ctx) -> str | None:
    """寄存器名决定位宽：xmm=128, ymm=256, zmm=512。"""
    conds = [c for c in card.get("conditions", []) if c.get("pred") == "vector_bits"]
    if not conds:
        return None
    text = window(lines, card["line_start"], card["line_end"])
    width = None
    for reg, bits in (("%%xmm", 128), ("%%ymm", 256), ("%%zmm", 512)):
        if reg in text:
            width = bits
            break
    if width is None:
        return "引文里找不到向量寄存器，无法支持 vector_bits"
    claimed = conds[0].get("value")
    if int(claimed) != width:
        return "引文是 %d 位，卡片却写 %s" % (width, claimed)
    return None


def r6_unroll_consistency(card, lines, ctx) -> str | None:
    """展开因子 × 每向量元素数 应等于 addq 的步长（double: 每 ymm 4 个）。"""
    params = [p for p in card.get("parameters", []) if p.get("name") == "unroll_factor"]
    if not params:
        return None
    text = window(lines, card["line_start"], card["line_end"])
    m = re.search(r"addq?\s+\$(\d+)", text)
    if not m:
        return None
    step = int(m.group(1))
    elems = 16 if "%%zmm" in text else 8 if "%%ymm" in text else 4
    # double 每元素 8 字节：step = unroll × elems_per_vec × (8 / elem_bytes)
    # 对 double，elems_per_vec 已是元素数（ymm=4），step = unroll × 4
    per_vec = elems // 2 if elems in (8, 16) else elems  # ymm→4, zmm→8, xmm→2
    claimed = int(params[0].get("value"))
    if claimed * per_vec != step:
        return "addq $%d 对应展开 %d，卡片却写 %d" % (step, step // per_vec, claimed)
    return None


RULES = [
    ("R1 引文在行内", r1_quote_in_range),
    ("R2 行号合法", r2_lines_valid),
    ("R3 精度一致", r3_dtype_consistent),
    ("R4 对齐一致", r4_alignment_consistent),
    ("R5 位宽一致", r5_vector_width),
    ("R6 展开/步长一致", r6_unroll_consistency),
]


def main() -> int:
    lines = load_lines(SRC)
    print("源文件: %s（%d 行）\n" % (SRC.name, len(lines)))
    rows = []
    for card in cards():
        reasons = []
        for label, fn in RULES:
            err = fn(card, lines, {})
            if err:
                reasons.append("%s: %s" % (label.split()[0], err))
        verdict = "PASS" if not reasons else "FAIL"
        rows.append((card["id"], card["name"], verdict, reasons))

    width = max(len(r[1]) for r in rows) + 2
    print("%-4s %-*s %-6s %s" % ("ID", width, "卡片", "判定", "原因"))
    print("-" * 110)
    for cid, name, verdict, reasons in rows:
        print("%-4s %-*s %-6s %s" % (cid, width, name, verdict, "；".join(reasons) or "—"))

    passed = sum(1 for r in rows if r[2] == "PASS")
    dist = {}
    for _, _, _, reasons in rows:
        for x in reasons:
            key = x.split(":", 1)[0]
            dist[key] = dist.get(key, 0) + 1
    print("\n通过 %d/%d；拒绝原因分布：%s" % (passed, len(rows), json.dumps(dist, ensure_ascii=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
