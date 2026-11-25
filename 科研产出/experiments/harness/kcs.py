"""experiments/harness/kcs.py —— 知识合规率（KCS, Knowledge Compliance Score）启发式 v0。

定义（E09 用，见 `E09_MPSA/README.md` §4）：

    KCS = #(检索到的策略中，"生成代码里能找到实现痕迹"的策略数)
          / #(检索到的策略中，"其实现痕迹可静态检查"的策略数)

背景：Lingxi（ISSTA 2026）的失败分析发现大量 "uninformed autonomy"——知识给了但模型没用。
KCS 用便宜的静态检查回答"生成代码到底用没用检索到的策略"，与 FCR/性能做相关性分析。

诚实边界
--------
- 这是**启发式 v0**：只检查关键词/指令/编译指示是否存在，不证明"策略被正确使用"；
  用于 E09 冒烟链路与 E7 失败分析的粗筛，论文里必须标注 heuristic、并给出人工抽检校准结果。
- `reorder / split / fuse / scalarize / reduce_register_pressure` 这类策略没有稳定的
  静态痕迹，归入 `uncheckable`，**不计入分母**（不虚高也不虚低）。
- 分母为 0（没检索到可检查策略）时返回 `None`，不返回 0，避免把"没知识"混同"没用知识"。
"""

from __future__ import annotations

import re
from typing import Any

# action -> [(signal_name, regex, strong_or_weak)]
# strong：出现即认为该策略被使用；weak：出现只算弱证据（仍计入 matched，但 detail 里标 weak）
_SIGNALS: dict[str, list[tuple[str, str, str]]] = {
    "vectorize": [
        ("rvv_intrinsic", r"\b(?:__riscv_v|vsetvl|vsetvli|vle\d+_v|vse\d+_v|vfmacc|vfadd|vfmul|vfred)", "strong"),
        ("x86_intrinsic", r"\b_(?:mm|mm256|mm512)_\w+", "strong"),
        ("arm_intrinsic", r"\b(?:vld1|vst1|vfma|vmla)_\w+", "strong"),
    ],
    "unroll": [
        ("pragma_unroll", r"#\s*pragma\s+(?:GCC\s+|clang\s+)?(?:loop\s+)?unroll", "strong"),
        ("unroll_macro", r"#\s*define\s+\w*UNROLL\w*", "weak"),
    ],
    "prefetch": [
        ("prefetch_builtin", r"__builtin_prefetch\s*\(", "strong"),
        ("prefetch_intrinsic", r"\b_(?:mm_prefetch|prefetch)\w*", "weak"),
    ],
    "pack": [
        ("rvv_reorder", r"\b(?:vrgather|vslideup|vslidedown|vzip|vunzip|vcompress)", "strong"),
        ("packed_buffer", r"\bmemcpy\s*\(|\bpacked?\b|\bpbuf\b|\bkbuf\b", "weak"),
    ],
    "tile": [
        ("tile_macro", r"#\s*define\s+\w*(?:TILE|BLOCK|NB|MB|KB|KC|MC|NC)\w*", "strong"),
        ("tile_loop", r"for\s*\([^;]*<\s*(?:[A-Z_]{2,}|\w*(?:tile|block|nb|kb|mc|nc)\w*)", "weak"),
    ],
    "tail_handle": [
        ("tail_loop", r"\b(?:remain|tail|rest|rem)\w*\b", "weak"),
        ("vsetvl_tail", r"vsetvl(?:i)?\s*\(?[^;\n]*(?:n\s*-\s*|vl_max|remaining)", "weak"),
    ],
    "specialize": [
        ("size_special_case", r"if\s*\(\s*(?:n|N)\s*(?:==|<=|>=|<|>)\s*\d+", "weak"),
        ("alignment_special_case", r"if\s*\([^)\n]*(?:&|%)\s*(?:0x)?(?:15|31|63|3|7)\b", "weak"),
    ],
}

# 有明确语义、但没有稳定静态痕迹的 action：不计入 KCS 分母
_UNCHECKABLE = {
    "reorder",
    "split",
    "fuse",
    "scalarize",
    "reduce_register_pressure",
}


def _iter_strategies(retrieved: Any) -> list[dict]:
    """兼容 retrieved 为 list / {"strategies": [...]} / {"strategy_ids": [...]} 三种形态。"""
    if retrieved is None:
        return []
    if isinstance(retrieved, list):
        items = [x for x in retrieved if isinstance(x, dict)]
        return items
    if isinstance(retrieved, dict):
        if isinstance(retrieved.get("strategies"), list):
            return [x for x in retrieved["strategies"] if isinstance(x, dict)]
        ids = retrieved.get("strategy_ids")
        if isinstance(ids, list):
            return [{"id": str(i)} for i in ids]
    return []


def _action_of(strategy: dict) -> str:
    action = str(strategy.get("action") or "").strip().lower()
    if action:
        return action
    # 兜底：从 id/name 里猜（如 stg:dot:vectorize_x / "SIMD向量指令"）
    blob = " ".join(str(strategy.get(k) or "") for k in ("id", "name", "family"))
    for key in list(_SIGNALS) + sorted(_UNCHECKABLE):
        if key in blob:
            return key
    return "unknown"


def estimate_kcs(generated_c: str, retrieved: Any) -> dict:
    """计算启发式 KCS，返回可 JSON 序列化的明细。"""
    code = generated_c or ""
    strategies = _iter_strategies(retrieved)
    total = len(strategies)

    matched: list[dict] = []
    weak_only: list[dict] = []
    unmatched: list[dict] = []
    uncheckable: list[dict] = []
    unclassified: list[dict] = []
    detected: dict[str, list[str]] = {}

    for st in strategies:
        sid = str(st.get("id") or st.get("strategy_id") or "")
        action = _action_of(st)
        rules = _SIGNALS.get(action)
        if not rules:
            rec = {"id": sid, "action": action}
            if action in _UNCHECKABLE:
                uncheckable.append(rec)
            else:
                unclassified.append(rec)
            continue
        hit = None
        for signal_name, pattern, strength in rules:
            if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                hit = {"signal": signal_name, "strength": strength}
                detected.setdefault(signal_name, []).append(sid)
                break
        rec = {"id": sid, "action": action, **(hit or {})}
        if hit and hit["strength"] == "strong":
            matched.append(rec)
        elif hit:  # weak：只记弱证据，不计入 KCS 分子
            weak_only.append(rec)
        else:
            unmatched.append(rec)

    denominator = len(matched) + len(weak_only) + len(unmatched)
    kcs = (len(matched) / denominator) if denominator else None
    return {
        "method": "heuristic-v0",
        "kcs": round(kcs, 4) if kcs is not None else None,
        "matched": matched,
        "weak_only": weak_only,
        "unmatched": unmatched,
        "uncheckable": uncheckable,
        "unclassified": unclassified,
        "denominator": denominator,
        "retrieved_count": total,
        "signals_detected": detected,
        "note": "启发式：只查 strong 实现痕迹，不证明策略被正确使用；weak 证据不计入分子；uncheckable 不计入分母",
    }
