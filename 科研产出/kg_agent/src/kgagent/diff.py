"""S2 DiffAgent：差分抽取（确定性 + 规则，无 LLM）。

同一算子的 generic C 与 RVV C 是同一算法的两个实现，**差异就是一条优化变换实例**。
本模块把 S1 的结构化视图做差，按 DELTA_RULES 产出 actions / delta / params / before-after 行号锚点，
并把动作翻成中文解释，作为 S3 命名与模板选择的输入（S3 不自由发挥，只能引用这些事实）。
"""

from __future__ import annotations

__all__ = ["run", "DELTA_RULES", "ACTION_INFO"]

#: action -> (中文名, family, 说明模板)
ACTION_INFO = {
    "vectorize": ("向量化内核", "SIMD向量指令", "用向量寄存器与向量指令替代标量逐元素循环（{intrs} 类 intrinsic）"),
    "fma": ("FMA 融合乘加", "SIMD向量指令", "把乘加合并为一条融合指令（fma 出现 {fma} 次），减少一次舍入与一条指令"),
    "blocking": ("寄存器分块", "分块Tiling", "累加器由 {old} 个扩展到 {new} 个（{kind}），提高每个装入值的 FMA 复用次数"),
    "unroll": ("k 循环展开", "循环展开", "k 循环步长分母由 {old} 变为 {new}，单次迭代处理更多 k 步以摊薄循环开销"),
    "tail_handle": ("尾部处理", "尾部处理", "用动态 vl/vsetvl 或掩码处理 n 不整除向量宽度时的尾部，替代标量收尾"),
    "lmul_tuning": ("LMUL 调优", "SIMD向量指令", "向量寄存器分组取 LMUL={lmul}（SEW={sew}），在单指令吞吐与寄存器组数间取平衡"),
    "stride_dispatch": ("步长分支派发", "指针/索引优化", "按 inc_x/inc_y 是否为 1 分支：单位步长走 vle/vse，跨步走 vlse/vsse"),
    "reduction_split": ("向量归约", "循环拆分", "把标量累加链改为向量累加器 + 归约指令（{red}），提高归约链并行度"),
    "prefetch": ("软件预取", "预取", "插入 __builtin_prefetch 拉开访存与计算的距离"),
}

#: 规则：(action, 判定函数, 人类可读理由)
DELTA_RULES = [
    ("vectorize", lambda d: d["rvv"]["intrinsic_count"] > 0 and d["generic"]["intrinsic_count"] == 0,
     "generic 版无任何向量 intrinsic，rvv 版出现 __riscv_v* 向量指令"),
    ("fma", lambda d: d["rvv"]["has_fma"] and not d["generic"]["has_fma"],
     "rvv 版出现 vfmacc/vfma 融合乘加，generic 版为独立乘法+加法"),
    ("blocking", lambda d: d["rvv"]["acc_count"] > d["generic"]["acc_count"] and d["rvv"]["acc_count"] > 0,
     "rvv 版累加器数量多于 generic 版（寄存器级输出分块变大）"),
    ("unroll", lambda d: d["rvv"]["k_den"] is not None and d["rvv"]["k_den"] != d["generic"]["k_den"],
     "两版 k 循环展开分母不同（bk/old -> bk/new）"),
    ("tail_handle", lambda d: d["rvv"]["has_tail"] and not d["generic"]["has_tail"],
     "rvv 版用动态 vl/vsetvl 或掩码承担尾部，generic 版没有尾部处理路径"),
    ("lmul_tuning", lambda d: d["rvv"]["lmul"] is not None and d["older"] is None,
     "rvv 版显式选择 LMUL/SEW 组合（e{sew}m{lmul}），generic 版无向量寄存器分组概念"),
    ("stride_dispatch", lambda d: d["rvv"]["stride_dispatch"] and not d["generic"]["stride_dispatch"],
     "rvv 版按 inc_x/inc_y 是否为 1 做分支派发（单位步长快速路径）"),
    ("reduction_split", lambda d: d["rvv"]["reduction"] and not d["generic"]["reduction"],
     "rvv 版用向量归约指令替代标量累加链"),
    ("prefetch", lambda d: d["rvv"]["prefetch"] and not d["generic"]["prefetch"],
     "rvv 版插入软件预取指令"),
]


def _delta_view(generic, rvv):
    return {
        "generic": generic,
        "rvv": rvv,
        "older": None,  # 占位：方便规则里引用，保持规则纯函数
    }


def _diff_one(op: str, spec: dict, generic: dict, rvv: dict, cfg) -> dict:
    view = _delta_view(generic, rvv)
    actions = []
    reasons = []
    for action, predicate, reason in DELTA_RULES:
        try:
            ok = bool(predicate(view))
        except (KeyError, TypeError):  # pragma: no cover - 防御
            ok = False
        if ok:
            actions.append(action)
            reasons.append(reason)
    # 归约算子的“向量归约”不一定满足 reduction 标志翻转，单独补规则
    if spec.get("reduction") and rvv["has_vector"] and "reduction_split" not in actions:
        actions.append("reduction_split")
        reasons.append("该算子是归约类（dot/nrm2），rvv 版用向量累加器 + 归约指令")
    if spec.get("reduction") and rvv["has_vector"] and "vectorize" not in actions:
        actions.insert(0, "vectorize")
        reasons.insert(0, "该算子为归约类，rvv 版改为向量化归约实现")
    actions = sorted(set(actions), key=actions.index)

    params = {}
    if rvv["acc_count"] > generic["acc_count"] and rvv["acc_count"] > 0:
        params["reg_block_n"] = rvv["acc_count"]
    if rvv["k_den"] is not None:
        params["k_unroll"] = rvv["k_den"]
    if rvv["lmul"] is not None:
        params["lmul"] = rvv["lmul"]
    if rvv["sew"] is not None:
        params["sew"] = rvv["sew"]
    if rvv["sew"] is not None and rvv["lmul"] is not None:
        params["vector_len"] = "VLEN*LMUL/SEW = 256*%d/%d = %d" % (rvv["lmul"], rvv["sew"], 256 * rvv["lmul"] // rvv["sew"])
    if rvv["k_den"] is not None:
        params["vector_len"] = params.get("vector_len", "VLEN*LMUL/SEW")

    return {
        "diff_id": "diff:%s:kernel" % op,
        "op": op,
        "variant_pair": [generic["rel_path"], rvv["rel_path"]],
        "actions": actions,
        "action_notes": reasons,
        "delta": {
            "intrinsics_new": rvv["intrinsics"][:80],
            "intrinsic_count_old": generic["intrinsic_count"],
            "intrinsic_count_new": rvv["intrinsic_count"],
            "vector_types_new": rvv["vector_types"],
            "acc_old": {"n": generic["acc_count"], "kind": generic["acc_kind"]},
            "acc_new": {"n": rvv["acc_count"], "kind": rvv["acc_kind"], "sew": rvv["sew"], "lmul": rvv["lmul"]},
            "k_unroll_old": generic["k_den"],
            "k_unroll_new": rvv["k_den"],
            "tail_old": generic["has_tail"],
            "tail_new": rvv["has_tail"],
            "mask_old": generic["has_mask_ops"],
            "mask_new": rvv["has_mask_ops"],
            "stride_dispatch_new": rvv["stride_dispatch"],
            "fma_new": rvv["fma_count"],
            "reduction_old": generic["reduction"],
            "reduction_new": rvv["reduction"],
            "loop_depth_old": generic["loop_depth"],
            "loop_depth_new": rvv["loop_depth"],
            "loop_count_old": generic["loop_count"],
            "loop_count_new": rvv["loop_count"],
            "sew_new": rvv["sew"],
            "lmul_new": rvv["lmul"],
        },
        "params": params,
        "anchors": {
            "before": {
                "file": generic["rel_path"],
                "artifact_id": generic["artifact_id"],
                "function": generic["func_name"],
                "line_start": generic["line_start"],
                "line_end": generic["line_end"],
            },
            "after": {
                "file": rvv["rel_path"],
                "artifact_id": rvv["artifact_id"],
                "function": rvv["func_name"],
                "line_start": rvv["line_start"],
                "line_end": rvv["line_end"],
            },
        },
        "provenance": {
            "source_file": rvv["rel_path"],
            "line_start": rvv["line_start"],
            "line_end": rvv["line_end"],
            "line_confidence": "exact",
            "commit": "OpenBLAS 0.3.27.dev",
            "extractor_version": cfg.extractor_version,
        },
        "no_action_reason": None if actions else "两版 kernel 在静态特征上无差异（可能不是配对 RVV 实现）",
    }


def run(cfg, impls: dict) -> list:
    """S2 主入口：返回每个算子的差分记录（含 actions=[] 的诚实记录）。"""
    diffs = []
    for op, pair in impls.items():
        spec = cfg.operators[op]
        diffs.append(_diff_one(op, spec, pair["generic"], pair["rvv"], cfg))
    return sorted(diffs, key=lambda d: d["diff_id"])
