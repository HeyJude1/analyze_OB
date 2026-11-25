"""S1 StaticFeatureAgent：规则静态特征（确定性，无 LLM）。

对 generic/RVV 两版 kernel 解析：函数签名与行号区间、循环嵌套深度、向量 intrinsic、SEW/LMUL、
累加器个数、k 循环展开分母、掩码/动态 vl 尾部处理、步长分支、FMA、预取、归约、打包函数、复数用法。
产出 `data/characteristics.jsonl`，并向 S2 提供结构化 `impl` 视图。

多架构扩展（task-11）：
  - `parse_arch_impl` 统一解析 x86 AVX2/AVX512、ARM NEON/SVE 的 C 内核（含内联汇编）与 `.S` 汇编内核；
  - 汇编无法做"generic vs optimized"差分，因此抽"特征存在"：指令 mnemonics 计数、向量寄存器宽度、
    FMA/prefetch 计数、寄存器分块线索（目标向量寄存器数）、循环/分支/宏块计数；
  - 这些特征全部可被 grep 复核，属于 S1 静态事实，不经过 LLM。
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from . import corpus

__all__ = [
    "run", "parse_impl", "find_function_span", "impl_summary",
    "parse_asm_impl", "parse_arch_c_impl", "parse_arch_impl", "arch_features_summary",
]

_DEF_RE = re.compile(
    r"^(?:int|void|double|float|BLASLONG|size_t|unsigned\s+int)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
_INCLUDE_RE = re.compile(r'#\s*include\s*[<"]([^>"]+)[>"]')
_RVV_INTRIN_RE = re.compile(r"__riscv_[a-z0-9_]+")
_X86_INTRIN_RE = re.compile(r"_mm(?:256|512)?_[a-z0-9_]+")
_VEC_TYPE_RE = re.compile(r"v(?:float|int|uint)(8|16|32|64)m([1248])_t")
_VSETVL_RE = re.compile(r"__riscv_vsetvl(?:_e|max_e)(\d+)m([1248])")
_ACC_RE = re.compile(r"\b(v?res\d+|v?acc\d+|vsum\d*|vdot\d*)\b")
_K_DEN_RE = re.compile(r"\bbk\s*/\s*(\d+)")
_TAIL_RE = re.compile(r"&\s*[137]\b|%\s*[0-9]+\b|\bif\s*\(\s*n\s*&")
_MASK_RE = re.compile(r"_(?:mu|m)\b|\b(?:vmerge|vfmerge|vmsbf|vmsif|vmsof|viota|vcompress)\b")
_FMA_RE = re.compile(r"(?:__riscv_)?(?:vfmacc|vfnmacc|vfma|vfmadd|vfnmsac|vfmul|_mm(?:256|512)_fmadd)\w*")
_REDUCTION_INTRIN_RE = re.compile(r"\b(?:vfred|vfwred|vred\w*)")
_REDUCTION_SCALAR_RE = re.compile(r"\b(?:dot|sum|asum|nrm|norm)\w*\s*\+?=", re.IGNORECASE)
_PACK_RE = re.compile(r"\b(?:oncopy|otcopy|pack|copy_k)\b")
_PREFETCH_RE = re.compile(r"__builtin_prefetch")
_ALPHA_RE = re.compile(r"\*\s*alpha\b|\balpha\s*\*")
_COMPLEX_RE = re.compile(r"\b(?:creal|cimag|real|imag)\s*\(")
_STRIDE_BRANCH_RE = re.compile(r"inc_[xy]\s*==\s*1|1\s*==\s*inc_[xy]")
_INC_RE = re.compile(r"\binc_[xy]\b")
_PTR_ARITH_RE = re.compile(r"\bptr[a-z]*\s*\+\+|\bptr[a-z]*\s*\+=|C[0-9]\s*=\s*C[0-9]\s*\+")
_LOOP_RE = re.compile(r"\b(for|while|do)\b")


def find_function_span(lines: list, func_name: str) -> tuple:
    """返回函数定义的行号区间 (start_line, end_line)，1-based；找不到返回 (1, len(lines))。"""
    start = None
    pattern = re.compile(
        r"^(?:int|void|double|float|BLASLONG|size_t|unsigned\s+int)\s+%s\s*\(" % re.escape(func_name)
    )
    for idx, line in enumerate(lines):
        if pattern.match(line.strip()) or pattern.match(line):
            start = idx
            break
    if start is None:
        for idx, line in enumerate(lines):
            if re.search(r"\b%s\s*\(" % re.escape(func_name), line):
                start = idx
                break
    if start is None:
        return 1, len(lines)
    code = "\n".join(lines[start:])
    depth = 0
    opened = False
    end_line = len(lines)
    for offset, char in enumerate(code):
        if char == "{":
            depth += 1
            opened = True
        elif char == "}":
            depth -= 1
            if opened and depth == 0:
                end_line = start + code[: offset + 1].count("\n") + 1
                break
    return start + 1, max(start + 1, end_line)


def _loop_nesting(text: str) -> list:
    """扫描字符流，返回 [(line_no, keyword, nesting)]；nesting=0 为最外层。"""
    loops = []
    loop_stack = []  # body brace depth of enclosing loops
    depth = 0
    line_no = 1
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\n":
            line_no += 1
            i += 1
            continue
        if ch == "{":
            depth += 1
            i += 1
            continue
        if ch == "}":
            while loop_stack and loop_stack[-1] >= depth:
                loop_stack.pop()
            depth = max(0, depth - 1)
            i += 1
            continue
        if ch in "fw" and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            m = _LOOP_RE.match(text, i)
            if m and m.group(1) in ("for", "while"):
                # 只记录带 '(' 的循环头
                j = m.end()
                while j < n and text[j] in " \t":
                    j += 1
                if j < n and text[j] == "(":
                    loops.append({"line": line_no, "keyword": m.group(1), "nesting": len(loop_stack)})
                    loop_stack.append(depth + 1)
                    i = j
                    continue
        i += 1
    return loops


def _most_common(values: list):
    if not values:
        return None
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts.items(), key=lambda kv: (kv[1], -(kv[0] if isinstance(kv[0], int) else 0)))[0]


def parse_impl(path, op: str, variant: str, dtype: str, cfg=None) -> dict:
    """解析一个 kernel 源文件，返回结构化特征视图。"""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    func_name = None
    for line in lines:
        m = _DEF_RE.match(line.strip())
        if m:
            func_name = m.group(1)
            break
    if func_name is None:  # 兜底：第一个看起来像定义的标识符
        m = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*$", text, re.MULTILINE)
        func_name = m.group(1) if m else "unknown"
    span = find_function_span(lines, func_name)

    loops = _loop_nesting(text)
    vec_types = _VEC_TYPE_RE.findall(text)
    vsetvl = _VSETVL_RE.findall(text)
    sew_candidates = [int(s) for s, _ in vsetvl] + [int(s) for s, _ in vec_types]
    lmul_candidates = [int(m) for _, m in vsetvl] + [int(m) for _, m in vec_types]
    intrinsics = _RVV_INTRIN_RE.findall(text)
    x86_intrinsics = _X86_INTRIN_RE.findall(text)
    acc_names = sorted(set(_ACC_RE.findall(text)))
    k_dens = [int(x) for x in _K_DEN_RE.findall(text)]
    has_mask = bool(_MASK_RE.search(text))
    fma_count = len(_FMA_RE.findall(text))
    reduction = bool(_REDUCTION_INTRIN_RE.search(text)) or bool(_REDUCTION_SCALAR_RE.search(text))
    include = [m for m in _INCLUDE_RE.findall(text)]
    vec_acc = bool([a for a in acc_names if a.startswith("v")])
    scalar_acc = bool([a for a in acc_names if not a.startswith("v")])
    # 归约累加器（vr/vsum 且没有 res 命名）
    if not acc_names:
        if re.search(r"\bvr\b", text) and reduction:
            acc_names = ["vr"]
            vec_acc = True
    return {
        "op": op,
        "variant": variant,
        "path": str(path),
        "file": path.name,
        "func_name": func_name,
        "line_start": span[0],
        "line_end": span[1],
        "line_count": len(lines),
        "signature": _signature(lines, span[0] - 1),
        "includes": sorted(set(include)),
        "loop_count": len(loops),
        "loop_depth": max([lo["nesting"] for lo in loops], default=0) + (1 if loops else 0),
        "loops": loops[:24],
        "intrinsics": sorted(set(intrinsics)),
        "intrinsic_count": len(intrinsics),
        "x86_intrinsics": sorted(set(x86_intrinsics)),
        "vector_types": sorted(set("%s/%s" % (s, m) for s, m in vec_types)),
        "sew": _most_common([s for s in sew_candidates if s in (8, 16, 32, 64)]),
        "lmul": _most_common([m for m in lmul_candidates if m in (1, 2, 4, 8)]),
        "vsetvl_count": len(vsetvl),
        "acc_names": acc_names,
        "acc_count": len(acc_names),
        "acc_kind": "vector" if vec_acc and not scalar_acc else ("scalar" if scalar_acc else "none"),
        "k_den": min(k_dens) if k_dens else None,
        "k_dens": sorted(set(k_dens)),
        "has_mask_ops": has_mask,
        "has_tail": bool(has_mask or _TAIL_RE.search(text) or (vsetvl and variant == "rvv")),
        "has_tail_branch": bool(_TAIL_RE.search(text)),
        "fma_count": fma_count,
        "has_fma": fma_count > 0,
        "has_vector": bool(intrinsics or vec_types or x86_intrinsics),
        "reduction": reduction,
        "packing": bool(_PACK_RE.search(text)),
        "prefetch": bool(_PREFETCH_RE.search(text)),
        "stride_dispatch": bool(_STRIDE_BRANCH_RE.search(text)),
        "uses_inc": bool(_INC_RE.search(text)),
        "pointer_arith": bool(_PTR_ARITH_RE.search(text)),
        "alpha_scale": bool(_ALPHA_RE.search(text)),
        "complex": bool(_COMPLEX_RE.search(text)) or (cfg is not None and dtype in ("c", "z")),
        "dtype": dtype,
        "arch": "riscv64" if variant == "rvv" else "generic",
        "march": None,
    }


def _signature(lines: list, start_idx: int) -> str:
    """拼接函数签名（def 行到第一个 ')'），限制长度。"""
    buf = []
    for line in lines[start_idx:start_idx + 8]:
        buf.append(line.strip())
        if ")" in line:
            break
    sig = " ".join(buf)
    return sig[:300]


def impl_summary(impl: dict) -> dict:
    keys = [
        "op", "variant", "file", "func_name", "line_start", "line_end", "loop_count", "loop_depth",
        "intrinsic_count", "sew", "lmul", "acc_count", "acc_kind", "k_den", "has_mask_ops",
        "has_tail", "has_fma", "reduction", "packing", "prefetch", "stride_dispatch",
        "pointer_arith", "alpha_scale", "dtype", "arch",
    ]
    return {k: impl.get(k) for k in keys}


_CHAR_CONFIDENCE = {
    "dtype": 1.0,
    "loop_depth": 0.9,
    "loop_count": 0.95,
    "vector_intrinsics": 1.0,
    "sew": 0.95,
    "lmul": 0.95,
    "reg_block_n": 0.9,
    "acc_kind": 0.9,
    "k_unroll": 0.9,
    "mask_ops": 0.9,
    "tail_handling": 0.85,
    "stride_dispatch": 1.0,
    "fma": 1.0,
    "prefetch": 1.0,
    "reduction": 0.85,
    "packing": 0.9,
    "complex": 0.9,
    "pointer_arith": 0.9,
}


def _records_for(op: str, impl: dict, source_file: str) -> list:
    span = "%s:%d" % (source_file, impl["line_start"])
    rows = []

    def add(char_type, value, confidence=None, span_text=None):
        if value is None:
            return
        rows.append({
            "op": op,
            "variant": impl["variant"],
            "file": source_file,
            "char_type": char_type,
            "value": value,
            "method": "static",
            "confidence": confidence if confidence is not None else _CHAR_CONFIDENCE.get(char_type, 0.8),
            "span": span_text or span,
        })

    add("dtype", impl["dtype"], 1.0)
    add("loop_depth", impl["loop_depth"])
    add("loop_count", impl["loop_count"])
    add("vector_intrinsics", impl["intrinsic_count"])
    add("sew", impl["sew"], span_text="%s:%d" % (source_file, impl["line_start"]))
    add("lmul", impl["lmul"])
    add("reg_block_n", impl["acc_count"] if impl["acc_count"] else None)
    add("acc_kind", impl["acc_kind"])
    add("k_unroll", impl["k_den"])
    add("mask_ops", impl["has_mask_ops"])
    add("tail_handling", impl["has_tail"])
    add("stride_dispatch", impl["stride_dispatch"])
    add("fma", impl["has_fma"])
    add("prefetch", impl["prefetch"])
    add("reduction", impl["reduction"])
    add("packing", impl["packing"])
    add("complex", impl["complex"])
    add("pointer_arith", impl["pointer_arith"])
    return rows


def run(cfg, artifacts: list) -> tuple:
    """S1 主入口：返回 (characteristics 记录, impls)。

    impls = {op: {"generic": impl, "rvv": impl}}
    """
    index = {a["artifact_id"]: a for a in artifacts}
    impls = {}
    rows = []
    for op, spec in cfg.operators.items():
        dtype = spec.get("dtype", "d")
        entry = {}
        for variant, artifact_id in (("generic", "art:%s:kernel_generic" % op), ("rvv", "art:%s:kernel_rvv" % op)):
            art = index[artifact_id]
            abs_path = cfg.workspace_root / art["path"]
            impl = parse_impl(abs_path, op, variant, dtype, cfg=cfg)
            impl["march"] = art.get("compile_march")
            impl["artifact_id"] = artifact_id
            impl["rel_path"] = art["path"]
            entry[variant] = impl
            rows.extend(_records_for(op, impl, art["path"]))
        impls[op] = entry
    return rows, impls


# ============================================================ 多架构（x86 / ARM）解析

_X86_REG_RE = re.compile(r"%(zmm|ymm|xmm)(\d+)")
_NEON_REG_RE = re.compile(r"\bv(\d+)\.(?:2d|4s|8h|16b|1d|2s|4h|8b|2s|4d)\b")
_SVE_REG_RE = re.compile(r"\bz(\d+)(?:\.[sdqhb])?\b")
_MNEMONIC_LINE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_.]*)(?:\s+(.*))?$")
_INLINE_ASM_STR_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
_ARM_INTRIN_RE = re.compile(
    r"\b(?:vld[1-4]q?_[a-z0-9_]+|vst[1-4]q?_[a-z0-9_]+|vfmaq?_[a-z0-9_]+|vmlaq?_[a-z0-9_]+|"
    r"vaddq?_[a-z0-9_]+|vmulq?_[a-z0-9_]+|vsubq?_[a-z0-9_]+|vdupq?_[a-z0-9_]+|vgetq?_[a-z0-9_]+|"
    r"vsetq?_[a-z0-9_]+|vfmaq_[a-z0-9_]+)"
)
_SVE_INTRIN_RE = re.compile(
    r"\bsv(?:ld1|st1|mla|mad|fma|whilelt|ptrue|add|mul|dup|get|set|cmplt|sel|reinterpret|brka|brkb|"
    r"ld1_u64|ld1_f64|st1_f64|st1_u64)[a-z0-9_]*"
)
_ASM_HEADER_SKIP = {
    "if", "else", "endif", "define", "include", "macro", "endm", "align", "p2align",
    "text", "data", "section", "globl", "global", "type", "size", "set", "equ",
    "ifdef", "ifndef", "elseif", "error", "warning",
    # 汇编宏（OpenBLAS common.h / asm 头里定义，不是指令本身）
    "prologue", "epilogue", "save_regs", "restore_regs", "kernel", "endkernel",
    "print", "asmmacro", "common", "function",
}
_ASM_BRANCH_PREFIXES = ("j", "b.", "cbz", "cbnz", "tbz", "tbnz", "bne", "beq", "blt", "bgt", "ble", "bge")
_ASM_LOAD_HINTS = ("vmovup", "vmovap", "vmovdqu", "vbroadcast", "ld1", "ldp", "ldr", "vld", "movsd", "movap")
_ASM_STORE_HINTS = ("st1", "stp", "str", "vst", "vmovup", "vmovap", "vmovdqu")
_BLOCK_DIMS_RE = re.compile(r"(?:kernel_)?(?:v)?(\d+)x(\d+)")
_PREFETCH_MACRO_RE = re.compile(r"\bprefetch\s*\(")
_SCALAR_UNROLL_RE = re.compile(r"\b(?:y|res|acc|t|v|x)\d+\b")


def _empty_asm_features() -> dict:
    return {
        "instruction_count": 0, "vector_instr_count": 0, "fma_count": 0, "prefetch_count": 0,
        "branch_count": 0, "label_count": 0, "macro_count": 0, "load_count": 0, "store_count": 0,
        "mnemonic_counts": {}, "dest_reg_count": 0, "max_vector_width": 0,
    }


def _scan_asm(text: str, arch: str, min_mnemonic_len: int = 2) -> dict:
    """扫描汇编指令体（`.S` 文件或 C 内联汇编字符串），抽 mnemonics 计数与向量特征。"""
    feats = _empty_asm_features()
    counts = Counter()
    vector_regs = set()
    dest_regs = set()
    in_comment = False
    for raw in text.splitlines():
        line = raw
        if in_comment:
            if "*/" in line:
                line = line.split("*/", 1)[1]
                in_comment = False
            else:
                continue
        while "/*" in line:
            start = line.find("/*")
            end = line.find("*/", start + 2)
            if end < 0:
                line = line[:start]
                in_comment = True
                break
            line = line[:start] + line[end + 2:]
        line = line.split("//")[0]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("."):
            if stripped.startswith(".macro"):
                feats["macro_count"] += 1
            continue
        if stripped.endswith(":"):
            feats["label_count"] += 1
            continue
        m = _MNEMONIC_LINE_RE.match(stripped)
        if not m:
            continue
        mnemonic, operands = m.group(1).lower(), (m.group(2) or "")
        if mnemonic in _ASM_HEADER_SKIP:
            continue
        if mnemonic.startswith("align") or mnemonic.startswith("_"):
            continue
        if len(mnemonic) < min_mnemonic_len:
            continue
        counts[mnemonic] += 1
        feats["instruction_count"] += 1
        if mnemonic.startswith(_ASM_BRANCH_PREFIXES) or mnemonic in ("b", "bl", "ret", "br"):
            feats["branch_count"] += 1
        if mnemonic.startswith(_ASM_LOAD_HINTS):
            feats["load_count"] += 1
        if mnemonic.startswith(_ASM_STORE_HINTS):
            feats["store_count"] += 1
        if "prefetch" in mnemonic or mnemonic.startswith("prfm"):
            feats["prefetch_count"] += 1
        regs = []
        if arch in ("x86", "x86_avx2", "x86_avx512"):
            hits = _X86_REG_RE.findall(operands)
            regs = [(w, int(n)) for w, n in hits]
            if regs:
                feats["vector_instr_count"] += 1
                for width_name, _ in regs:
                    feats["max_vector_width"] = max(
                        feats["max_vector_width"], {"zmm": 512, "ymm": 256, "xmm": 128}[width_name])
                vector_regs.update("x%s" % n for _, n in regs)
        elif arch == "arm_neon":
            hits = [int(n) for n in _NEON_REG_RE.findall(operands)]
            if hits:
                feats["vector_instr_count"] += 1
                feats["max_vector_width"] = max(feats["max_vector_width"], 128)
                vector_regs.update("v%d" % n for n in hits)
        else:  # arm_sve
            hits = [int(n) for n in _SVE_REG_RE.findall(operands)]
            if hits:
                feats["vector_instr_count"] += 1
                vector_regs.update("z%d" % n for n in hits)
        if any(k in mnemonic for k in ("fmadd", "fmsub", "fmla", "fmls", "fnmadd", "fnmsub")):
            if regs or (arch == "arm_neon" and operands) or (arch == "arm_sve" and operands):
                feats["fma_count"] += 1
                if arch in ("x86", "x86_avx2", "x86_avx512") and regs:
                    dest_regs.add("x%s" % regs[-1][1])
                elif arch == "arm_neon":
                    hits = [int(n) for n in _NEON_REG_RE.findall(operands)]
                    if hits:
                        dest_regs.add("v%d" % hits[0])
                else:
                    hits = [int(n) for n in _SVE_REG_RE.findall(operands)]
                    if hits:
                        dest_regs.add("z%d" % hits[0])
    feats["mnemonic_counts"] = {k: v for k, v in counts.most_common(16)}
    feats["dest_reg_count"] = len(dest_regs)
    feats["vector_reg_count"] = len(vector_regs)
    return feats


def _block_dims(name: str):
    m = _BLOCK_DIMS_RE.search(name)
    if not m:
        return None
    try:
        a, b = int(m.group(1)), int(m.group(2))
    except ValueError:  # pragma: no cover
        return None
    if a <= 0 or b <= 0 or a > 64 or b > 64:
        return None
    return [a, b]


def _isa_from_width(arch: str, width: int) -> str:
    if arch.startswith("x86"):
        if width >= 512:
            return "AVX512F"
        if width >= 256:
            return "AVX2"
        if width >= 128:
            return "SSE2"
        return "x86_64 scalar"
    return _detected_isa(arch)


def parse_asm_impl(path, op: str, arch: str, arch_family: str, vector_bits: int = None) -> dict:
    """解析汇编内核（.S）：mnemonics 计数、向量宽度、FMA/prefetch、寄存器分块线索。"""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    feats = _scan_asm(text, arch)
    width = feats["max_vector_width"] or {"arm_neon": 128, "arm_sve": vector_bits or 256}.get(arch, vector_bits or 128)
    block_dims = _block_dims(path.name)
    detected = _isa_from_width(arch, width) if arch.startswith("x86") else _detected_isa(arch)
    return {
        "op": op,
        "arch": arch,
        "arch_family": arch_family,
        "kind": "asm",
        "file": path.name,
        "path": str(path),
        "func_name": "asm_kernel_%s" % path.stem,
        "signature": "asm::" + path.name,
        "line_start": 1,
        "line_end": len(lines),
        "line_count": len(lines),
        "dtype": "d",
        "loop_depth": 1,
        "loop_count": max(1, feats["label_count"]),
        "vector_width_bits": width,
        "vector_instr_count": feats["vector_instr_count"],
        "fma_count": feats["fma_count"],
        "prefetch_count": feats["prefetch_count"],
        "reg_block_estimate": max(feats["dest_reg_count"], (block_dims[0] * block_dims[1]) if block_dims else 0),
        "block_dims": block_dims,
        "unroll_hint": bool(block_dims or feats["macro_count"] >= 2),
        "macro_count": feats["macro_count"],
        "instruction_count": feats["instruction_count"],
        "branch_count": feats["branch_count"],
        "label_count": feats["label_count"],
        "load_count": feats["load_count"],
        "store_count": feats["store_count"],
        "mnemonic_counts": feats["mnemonic_counts"],
        "has_vector": feats["vector_instr_count"] > 0,
        "has_fma": feats["fma_count"] > 0,
        "has_prefetch": feats["prefetch_count"] > 0,
        "stride_dispatch": False,
        "reduction": False,
        "detected_isa": detected,
    }


def _detected_isa(arch: str, text: str = "", asm: bool = False) -> str:
    if arch == "x86_avx2":
        return "AVX2"
    if arch == "x86_avx512":
        return "AVX512F"
    if arch == "arm_neon":
        return "NEON"
    if arch == "arm_sve":
        return "SVE"
    if arch == "rvv":
        return "RVV 1.0"
    return "unknown"


def parse_arch_c_impl(path, op: str, arch: str, arch_family: str, vector_bits: int = None,
                      dtype: str = "d") -> dict:
    """解析多架构 C 内核：通用静态特征 + ISA intrinsic + C 内联汇编 mnemonics。"""
    path = Path(path)
    base = parse_impl(path, op, arch, dtype)
    text = path.read_text(encoding="utf-8", errors="replace")
    intrin_counts = {
        "x86": len(_X86_INTRIN_RE.findall(text)),
        "arm": len(_ARM_INTRIN_RE.findall(text)),
        "sve": len(_SVE_INTRIN_RE.findall(text)),
    }
    raw_strings = _INLINE_ASM_STR_RE.findall(text)
    arch_reg_re = {
        "x86_avx2": re.compile(r"%(?:ymm|zmm|xmm)\d+"),
        "x86_avx512": re.compile(r"%(?:zmm|ymm|xmm)\d+"),
        "arm_neon": _NEON_REG_RE,
        "arm_sve": _SVE_REG_RE,
    }.get(arch)
    asm_strings = [s for s in raw_strings if arch_reg_re and arch_reg_re.search(s)]
    quoted = "\n".join(asm_strings)
    # C 内联汇编字符串里的 \n \t 是转义序列，先还原成真正的换行/制表，再按行扫指令
    quoted = quoted.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r").replace('\\"', '"')
    asm_feats = _scan_asm(quoted, arch, min_mnemonic_len=3) if asm_strings else _empty_asm_features()
    detected = _detected_isa(arch)
    width = {"x86_avx2": 256, "x86_avx512": 512, "arm_neon": 128, "arm_sve": vector_bits or 256}.get(arch, vector_bits or 128)
    block_dims = _block_dims(path.name)
    # 标量展开线索（例如 daxpy_thunderx.c 用 y0..y7 手工 8 路展开）
    scalar_unroll = 0
    if base.get("acc_count"):
        scalar_unroll = base["acc_count"]
    else:
        distinct = {m for m in _SCALAR_UNROLL_RE.findall(text)}
        scalar_unroll = len(distinct) if len(distinct) >= 4 else 0
    vector_instr = max(asm_feats["vector_instr_count"], intrin_counts["x86"] + intrin_counts["arm"] + intrin_counts["sve"])
    fma_count = asm_feats["fma_count"] + base.get("fma_count", 0)
    has_fma = bool(fma_count) or bool(re.search(r"_mm(?:256|512)_fmadd|vfmaq?_f|svmla|svmad", text))
    reg_block = max(base.get("acc_count") or 0, asm_feats["dest_reg_count"],
                    (block_dims[0] * block_dims[1]) if block_dims else 0, scalar_unroll)
    prefetch_count = asm_feats["prefetch_count"] + (1 if base.get("prefetch") else 0)
    macro_prefetch = len(_PREFETCH_MACRO_RE.findall(text))
    if macro_prefetch:
        prefetch_count = max(prefetch_count, macro_prefetch)
    features = dict(base)
    features.update({
        "arch": arch,
        "arch_family": arch_family,
        "kind": "c",
        "vector_width_bits": width,
        "intrinsic_count": sum(intrin_counts.values()),
        "vector_instr_count": vector_instr,
        "fma_count": fma_count,
        "prefetch_count": prefetch_count,
        "reg_block_estimate": reg_block,
        "block_dims": block_dims,
        "scalar_unroll": scalar_unroll,
        "unroll_hint": bool(block_dims or base.get("k_den") or scalar_unroll >= 4),
        "instruction_count": asm_feats["instruction_count"],
        "branch_count": asm_feats["branch_count"],
        "label_count": asm_feats["label_count"],
        "load_count": asm_feats["load_count"],
        "store_count": asm_feats["store_count"],
        "mnemonic_counts": asm_feats["mnemonic_counts"],
        "has_vector": vector_instr > 0,
        "has_fma": has_fma,
        "has_prefetch": prefetch_count > 0,
        "detected_isa": detected,
    })
    return features


def parse_arch_impl(path, op: str, arch: str, arch_family: str, vector_bits: int = None,
                    dtype: str = "d") -> dict:
    """按后缀分派：.S -> 汇编解析；.c/.h -> C（含内联汇编）解析；rvv -> 复用 RVV C 解析。"""
    suffix = Path(path).suffix.lower()
    if arch == "rvv" and suffix not in (".s", ".asm", ".inc"):
        base = parse_impl(path, op, "rvv", dtype)
        base.update({
            "arch": arch,
            "arch_family": arch_family,
            "kind": "c",
            "vector_width_bits": vector_bits or 256,
            "vector_instr_count": base.get("intrinsic_count", 0),
            "fma_count": base.get("fma_count", 0),
            "prefetch_count": 1 if base.get("prefetch") else 0,
            "reg_block_estimate": base.get("acc_count") or 0,
            "block_dims": None,
            "unroll_hint": bool(base.get("k_den")),
            "has_vector": bool(base.get("has_vector")),
            "has_fma": bool(base.get("has_fma")),
            "has_prefetch": bool(base.get("prefetch")),
            "detected_isa": "RVV 1.0",
            "instruction_count": 0,
            "branch_count": 0,
            "mnemonic_counts": {},
        })
        return base
    if suffix in (".s", ".asm", ".inc"):
        return parse_asm_impl(path, op, arch, arch_family, vector_bits=vector_bits)
    return parse_arch_c_impl(path, op, arch, arch_family, vector_bits=vector_bits, dtype=dtype)


def arch_char_records(op: str, arch: str, feature: dict, source_file: str) -> list:
    """把多架构实现特征转成 characteristics.jsonl 记录（S1 产物）。"""
    rows = []

    def add(char_type, value, confidence=0.9):
        if value is None:
            return
        rows.append({
            "op": op,
            "variant": arch,
            "file": source_file,
            "char_type": char_type,
            "value": value,
            "method": "static",
            "confidence": confidence,
            "span": "%s:%d" % (source_file, feature.get("line_start") or 1),
        })

    add("dtype", feature.get("dtype"), 1.0)
    add("vector_width_bits", feature.get("vector_width_bits"), 0.95)
    add("vector_instr_count", feature.get("vector_instr_count"), 0.95)
    add("fma_count", feature.get("fma_count"), 0.95)
    add("prefetch_count", feature.get("prefetch_count"), 0.9)
    add("reg_block_estimate", feature.get("reg_block_estimate"), 0.85)
    add("instruction_count", feature.get("instruction_count"), 0.9)
    add("branch_count", feature.get("branch_count"), 0.9)
    add("unroll_hint", bool(feature.get("unroll_hint")), 0.8)
    return rows


def arch_features_summary(feature: dict) -> dict:
    keys = [
        "arch", "arch_family", "kind", "file", "detected_isa", "vector_width_bits",
        "vector_instr_count", "fma_count", "prefetch_count", "reg_block_estimate",
        "block_dims", "instruction_count", "branch_count", "mnemonic_counts",
    ]
    return {k: feature.get(k) for k in keys}

