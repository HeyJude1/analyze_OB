"""多架构层（task-11）：x86 AVX2/AVX512、ARM NEON/SVE、RISC-V RVV 的发现/入库/差分/策略。

设计要点：
1. **自动发现**：`config.multiarch.arch_sources.<arch>.files.<op>` 是 glob 模式列表，
   再加 `auto_microk_globs` / `auto_globs` 自动补登记 microkernel 与变体文件；
2. **自包含**：命中的源文件复制到 `data/corpus_multiarch/<arch>/<op>/`，图谱只引用复制后的路径 + sha256；
3. **arch 维度**：每个 implementation 带 `arch / arch_family / isa / uarch(profile)`；
4. **C 可差分**：同架构 generic 基线存在时做 generic vs optimized 差分（method=diff）；
   **汇编 only** 用特征存在标注（method=static：mnemonics 计数、向量宽度、FMA/prefetch、寄存器分块线索）；
5. **rvv 复用**：12 个已有算子的 RVV 实现来自 cx_operator_tool 语料，不重复复制/重复抽策略；
   仅新算子 asum/axpby 从 openblas_src 复制。
"""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

from . import static

__all__ = [
    "run",
    "discover",
    "materialize",
    "parse_impls",
    "diffs",
    "strategies",
    "ARCH_VECTOR_BITS",
    "ARCH_LABELS",
    "MA_ACTIONS",
    "arch_summary",
]

#: 架构默认向量宽度（SVE 按实现 VL=256 记录，见 hardware_profiles.yaml 的 note）
ARCH_VECTOR_BITS = {
    "x86_avx2": 256,
    "x86_avx512": 512,
    "arm_neon": 128,
    "arm_sve": 256,
    "rvv": 256,
}
ARCH_LABELS = {
    "x86_avx2": "x86 AVX2",
    "x86_avx512": "x86 AVX512",
    "arm_neon": "ARM NEON",
    "arm_sve": "ARM SVE",
    "rvv": "RISC-V RVV",
}
#: 多架构 action -> (family, 规范 action, 名称后缀, ISA 模板变体)
MA_ACTIONS = {
    "vectorize": ("SIMD向量指令", "vectorize", "向量化内核", "arch"),
    "tile": ("分块Tiling", "tile", "寄存器分块内核", "arch"),
    "unroll": ("循环展开", "unroll", "循环展开/多路累加", "portable"),
    "prefetch": ("预取", "prefetch", "软件预取", "portable"),
    "tail_handle": ("尾部处理", "tail_handle", "尾部/掩码处理", "portable"),
    "specialize": ("指针/索引优化", "specialize", "步长分支/布局特化", "arch"),
    "pack": ("数据重排", "pack", "打包/拷贝缓冲", "portable"),
}


#: detected_isa -> ISACapability 节点（与 normalize.ISA_CATALOG 对应）
ISA_NODE_BY_ISA = {
    "AVX2": "isa:avx2.fma.f64",
    "AVX512F": "isa:avx512.fma.f64",
    "NEON": "isa:neon.fma.f64",
    "SVE": "isa:sve.fma.f64",
    "RVV 1.0": "isa:rvv1.0.fma.f64",
}


def _arch_display_label(arch_label: str, detected_isa: str) -> str:
    """x86 汇编可能只检测到 SSE2/标量基线，名称里如实标注。"""
    if detected_isa == "SSE2":
        return "x86-64 baseline (SSE2)"
    if detected_isa == "x86_64 scalar":
        return "x86-64 scalar baseline"
    return arch_label


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_lines(path: Path) -> int:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return sum(1 for _ in fh)


def _fmt(pattern: str, kernel_dir: str, op: str) -> str:
    return pattern.format(kernel_dir=kernel_dir, op=op)


def discover(cfg) -> list:
    """按 config.multiarch.arch_sources 自动发现各架构实现文件（glob，确定性排序）。"""
    repo = cfg.multiarch_repo
    if not repo.exists():
        raise FileNotFoundError("多架构源仓库不存在：%s（检查 config.multiarch.repo）" % repo)
    all_ops = list(cfg.all_operator_specs)
    records = []
    seen = set()
    for arch, spec in cfg.arch_sources.items():
        kernel_dir = spec.get("kernel_dir", "")
        files_map = spec.get("files", {}) or {}
        reuse = set(spec.get("reuse_existing_ops", []) or [])
        patterns = []  # (op, pattern)
        for op, pats in files_map.items():
            for pat in pats:
                patterns.append((op, pat))
        for op in files_map:
            for glob in spec.get("auto_microk_globs", []) or []:
                patterns.append((op, _fmt(glob, kernel_dir, op)))
        for op in all_ops:
            if op in files_map or op in reuse:
                continue
            for glob in spec.get("auto_globs", []) or []:
                patterns.append((op, _fmt(glob, kernel_dir, op)))
        for op, pattern in patterns:
            for path in sorted(repo.glob(pattern)):
                if not path.is_file():
                    continue
                key = (arch, op, path.as_posix())
                if key in seen:
                    continue
                seen.add(key)
                rel = path.relative_to(repo).as_posix()
                suffix = path.suffix.lower()
                name = path.name
                records.append({
                    "op": op,
                    "arch": arch,
                    "arch_family": spec.get("arch_family", "x86"),
                    "arch_label": spec.get("label", ARCH_LABELS.get(arch, arch)),
                    "isa": spec.get("isa"),
                    "isa_node": spec.get("isa_node"),
                    "profiles": list(spec.get("profiles") or []),
                    "compiler": spec.get("compiler"),
                    "vector_bits": ARCH_VECTOR_BITS.get(arch, 128),
                    "kernel_dir": kernel_dir,
                    "rel_repo_path": rel,
                    "abs_path": path,
                    "kind": "asm" if suffix in (".s", ".asm", ".inc") else "c",
                    "role": "microk" if ("microk" in name or "kernel" in name) else "main",
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                    "line_count": _count_lines(path),
                })
    return sorted(records, key=lambda r: (r["arch"], r["op"], r["rel_repo_path"]))


def materialize(cfg, records: list) -> list:
    """把发现的源文件复制到 data/corpus_multiarch/<arch>/<op>/，返回 SourceArtifact 记录。"""
    out_root = cfg.multiarch_corpus_dir
    artifacts = []
    for rec in records:
        dest_dir = out_root / rec["arch"] / rec["op"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / Path(rec["rel_repo_path"]).name
        source = Path(rec["abs_path"])
        if not dest.exists() or _sha256(dest) != rec["sha256"]:
            shutil.copyfile(source, dest)
        artifact_id = "art:ma:%s:%s:%s" % (
            rec["arch"], rec["op"], re.sub(r"[^A-Za-z0-9_]+", "_", dest.name).strip("_"))
        artifacts.append({
            "artifact_id": artifact_id,
            "op": rec["op"],
            "kind": "multiarch_impl",
            "variant": rec["role"],
            "arch": rec["arch"],
            "arch_family": rec["arch_family"],
            "isa": rec["isa"],
            "isa_node": rec["isa_node"],
            "profiles": rec["profiles"],
            "path": cfg.rel_root(dest),
            "repo_path": rec["rel_repo_path"],
            "sha256": rec["sha256"],
            "bytes": dest.stat().st_size,
            "line_count": _count_lines(dest),
            "commit": cfg.multiarch_cfg.get("commit"),
            "repo": "OpenBLAS",
            "extractor_version": cfg.extractor_version,
            "compile_march": rec.get("compiler"),
        })
    return sorted(artifacts, key=lambda a: a["artifact_id"])


def _merge_features(files: list) -> dict:
    """把同一 (op, arch) 下多个实现文件的特征合并成一份（用于一条策略/一次差分）。"""
    if not files:
        return {}
    merged = dict(files[0])
    for field in ("vector_instr_count", "fma_count", "prefetch_count", "instruction_count",
                  "branch_count", "reg_block_estimate", "scalar_unroll", "intrinsic_count"):
        merged[field] = max(int(f.get(field) or 0) for f in files)
    merged["vector_width_bits"] = max(int(f.get("vector_width_bits") or 0) for f in files)
    merged["has_vector"] = any(f.get("has_vector") for f in files)
    merged["has_fma"] = any(f.get("has_fma") for f in files)
    merged["has_prefetch"] = any(f.get("has_prefetch") for f in files)
    merged["unroll_hint"] = any(f.get("unroll_hint") for f in files)
    merged["block_dims"] = next((f.get("block_dims") for f in files if f.get("block_dims")), None)
    merged["loop_depth"] = max(int(f.get("loop_depth") or 0) for f in files)
    merged["mnemonic_counts"] = next((f.get("mnemonic_counts") for f in files if f.get("mnemonic_counts")), {})
    merged["kinds"] = sorted({f.get("kind") for f in files})
    merged["files"] = files
    merged["op"] = files[0].get("op")
    merged["arch"] = files[0].get("arch")
    merged["arch_family"] = files[0].get("arch_family")
    return merged


def _primary(files: list) -> dict:
    """选择"代表实现"：优先带向量的 C microk，其次带向量的任意文件，最后第一个。"""
    def score(f):
        return (
            1 if (f.get("kind") == "c" and f.get("has_vector")) else 0,
            1 if f.get("has_vector") else 0,
            1 if f.get("role") == "microk" else 0,
            f.get("vector_instr_count") or 0,
        )
    return sorted(files, key=score, reverse=True)[0]


def parse_impls(cfg, artifacts: list) -> tuple:
    """S1（arch 维度）：解析每个 (op, arch) 的 C/汇编实现，返回 (characteristics, impls)。"""
    specs = cfg.arch_sources
    impls = {}
    rows = []
    for art in artifacts:
        if art.get("kind") != "multiarch_impl":
            continue
        spec = specs.get(art["arch"], {})
        abs_path = cfg.root / art["path"]
        feature = static.parse_arch_impl(
            abs_path, art["op"], art["arch"], art["arch_family"],
            vector_bits=ARCH_VECTOR_BITS.get(art["arch"], 128),
            dtype=cfg.all_operator_specs.get(art["op"], {}).get("dtype", "d"),
        )
        feature["artifact_id"] = art["artifact_id"]
        feature["rel_path"] = art["path"]
        feature["role"] = art.get("variant")
        feature["arch_label"] = spec.get("label", art["arch"])
        feature["profiles"] = art.get("profiles") or []
        feature["isa_node"] = art.get("isa_node")
        feature["isa"] = art.get("isa")
        impls.setdefault((art["op"], art["arch"]), []).append(feature)
        rows.extend(static.arch_char_records(art["op"], art["arch"], feature, art["path"]))
    merged = {key: _merge_features(files) for key, files in impls.items()}
    return rows, merged


def diffs(cfg, ma_impls: dict, generic_impls: dict) -> list:
    """S2（arch 维度）：C 对可用时做 generic vs optimized 差分；汇编-only 用特征存在（method=static）。"""
    out = []
    for (op, arch), features in sorted(ma_impls.items()):
        spec = cfg.arch_sources.get(arch, {})
        files = features.get("files") or []
        has_c = any(f.get("kind") == "c" for f in files)
        generic = (generic_impls.get(op) or {}).get("generic")
        method = "diff" if (has_c and generic) else "static"
        actions, notes = _derive_actions(features, method, generic)
        if not actions:
            actions = []
            notes.append("静态特征未达到阈值（无向量指令/预取/分块线索）")
        primary = _primary(files)
        before = None
        if method == "diff" and generic:
            before = {
                "file": generic.get("rel_path"),
                "line_start": generic.get("line_start"),
                "line_end": generic.get("line_end"),
                "function": generic.get("func_name"),
            }
        out.append({
            "diff_id": "ma_diff:%s:%s" % (arch, op),
            "op": op,
            "arch": arch,
            "arch_family": spec.get("arch_family"),
            "arch_label": spec.get("label", ARCH_LABELS.get(arch, arch)),
            "isa": spec.get("isa"),
            "isa_node": spec.get("isa_node"),
            "profiles": list(spec.get("profiles") or []),
            "method": method,
            "actions": actions,
            "action_notes": notes,
            "features": static.arch_features_summary(features),
            "impl_files": [
                {
                    "file": f.get("file"), "repo_path": f.get("rel_path"), "kind": f.get("kind"),
                    "artifact_id": f.get("artifact_id"), "role": f.get("role"),
                    "line_start": f.get("line_start"), "line_end": f.get("line_end"),
                }
                for f in files
            ],
            "anchors": {
                "before": before,
                "after": {
                    "file": primary.get("rel_path"),
                    "artifact_id": primary.get("artifact_id"),
                    "function": primary.get("func_name"),
                    "line_start": primary.get("line_start"),
                    "line_end": primary.get("line_end"),
                },
            },
            "provenance": {
                "source_file": primary.get("rel_path"),
                "line_start": primary.get("line_start"),
                "line_end": primary.get("line_end"),
                "line_confidence": "exact",
                "commit": cfg.multiarch_cfg.get("commit"),
                "extractor_version": cfg.extractor_version,
            },
            "delta": _delta_vs_generic(features, generic) if before else None,
        })
    return out


def _derive_actions(features: dict, method: str, generic: dict | None) -> tuple:
    actions = []
    notes = []
    if features.get("has_vector"):
        actions.append("vectorize")
        notes.append("检测到向量指令/向量寄存器（%s 条），实现数据级并行" % features.get("vector_instr_count"))
    if features.get("has_fma"):
        actions.append("fma")
        notes.append("检测到融合乘加（FMA/MLA）指令 %s 条" % features.get("fma_count"))
    if features.get("has_prefetch"):
        actions.append("prefetch")
        notes.append("检测到软件预取指令/宏 %s 处" % features.get("prefetch_count"))
    reg_block = features.get("reg_block_estimate") or 0
    if reg_block >= 4 or features.get("block_dims"):
        actions.append("tile")
        dims = features.get("block_dims")
        notes.append("寄存器分块线索：%s / 估计 %d 个向量累加器" % (dims, reg_block))
    if features.get("unroll_hint") or (features.get("scalar_unroll") or 0) >= 4:
        actions.append("unroll")
        notes.append("多路展开线索：scalar_unroll=%s, macro=%s" % (
            features.get("scalar_unroll"), features.get("macro_count")))
    if method == "diff" and generic:
        if (features.get("reg_block_estimate") or 0) > (generic.get("acc_count") or 0):
            if "tile" not in actions:
                actions.append("tile")
                notes.append("相对 generic 累加器数增加：%s -> %s" % (
                    generic.get("acc_count"), features.get("reg_block_estimate")))
        if generic.get("loop_depth") and features.get("loop_depth"):
            pass
    # fma 合并进 vectorize，避免同 (op, arch, action) 重复
    if "fma" in actions and "vectorize" in actions:
        actions.remove("fma")
    return actions, notes


def _delta_vs_generic(features: dict, generic: dict) -> dict:
    return {
        "intrinsic_count_old": generic.get("intrinsic_count"),
        "intrinsic_count_new": features.get("intrinsic_count"),
        "acc_old": generic.get("acc_count"),
        "acc_new": features.get("reg_block_estimate"),
        "loop_depth_old": generic.get("loop_depth"),
        "loop_depth_new": features.get("loop_depth"),
        "vector_width_bits": features.get("vector_width_bits"),
        "fma_count_new": features.get("fma_count"),
        "prefetch_count_new": features.get("prefetch_count"),
    }


def _param_list(action: str, features: dict, arch: str) -> list:
    params = []
    width = features.get("vector_width_bits")
    dims = features.get("block_dims")
    if action in ("vectorize", "specialize"):
        if width:
            params.append({
                "name": "vector_width", "label": "向量宽度", "kind": "vector_len", "unit": "bits",
                "value_in_source": width, "range": [128, 256, 512],
                "derived_from": "ISA 向量寄存器宽度",
                "effect_direction": "宽度越大单指令处理元素越多，但寄存器压力与尾部处理成本上升",
            })
    if action == "tile":
        if dims:
            params.append({
                "name": "reg_block_m", "label": "M 方向寄存器分块", "kind": "blocking", "unit": "elements",
                "value_in_source": dims[0], "range": [1, 2, 4, 8, 16],
                "derived_from": "内核文件名中的 block 维度（%dx%d）" % (dims[0], dims[1]),
                "effect_direction": "M 块越大装入值复用越多，但需要更多累加器寄存器",
            })
            params.append({
                "name": "reg_block_n", "label": "N 方向寄存器分块", "kind": "blocking", "unit": "elements",
                "value_in_source": dims[1], "range": [1, 2, 4, 8, 16],
                "derived_from": "内核文件名中的 block 维度（%dx%d）" % (dims[0], dims[1]),
                "effect_direction": "N 块越大 FMA 复用越多，受向量寄存器数量限制",
            })
        else:
            params.append({
                "name": "reg_block_n", "label": "向量累加器个数", "kind": "blocking", "unit": "accumulators",
                "value_in_source": features.get("reg_block_estimate"), "range": [2, 4, 8, 16, 32],
                "derived_from": "汇编中不同目标向量寄存器数量",
                "effect_direction": "累加器越多复用越高，超过 32 个向量寄存器会溢出",
            })
    if action == "unroll":
        value = (dims[0] * dims[1]) if dims else (features.get("scalar_unroll") or features.get("macro_count") or 2)
        params.append({
            "name": "unroll_factor", "label": "展开因子", "kind": "unroll", "unit": "iterations",
            "value_in_source": value, "range": [2, 4, 8, 16],
            "derived_from": "块维度/标量展开变量数/宏块数",
            "effect_direction": "展开越大循环开销越低、寄存器压力越大",
        })
    if action == "prefetch":
        params.append({
            "name": "prefetch_distance", "label": "预取距离", "kind": "prefetch_dist", "unit": "cacheline",
            "value_in_source": features.get("prefetch_count"), "range": [1, 2, 4, 8],
            "derived_from": "代码中的 prefetch 指令条数/距离常量",
            "effect_direction": "预取太近掩盖不了延迟，太远会污染 cache",
        })
    return params


def strategies(cfg, ma_diffs: list, ma_impls: dict, artifacts: list) -> list:
    """S3（arch 维度）：从每个 (op, arch) 的特征/动作生成 arch_specific 策略。"""
    by_artifact = {a["artifact_id"]: a for a in artifacts if a.get("kind") == "multiarch_impl"}
    out = []
    for diff in ma_diffs:
        features = ma_impls.get((diff["op"], diff["arch"]), {})
        files = features.get("files") or []
        primary = _primary(files)
        if not diff["actions"]:
            continue
        spec = cfg.all_operator_specs.get(diff["op"], {})
        detected_isa = features.get("detected_isa") or diff.get("isa") or ""
        display_label = _arch_display_label(diff["arch_label"], detected_isa)
        isa_node = ISA_NODE_BY_ISA.get(detected_isa)
        isa_variant = {
            "x86_avx2": "avx2", "x86_avx512": "avx512",
            "arm_neon": "neon", "arm_sve": "sve", "rvv": "rvv",
        }.get(diff["arch"], "portable")
        arch_evidence = []
        for f in files:
            arch_evidence.append({
                "arch": diff["arch"],
                "arch_family": diff["arch_family"],
                "arch_label": display_label,
                "isa": detected_isa or diff["isa"],
                "isa_node": isa_node,
                "artifact_id": f.get("artifact_id"),
                "file": f.get("rel_path"),
                "repo_path": (by_artifact.get(f.get("artifact_id")) or {}).get("repo_path"),
                "role": f.get("role"),
                "kind": f.get("kind"),
                "line_start": f.get("line_start"),
                "line_end": f.get("line_end"),
                "method": diff["method"],
            })
        for action in diff["actions"]:
            if action not in MA_ACTIONS:
                continue
            family, mapped_action, name_suffix, isa_key = MA_ACTIONS[action]
            params = _param_list(mapped_action, features, diff["arch"])
            from .extract import canonical_template  # 延迟导入，避免循环依赖

            template = canonical_template(mapped_action, isa_variant, {
                p["name"]: p.get("value_in_source") for p in params
            })
            conditions = _conditions(diff["op"], spec, features)
            note = next((n for n in diff["action_notes"] if _note_matches(action, n)), diff["action_notes"][0])
            out.append({
                "raw_key": "multiarch:%s:%s:%s" % (diff["arch"], diff["op"], action),
                "op": diff["op"],
                "source": "multiarch",
                "arch": diff["arch"],
                "arch_family": diff["arch_family"],
                "arch_scope": "arch_specific",
                "arch_evidence": arch_evidence,
                "architecture": diff["arch"],
                "level": "instruction" if diff["method"] == "static" else "code",
                "family": family,
                "action": mapped_action,
                "stage": "core" if mapped_action in ("vectorize", "tile", "unroll", "specialize") else (
                    "prep" if mapped_action == "prefetch" else "transform"),
                "name": "%s %s" % (display_label, name_suffix),
                "description": "%s %s实现（%s）。%s" % (
                    display_label, name_suffix, detected_isa or diff["isa"] or diff["arch"], note),
                "rationale": "S1 静态特征：vector_instr=%s, fma=%s, prefetch=%s, reg_block=%s, block_dims=%s, "
                             "instruction_count=%s（method=%s, detected_isa=%s）。" % (
                                 features.get("vector_instr_count"), features.get("fma_count"),
                                 features.get("prefetch_count"), features.get("reg_block_estimate"),
                                 features.get("block_dims"), features.get("instruction_count"),
                                 diff["method"], detected_isa or "unknown"),
                "trade_offs": "该实现绑定 %s；跨架构迁移需要重新参数化（向量宽度/寄存器块不同）。" % display_label,
                "conditions": conditions,
                "params": params,
                "template": template,
                "hardware_profile_ids": list(diff["profiles"]),
                "isa_requirements": ([{
                    "isa": isa_node, "hard": True,
                    "note": "%s：%s" % (display_label, detected_isa),
                }] if isa_node else []),
                "origins": arch_evidence,
                "provenance": {
                    "source_file": primary.get("rel_path"),
                    "line_start": primary.get("line_start"),
                    "line_end": primary.get("line_end"),
                    "line_confidence": "exact",
                    "commit": cfg.multiarch_cfg.get("commit"),
                    "extractor_version": cfg.extractor_version,
                    "origin_file": (by_artifact.get(primary.get("artifact_id")) or {}).get("repo_path"),
                },
                "support_count": len(files),
                "status": "verified" if diff["method"] == "diff" else "inferred",
                "confidence": 0.85 if diff["method"] == "diff" else 0.7,
                "extraction_method": diff["method"],
                "method": diff["method"],
                "pattern_stage": "core",
            })
    return out


def _note_matches(action: str, note: str) -> bool:
    keywords = {
        "vectorize": ("向量", "向量指令"),
        "tile": ("分块", "累加器"),
        "unroll": ("展开",),
        "prefetch": ("预取",),
    }
    return any(k in note for k in keywords.get(action, ()))


def _conditions(op: str, spec: dict, features: dict) -> list:
    depth = 3 if spec.get("op_class") in ("matrix", "triangular") else max(1, min(2, features.get("loop_depth") or 1))
    conditions = [
        {"pred": "dtype", "op": "in", "value": [spec.get("dtype", "d")], "hard": True},
        {"pred": "loop_depth", "op": "ge", "value": max(1, depth - 1 if depth > 2 else depth), "hard": True},
        {"pred": "size_regime", "op": "in", "value": ["medium", "large"], "hard": True},
    ]
    if spec.get("structural") and spec["structural"] != "none":
        conditions.append({"pred": "structural", "op": "eq", "value": spec["structural"], "hard": True})
    if spec.get("reduction"):
        conditions.append({"pred": "reduction", "op": "eq", "value": True, "hard": True})
    return conditions


def run(cfg, artifacts: list):
    """S0b 入口：发现 + 复制 + 返回 (源工件, characteristics, impls, diffs, 策略原始记录)。"""
    records = discover(cfg)
    ma_artifacts = materialize(cfg, records)
    chars, ma_impls = parse_impls(cfg, ma_artifacts)
    return ma_artifacts, chars, ma_impls


def arch_summary(ma_impls: dict) -> dict:
    """arch × op 覆盖与实现数（供统计/验收）。"""
    matrix = {}
    for (op, arch), features in ma_impls.items():
        matrix.setdefault(arch, {})[op] = len(features.get("files") or [])
    return matrix
