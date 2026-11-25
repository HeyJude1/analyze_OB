"""唯一命令行入口：python -m kgagent.cli <cmd>（SPEC §4）。

命令：
  build --offline|--llm [--data-dir DIR]
  validate [--data-dir DIR]
  retrieve --op dgemm --profile x86_avx2 [--top-k 6] [--trace-out DIR] [--code-features JSON]
  export-viz [--data-dir DIR] [--viz-dir DIR]
  stats [--data-dir DIR]
  mpsa --op daxpy --profile k1_rvv256 --code generic.c --mode b3|b4 --provider dashscope|deepseek|mock --out DIR
       [--top-k 6] [--analysis-k 3] [--model qwen-plus-2025-04-28] [--mock-code FILE]
  mpsa-repair --op daxpy --profile k1_rvv256 --code generated.c --error build.stderr.log
       --mode b3|b4 --provider dashscope|deepseek|mock --out DIR [--signature "..."] [--mock-code FILE]

退出码：0 成功；1 校验失败；2 参数/运行异常；
mpsa 额外：3 缺少对应 provider 的 API key；4 LLM 调用/解析失败；5 检索失败（见 stderr）。
mpsa-repair 额外：3 缺少对应 provider 的 API key；4 LLM 调用/解析失败。
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from . import __version__
from .config import load_config


def _add_common(parser):
    parser.add_argument("--config", default=None, help="config.yaml 路径（默认包内 kg_agent/config.yaml）")
    parser.add_argument("--data-dir", default=None, help="data 目录（默认 config.paths.data_dir）")


def _print(obj) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def cmd_build(args) -> int:
    from . import validate as validate_mod

    cfg = load_config(args.config)
    mode = "llm" if args.llm else "offline"
    if args.llm and args.offline:
        sys.stderr.write("[warn] 同时给了 --llm 与 --offline，按 --llm 处理\n")
    llm_client = None
    if mode == "llm":
        from .llm import LLMClient

        llm_client = LLMClient(cfg, backend="api")
        if cfg.llm_cfg.get("backend") == "offline":
            sys.stderr.write("[warn] config.llm.backend=offline，--llm 将降级为规则抽取\n")
    report = validate_mod.build(cfg, mode=mode, data_dir=args.data_dir, llm_client=llm_client,
                                verbose=True, split=args.split)
    stats = report["stats"]
    source = report["source_stats"]
    summary = {
        "status": "ok",
        "mode": mode,
        "split": args.split,
        "data_dir": str(Path(args.data_dir) if args.data_dir else (
            cfg.data_dir.with_name("data_" + args.split) if args.split else cfg.data_dir)),
        "nodes": stats["nodes"],
        "edges": stats["edges"],
        "strategies": stats["strategies"],
        "acceptance_count": stats["acceptance_count"],
        "condition_executable_rate": stats["condition_executable_rate"],
        "arch": {
            "implementation_files": report["arch"]["implementation_files"],
            "ops_with_multi_arch": report["arch"]["ops_with_multi_arch"],
            "ops_with_multi_arch_list": report["arch"]["ops_with_multi_arch_list"],
            "strategies_by_arch_scope": report["arch"]["strategies_by_arch_scope"],
            "strategies_by_arch": report["arch"]["strategies_by_arch"],
            "cross_arch_shared_count": report["arch"]["cross_arch_shared_count"],
        },
        "source": source,
        "warnings": report["warnings"][:10],
    }
    _print(summary)
    return 0


def cmd_validate(args) -> int:
    from . import validate as validate_mod

    cfg = load_config(args.config)
    report = validate_mod.validate_data_dir(cfg, data_dir=args.data_dir, require_acceptance=True)
    out = {
        "ok": report["ok"],
        "errors": report["errors"],
        "warnings": report["warnings"][:20],
        "stats": report.get("stats", {}),
    }
    _print(out)
    return 0 if report["ok"] else 1


def cmd_retrieve(args) -> int:
    from . import retrieve as retrieve_mod

    cfg = load_config(args.config)
    code_features = None
    if args.code_features:
        code_features = json.loads(args.code_features)
    trace = retrieve_mod.retrieve(
        cfg, args.op, profile=args.profile, top_k=args.top_k, code_features=code_features,
        data_dir=args.data_dir, traces_dir=args.trace_out, write=True, strict=args.strict,
        source_op=args.source_op, independent=args.independent,
    )
    pack = trace["result"]["pack"]
    summary = {
        "trace_id": trace["trace_id"],
        "trace_path": trace.get("trace_path"),
        "steps": len(trace["steps"]),
        "step_names": [s["name"] for s in trace["steps"]],
        "dropped_per_step": {s["name"]: len(s.get("dropped") or []) for s in trace["steps"]},
        "selected": [s["id"] for s in pack["strategies"]],
        "selected_names": [s["name"] for s in pack["strategies"]],
        "scores": {s["id"]: s["score"] for s in pack["strategies"]},
        "prompt_preview": trace["result"]["prompt_preview"],
        "plan": pack.get("plan"),
    }
    _print(summary)
    return 0


def cmd_mpsa(args) -> int:
    from . import mpsa

    cfg = load_config(args.config)
    result = mpsa.run_mpsa(
        cfg,
        op=args.op,
        profile=args.profile,
        mode=args.mode,
        provider=args.provider,
        model=args.model,
        code_path=args.code,
        top_k=args.top_k,
        analysis_k=args.analysis_k,
        out_dir=args.out,
        mock_code_path=args.mock_code,
        timeout_s=args.timeout,
        data_dir=args.data_dir,
    )
    for warning in result.get("warnings") or []:
        sys.stderr.write("[warn] %s\n" % warning)
    analysis = result.get("analysis") or {}
    synthesis = result.get("synthesis") or {}
    summary = {
        "status": "ok",
        "op": result["op"],
        "op_canonical": result["op_canonical"],
        "profile": result["profile"],
        "mode": result["mode"],
        "provider": result["provider"],
        "model": result["model"],
        "out_dir": result["artifacts"].get("out_dir"),
        "result_path": result["artifacts"].get("result_path"),
        "generated_c_path": result["artifacts"].get("generated_c_path"),
        "retrieved": {
            "count": result["retrieved"]["count"],
            "top_k": result["retrieved"]["top_k"],
            "strategy_ids": result["retrieved"]["strategy_ids"],
            "trace_id": result["retrieved"].get("trace_id"),
        },
        "analysis": None if result["mode"] == "b3" else {
            "analysis_k": analysis.get("analysis_k"),
            "count": analysis.get("count"),
            "strategy_ids": [item.get("strategy_id") for item in analysis.get("items") or []],
        },
        "synthesis": None if result["mode"] == "b3" else {
            "criteria_scores": synthesis.get("criteria_scores"),
            "overall": synthesis.get("overall"),
        },
        "usage": result["usage"],
        "elapsed_s": result["elapsed_s"],
        "code_meta": result["code_meta"],
        "code_preview": result["code"][:200],
    }
    _print(summary)
    return 0


def cmd_mpsa_repair(args) -> int:
    from . import mpsa

    cfg = load_config(args.config)
    result = mpsa.repair_code(
        cfg,
        op=args.op,
        profile=args.profile,
        code=args.code,
        error=args.error,
        mode=args.mode,
        provider=args.provider,
        model=args.model,
        signature=args.signature,
        out_dir=args.out,
        mock_code_path=args.mock_code,
        timeout_s=args.timeout,
    )
    for warning in result.get("warnings") or []:
        sys.stderr.write("[warn] %s\n" % warning)
    summary = {
        "status": "ok",
        "op": result["op"],
        "op_canonical": result["op_canonical"],
        "profile": result["profile"],
        "mode": result["mode"],
        "provider": result["provider"],
        "model": result["model"],
        "out_dir": result["artifacts"].get("out_dir"),
        "result_path": result["artifacts"].get("result_path"),
        "repaired_c_path": result["artifacts"].get("repaired_c_path"),
        "error_source": result["error_source"],
        "input_error_preview": result["input_error"][:200],
        "fix_summary": result["fix_summary"],
        "calls": result["calls"],
        "usage": result["usage"],
        "elapsed_s": result["elapsed_s"],
        "code_meta": result["code_meta"],
        "code_preview": result["code"][:200],
    }
    _print(summary)
    return 0


def cmd_export_viz(args) -> int:
    from . import viz_export

    cfg = load_config(args.config)
    result = viz_export.export(cfg, data_dir=args.data_dir, viz_dir=args.viz_dir)
    _print(result)
    return 0


def cmd_stats(args) -> int:
    from . import validate as validate_mod

    cfg = load_config(args.config)
    report = validate_mod.validate_data_dir(cfg, data_dir=args.data_dir, require_acceptance=False)
    stats = report.get("stats", {})
    arch = report.get("arch") or stats.get("arch") or {}
    out = {
        "data_dir": str(Path(args.data_dir) if args.data_dir else cfg.data_dir),
        "stats": stats,
        "arch": {
            "matrix": arch.get("arch_matrix"),
            "ops_with_multi_arch": arch.get("ops_with_multi_arch"),
            "ops_with_multi_arch_list": arch.get("ops_with_multi_arch_list"),
            "strategies_by_arch_scope": arch.get("strategies_by_arch_scope"),
            "strategies_by_arch": arch.get("strategies_by_arch"),
            "cross_arch_shared_count": arch.get("cross_arch_shared_count"),
            "cross_arch_shared": arch.get("cross_arch_shared"),
        },
        "acceptance_sample": report.get("acceptance", [])[:20],
        "errors": report.get("errors", [])[:10],
    }
    if args.matrix:
        sys.stdout.write(_format_arch_matrix(arch) + "\n")
    _print(out)
    return 0 if not report.get("errors") else 1


def _format_arch_matrix(arch: dict) -> str:
    """把 arch × operator 覆盖矩阵与跨架构共有策略表打印成人可读的 ASCII 表格。"""
    matrix = (arch or {}).get("arch_matrix") or {}
    ops = sorted({op for row in matrix.values() for op in row})
    arches = sorted(matrix.keys())
    if not ops or not arches:
        return "[arch matrix] 空（数据未包含多架构层）"
    lines = ["[arch × operator 覆盖矩阵] 单元格 = 该架构实现的源文件数（>=1 即覆盖）"]
    header = "arch".ljust(12) + "".join(op.center(8) for op in ops)
    lines.append(header)
    lines.append("-" * len(header))
    for arch_name in arches:
        row = matrix.get(arch_name, {})
        lines.append(arch_name.ljust(12) + "".join(
            (str(row[op]) if op in row else "-").center(8) for op in ops))
    lines.append("")
    lines.append("[跨架构共有策略] parameterized 策略（arch_scope=parameterized）")
    shared = (arch or {}).get("cross_arch_shared") or []
    if not shared:
        lines.append("  （无）")
    for item in shared:
        lines.append("  %-32s op=%-6s action=%-10s arches=%s" % (
            item["id"], item["op"], item["action"], ",".join(item["arches"])))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kgagent.cli",
        description="kg_agent v%s：算子优化知识图谱 build/validate/retrieve/export-viz/stats/mpsa/mpsa-repair"
        % __version__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="S0→S6 构建图谱（--offline 完全离线）")
    p_build.add_argument("--offline", action="store_true", help="纯规则离线构建（默认）")
    p_build.add_argument("--llm", action="store_true", help="S3 可调 LLM（有缓存，失败降级 offline）")
    p_build.add_argument("--split", choices=("thesis",), help="隔离构建论文训练图谱")
    _add_common(p_build)
    p_build.set_defaults(func=cmd_build)

    p_validate = sub.add_parser("validate", help="schema + 不变量校验（非 0 退出码=失败）")
    _add_common(p_validate)
    p_validate.set_defaults(func=cmd_validate)

    p_retrieve = sub.add_parser("retrieve", help="可解释检索 + 7 步 trace")
    p_retrieve.add_argument("--op", required=True, help="算子名，如 dgemm / gemm")
    p_retrieve.add_argument("--profile", default="x86_avx2", help="硬件 profile，如 x86_avx2 / k1_rvv256")
    p_retrieve.add_argument("--top-k", type=int, default=None, help="返回策略数（默认取 config）")
    p_retrieve.add_argument("--trace-out", default=None, help="trace 输出目录（默认 data/traces/）")
    p_retrieve.add_argument("--code-features", default=None, help="覆盖默认代码特征（JSON 字符串）")
    p_retrieve.add_argument("--strict", action="store_true", help="仅用任务显式提供的代码特征，返回候选优化计划")
    p_retrieve.add_argument("--source-op", help="仅从登记的来源算子抽象策略作类比（需 --strict）")
    p_retrieve.add_argument("--independent", action="store_true", help="按单卡排序，作为关系规划对照（需 --strict）")
    _add_common(p_retrieve)
    p_retrieve.set_defaults(func=cmd_retrieve)

    p_viz = sub.add_parser("export-viz", help="生成 viz/graph_data.js + viz/traces.js")
    p_viz.add_argument("--viz-dir", default=None, help="viz 目录（默认 kg_agent/viz）")
    _add_common(p_viz)
    p_viz.set_defaults(func=cmd_export_viz)

    p_stats = sub.add_parser("stats", help="节点/边/覆盖率/条件可执行率 + arch×op 矩阵")
    p_stats.add_argument("--matrix", action="store_true", help="额外打印 arch×operator 覆盖矩阵与跨架构共有策略表")
    _add_common(p_stats)
    p_stats.set_defaults(func=cmd_stats)

    p_mpsa = sub.add_parser(
        "mpsa", help="MPSA 多视角策略分析：B3（直接生成）/ B4（逐条分析+综合后生成）"
    )
    p_mpsa.add_argument("--op", required=True, help="算子名，如 daxpy / ddot / dgemm")
    p_mpsa.add_argument("--profile", default="k1_rvv256", help="硬件 profile，如 k1_rvv256 / x86_avx2")
    p_mpsa.add_argument("--code", default=None, help="基线 generic.c（提供任务函数签名与语义参考）")
    p_mpsa.add_argument("--mode", choices=("b3", "b4"), default="b3",
                        help="b3=StrategyPack 直接生成；b4=逐条适用性分析+综合后生成")
    p_mpsa.add_argument("--provider", choices=("dashscope", "deepseek", "mock"), default="dashscope",
                        help="dashscope/deepseek=真实 API（分别需 DASHSCOPE_API_KEY / DEEPSEEK_API_KEY）；mock=离线确定性输出")
    p_mpsa.add_argument("--out", required=True, help="输出目录（写 mpsa_result.json + generated.c）")
    p_mpsa.add_argument("--top-k", type=int, default=None, help="检索策略数（默认 config/6）")
    p_mpsa.add_argument("--analysis-k", type=int, default=None, help="B4 逐条分析策略数（默认 config/3）")
    p_mpsa.add_argument("--model", default=None, help="模型名（dashscope 默认 qwen-plus-2025-04-28；deepseek 默认 deepseek-chat）")
    p_mpsa.add_argument("--mock-code", default=None, help="mock provider 指定返回代码文件（优先于内置 mock 实现）")
    p_mpsa.add_argument("--timeout", type=int, default=None, help="LLM 超时秒数（默认 config/120）")
    _add_common(p_mpsa)
    p_mpsa.set_defaults(func=cmd_mpsa)

    p_repair = sub.add_parser(
        "mpsa-repair", help="MPSA 修复回路：把编译器 stderr / 运行错误回灌给 LLM 做 1 轮修复"
    )
    p_repair.add_argument("--op", required=True, help="算子名，如 daxpy / ddot")
    p_repair.add_argument("--profile", default="k1_rvv256", help="硬件 profile，如 k1_rvv256")
    p_repair.add_argument("--code", required=True, help="待修复的原 C 代码（文件路径或内联文本）")
    p_repair.add_argument("--error", required=True,
                          help="编译器 stderr / QEMU 运行输出（文件路径或内联文本）")
    p_repair.add_argument("--mode", choices=("b3", "b4"), default="b3",
                          help="生成条件来源（写入结果；修复行为一致）")
    p_repair.add_argument("--provider", choices=("dashscope", "deepseek", "mock"), default="dashscope",
                          help="dashscope/deepseek=真实 API（分别需 DASHSCOPE_API_KEY / DEEPSEEK_API_KEY）；mock=离线确定性修复")
    p_repair.add_argument("--out", required=True, help="输出目录（写 repair_result.json + repaired.c）")
    p_repair.add_argument("--model", default=None, help="模型名（dashscope 默认 qwen-plus-2025-04-28；deepseek 默认 deepseek-chat）")
    p_repair.add_argument("--signature", default=None,
                          help="任务函数签名（可选；默认从 --code 提取，再退回内置签名）")
    p_repair.add_argument("--mock-code", default=None,
                          help="mock provider 指定修复后代码文件（优先于内置 mock 修复）")
    p_repair.add_argument("--timeout", type=int, default=None, help="LLM 超时秒数（默认 config/120）")
    _add_common(p_repair)
    p_repair.set_defaults(func=cmd_mpsa_repair)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except SystemExit:
        raise
    except FileNotFoundError as exc:
        sys.stderr.write("[error] %s\n" % exc)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI 需要把异常转成可读输出
        # MPSA 异常带 exit_code（3=缺 key / 4=LLM 失败 / 5=检索失败），其余按 2
        exit_code = int(getattr(exc, "exit_code", 2) or 2)
        sys.stderr.write("[error] %s: %s\n" % (type(exc).__name__, exc))
        if "--debug" in sys.argv:
            traceback.print_exc()
        return exit_code


if __name__ == "__main__":
    sys.exit(main())
