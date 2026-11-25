#!/usr/bin/env python3
"""E09_MPSA/run_smoke.py —— MPSA B3 vs B4 冒烟/批跑脚本（Windows 可跑）。

一条链路：
    kg_agent mpsa CLI（生成代码）
      → 本脚本读 mpsa_result.json 的 code 写 generated_initial.c / generated.c
      → experiments/harness/qemu.py（WSL 交叉编译 + QEMU 运行 + 正确性解析）
      → [修复回路 --repair-rounds N] compile_failed 用 build.stderr、wrong_answer 用 qemu.stdout
         调 kg_agent mpsa-repair → 修复后重新编译/QEMU（保留 generated_initial.c）
      → results.jsonl 追加一行 + runs/<run_id>.md 报告 + summary.md 自动汇总

用法（在 `开题` 根目录）：
    python experiments\\E09_MPSA\\run_smoke.py --provider mock --ops daxpy --modes b3 --seed 1
    python experiments\\E09_MPSA\\run_smoke.py --provider mock --ops daxpy,ddot --modes b3,b4 --seed 1
    python experiments\\E09_MPSA\\run_smoke.py --provider dashscope --ops daxpy --modes b3 --seed 1

参数语义
--------
- `--provider mock`：不联网，验证链路（mock 代码是确定性标量参考实现，**不代表方法效果**）；
- `--provider dashscope`：真实 API，需环境变量 `DASHSCOPE_API_KEY`（不写进任何文件）；
- `--ops/--modes/--seed/--sizes`：逗号分隔；默认 2 算子 × 2 条件 × seed=1 × 每个 task.yaml 的中位尺寸；
- `--repair-rounds N`：修复回路（默认 0=关闭；1=初始失败后修复一次再评）；
- `--mpsa-backend auto|cli|stub`：auto 时若 CLI 缺 mpsa 子命令且 provider=mock，退回本地 stub（仅链路自检）。

输出（全部在 `experiments/E09_MPSA/` 下）：
- `results.jsonl`：append-only，一行一个运行单元；
- `artifacts/<run_id>/repro/`：可复现包（输入/硬件/检索链/每轮迭代/最终算子）；
- `runs/<YYYY-MM-DD_HHMM>__<run_id>__<STATUS>.md`：人读报告（跑完自动整理，一个 run 一个文件）；`runs/INDEX.md` 为索引；
- `artifacts/<run_id>/<op>_<cond>[_n<size>]/`：`mpsa_result.json`、`generated.c`、`mpsa_*.log`、`harness/`；
- `summary.md`：AUTO-SUMMARY 区间自动刷新。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent          # experiments/E09_MPSA
EXPERIMENTS = HERE.parent                       # experiments/
PROJ = EXPERIMENTS.parent                       # 科研产出/
KG_SRC = PROJ / "kg_agent" / "src"
sys.path.insert(0, str(EXPERIMENTS))

from harness.qemu import compile_run, read_task_meta  # noqa: E402
from harness.kcs import estimate_kcs  # noqa: E402
import harness.mock_mpsa as mock_mpsa  # noqa: E402
import harness.mock_repair as mock_repair  # noqa: E402
from harness.repro_package import build_run_package  # noqa: E402

EXPERIMENT_ID = "E09_MPSA"
PROFILE = "k1_rvv256"
RESULTS_PATH = HERE / "results.jsonl"
RUNS_DIR = HERE / "runs"
SUMMARY_PATH = HERE / "summary.md"
TASK_DIRS = {"daxpy": HERE / "tasks" / "daxpy", "ddot": HERE / "tasks" / "ddot"}
MODE_LABEL = {
    "b3": "B3 · StrategyPack 直注入 → 单次生成",
    "b4": "B4 · 多视角分析 + 综合 → 生成（MPSA）",
}
AUTO_BEGIN = "<!-- AUTO-SUMMARY:BEGIN -->"
AUTO_END = "<!-- AUTO-SUMMARY:END -->"
STATUSES = {
    "ok",
    "compile_failed",
    "wrong_answer",
    "qemu_timeout",
    "mpsa_failed",
    "no_code",
    "harness_error",
}


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _csv_ints(raw: str) -> list[int]:
    out = []
    for tok in str(raw).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _csv_strs(raw: str) -> list[str]:
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _rel(path: Path) -> str:
    """相对 E09_MPSA 的 POSIX 路径，便于 results.jsonl 跨机器引用。"""
    try:
        return path.resolve().relative_to(HERE).as_posix()
    except ValueError:
        return path.as_posix()


def _fmt_pct(num: int, den: int) -> str:
    if not den:
        return "—"
    return "%.1f%%" % (100.0 * num / den)


def _trim(text: str, n: int = 300) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n] + " …"


def _tail_lines(text: str, n: int = 15) -> str:
    lines = (text or "").strip().splitlines()
    return "\n".join(lines[-n:]) if lines else ""


def _mpsa_env() -> dict:
    env = os.environ.copy()
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(KG_SRC) + (os.pathsep + old if old else "")
    return env


def _mpsa_provenance() -> dict:
    """记录 kg_agent/mpsa 实现版本（CLI 可能在实验中升级，版本混杂会让结果不可追溯）。"""
    info = {
        "kgagent_version": None,
        "mpsa_py_sha256": None,
        "kg_src": str(KG_SRC),
    }
    try:
        init_text = (KG_SRC / "kgagent" / "__init__.py").read_text(encoding="utf-8", errors="replace")
        m = re.search(r"""__version__\s*=\s*["']([^"']+)["']""", init_text)
        info["kgagent_version"] = m.group(1) if m else None
    except OSError:
        pass
    mpsa_py = KG_SRC / "kgagent" / "mpsa.py"
    try:
        info["mpsa_py_sha256"] = hashlib.sha256(mpsa_py.read_bytes()).hexdigest()
    except OSError:
        pass
    cli_py = KG_SRC / "kgagent" / "cli.py"
    try:
        info["cli_py_sha256"] = hashlib.sha256(cli_py.read_bytes()).hexdigest()
    except OSError:
        pass
    return info


# ---------------------------------------------------------------------------
# 调 mpsa CLI
# ---------------------------------------------------------------------------


def _mpsa_cmd(op: str, mode: str, code_path: Path, out_dir: Path, args) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "kgagent.cli",
        "mpsa",
        "--op",
        op,
        "--profile",
        PROFILE,
        "--code",
        str(code_path),
        "--mode",
        mode,
        "--provider",
        args.provider,
        "--out",
        str(out_dir),
        "--top-k",
        str(args.top_k),
    ]
    if mode == "b4":
        cmd += ["--analysis-k", str(args.analysis_k)]
    if args.model:
        cmd += ["--model", args.model]
    if args.provider == "mock" and args.mock_code:
        cmd += ["--mock-code", str(Path(args.mock_code).resolve())]
    return cmd


def _call_mpsa(op: str, mode: str, code_path: Path, out_dir: Path, args) -> dict:
    """调 CLI 或（mock 兜底）stub，返回 {exit, stdout, stderr, elapsed_s, backend, cmd}。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = _mpsa_cmd(op, mode, code_path, out_dir, args)
    # 强制 stub：只允许 mock，不调 CLI（用于调试 harness 链路）
    if args.provider == "mock" and args.mpsa_backend == "stub":
        stub = mock_mpsa.run_stub(op, mode, code_path, out_dir, PROFILE, reason="--mpsa-backend stub 强制")
        (out_dir / "mpsa_stdout.log").write_text(stub["stdout"], encoding="utf-8")
        (out_dir / "mpsa_stderr.log").write_text("", encoding="utf-8")
        return {
            "exit": stub["exit"],
            "stdout": stub["stdout"],
            "stderr": "",
            "elapsed_s": stub["elapsed_s"],
            "backend": "stub",
            "cmd": cmd,
            "timeout": False,
        }
    t0 = time.time()
    stdout = stderr = ""
    exit_code = None
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(KG_SRC),
            env=_mpsa_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=args.timeout_mpsa,
            check=False,
        )
        exit_code, stdout, stderr = proc.returncode, proc.stdout or "", proc.stderr or ""
    except FileNotFoundError as exc:
        stderr = "[error] 无法启动 python/kgagent：%s" % exc
        exit_code = 127
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = "[error] mpsa 超过 %.0fs 未返回" % args.timeout_mpsa
        exit_code = 124
    elapsed = round(time.time() - t0, 3)

    (out_dir / "mpsa_stdout.log").write_text(stdout, encoding="utf-8")
    (out_dir / "mpsa_stderr.log").write_text(stderr, encoding="utf-8")

    # 本地兜底：只对 mock 生效；真实 API 绝不 stub
    cli_missing = bool(
        re.search(r"invalid choice:\s*'?mpsa", stderr) or "No module named 'kgagent'" in stderr
    )
    if args.provider == "mock" and args.mpsa_backend != "cli" and cli_missing:
        stub = mock_mpsa.run_stub(
            op, mode, code_path, out_dir, PROFILE, reason="mpsa CLI 不可用（invalid choice）"
        )
        return {
            "exit": stub["exit"],
            "stdout": stdout + stub["stdout"],
            "stderr": stderr,
            "elapsed_s": round(elapsed + stub["elapsed_s"], 3),
            "backend": "stub",
            "cmd": cmd,
            "timeout": False,
        }
    return {
        "exit": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "elapsed_s": elapsed,
        "backend": "cli",
        "cmd": cmd,
        "timeout": timed_out,
    }


# ---------------------------------------------------------------------------
# 修复回路：compile_failed → build.stderr 调 mpsa-repair；wrong_answer → qemu.stdout 调 repair
# ---------------------------------------------------------------------------

# 可触发修复回路的初始失败类型 → (错误类型标签, 日志文件名)
REPAIR_ERROR_FILES = {
    "compile_failed": ("build_stderr", "build.stderr.log"),
    "wrong_answer": ("qemu_stdout", "qemu.stdout.log"),
}


def _repair_error_source(
    status: str, harness: dict | None, harness_dir: Path
) -> tuple[str | None, Path | None]:
    """按 E09 协议选修复输入：编译失败用 build.stderr，数值错误用 qemu.stdout（含 maxdiff/reldiff）。"""
    item = REPAIR_ERROR_FILES.get(status)
    if not item:
        return None, None
    kind, fname = item
    logs = (harness or {}).get("logs") or {}
    path = Path(logs.get(kind)) if logs.get(kind) else harness_dir / fname
    return kind, path


def _repair_cmd(op: str, mode: str, code_path: Path, error_path: Path, out_dir: Path, args) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "kgagent.cli",
        "mpsa-repair",
        "--op",
        op,
        "--profile",
        PROFILE,
        "--code",
        str(code_path),
        "--error",
        str(error_path),
        "--mode",
        mode,
        "--provider",
        args.provider,
        "--out",
        str(out_dir),
    ]
    if args.model:
        cmd += ["--model", args.model]
    if args.provider == "mock" and args.repair_mock_code:
        cmd += ["--mock-code", str(Path(args.repair_mock_code).resolve())]
    return cmd


def _repair_fallback_code(op: str, args) -> Path:
    """mock 兜底修复返回的代码：--repair-mock-code > 任务卡 generic.c。"""
    if args.repair_mock_code:
        return Path(args.repair_mock_code).resolve()
    return (TASK_DIRS[op] / "input" / "generic.c").resolve()


def _call_repair(
    op: str, mode: str, code_path: Path, error_path: Path, out_dir: Path, args
) -> dict:
    """调 mpsa-repair CLI（或 mock 兜底 stub）。

    返回 {exit, stdout, stderr, elapsed_s, backend, cmd, timeout}；
    backend = "cli"（真 CLI）| "stub"（本地 mock 兜底，仅链路自检）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = _repair_cmd(op, mode, code_path, error_path, out_dir, args)

    # 强制 stub（仅 mock，用于链路自检/无 CLI 调试）
    if args.provider == "mock" and args.repair_backend == "stub":
        stub = mock_repair.run_stub(
            op, mode, code_path, error_path, out_dir, PROFILE,
            fixed_code_path=_repair_fallback_code(op, args),
            reason="--repair-backend stub 强制",
        )
        (out_dir / "repair_stdout.log").write_text(stub["stdout"], encoding="utf-8")
        (out_dir / "repair_stderr.log").write_text("", encoding="utf-8")
        return {
            "exit": stub["exit"],
            "stdout": stub["stdout"],
            "stderr": "",
            "elapsed_s": stub["elapsed_s"],
            "backend": "stub",
            "cmd": cmd,
            "timeout": False,
        }

    t0 = time.time()
    stdout = stderr = ""
    exit_code = None
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(KG_SRC),
            env=_mpsa_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=args.timeout_repair,
            check=False,
        )
        exit_code, stdout, stderr = proc.returncode, proc.stdout or "", proc.stderr or ""
    except FileNotFoundError as exc:
        stderr = "[error] 无法启动 python/kgagent：%s" % exc
        exit_code = 127
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = "[error] mpsa-repair 超过 %.0fs 未返回" % args.timeout_repair
        exit_code = 124
    elapsed = round(time.time() - t0, 3)

    (out_dir / "repair_stdout.log").write_text(stdout, encoding="utf-8")
    (out_dir / "repair_stderr.log").write_text(stderr, encoding="utf-8")

    # 本地兜底：只对 mock 生效；真实 API 绝不 stub
    cli_missing = bool(
        re.search(r"invalid choice:\s*'?mpsa-repair", stderr)
        or "No module named 'kgagent'" in stderr
    )
    if args.provider == "mock" and args.repair_backend != "cli" and cli_missing:
        stub = mock_repair.run_stub(
            op, mode, code_path, error_path, out_dir, PROFILE,
            fixed_code_path=_repair_fallback_code(op, args),
            reason="mpsa-repair CLI 不可用（invalid choice）",
        )
        return {
            "exit": stub["exit"],
            "stdout": stdout + stub["stdout"],
            "stderr": stderr,
            "elapsed_s": round(elapsed + stub["elapsed_s"], 3),
            "backend": "stub",
            "cmd": cmd,
            "timeout": False,
        }
    return {
        "exit": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "elapsed_s": elapsed,
        "backend": "cli",
        "cmd": cmd,
        "timeout": timed_out,
    }


def _sum_usage(usages: list[dict]) -> dict:
    keys = ("calls", "prompt_tokens", "completion_tokens", "total_tokens")
    agg = {k: 0 for k in keys}
    for u in usages:
        if not isinstance(u, dict):
            continue
        for k in keys:
            agg[k] += int(u.get(k) or 0)
    return agg


def _repair_result_paths(out_dir: Path) -> list[Path]:
    """修复产物：契约是 repair_result.json；代码文件兼容 repaired.c / generated.c。"""
    return [out_dir / "repair_result.json", out_dir / "repaired.c", out_dir / "generated.c"]


def _evaluate(
    task_dir: Path, gen_path: Path, harness_dir: Path, size: int, reps: int
) -> tuple[dict | None, str, str | None]:
    """编译 + QEMU，返回 (harness_result, status, error)；harness 自身异常时 h=None。"""
    try:
        h = compile_run(task_dir, gen_path, harness_dir, n=size, reps=reps)
    except Exception as exc:  # noqa: BLE001 - 任何 harness 异常都要落库可诊断
        return None, "harness_error", "harness 异常：%s: %s" % (type(exc).__name__, exc)
    if not h.get("compile_ok"):
        return h, "compile_failed", h.get("error")
    if h.get("timeout"):
        return h, "qemu_timeout", h.get("error")
    if not h.get("correct"):
        return h, "wrong_answer", h.get("error")
    return h, "ok", None


def _apply_harness(rec: dict, h: dict, status: str, error: str | None) -> None:
    rec.update(
        {
            "compile_ok": bool(h.get("compile_ok")),
            "correct": bool(h.get("correct")),
            "maxdiff": h.get("maxdiff"),
            "reldiff": h.get("reldiff"),
            "qemu_exit": h.get("qemu_exit"),
            "timeout": bool(h.get("timeout")),
            "best_ms_qemu_only": h.get("best_ms"),
        }
    )
    rec["status"] = status
    rec["error"] = error


def _set_harness_paths(rec: dict, harness_dir: Path) -> None:
    rec["harness_dir"] = _rel(harness_dir)
    rec["paths"]["build_script"] = _rel(harness_dir / "build.sh")
    rec["paths"]["run_script"] = _rel(harness_dir / "run.sh")
    rec["paths"]["build_stderr"] = _rel(harness_dir / "build.stderr.log")
    rec["paths"]["qemu_stdout"] = _rel(harness_dir / "qemu.stdout.log")


# ---------------------------------------------------------------------------
# 从 mpsa_result.json 提取摘要
# ---------------------------------------------------------------------------


def _result_summary(res: dict | None) -> dict:
    if not isinstance(res, dict):
        return {}
    retrieved = res.get("retrieved") or {}
    strategies = retrieved.get("strategies") or []
    ids = retrieved.get("strategy_ids") or [
        s.get("id") for s in strategies if isinstance(s, dict) and s.get("id")
    ]
    usage = res.get("usage") or {}
    prompt = res.get("prompt") or {}
    analysis = res.get("analysis")
    items = (analysis or {}).get("items") if isinstance(analysis, dict) else None
    synthesis = (res.get("synthesis") or {}) if isinstance(res.get("synthesis"), dict) else {}
    analysis_prompts = prompt.get("analysis") or []
    return {
        "model": res.get("model"),
        "op_canonical": res.get("op_canonical"),
        "elapsed_s": res.get("elapsed_s"),
        "retrieved_count": retrieved.get("count", len(ids)),
        "retrieved_ids": ids,
        "retrieved_strategies": [
            {
                "id": s.get("id"),
                "name": s.get("name"),
                "action": s.get("action"),
                "score": s.get("score"),
            }
            for s in strategies
            if isinstance(s, dict)
        ],
        "analysis_count": (analysis or {}).get("analysis_k")
        if isinstance(analysis, dict)
        else None,
        "analysis_items": items or [],
        "analysis_calls": (analysis or {}).get("calls") if isinstance(analysis, dict) else None,
        "synthesis_criteria": synthesis.get("criteria_scores"),
        "synthesis_overall": synthesis.get("overall"),
        "synthesis_recommended": synthesis.get("recommended"),
        "usage": usage,
        "prompt_chars": {
            "generate": len(prompt.get("generate") or ""),
            "analysis": sum(len(x) for x in analysis_prompts if isinstance(x, str))
            if isinstance(analysis_prompts, list)
            else 0,
            "synthesis": len(prompt.get("synthesis") or "")
            if isinstance(prompt.get("synthesis"), str)
            else 0,
        },
        "code_meta": res.get("code_meta"),
        "warnings": res.get("warnings") or [],
        "stub": bool(res.get("stub")),
    }


# ---------------------------------------------------------------------------
# 单个 cell：mpsa → 编译/QEMU → 记录
# ---------------------------------------------------------------------------


def _run_cell(op: str, mode: str, seed: int, size: int, cell_dir: Path, args) -> dict:
    task_dir = TASK_DIRS[op]
    generic = task_dir / "input" / "generic.c"
    cell_dir.mkdir(parents=True, exist_ok=True)

    rec = {
        "ts": _now(),
        "experiment": EXPERIMENT_ID,
        "run_id": args._run_id,
        "provider": args.provider,
        "model": args.model or "default",
        "profile": PROFILE,
        "op": op,
        "condition": mode,
        "seed": seed,
        "size": size,
        "reps": args.reps,
        "mpsa_backend": None,
        "mpsa_exit": None,
        "mpsa_elapsed_s": None,
        "mpsa_cmd": None,
        "mpsa_impl": _mpsa_provenance(),
        "usage": {},
        "retrieved_count": None,
        "retrieved_ids": [],
        "analysis_count": None,
        "status": None,
        "compile_ok": False,
        "correct": False,
        "maxdiff": None,
        "reldiff": None,
        "qemu_exit": None,
        "timeout": False,
        "best_ms_qemu_only": None,
        "qemu_only": True,
        "kcs": None,
        "kcs_detail": None,
        "generated_code_sha256": None,
        "generated_code_bytes": None,
        "generated_initial_sha256": None,
        "generated_initial_bytes": None,
        "out_dir": _rel(cell_dir),
        "harness_dir": None,
        "initial_harness_dir": None,
        "paths": {
            "mpsa_result": _rel(cell_dir / "mpsa_result.json"),
            "generated_initial": _rel(cell_dir / "generated_initial.c"),
            "generated_c": _rel(cell_dir / "generated.c"),
        },
        # --- 修复回路（--repair-rounds）---
        "status_initial": None,
        "compile_ok_initial": False,
        "correct_initial": False,
        "maxdiff_initial": None,
        "qemu_exit_initial": None,
        "kcs_initial": None,
        "error_initial": None,
        "repair_backend": None,
        "repair_rounds": 0,
        "repair_tokens": 0,
        "repair_usage": {},
        "repair_out_dir": None,
        "repairs": [],
        "error": None,
        "notes": "",
    }
    detail = {"result": None, "harness": None, "summary": {}, "repairs": []}

    # 1) mpsa（生成代码）
    mpsa = _call_mpsa(op, mode, generic, cell_dir, args)
    rec["mpsa_backend"] = mpsa["backend"]
    rec["mpsa_exit"] = mpsa["exit"]
    rec["mpsa_elapsed_s"] = mpsa["elapsed_s"]
    rec["mpsa_cmd"] = " ".join(mpsa["cmd"])

    result_path = cell_dir / "mpsa_result.json"
    res = _read_json(result_path)
    detail["result"] = res
    if res is None:
        rec["status"] = "mpsa_failed"
        rec["error"] = "mpsa 失败（exit=%r）：%s" % (mpsa["exit"], _trim(mpsa["stderr"], 500))
        return {"record": rec, "detail": detail}

    summary = _result_summary(res)
    detail["summary"] = summary
    rec["usage"] = summary.get("usage") or {}
    rec["retrieved_count"] = summary.get("retrieved_count")
    rec["retrieved_ids"] = summary.get("retrieved_ids") or []
    rec["analysis_count"] = summary.get("analysis_count")
    rec["model"] = summary.get("model") or rec["model"]

    code = res.get("code") or ""
    if not code:
        gen_file = cell_dir / "generated.c"
        code = gen_file.read_text(encoding="utf-8", errors="replace") if gen_file.is_file() else ""
    if not code:
        rec["status"] = "no_code"
        rec["error"] = "mpsa_result.json 无 code 且 generated.c 不存在"
        return {"record": rec, "detail": detail}

    gen_path = cell_dir / "generated.c"
    init_path = cell_dir / "generated_initial.c"
    gen_path.write_text(code, encoding="utf-8", newline="\n")
    init_path.write_text(code, encoding="utf-8", newline="\n")
    detail["code_initial"] = code
    detail["code_final"] = code
    rec["generated_initial_sha256"] = _sha256_text(code)
    rec["generated_initial_bytes"] = len(code.encode("utf-8"))
    rec["generated_code_sha256"] = rec["generated_initial_sha256"]
    rec["generated_code_bytes"] = rec["generated_initial_bytes"]
    rec["kcs_detail"] = estimate_kcs(code, res.get("retrieved"))
    rec["kcs"] = rec["kcs_detail"].get("kcs")

    # 2) 初始实现：编译 + QEMU
    harness_dir = cell_dir / "harness"
    rec["initial_harness_dir"] = _rel(harness_dir)
    h, status, error = _evaluate(task_dir, gen_path, harness_dir, size, args.reps)
    if h is None:
        rec["status"] = rec["status_initial"] = "harness_error"
        rec["error"] = error
        return {"record": rec, "detail": detail}
    detail["harness"] = h
    _apply_harness(rec, h, status, error)
    _set_harness_paths(rec, harness_dir)

    # 初始快照（修复前的真实结果，供 FCR_initial 与 run md 对照）
    rec["status_initial"] = rec["status"]
    rec["compile_ok_initial"] = rec["compile_ok"]
    rec["correct_initial"] = rec["correct"]
    rec["maxdiff_initial"] = rec["maxdiff"]
    rec["qemu_exit_initial"] = rec["qemu_exit"]
    rec["kcs_initial"] = rec["kcs"]
    rec["error_initial"] = rec["error"]

    # 3) 修复回路：compile_failed → build.stderr；wrong_answer → qemu.stdout；修复后重新编译/QEMU
    repair_usages: list[dict] = []
    for attempt in range(1, max(0, int(args.repair_rounds)) + 1):
        if rec["status"] not in REPAIR_ERROR_FILES:
            break
        round_dir = cell_dir / ("repair_r%d" % attempt)
        kind, err_path = _repair_error_source(rec["status"], h, harness_dir)
        round_info = {
            "round": attempt,
            "input_status": rec["status"],
            "error_kind": kind,
            "error_path": _rel(err_path) if err_path else None,
            "repair_backend": None,
            "repair_exit": None,
            "repair_elapsed_s": None,
            "repair_usage": {},
            "repair_tokens": 0,
            "result_path": _rel(round_dir / "repair_result.json"),
            "out_dir": _rel(round_dir),
            "status": None,
            "compile_ok": False,
            "correct": False,
            "maxdiff": None,
            "qemu_exit": None,
            "code_sha256": None,
            "error": None,
        }
        if err_path is None or not err_path.is_file():
            round_info["error"] = "找不到修复输入日志（%s）" % kind
            rec["repairs"].append(round_info)
            break

        rep = _call_repair(op, mode, gen_path, err_path, round_dir, args)
        round_info["repair_backend"] = rep["backend"]
        round_info["repair_exit"] = rep["exit"]
        round_info["repair_elapsed_s"] = rep["elapsed_s"]
        rec["repair_backend"] = rep["backend"]
        rr = _read_json(round_dir / "repair_result.json")
        detail["repairs"].append({"round": attempt, "result": rr, "call": rep})
        round_info["fix_summary"] = (rr or {}).get("fix_summary")

        repaired_code = (rr or {}).get("code") or ""
        if not repaired_code:
            for cand in (round_dir / "repaired.c", round_dir / "generated.c"):
                if cand.is_file():
                    repaired_code = cand.read_text(encoding="utf-8", errors="replace")
                    break
        usage = (rr or {}).get("usage") or {}
        repair_usages.append(usage)
        round_info["repair_usage"] = usage
        round_info["repair_tokens"] = int(usage.get("total_tokens") or 0)

        if not repaired_code:
            round_info["error"] = "修复失败（exit=%r）：%s" % (rep["exit"], _trim(rep["stderr"], 400))
            rec["repairs"].append(round_info)
            rec["repair_rounds"] = attempt
            rec["repair_out_dir"] = _rel(round_dir)
            break

        # 修复后代码：写回 generated.c（generated_initial.c 保留初版）
        round_dir.mkdir(parents=True, exist_ok=True)
        (round_dir / "generated.c").write_text(repaired_code, encoding="utf-8", newline="\n")
        gen_path.write_text(repaired_code, encoding="utf-8", newline="\n")
        detail["code_final"] = repaired_code
        rec["generated_code_sha256"] = _sha256_text(repaired_code)
        rec["generated_code_bytes"] = len(repaired_code.encode("utf-8"))
        rec["kcs_detail"] = estimate_kcs(repaired_code, res.get("retrieved"))
        rec["kcs"] = rec["kcs_detail"].get("kcs")

        # 重新编译 + QEMU
        r_harness = cell_dir / ("harness_r%d" % attempt)
        rh, rstatus, rerror = _evaluate(task_dir, gen_path, r_harness, size, args.reps)
        round_info["code_sha256"] = rec["generated_code_sha256"]
        if rh is None:
            round_info["error"] = rerror
            rec["repairs"].append(round_info)
            rec["repair_rounds"] = attempt
            rec["repair_out_dir"] = _rel(round_dir)
            rec["status"] = "harness_error"
            rec["error"] = rerror
            break
        _apply_harness(rec, rh, rstatus, rerror)
        _set_harness_paths(rec, r_harness)
        h = rh
        round_info.update(
            {
                "status": rstatus,
                "compile_ok": bool(rh.get("compile_ok")),
                "correct": bool(rh.get("correct")),
                "maxdiff": rh.get("maxdiff"),
                "qemu_exit": rh.get("qemu_exit"),
                "error": rerror,
            }
        )
        rec["repairs"].append(round_info)
        rec["repair_rounds"] = attempt
        rec["repair_out_dir"] = _rel(round_dir)
        if rec["status"] == "ok":
            break

    rec["repair_usage"] = _sum_usage(repair_usages)
    rec["repair_tokens"] = int(rec["repair_usage"].get("total_tokens") or 0)
    return {"record": rec, "detail": detail}


# ---------------------------------------------------------------------------
# run 级别：run_id、报告、summary 自动区
# ---------------------------------------------------------------------------


def _resolve_run_id(raw: str | None, provider: str, seeds: list[int], tag: str | None) -> str:
    if raw:
        base = raw.strip()
    else:
        tag = (tag or ("smoke" if provider == "mock" else "run")).strip("_")
        seed_part = "seed%d" % seeds[0] if len(seeds) == 1 else "seeds%s" % "-".join(map(str, seeds))
        base = "%s_%s_%s" % (tag, seed_part, _stamp())
    # 唯一化：runs/<id>.md 已存在则加 _r2/_r3 ...
    candidate = base
    n = 2
    while (RUNS_DIR / (candidate + ".md")).exists() or (RUNS_DIR / candidate).exists():
        candidate = "%s_r%d" % (base, n)
        n += 1
    return candidate


def _render_run_md(run_meta: dict, cells: list[dict]) -> str:
    records = [c["record"] for c in cells]
    n_cells = len(records)
    n_ok = sum(1 for r in records if r["status"] == "ok")
    n_compile = sum(1 for r in records if r["compile_ok"])
    n_correct = sum(1 for r in records if r["correct"])
    tokens = sum(int((r.get("usage") or {}).get("total_tokens") or 0) for r in records)
    calls = sum(int((r.get("usage") or {}).get("calls") or 0) for r in records)
    kcs_vals = [r["kcs"] for r in records if r.get("kcs") is not None]
    kcs_avg = (sum(kcs_vals) / len(kcs_vals)) if kcs_vals else None

    env = {}
    if cells:
        h0 = (cells[0].get("detail") or {}).get("harness") or {}
        env = h0.get("env") or {}
    lines = []
    lines.append("# E09 MPSA run `%s`" % run_meta["run_id"])
    lines.append("")
    lines.append("> 可复现包：`../artifacts/%s/repro/README.md`（输入/硬件/检索链/每轮迭代/最终算子）" % run_meta["run_id"])
    lines.append("")
    lines.append("> 生成时间：%s ｜ 实验：%s ｜ provider：`%s` ｜ profile：`%s`" % (
        run_meta["created_at"], EXPERIMENT_ID, run_meta["provider"], PROFILE))
    if run_meta.get("note"):
        lines.append(">")
        lines.append("> ⚠️ %s" % run_meta["note"])
    lines.append("")
    lines.append("## 1. 配置")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append("| run_id | `%s` |" % run_meta["run_id"])
    models = ", ".join(sorted({str(r.get("model")) for r in records})) or run_meta["model"]
    lines.append("| provider / model | `%s` / `%s` |" % (run_meta["provider"], models))
    lines.append("| 条件 | %s |" % ", ".join("%s（%s）" % (m, MODE_LABEL.get(m, m)) for m in run_meta["modes"]))
    lines.append("| 算子 | %s |" % ", ".join(run_meta["ops"]))
    lines.append("| seeds | %s |" % ", ".join(map(str, run_meta["seeds"])))
    lines.append("| sizes / reps | %s / %s |" % (", ".join(map(str, run_meta["sizes"])), run_meta["reps"]))
    lines.append("| mpsa backend | %s |" % ", ".join(sorted({str(r.get("mpsa_backend")) for r in records})))
    impl = (records[0].get("mpsa_impl") or {}) if records else {}
    lines.append("| mpsa impl | kgagent `%s` / mpsa.py sha256 `%s` / cli.py `%s` |" % (
        impl.get("kgagent_version") or "—",
        (impl.get("mpsa_py_sha256") or "—")[:16],
        (impl.get("cli_py_sha256") or "—")[:16],
    ))
    if run_meta.get("repair_rounds"):
        lines.append("| repair | rounds=%d / backend=%s |" % (
            run_meta["repair_rounds"], run_meta.get("repair_backend", "auto")))
    lines.append("| WSL / CC | `%s` / `%s` |" % (env.get("wsl_distro", "?"), env.get("cc", "?")))
    lines.append("| CFLAGS | `%s` |" % " ".join(env.get("cflags", [])))
    lines.append("| QEMU | `%s` `-cpu %s` |" % (env.get("qemu", "?"), env.get("qemu_cpu", "?")))
    lines.append("")
    lines.append("## 2. 命令")
    lines.append("")
    lines.append("mpsa（示例，实际每 cell 的完整命令见 `results.jsonl.mpsa_cmd`）：")
    lines.append("")
    lines.append("```powershell")
    if records:
        lines.append(records[0]["mpsa_cmd"] or "")
    lines.append("```")
    lines.append("")
    lines.append("编译 + QEMU（`experiments/harness/qemu.py` 生成，产物在 `harness/`）：")
    lines.append("")
    lines.append("```bash")
    if cells and cells[0].get("detail", {}).get("harness"):
        h = cells[0]["detail"]["harness"]
        lines.append("%s %s ... -o bin" % (h["env"]["cc"], " ".join(h["env"]["cflags"])))
        lines.append("%s -cpu %s ./bin %d %d" % (
            h["env"]["qemu"], h["env"]["qemu_cpu"], run_meta["sizes"][0], run_meta["reps"]))
    lines.append("```")
    lines.append("")
    lines.append("## 3. 结果总览")
    lines.append("")
    lines.append("| op | 条件 | size | initial | repair_rounds | compile_ok | correct | maxdiff | qemu_exit | gen_tokens | repair_tokens | mpsa_s | KCS | status |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in records:
        lines.append("| %s | %s | %s | `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s | `%s` |" % (
            r["op"], r["condition"], r["size"],
            r.get("status_initial") or r["status"],
            r.get("repair_rounds") or 0,
            "✅" if r["compile_ok"] else "❌",
            "✅" if r["correct"] else "❌",
            "—" if r["maxdiff"] is None else "%.2e" % r["maxdiff"],
            r["qemu_exit"],
            int((r.get("usage") or {}).get("total_tokens") or 0),
            int(r.get("repair_tokens") or 0),
            r["mpsa_elapsed_s"],
            "—" if r["kcs"] is None else "%.2f" % r["kcs"],
            r["status"],
        ))
    lines.append("")
    n_init_correct = sum(1 for r in records if r.get("correct_initial", r.get("correct")))
    n_init_compile = sum(1 for r in records if r.get("compile_ok_initial", r.get("compile_ok")))
    repaired_cells = [r for r in records if int(r.get("repair_rounds") or 0) > 0]
    n_turned = sum(1 for r in repaired_cells if r.get("correct") and not r.get("correct_initial"))
    repair_tokens_total = sum(int(r.get("repair_tokens") or 0) for r in records)
    lines.append("汇总：**cells=%d，ok=%d，FCR_final=%.1f%%（initial %.1f%%）**，编译通过率 final %s（initial %s），总 gen tokens=%d（LLM calls=%d）%s" % (
        n_cells,
        n_ok,
        100.0 * n_correct / n_cells if n_cells else 0.0,
        100.0 * n_init_correct / n_cells if n_cells else 0.0,
        _fmt_pct(n_compile, n_cells),
        _fmt_pct(n_init_compile, n_cells),
        tokens,
        calls,
        "" if kcs_avg is None else "，KCS 均值 %.2f（n=%d）" % (kcs_avg, len(kcs_vals)),
    ))
    if run_meta.get("repair_rounds"):
        lines.append("修复回路：**%d cells 触发修复，修复后转正 %d，修复 tokens=%d**（初始失败类型见第 4 节）。" % (
            len(repaired_cells), n_turned, repair_tokens_total))
    lines.append("")
    lines.append("> `ok`/FCR 只表示“编译通过 + 数值正确”，**不含性能含义**；QEMU 的 `best=` 毫秒数不得进性能表（见 `08b` §5.1）。")
    lines.append("")
    lines.append("## 4. 逐 cell 详情")
    lines.append("")
    for c in cells:
        r, d = c["record"], c["detail"]
        s = d.get("summary") or {}
        h = d.get("harness") or {}
        lines.append("### %s × %s（seed=%s，n=%s）" % (r["op"], r["condition"], r["seed"], r["size"]))
        lines.append("")
        if run_meta.get("repair_rounds"):
            lines.append("- **initial**：status=`%s`；compile_ok=%s，correct=%s，maxdiff=%s，qemu_exit=%s，kcs=%s" % (
                r.get("status_initial") or r["status"],
                "✅" if r.get("compile_ok_initial") else "❌",
                "✅" if r.get("correct_initial") else "❌",
                "—" if r.get("maxdiff_initial") is None else "%.3e" % r["maxdiff_initial"],
                "—" if r.get("qemu_exit_initial") is None else r["qemu_exit_initial"],
                "—" if r.get("kcs_initial") is None else "%.2f" % r["kcs_initial"],
            ))
            repairs = r.get("repairs") or []
            if repairs:
                lines.append("- **修复回路**：backend=`%s`，执行 %d 轮" % (
                    r.get("repair_backend") or "—", r.get("repair_rounds") or 0))
                for rp in repairs:
                    fs = rp.get("fix_summary") or {}
                    fs_txt = ""
                    if isinstance(fs, dict) and fs:
                        fs_txt = "；fix_summary changed=%s chars %s→%s" % (
                            fs.get("changed"),
                            fs.get("original_chars"),
                            fs.get("repaired_chars"),
                        )
                    lines.append(
                        "  - r%s：input=`%s`，error=%s（`%s`）→ repair exit=%s、%.2fs、tokens=%s → status=`%s` correct=%s%s%s"
                        % (
                            rp.get("round"), rp.get("input_status"), rp.get("error_kind"),
                            rp.get("error_path"), rp.get("repair_exit"),
                            rp.get("repair_elapsed_s") or 0.0, rp.get("repair_tokens", 0),
                            rp.get("status"), "✅" if rp.get("correct") else "❌",
                            fs_txt,
                            "" if not rp.get("error") else "；error：%s" % _trim(str(rp["error"]), 200),
                        )
                    )
            else:
                lines.append("- **修复回路**：未触发（初始 `%s`，无需/不可修复）" % (
                    r.get("status_initial") or r["status"]))
        lines.append("- **status（final）**：`%s`%s" % (
            r["status"], "" if not r.get("error") else "；error：%s" % _trim(str(r["error"]), 300)))
        lines.append("- **mpsa**：backend=`%s` exit=%s elapsed=%.2fs，model=`%s`" % (
            r["mpsa_backend"], r["mpsa_exit"], r["mpsa_elapsed_s"] or 0.0, r["model"]))
        lines.append("- **检索**：%s 条策略 %s" % (
            s.get("retrieved_count", len(r.get("retrieved_ids") or [])),
            "；".join("`%s` %s" % (x.get("id"), x.get("name") or "") for x in (s.get("retrieved_strategies") or [])[:6]) or "（无）",
        ))
        if r["condition"] == "b4":
            items = s.get("analysis_items") or []
            applies = sum(1 for it in items if isinstance(it, dict) and it.get("applies"))
            lines.append("- **B4 分析**：%d 条（applies=%d）；综合 criteria=%s" % (
                len(items), applies,
                json.dumps(s.get("synthesis_criteria") or {}, ensure_ascii=False),
            ))
            if s.get("synthesis_overall"):
                lines.append("  - overall：%s" % _trim(s["synthesis_overall"], 240))
        else:
            lines.append("- **B3 分析**：无（策略包直注入）")
        lines.append("- **prompt**：generate=%s chars，analysis=%s chars，synthesis=%s chars" % (
            (s.get("prompt_chars") or {}).get("generate", "?"),
            (s.get("prompt_chars") or {}).get("analysis", "?"),
            (s.get("prompt_chars") or {}).get("synthesis", "?"),
        ))
        lines.append("- **生成代码**：%s bytes，sha256=`%s`，存档 `%s`" % (
            r["generated_code_bytes"], (r["generated_code_sha256"] or "")[:16], r["paths"]["generated_c"]))
        if h:
            lines.append("- **编译（final）**：%s（%.2fs）；正确性：%s，maxdiff=%s，reldiff=%s，qemu_exit=%s，timeout=%s" % (
                "✅ ok" if r["compile_ok"] else "❌ failed",
                h.get("elapsed_build_s") or 0.0,
                "✅" if r["correct"] else "❌",
                "—" if r["maxdiff"] is None else "%.3e" % r["maxdiff"],
                "—" if r["reldiff"] is None else "%.3e" % r["reldiff"],
                r["qemu_exit"],
                r["timeout"],
            ))
            lines.append("- **QEMU 计时（仅存档，不是性能）**：best=%s ms（qemu_only=True）" % (
                "—" if r["best_ms_qemu_only"] is None else "%.4f" % r["best_ms_qemu_only"]))
            if not r["compile_ok"] and r["paths"].get("build_stderr"):
                log = HERE / r["paths"]["build_stderr"]
                try:
                    tail = _tail_lines(log.read_text(encoding="utf-8", errors="replace"), 10)
                except OSError:
                    tail = ""
                if tail:
                    lines.append("")
                    lines.append("  ```text")
                    lines.extend("  " + ln for ln in tail.splitlines())
                    lines.append("  ```")
        lines.append("- **token**：total=%s（prompt=%s / completion=%s / calls=%s）" % (
            (r.get("usage") or {}).get("total_tokens", 0),
            (r.get("usage") or {}).get("prompt_tokens", 0),
            (r.get("usage") or {}).get("completion_tokens", 0),
            (r.get("usage") or {}).get("calls", 0),
        ))
        if r.get("kcs") is not None:
            kd = r.get("kcs_detail") or {}
            lines.append("- **KCS（启发式 v0）**：%.2f（matched %d / checkable %d；retrieved %d；weak-only %d；uncheckable %d）" % (
                r["kcs"], len(kd.get("matched") or []), kd.get("denominator") or 0,
                kd.get("retrieved_count") or 0, len(kd.get("weak_only") or []),
                len(kd.get("uncheckable") or [])))
        else:
            lines.append("- **KCS**：null（无可检查策略或 stub 无检索）")
        code_final = d.get("code_final") or ((d.get("result") or {}).get("code") or "")
        if code_final:
            code_head = "\n".join(code_final.splitlines()[:20])
            lines.append("")
            lines.append("  最终代码前 20 行（全文见 `%s`；初版见 `%s`）：" % (
                r["paths"]["generated_c"], r["paths"].get("generated_initial", "—")))
            lines.append("")
            lines.append("  ```c")
            lines.extend("  " + ln for ln in code_head.splitlines())
            lines.append("  ```")
        lines.append("")

    lines.append("## 5. 结论与下一步")
    lines.append("")
    if run_meta["provider"] == "mock":
        lines.append("- 本次为 **mock 冒烟**：mock 代码是确定性标量参考实现，FCR/KCS/token **不代表 B3 vs B4 的方法效果**，仅验证链路与记录格式。")
    elif any(str(r.get("mpsa_backend")) == "stub" for r in records):
        lines.append("- ⚠️ 本次含 `stub` 兜底（mpsa CLI 不可用），结果不代表方法效果。")
    else:
        lines.append("- 本次为真实 API 运行：可直接进入 B3 vs B4 的配对比较（同 op/seed）；注意 token 成本已计入。")
    lines.append("- QEMU 只作正确性前置；性能（S_comp/EGC）须在 K1 真机按 `08b` 协议另测。")
    lines.append("- 失败单元按 `status` 分类处理：`compile_failed`/`wrong_answer` 应进入 E7 失败分析。")
    if run_meta.get("repair_rounds"):
        lines.append(
            "- **修复回路**：初始 FCR %.1f%% → 最终 %.1f%%；触发 %d cells（%d 轮），修复后转正 %d，repair tokens=%d。"
            % (
                100.0 * n_init_correct / n_cells if n_cells else 0.0,
                100.0 * n_correct / n_cells if n_cells else 0.0,
                len(repaired_cells),
                sum(int(r.get("repair_rounds") or 0) for r in repaired_cells),
                n_turned,
                repair_tokens_total,
            )
        )
    lines.append("- 产物目录：`%s/`；流水：`results.jsonl`。" % _rel(RUNS_DIR / run_meta["run_id"]))
    lines.append("")
    return "\n".join(lines)


def _update_summary() -> bool:
    """把 results.jsonl 的聚合写进 summary.md 的 AUTO-SUMMARY 区间（保留人写区）。"""
    try:
        raw_lines = RESULTS_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    records = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not records:
        block = "（尚无 run；运行 `python experiments/E09_MPSA/run_smoke.py --provider mock --ops daxpy --modes b3 --seed 1` 后自动更新）"
    else:
        groups: dict[tuple, dict] = {}
        for r in records:
            key = (
                r.get("provider"),
                r.get("mpsa_backend"),
                r.get("repair_backend"),
                r.get("condition"),
            )
            g = groups.setdefault(
                key,
                {
                    "cells": 0,
                    "correct": 0,
                    "correct_initial": 0,
                    "compile_final": 0,
                    "tokens": 0,
                    "repair_tokens": 0,
                    "repaired": 0,
                    "turned": 0,
                    "repair_rounds": 0,
                    "kcs": [],
                    "runs": set(),
                },
            )
            g["cells"] += 1
            correct_initial = bool(r.get("correct_initial", r.get("correct")))
            correct_final = bool(r.get("correct"))
            g["correct"] += 1 if correct_final else 0
            g["correct_initial"] += 1 if correct_initial else 0
            g["compile_final"] += 1 if r.get("compile_ok") else 0
            g["tokens"] += int((r.get("usage") or {}).get("total_tokens") or 0)
            g["repair_tokens"] += int(r.get("repair_tokens") or 0)
            rounds = int(r.get("repair_rounds") or 0)
            if rounds > 0:
                g["repaired"] += 1
                g["repair_rounds"] += rounds
                if correct_final and not correct_initial:
                    g["turned"] += 1
            if r.get("kcs") is not None:
                g["kcs"].append(r["kcs"])
            if r.get("run_id"):
                g["runs"].add(r["run_id"])

        last_run = records[-1].get("run_id")
        last_ts = records[-1].get("ts")
        block_lines = [
            "共 **%d** 条运行单元；最近 run：`%s`（%s）。" % (len(records), last_run, last_ts),
            "",
            "| provider / mpsa / repair | 条件 | cells | FCR_initial | FCR_final | 编译通过率(final) | 修复触发 | 修复转正 | repair_rounds | 平均 gen tokens/cell | 平均 repair tokens/cell | KCS 均值 (n) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for (provider, backend, rbackend, condition), g in sorted(groups.items(), key=lambda kv: str(kv[0])):
            kcs_avg = (
                "%.2f (%d)" % (sum(g["kcs"]) / len(g["kcs"]), len(g["kcs"])) if g["kcs"] else "—"
            )
            block_lines.append(
                "| %s / %s / %s | %s | %d | %s | %s | %s | %d | %d | %d | %.1f | %.1f | %s |"
                % (
                    provider,
                    backend,
                    rbackend or "—",
                    condition,
                    g["cells"],
                    _fmt_pct(g["correct_initial"], g["cells"]),
                    _fmt_pct(g["correct"], g["cells"]),
                    _fmt_pct(g["compile_final"], g["cells"]),
                    g["repaired"],
                    g["turned"],
                    g["repair_rounds"],
                    g["tokens"] / g["cells"] if g["cells"] else 0.0,
                    g["repair_tokens"] / g["cells"] if g["cells"] else 0.0,
                    kcs_avg,
                )
            )
        block_lines += [
            "",
            "> FCR_initial = 修复前 correct 比例；FCR_final = 最终 correct 比例；修复转正 = 初始失败且最终正确。",
            "> `ok` 仅表示编译+正确，**不含性能**。mock/stub 行只验证链路，不代表方法效果。",
            "> 每个 run 的详细报告在 `runs/<YYYY-MM-DD_HHMM>__<run_id>__<STATUS>.md`（见 `runs/INDEX.md`）；逐 cell 流水在 `results.jsonl`。",
        ]
        block = "\n".join(block_lines)

    try:
        text = SUMMARY_PATH.read_text(encoding="utf-8")
    except OSError:
        text = "# E09_MPSA 摘要\n\n%s\n%s\n" % (AUTO_BEGIN, AUTO_END)
    new_block = AUTO_BEGIN + "\n" + block + "\n" + AUTO_END
    if AUTO_BEGIN in text and AUTO_END in text:
        pre = text.split(AUTO_BEGIN, 1)[0]
        post = text.split(AUTO_END, 1)[1]
        new_text = pre + new_block + post
    else:
        new_text = text.rstrip("\n") + "\n\n" + new_block + "\n"
    SUMMARY_PATH.write_text(new_text, encoding="utf-8", newline="\n")
    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="E09_MPSA：MPSA B3 vs B4 冒烟/批跑（kg_agent mpsa → 编译 → QEMU → 记录）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ops", default="daxpy,ddot", help="算子（逗号分隔，可选 daxpy/ddot）")
    p.add_argument("--modes", default="b3,b4", help="条件（逗号分隔：b3,b4）")
    p.add_argument("--seed", default="1", help="随机种子（逗号分隔可多个，如 1,2,3）")
    p.add_argument("--sizes", default=None, help="规模（逗号分隔；默认取每个 task.yaml sizes 的中位数，即 16384）")
    p.add_argument("--reps", type=int, default=10, help="driver 计时重复次数（QEMU 上只求能跑完）")
    p.add_argument("--provider", choices=["mock", "dashscope", "deepseek"], default="deepseek",
                   help="mock=不联网验证链路；deepseek/dashscope=真实 API（分别需 DEEPSEEK_API_KEY / DASHSCOPE_API_KEY）")
    p.add_argument("--model", default=None, help="模型名（deepseek 默认 deepseek-chat；dashscope 默认 qwen-plus-2025-04-28）")
    p.add_argument("--top-k", type=int, default=6, help="检索策略数")
    p.add_argument("--analysis-k", type=int, default=3, help="B4 分析视角数")
    p.add_argument("--repair-rounds", type=int, default=0,
                   help="修复回路轮数（0=关闭；compile_failed 用 build.stderr、wrong_answer 用 qemu.stdout 调 mpsa-repair，修复后重编译/QEMU）")
    p.add_argument("--repair-backend", choices=["auto", "cli", "stub"], default="auto",
                   help="auto=CLI 缺 mpsa-repair 且 mock 时退回本地 stub；cli=强制 CLI；stub=强制 stub（仅 mock 链路自检）")
    p.add_argument("--repair-mock-code", default=None,
                   help="mock 兜底修复返回的代码文件（仅 repair-backend stub；默认用任务卡 generic.c）")
    p.add_argument("--timeout-repair", type=float, default=None,
                   help="单次 mpsa-repair 调用超时秒数（默认 mock 180 / dashscope 600）")
    p.add_argument("--mock-code", default=None, help="mock 时指定的生成代码文件（可选，固定用某份代码）")
    p.add_argument("--run-id", default=None, help="run id（默认 <tag>_seed<N>_<时间戳>；已存在时自动加 _r2）")
    p.add_argument("--tag", default=None, help="run_id 前缀标签（默认 mock→smoke，dashscope→run）")
    p.add_argument("--timeout-mpsa", type=float, default=None, help="单次 mpsa 调用超时秒数（默认 mock 180 / dashscope 600）")
    p.add_argument("--mpsa-backend", choices=["auto", "cli", "stub"], default="auto",
                   help="auto=CLI 缺 mpsa 且 mock 时退回本地 stub；cli=强制 CLI；stub=强制 stub（仅 mock）")
    p.add_argument("--no-summary", action="store_true", help="不更新 summary.md 自动区")
    p.add_argument("--no-organize", action="store_true", help="不自动整理 runs/（默认跑完自动整理：一 run 一 md + INDEX）")
    p.add_argument("--no-repro", action="store_true", help="不生成可复现包 artifacts/<run_id>/repro/（默认生成）")
    p.add_argument("--dry-run", action="store_true", help="只打印将执行的 cell，不调用 mpsa/编译")
    return p


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - 非交互环境忽略
        pass
    args = build_parser().parse_args(argv)

    ops = _csv_strs(args.ops)
    modes = _csv_strs(args.modes)
    seeds = _csv_ints(args.seed)
    for op in ops:
        if op not in TASK_DIRS:
            print("[error] 未知算子 %r（可用：%s）" % (op, ", ".join(TASK_DIRS)), file=sys.stderr)
            return 2
        if not (TASK_DIRS[op] / "task.yaml").is_file():
            print("[error] 任务卡缺失：%s" % (TASK_DIRS[op] / "task.yaml"), file=sys.stderr)
            return 2
    bad_modes = [m for m in modes if m not in MODE_LABEL]
    if bad_modes:
        print("[error] 未知条件 %r（可用：b3, b4）" % bad_modes, file=sys.stderr)
        return 2
    if not seeds:
        print("[error] --seed 不能为空", file=sys.stderr)
        return 2
    if args.mpsa_backend == "stub" and args.provider != "mock":
        print("[error] stub 仅用于 provider=mock 的链路自检，真实 API 不允许 stub", file=sys.stderr)
        return 2
    if args.repair_rounds < 0:
        print("[error] --repair-rounds 不能为负（0=关闭修复回路）", file=sys.stderr)
        return 2
    if args.repair_backend == "stub" and args.provider != "mock":
        print("[error] repair stub 仅用于 provider=mock 的链路自检，真实 API 不允许 stub", file=sys.stderr)
        return 2
    if args.repair_mock_code and not Path(args.repair_mock_code).is_file():
        print("[error] --repair-mock-code 文件不存在：%s" % args.repair_mock_code, file=sys.stderr)
        return 2
    if args.provider in ("dashscope", "deepseek"):
        key_env = "DASHSCOPE_API_KEY" if args.provider == "dashscope" else "DEEPSEEK_API_KEY"
        if not os.environ.get(key_env):
            print(f"[error] provider={args.provider} 需要环境变量 {key_env}（请在本会话 export/set，勿写入文件）",
                  file=sys.stderr)
            return 2
    if args.timeout_mpsa is None:
        args.timeout_mpsa = 600.0 if args.provider in ("dashscope", "deepseek") else 180.0
    if args.timeout_repair is None:
        args.timeout_repair = 600.0 if args.provider in ("dashscope", "deepseek") else 180.0

    # 每个算子的尺寸：--sizes > task.yaml 中位数
    sizes_by_op: dict[str, list[int]] = {}
    if args.sizes:
        sizes = _csv_ints(args.sizes)
        if not sizes:
            print("[error] --sizes 解析为空", file=sys.stderr)
            return 2
        for op in ops:
            sizes_by_op[op] = sizes
    else:
        for op in ops:
            meta = read_task_meta(TASK_DIRS[op])
            s = meta.get("sizes") or [16384]
            sizes_by_op[op] = [sorted(s)[len(s) // 2]]

    run_id = _resolve_run_id(args.run_id, args.provider, seeds, args.tag)
    args._run_id = run_id
    run_dir = RUNS_DIR / run_id

    cells_plan = [
        (op, mode, seed, size)
        for op in ops
        for mode in modes
        for seed in seeds
        for size in sizes_by_op[op]
    ]
    run_meta = {
        "run_id": run_id,
        "created_at": _now(),
        "provider": args.provider,
        "model": args.model or "default",
        "ops": ops,
        "modes": modes,
        "seeds": seeds,
        "sizes": sorted({size for _, _, _, size in cells_plan}),
        "reps": args.reps,
        "repair_rounds": args.repair_rounds,
        "repair_backend": args.repair_backend,
        "note": "mock 冒烟：结果仅验证链路，不代表方法效果" if args.provider == "mock" else None,
    }

    print("[E09] run_id=%s" % run_id)
    print("[E09] 计划 %d 个 cell：%s" % (
        len(cells_plan),
        ", ".join("%s×%s seed%d n=%d" % (op, m, seed, n) for op, m, seed, n in cells_plan)))
    repair_desc = ("off" if not args.repair_rounds
                   else "%d 轮 / backend=%s" % (args.repair_rounds, args.repair_backend))
    print("[E09] provider=%s backend=%s repair=%s profile=%s 输出=%s" % (
        args.provider, args.mpsa_backend, repair_desc, PROFILE, run_dir))
    if args.dry_run:
        for op, mode, seed, size in cells_plan:
            cdir = run_dir / ("%s_%s" % (op, mode) if len(sizes_by_op[op]) == 1 else "%s_%s_n%d" % (op, mode, size))
            print("  [dry-run] %s" % cdir)
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)

    cells: list[dict] = []
    t0 = time.time()
    for idx, (op, mode, seed, size) in enumerate(cells_plan, 1):
        suffix = "%s_%s" % (op, mode) if len(sizes_by_op[op]) == 1 else "%s_%s_n%d" % (op, mode, size)
        cell_dir = run_dir / suffix
        print("[E09] (%d/%d) %s × %s seed=%d n=%d ..." % (idx, len(cells_plan), op, mode, seed, size), flush=True)
        cell = _run_cell(op, mode, seed, size, cell_dir, args)
        cells.append(cell)
        rec = cell["record"]
        # 追加 results.jsonl（append-only，一行一个运行单元）
        with RESULTS_PATH.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        repair_txt = "—"
        if rec.get("repair_rounds"):
            repair_txt = "%d轮/%d tok" % (rec["repair_rounds"], rec.get("repair_tokens") or 0)
        print(
            "        status=%-14s initial=%-14s compile=%s correct=%s maxdiff=%s tokens=%s repair=%s kcs=%s mpsa=%.2fs"
            % (
                rec["status"],
                rec.get("status_initial") or rec["status"],
                "ok" if rec["compile_ok"] else "FAIL",
                "ok" if rec["correct"] else "FAIL",
                "—" if rec["maxdiff"] is None else "%.2e" % rec["maxdiff"],
                (rec.get("usage") or {}).get("total_tokens", 0),
                repair_txt,
                "—" if rec["kcs"] is None else "%.2f" % rec["kcs"],
                rec["mpsa_elapsed_s"] or 0.0,
            ),
            flush=True,
        )

    # run 报告
    md_path = RUNS_DIR / (run_id + ".md")
    md_path.write_text(_render_run_md(run_meta, cells), encoding="utf-8", newline="\n")
    print("[E09] run 报告：%s" % md_path)

    # summary 自动汇总
    if not args.no_summary:
        try:
            _update_summary()
            print("[E09] summary 自动区已更新：%s" % SUMMARY_PATH)
        except Exception as exc:  # noqa: BLE001 - 汇总失败不影响主结果
            print("[warn] summary 更新失败：%s" % exc, file=sys.stderr)

    n = len(cells)
    n_ok = sum(1 for c in cells if c["record"]["status"] == "ok")
    n_compile = sum(1 for c in cells if c["record"]["compile_ok"])
    n_correct = sum(1 for c in cells if c["record"]["correct"])
    n_init_correct = sum(
        1 for c in cells if c["record"].get("correct_initial", c["record"]["correct"])
    )
    repaired = [c["record"] for c in cells if int(c["record"].get("repair_rounds") or 0) > 0]
    n_turned = sum(
        1 for r in repaired if r.get("correct") and not r.get("correct_initial")
    )
    repair_tokens = sum(int(c["record"].get("repair_tokens") or 0) for c in cells)
    n_mpsa_fail = sum(1 for c in cells if c["record"]["status"] in ("mpsa_failed", "no_code"))
    tokens = sum(int((c["record"].get("usage") or {}).get("total_tokens") or 0) for c in cells)
    print("")
    print("[E09] 完成：%d cells，%d ok，FCR=初始 %.1f%% → 最终 %.1f%%；编译通过率(final)=%s" % (
        n, n_ok,
        100.0 * n_init_correct / n if n else 0.0,
        100.0 * n_correct / n if n else 0.0,
        _fmt_pct(n_compile, n)))
    if args.repair_rounds:
        print("[E09] 修复回路：触发 %d cells（%d 轮），修复后转正 %d，repair tokens=%d" % (
            len(repaired), sum(int(r.get("repair_rounds") or 0) for r in repaired),
            n_turned, repair_tokens))
    print("[E09] 总 gen tokens=%d，用时 %.1fs" % (tokens, time.time() - t0))
    print("[E09] 流水：%s" % RESULTS_PATH)

    # 自动整理：runs/ 一个 run 一个 md + artifacts/ + INDEX.md
    if not getattr(args, "no_organize", False):
        try:
            from organize_runs import organize
            res = organize(HERE)
            print("[E09] runs 已整理：normal=%d mock=%d selftest=%d；索引：%s"
                  % (len(res["normal"]), len(res["mock"]), len(res["selftest"]), res["index"]))
            latest = res["normal"][0] if res["normal"] else None
            if latest:
                print("[E09] 最新正式 run：%s（%s，cells=%d ok=%d）"
                      % (latest["run_id"], latest["status"], latest["cells"], latest["ok"]))
        except Exception as exc:  # noqa: BLE001 - 整理失败不影响主结果
            print("[warn] runs 整理失败：%s" % exc, file=sys.stderr)

    # 可复现包：输入 + 检索链 + 每轮迭代 + 最终优化算子
    if not getattr(args, "no_repro", False):
        try:
            rp = build_run_package(HERE, run_id)
            if rp:
                print("[E09] 可复现包：%s" % rp)
        except Exception as exc:  # noqa: BLE001 - 打包失败不影响主结果
            print("[warn] 可复现包生成失败：%s" % exc, file=sys.stderr)

    return 0 if n_mpsa_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
