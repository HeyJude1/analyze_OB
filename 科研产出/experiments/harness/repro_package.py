#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为一次实验 run 生成"可复现包"：输入 + 检索链 + 每轮迭代 + 最终优化算子。

用户要求：
1. 留存输入算子（generic.c）、task.yaml、driver、硬件 profile、代码特征；
2. 留存检索链（7 步 trace + StrategyPack），用户可自己复现；
3. 留存每一轮迭代（round0 初始生成、round1.. 修复）的代码、编译、QEMU 结果；
4. 最终优化算子可直接 build/run；
5. 最后才是人读 md（在 runs/ 下）。

用法：
  python experiments/harness/repro_package.py --experiment experiments/E09_MPSA --run-id real_grid_seed_sizes
  python experiments/harness/repro_package.py --experiment experiments/E09_MPSA --all-normal
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

CFLAGS = "-O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -static"
QEMU_CPU = "rv64,v=true,vlen=256,zba=true,zbb=true,zbc=true,zbs=true"


def _w(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _copy(src: Path | None, dst: Path):
    if src and src.exists() and src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def _load_records(exp: Path, run_id: str):
    p = exp / "results.jsonl"
    rows = []
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("run_id") == run_id:
                rows.append(r)
    return rows


def _cell_suffix(op: str, mode: str, size: int, sizes_for_op: list[int]) -> str:
    return f"{op}_{mode}" if len(sizes_for_op) == 1 else f"{op}_{mode}_n{size}"


def _extract_profile_block(yaml_path: Path, profile: str) -> str | None:
    """从 hardware_profiles.yaml 抽出某个 profile 的原始 YAML 块（保留注释与数值）。"""
    if not yaml_path.exists():
        return None
    lines = yaml_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out = []
    inside = False
    for ln in lines:
        if not inside and ln.strip() == f"{profile}:" and ln.startswith("  "):
            inside = True
            out.append(f"  {profile}:")
            continue
        if inside:
            if ln.startswith("  ") and not ln.startswith("    ") and ln.strip().endswith(":"):
                break
            out.append(ln)
    return "\n".join(out).rstrip() + "\n" if out else None


def _build_scripts(round_dir: Path, op: str, n: int, reps: int):
    _w(round_dir / "build.sh",
       "#!/usr/bin/env bash\n"
       "set -euo pipefail\n"
       'cd "$(dirname "$0")"\n'
       "riscv64-linux-gnu-gcc " + CFLAGS + " driver/test_" + op + ".c generated.c -lm -o bin\n"
       "echo \"[build] ok: $(pwd)/bin\"\n")
    _w(round_dir / "run.sh",
       "#!/usr/bin/env bash\n"
       "set -euo pipefail\n"
       'cd "$(dirname "$0")"\n'
       "qemu-riscv64 -cpu " + QEMU_CPU + " ./bin " + str(n) + " " + str(reps) + "\n")
    _w(round_dir / "RUN.md",
       f"# {op} 复现\n\n"
       f"```bash\ncd {round_dir.as_posix()}\nbash build.sh && bash run.sh\n```\n\n"
       f"- 期望：`maxdiff` 在容差内（见 `result.json` / `task.yaml`）\n"
       f"- QEMU 计时只用于确认能跑通，不是性能结论。\n")


def _round_from_cell(cell_dir: Path, dst: Path, op: str, n: int, reps: int,
                     code_name: str, result_dir: str, driver_src: Path,
                     extra_copy: list[tuple[Path | None, str]] | None = None):
    dst.mkdir(parents=True, exist_ok=True)
    code = cell_dir / code_name
    _copy(code if code.exists() else cell_dir / "generated.c", dst / "generated.c")
    _copy(driver_src, dst / "driver" / f"test_{op}.c")
    _copy(cell_dir / result_dir / "result.json", dst / "result.json")
    for f in ("build.stdout.log", "build.stderr.log", "qemu.stdout.log", "qemu.stderr.log"):
        _copy(cell_dir / result_dir / f, dst / "logs" / f)
    _build_scripts(dst, op, n, reps)
    if extra_copy:
        for src, rel in extra_copy:
            _copy(src, dst / rel)


def _get_trace(exp: Path, op: str, profile: str, top_k: int,
               code_features: dict | None = None) -> dict | None:
    """用 kg_agent 重新生成检索 trace（7 步）与 pack，供用户复现。"""
    kg_src = exp.parent.parent / "kg_agent" / "src"
    if not kg_src.exists():
        return None
    try:
        sys.path.insert(0, str(kg_src))
        from kgagent.cli import load_config  # type: ignore
        from kgagent import retrieve as R  # type: ignore
        cfg = load_config()
        tr = R.retrieve(cfg, op, profile=profile, top_k=top_k,
                        code_features=code_features, write=False)
        return tr
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def _env_info() -> list[str]:
    lines = []
    for label, cmd in [
        ("gcc", "riscv64-linux-gnu-gcc --version | head -1"),
        ("qemu", "qemu-riscv64 --version | head -1"),
        ("os", "cat /etc/os-release | head -2"),
    ]:
        try:
            out = subprocess.run(["wsl.exe", "-d", "Ubuntu", "-e", "bash", "-lc", cmd],
                                 capture_output=True, text=True, timeout=30)
            lines.append(f"{label}: {(out.stdout or out.stderr).strip()}")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"{label}: ERROR {exc}")
    return lines


def build_run_package(exp: Path, run_id: str) -> Path | None:
    exp = exp.resolve()
    records = _load_records(exp, run_id)
    if not records:
        print(f"[repro] no records for run_id={run_id}", file=sys.stderr)
        return None
    art = exp / "artifacts" / run_id
    if not art.is_dir():
        for sub in ("_mock", "_selftest"):
            cand = exp / "artifacts" / sub / run_id
            if cand.is_dir():
                art = cand
                break
    if not art.is_dir():
        print(f"[repro] artifact dir not found: {art}", file=sys.stderr)
        return None
    repro = art / "repro"
    if repro.exists():
        shutil.rmtree(repro)
    repro.mkdir(parents=True, exist_ok=True)

    # 1) 输入：task/generic/driver/hardware
    ops = sorted({r["op"] for r in records})
    for op in ops:
        task = exp / "tasks" / op
        _copy(task / "input" / "generic.c", repro / "inputs" / op / "generic.c")
        _copy(task / "task.yaml", repro / "inputs" / op / "task.yaml")
        _copy(task / "driver" / f"test_{op}.c", repro / "inputs" / op / "driver" / f"test_{op}.c")
    hw = exp.parent.parent / "kg_agent" / "schema" / "hardware_profiles.yaml"
    _copy(hw, repro / "inputs" / "hardware" / "hardware_profiles.yaml")
    profile = records[0].get("profile", "k1_rvv256")
    block = _extract_profile_block(hw, profile)
    if block:
        _w(repro / "inputs" / "hardware" / f"{profile}.yaml", block)
    _w(repro / "inputs" / "hardware" / "README.md",
       "# 硬件 profile\n\n"
       f"本次所有 cell 使用 `{profile}`；该 profile 的原始 YAML 见 `{profile}.yaml`，"
       "完整 x86/ARM/RVV/QEMU profiles 见 `hardware_profiles.yaml`。\n")

    # 2) 检索链：每个 op 一次（同一 profile/top_k）
    top_k = 6
    meta0 = {}
    for r in records:
        cell = art / _cell_suffix(r["op"], r["condition"], r["size"],
                                  sorted({x["size"] for x in records if x["op"] == r["op"]}))
        mr = cell / "mpsa_result.json"
        if mr.exists():
            try:
                d = json.loads(mr.read_text(encoding="utf-8"))
                top_k = int((d.get("retrieved") or {}).get("top_k") or top_k)
                meta0 = d
                break
            except Exception:
                pass

    traces = {}
    features_by_op: dict[str, dict | None] = {}
    for op in ops:
        # 从该 op 的第一个 mpsa_result 取实际 code_features（尽量复现同一查询）
        feat = None
        for r in records:
            if r["op"] != op:
                continue
            sizes_for_op = sorted({x["size"] for x in records if x["op"] == op})
            cell = art / _cell_suffix(op, r["condition"], r["size"], sizes_for_op)
            mr = cell / "mpsa_result.json"
            if mr.exists():
                try:
                    d = json.loads(mr.read_text(encoding="utf-8"))
                    feat = ((d.get("retrieved") or {}).get("query") or {}).get("code_features")
                except Exception:
                    feat = None
                break
        features_by_op[op] = feat
        profile = records[0].get("profile", "k1_rvv256")
        tr = _get_trace(exp, op, profile, top_k, code_features=feat)
        if tr:
            traces[op] = tr
            _w(repro / "inputs" / "retrieval" / f"{op}_{profile}_k{top_k}_trace.json",
               json.dumps(tr, ensure_ascii=False, indent=2))
            if isinstance(tr, dict) and "result" in tr and "pack" in tr["result"]:
                _w(repro / "inputs" / "retrieval" / f"{op}_{profile}_k{top_k}_pack.json",
                   json.dumps(tr["result"]["pack"], ensure_ascii=False, indent=2))

    # 3) 每个 cell：query/strategies/prompts + round0/round1/final
    cells_manifest = []
    for r in records:
        op, mode, size = r["op"], r["condition"], int(r["size"])
        seed = int(r["seed"])
        sizes_for_op = sorted({x["size"] for x in records if x["op"] == op})
        suffix = _cell_suffix(op, mode, size, sizes_for_op)
        src = art / suffix
        if not src.is_dir():
            continue
        dst = repro / "cells" / f"{suffix}_seed{seed}"
        dst.mkdir(parents=True, exist_ok=True)
        mr = src / "mpsa_result.json"
        md = {}
        if mr.exists():
            try:
                md = json.loads(mr.read_text(encoding="utf-8"))
            except Exception:
                md = {}
        _w(dst / "query.json", json.dumps({
            "op": op, "mode": mode, "profile": r.get("profile"), "seed": seed,
            "n": size, "reps": r.get("reps"), "top_k": top_k,
            "analysis_k": r.get("analysis_count"),
            "query": (md.get("retrieved") or {}).get("query"),
        }, ensure_ascii=False, indent=2))
        if md.get("retrieved"):
            _w(dst / "strategies.json", json.dumps(md["retrieved"].get("strategies") or [],
                                                    ensure_ascii=False, indent=2))
        prompts = md.get("prompt") or {}
        for k, v in prompts.items():
            if isinstance(v, str) and v:
                _w(dst / "prompts" / f"{k}.txt", v)
        if md.get("analysis"):
            _w(dst / "analysis.json", json.dumps(md["analysis"], ensure_ascii=False, indent=2))
        if md.get("synthesis"):
            _w(dst / "synthesis.json", json.dumps(md["synthesis"], ensure_ascii=False, indent=2))
        _w(dst / "usage.json", json.dumps({
            "usage": md.get("usage"), "code_meta": md.get("code_meta"),
            "provider": md.get("provider"), "model": md.get("model"),
            "elapsed_s": md.get("elapsed_s"),
        }, ensure_ascii=False, indent=2))

        driver = exp / "tasks" / op / "driver" / f"test_{op}.c"
        # round0：初始生成
        _round_from_cell(src, dst / "round0", op, size, int(r.get("reps") or 10),
                         "generated_initial.c", "harness", driver)

        # round1..N：每一轮修复都保存
        repair_dirs = sorted([p for p in src.glob("repair_r*") if p.is_dir()],
                             key=lambda p: int(re.sub(r"\D", "", p.name) or 0))
        for rp in repair_dirs:
            k = int(re.sub(r"\D", "", rp.name) or 0)
            hdir = src / f"harness_r{k}"
            _round_from_cell(
                src, dst / f"round{k}", op, size, int(r.get("reps") or 10),
                f"repair_r{k}/repaired.c" if (rp / "repaired.c").exists() else "generated.c",
                f"harness_r{k}" if hdir.is_dir() else "harness", driver,
                [(rp / "repair_result.json", "repair_result.json")])

        # final：最终代码 + 可执行文件（取最后一轮 harness_rK，否则 harness）
        final = dst / "final"
        final.mkdir(parents=True, exist_ok=True)
        _copy(src / "generated.c", final / "generated.c")
        _copy(driver, final / "driver" / f"test_{op}.c")
        fres = "harness"
        for p in sorted(src.glob("harness_r*")):
            if (p / "result.json").exists():
                fres = p.name
        _copy(src / fres / "result.json", final / "result.json")
        _build_scripts(final, op, size, int(r.get("reps") or 10))
        bin_name = f"{op}_test"
        _copy(src / fres / bin_name, final / "bin")
        _w(final / "expected.txt",
           f"op={op} mode={mode} seed={seed} n={size}\n"
           f"status={r.get('status')} compile_ok={r.get('compile_ok')} correct={r.get('correct')} "
           f"maxdiff={r.get('maxdiff')} reldiff={r.get('reldiff')}\n")

        rounds_txt = "\n".join(
            f"- `{p.name}/`：{'初始生成' if p.name == 'round0' else '第 ' + p.name[5:] + ' 轮修复'}"
            for p in sorted(dst.glob("round*")))
        _w(dst / "README.md",
           f"# {suffix} seed{seed}\n\n"
           f"- 输入：`../../inputs/{op}/generic.c`；硬件：`../../inputs/hardware/hardware_profiles.yaml`\n"
           f"- 检索链：`../../inputs/retrieval/{op}_{r.get('profile')}_k{top_k}_trace.json`"
           f"（pack 同目录）\n\n"
           f"## 迭代\n"
           f"{rounds_txt}\n"
           f"- `final/`：最终代码 + `bin`，可直接 `bash build.sh && bash run.sh`\n\n"
           f"## 复现\n```bash\ncd final && bash build.sh && bash run.sh\n```\n")
        cells_manifest.append({
            "cell": f"{suffix}_seed{seed}",
            "op": op, "mode": mode, "seed": seed, "size": size,
            "status": r.get("status"), "correct": r.get("correct"),
            "maxdiff": r.get("maxdiff"), "compile_ok": r.get("compile_ok"),
            "repair_rounds": r.get("repair_rounds"),
            "input": f"inputs/{op}/generic.c",
            "retrieval_trace": f"inputs/retrieval/{op}_{r.get('profile')}_k{top_k}_trace.json",
            "final_dir": f"cells/{suffix}_seed{seed}/final",
        })

    # 4) run 级 manifest + README + env
    env_lines = _env_info()
    manifest = {
        "schema": "e09-repro-v1",
        "experiment": exp.name,
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_records": "results.jsonl",
        "provider": records[0].get("provider"),
        "model": records[0].get("model"),
        "profile": records[0].get("profile"),
        "top_k": top_k,
        "repair_rounds": max(int(r.get("repair_rounds") or 0) for r in records),
        "mpsa_impl": records[0].get("mpsa_impl"),
        "cells": cells_manifest,
        "env": env_lines,
        "commands": {
            "retrieve_example": "cd kg_agent/src && python -m kgagent.cli retrieve --op daxpy --profile k1_rvv256 --top-k 6",
            "build_example": "cd cells/daxpy_b3_seed1_n1024/final && bash build.sh",
            "run_example": "cd cells/daxpy_b3_seed1_n1024/final && bash run.sh",
        },
    }
    _w(repro / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    # 逐 cell 原命令（mpsa 生成 + build/run 模板），方便用户完全照抄
    cmd_lines = ["# 自动记录：每个 cell 的 mpsa 生成命令", ""]
    for r in records:
        suf = _cell_suffix(r["op"], r["condition"], r["size"],
                           sorted({x["size"] for x in records if x["op"] == r["op"]}))
        cmd_lines.append(f"## {suf}_seed{r['seed']}")
        if r.get("mpsa_cmd"):
            cmd_lines.append("mpsa: " + str(r["mpsa_cmd"]))
        cmd_lines.append(f"build: cd cells/{suf}_seed{r['seed']}/final && bash build.sh")
        cmd_lines.append(f"run:   cd cells/{suf}_seed{r['seed']}/final && bash run.sh")
        cmd_lines.append("")
    _w(repro / "commands.txt", "\n".join(cmd_lines) + "\n")
    _w(repro / "env.txt", "\n".join(env_lines) + "\n")
    ex_cell = cells_manifest[0]["cell"] if cells_manifest else "daxpy_b3_n1024_seed1"
    ex_op = cells_manifest[0]["op"] if cells_manifest else "daxpy"
    _w(repro / "README.md",
       f"# 可复现包：{run_id}\n\n"
       f"> 生成时间：{manifest['created_at']}｜provider={manifest['provider']}｜model={manifest['model']}｜profile={manifest['profile']}\n\n"
       "## 目录\n\n"
       "```text\n"
       "inputs/            输入算子 generic.c、task.yaml、driver、硬件 profile、检索链 trace/pack\n"
       "cells/<cell>/      query/strategies/prompts + round0(初始) + round1+(修复) + final(最终代码+bin)\n"
       "manifest.json      本次 run 的全部映射（哪些 cell、状态、输入/检索/最终目录）\n"
       "commands.txt       每个 cell 的原始 mpsa 命令 + build/run 命令（历史 out 路径已迁到本包）\n"
       "env.txt            gcc/qemu 版本\n"
       "```\n\n"
       "## 1. 复现检索链\n\n"
       "```bash\n"
       f"cd kg_agent/src\n"
       f"python -m kgagent.cli retrieve --op {ex_op} --profile {manifest['profile']} --top-k {top_k} --trace-out /tmp/traces\n"
       "```\n"
       "本包已保存检索结果 `inputs/retrieval/<op>_<profile>_k<top_k>_trace.json` 与 `_pack.json`；"
       "硬件 profile 见 `inputs/hardware/<profile>.yaml`；原始生成命令见 `commands.txt`。"
       "可视化里回放请用 `kg_agent/viz/graph.html?panel=replay`。\n\n"
       "## 2. 复现编译 + QEMU\n\n"
       "```bash\n"
       f"cd cells/{ex_cell}/final\n"
       "bash build.sh && bash run.sh\n"
       "```\n"
       "每个 round 目录都有独立的 `build.sh`/`run.sh`；`round0` 是初始生成，`round1..` 是修复轮，`final` 是最终结果。\n"
       "QEMU 只验证正确性，计时不是性能结论。\n")

    print(f"[repro] {repro}")
    print(f"[repro] cells={len(cells_manifest)} env={len(env_lines)}")
    return repro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--run-id")
    ap.add_argument("--all-normal", action="store_true", help="为 runs/ 下所有正式 run 生成复现包")
    args = ap.parse_args()
    exp = Path(args.experiment).resolve()
    if args.all_normal:
        runs = exp / "runs"
        for md in sorted(runs.glob("*.md")):
            if md.stem in ("INDEX", "README"):
                continue
            # 从文件名取 run_id：YYYY-MM-DD_HHMM__<run_id>__<STATUS>.md
            parts = md.stem.split("__")
            if len(parts) >= 3:
                rid = parts[1]
                print(f"[repro] === {rid} ===")
                build_run_package(exp, rid)
    elif args.run_id:
        build_run_package(exp, args.run_id)
    else:
        ap.error("--run-id or --all-normal required")


if __name__ == "__main__":
    main()
