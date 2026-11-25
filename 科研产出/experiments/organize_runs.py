#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E09 / 通用实验目录整理器。

目标（用户要求）：
- `runs/` 下**每次实验只对应一个 .md 文件**，文件名带时间 + run_id + 状态；
- 原始产物（generated.c / harness / mpsa_result.json 等）统一放到 `<experiment>/artifacts/<run_id>/`；
- 自动生成 `runs/INDEX.md`，一眼看出最新、OK、MIXED、MOCK、SELFTEST；
- 旧 run 的 md 内 `runs/<run_id>/` 链接自动改写为 `../artifacts/<run_id>/`；
- 幂等：可以反复执行。

用法：
  python experiments/organize_runs.py --experiment experiments/E09_MPSA
"""
from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path

STATUS_ORDER = {"OK": 0, "MIXED": 1, "FAIL": 2, "MOCK": 3, "SELFTEST": 4}


def parse_meta(md: Path, text: str) -> dict:
    run_id = ""
    m = re.search(r"run `([^`]+)`", text)
    if m:
        run_id = m.group(1).strip()
    if not run_id:
        run_id = md.stem
    tm = re.search(r"生成时间：([0-9T:+\-\.]+)", text)
    if tm:
        ts = tm.group(1)
    else:
        ts = datetime.fromtimestamp(md.stat().st_mtime).isoformat(timespec="seconds")
    provider = "mock" if "provider：`mock`" in text or "provider: `mock`" in text else "dashscope"
    m = re.search(r"cells=(\d+)，ok=(\d+)", text)
    cells = int(m.group(1)) if m else 0
    ok = int(m.group(2)) if m else 0
    if not cells:
        # 兼容旧格式：数表格行 / 汇总行
        m = re.search(r"(\d+) 个 cell", text)
        cells = int(m.group(1)) if m else 0
        ok = cells if "全部 ok" in text or "全 ok" in text else 0
    is_selftest = run_id.startswith("selftest") or "selftest" in md.stem
    if is_selftest:
        status = "SELFTEST"
    elif provider == "mock":
        status = "MOCK"
    elif cells and ok == cells:
        status = "OK"
    elif ok:
        status = "MIXED"
    else:
        status = "FAIL"
    return {"run_id": run_id, "ts": ts, "provider": provider, "cells": cells,
            "ok": ok, "status": status, "source": md}


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", s)[:120].strip("_") or "run"


def rewrite_links(md_path: Path, run_id: str):
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    new = text.replace(f"runs/{run_id}/", f"../artifacts/{run_id}/")
    new = new.replace(f"runs\\{run_id}\\", f"../artifacts/{run_id}/")
    if new != text:
        md_path.write_text(new, encoding="utf-8")


def organize(exp_dir: Path, move_artifacts: bool = True) -> dict:
    runs = exp_dir / "runs"
    artifacts = exp_dir / "artifacts"
    runs.mkdir(exist_ok=True)
    artifacts.mkdir(exist_ok=True)

    # 1) 收集现有 run md（顶层，不含 _mock/_selftest 子目录）
    md_files = sorted([p for p in runs.glob("*.md") if p.is_file() and p.stem not in ("INDEX", "README")])
    metas = []
    for md in md_files:
        try:
            text = md.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""
        meta = parse_meta(md, text)
        meta["md"] = md
        metas.append(meta)

    # 2) 解析时间戳为排序键
    def ts_key(meta):
        try:
            return datetime.fromisoformat(meta["ts"].replace("Z", "+00:00"))
        except Exception:
            return datetime.fromtimestamp(meta["md"].stat().st_mtime)

    metas.sort(key=ts_key, reverse=True)

    # 3) 移动 md 到分类子目录，重命名
    moved = []
    for meta in metas:
        md = meta["md"]
        run_id = meta["run_id"]
        status = meta["status"]
        dt = ts_key(meta)
        prefix = dt.strftime("%Y-%m-%d_%H%M")
        if status == "SELFTEST":
            target_dir = runs / "_selftest"
        elif status == "MOCK":
            target_dir = runs / "_mock"
        else:
            target_dir = runs
        target_dir.mkdir(exist_ok=True)
        new_name = f"{prefix}__{safe_name(run_id)}__{status}.md"
        target = target_dir / new_name
        if md.resolve() != target.resolve():
            if target.exists():
                target = target_dir / f"{prefix}__{safe_name(run_id)}__{status}__{md.stat().st_mtime_ns}.md"
            shutil.move(str(md), str(target))
        rewrite_links(target, run_id)
        meta["report"] = target
        moved.append(meta)

        if move_artifacts:
            src = runs / run_id
            if src.is_dir():
                if status == "SELFTEST":
                    dst_parent = artifacts / "_selftest"
                elif status == "MOCK":
                    dst_parent = artifacts / "_mock"
                else:
                    dst_parent = artifacts
                dst_parent.mkdir(exist_ok=True)
                dst = dst_parent / run_id
                if dst.exists():
                    dst = dst_parent / f"{run_id}__{int(dt.timestamp())}"
                shutil.move(str(src), str(dst))
                meta["artifact"] = dst
            else:
                # 重复整理时：产物已在 artifacts/ 下，补上索引链接
                cand = artifacts / run_id
                if status == "SELFTEST":
                    cand = artifacts / "_selftest" / run_id
                elif status == "MOCK":
                    cand = artifacts / "_mock" / run_id
                meta["artifact"] = cand if cand.is_dir() else None
        else:
            meta["artifact"] = None

    # 4) 扫描已归档的 _mock/_selftest 报告（保证重复整理时索引仍完整）
    seen = {m["run_id"] for m in moved}
    for sub, status_hint in (("_mock", "MOCK"), ("_selftest", "SELFTEST")):
        for md in sorted((runs / sub).glob("*.md")):
            if md.stem in ("INDEX", "README"):
                continue
            try:
                text = md.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text = ""
            meta = parse_meta(md, text)
            if meta["run_id"] in seen:
                continue
            meta["md"] = md
            meta["report"] = md
            meta["artifact"] = None
            # 尝试找同名产物目录
            cand = artifacts / sub / meta["run_id"]
            if cand.is_dir():
                meta["artifact"] = cand
            moved.append(meta)
            seen.add(meta["run_id"])

    # 5) 若 artifacts 里已有同名但未移动的旧目录，保持
    # 6) 生成 INDEX.md
    def rel(p):
        if not p:
            return "—"
        return p.relative_to(exp_dir).as_posix()

    normal = [m for m in moved if m["status"] not in ("MOCK", "SELFTEST")]
    mock = [m for m in moved if m["status"] == "MOCK"]
    self_ = [m for m in moved if m["status"] == "SELFTEST"]
    normal.sort(key=ts_key, reverse=True)
    mock.sort(key=ts_key, reverse=True)
    self_.sort(key=ts_key, reverse=True)

    lines = ["# E09 runs 索引", "",
             "> 命名规则：`YYYY-MM-DD_HHMM__<run_id>__<状态>.md`；每个 run **一个 md 文件**。",
             "> 状态：`OK` 全部编译+正确；`MIXED` 部分失败；`FAIL` 全部失败；`MOCK` 离线链路；`SELFTEST` 框架自测。",
             "> 原始产物（generated.c / harness / mpsa_result.json）在 `../artifacts/<run_id>/`。", ""]
    if normal:
        latest = normal[0]
        lines += [f"> **最新正式 run**：`{latest['run_id']}`（{latest['status']}，cells={latest['cells']}，ok={latest['ok']}，{latest['ts'][:19]}）", ""]
    lines += ["## 正式 run（按时间从新到旧）", "",
              "| 时间 | run_id | provider | cells | ok | 状态 | 报告文件 | 产物 | 复现包 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for m in normal:
        repro = exp_dir / "artifacts" / m["run_id"] / "repro" / "README.md"
        repro_cell = f"[repro](../{repro.relative_to(exp_dir).as_posix()})" if repro.exists() else "—"
        lines.append(f"| {m['ts'][:19]} | `{m['run_id']}` | {m['provider']} | {m['cells']} | {m['ok']} | **{m['status']}** | [{m['report'].name}]({m['report'].name}) | `{rel(m['artifact'])}` | {repro_cell} |")
    if not normal:
        lines.append("| — | — | — | — | — | — | — | — |")

    def sub_section(title, items, folder):
        out = ["", f"## {title}", ""]
        if not items:
            out.append("（无）")
            return out
        out += ["| 时间 | run_id | cells | ok | 状态 | 报告文件 |",
                "|---|---|---|---|---|---|"]
        for m in items:
            rp = m["report"].relative_to(exp_dir / "runs").as_posix()
            out.append(f"| {m['ts'][:19]} | `{m['run_id']}` | {m['cells']} | {m['ok']} | {m['status']} | [{m['report'].name}]({folder}/{m['report'].name}) |")
        return out

    lines += sub_section("Mock 链路（不花 token，不代表方法效果）", mock, "_mock")
    lines += sub_section("框架自测（repair 回路/无修复回归）", self_, "_selftest")
    lines += ["", "---", f"*最后整理：{datetime.now().isoformat(timespec='seconds')}*", ""]
    (runs / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")

    # 6) runs/README.md 说明
    (runs / "README.md").write_text(
        "# runs 目录说明\n\n"
        "- 每个 run 只有一个 `.md` 文件：`YYYY-MM-DD_HHMM__<run_id>__<状态>.md`。\n"
        "- `INDEX.md` 是总索引；正式 run 在根目录，mock/selftest 在 `_mock/`、`_selftest/`。\n"
        "- 原始产物在 `../artifacts/<run_id>/`；可复现包在 `../artifacts/<run_id>/repro/README.md`（输入/硬件/检索链/每轮迭代/最终算子）。\n"
        "- 报告内链接已自动改写为 `../artifacts/...`。\n"
        "- 状态：OK / MIXED / FAIL / MOCK / SELFTEST。\n",
        encoding="utf-8")
    return {"normal": normal, "mock": mock, "selftest": self_,
            "index": runs / "INDEX.md"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, help="实验目录，如 experiments/E09_MPSA")
    ap.add_argument("--no-artifacts", action="store_true", help="只整理 md/索引，不移动产物目录")
    args = ap.parse_args()
    exp = Path(args.experiment).resolve()
    res = organize(exp, move_artifacts=not args.no_artifacts)
    print(f"[organize] {exp}")
    print(f"  normal={len(res['normal'])} mock={len(res['mock'])} selftest={len(res['selftest'])}")
    print(f"  index={res['index']}")
    for m in res["normal"][:5]:
        print(f"  - {m['ts'][:19]} {m['run_id']} {m['status']} cells={m['cells']} ok={m['ok']}")


if __name__ == "__main__":
    main()
