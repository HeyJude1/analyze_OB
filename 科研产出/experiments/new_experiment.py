#!/usr/bin/env python3
"""experiments/new_experiment.py —— 一键创建一个符合 experiments/README.md 规范的实验目录。

用法
----
    python experiments/new_experiment.py --id E10 --name my_experiment
    python experiments/new_experiment.py --id E10_MOCK --name mock     # id 自带后缀时按原样建目录
    python experiments/new_experiment.py --id E11 --name sweep --owner 张三 --register

生成
----
    experiments/E10_my_experiment/
    ├── README.md        # 实验协议模板（假设/条件/指标/步骤/限制）
    ├── summary.md       # 结论模板（含 AUTO-SUMMARY 自动区）
    ├── results.jsonl    # 空文件，append-only
    └── runs/            # 每次 run 的产物

只写 experiments/**；已存在目录默认拒绝，需 --force 才覆盖模板（结果文件永不覆盖）。
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent          # experiments/
DEFAULT_ROOT = HERE

_ID_RE = re.compile(r"^E\d{2}(?:_[A-Za-z0-9]+)*$")
_STATUS = "🟡 进行中"


def _slug(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff]+", "_", name.strip())
    return s.strip("_") or "exp"


def _full_id(exp_id: str, name: str | None) -> str:
    """E10 + name → E10_name；E10_MOCK（id 已含后缀）→ 原样。"""
    if "_" in exp_id:
        return exp_id
    if not name:
        raise SystemExit("[error] 只给 --id 时必须给 --name（否则不知道目录后缀），例如 --id E10 --name my_experiment")
    return "%s_%s" % (exp_id, _slug(name))


README_TMPL = """# {full_id}：{name}

> 创建：{created}｜负责人：{owner}｜状态：{status}
> 规范见 `../README.md`，指标和双平台测量口径见 `../../03_实验与验收.md`。

## 1. 科学问题与假设

| 编号 | 假设 | 判据（预注册） |
|---|---|---|
| H1 | TODO：写出可证伪的假设 | TODO：指标 + 阈值（例：FCR 提升 ≥5pp，配对检验） |
| H2 | TODO | TODO |

## 2. 条件与对照

| 条件 | 定义 | 说明 |
|---|---|---|
| TODO | TODO | TODO |

（写清：同 LLM / 同 token 预算 / 同尝试次数；**QEMU 只作正确性前置，不作性能结论**。）

## 3. 任务与数据

- 任务：`tasks/<op>/{{task.yaml, input/generic.c, driver/test_<op>.c}}`
- 数据来源：OpOptBench（复制后在 task.yaml 记录来源与容差）
- 规模/seed：TODO

## 4. 指标

| 指标 | 定义 | 数据来源 |
|---|---|---|
| FCR | `compile_ok & correct` 的运行单元占比 | `results.jsonl` |
| KCS | 知识合规率（启发式 v0，见 `../harness/kcs.py`） | `results.jsonl.kcs` |
| 成本 | token 数 / LLM 调用次数 / 墙钟 | `results.jsonl.usage` |

> 按现行协议在目标真机上测量性能。QEMU 结果用于兼容性与正确性验证。

## 5. 执行步骤

```powershell
# 冒烟（mock，不联网）：
python experiments/{full_id}/run_smoke.py --provider mock --ops TODO --modes TODO --seed 1

# 真实 API（需先 $env:DASHSCOPE_API_KEY）：
python experiments/{full_id}/run_smoke.py --provider dashscope --ops TODO --modes TODO --seed 1
```

1. 跑 run → 检查 `runs/<run_id>.md` 与 `results.jsonl`；
2. 结论写进 `summary.md` 手写区（AUTO 区间由脚本更新）；
3. 更新 `../EXPERIMENTS.md` 状态。

## 6. 已知限制与排错

- TODO：mock/stub 只验证链路，不代表方法效果；
- 编译失败看 `runs/<run_id>/<op>_<cond>/build.stderr.log`；超时看 `qemu.*.log`；
- WSL 网络警告与 QEMU `vector version` 警告可忽略。

## 7. 结果入口

- 流水：`results.jsonl`（append-only，一行一个运行单元）
- 每次 run 报告：`runs/<run_id>.md`
- 汇总：`summary.md`
"""

SUMMARY_TMPL = """# {full_id} 摘要

> 人写区：结论、关键数字、图、下一步。`AUTO-SUMMARY` 区间由脚本根据 `results.jsonl` 刷新，不要手改。

## 结论（待填）

- TODO

<!-- AUTO-SUMMARY:BEGIN -->
（尚无 run）
<!-- AUTO-SUMMARY:END -->

## 下一步（待填）

- TODO
"""


def _write_if_absent(path: Path, text: str, force: bool) -> tuple[bool, str]:
    if path.exists() and not force:
        return False, "已存在，跳过"
    path.write_text(text, encoding="utf-8", newline="\n")
    return True, "已写入"


def _register(full_id: str, name: str, owner: str, day: str, root: Path) -> bool:
    """在 <root>/EXPERIMENTS.md 总表追加一行（幂等：已有该 id 则跳过）。"""
    experiments_md = root / "EXPERIMENTS.md"
    if not experiments_md.is_file():
        return False
    text = experiments_md.read_text(encoding="utf-8")
    if re.search(r"^\|\s*%s\s*\|" % re.escape(full_id), text, re.MULTILINE):
        return False
    rel_dir = "%s/" % full_id
    row = "| %s | TODO：实验名称 | %s | %s | %s | `%s` | `%s/results.jsonl` | 由 new_experiment.py 创建，待补协议 |\n" % (
        full_id,
        _STATUS,
        day,
        owner,
        rel_dir,
        rel_dir.rstrip("/"),
    )
    # 插到第一个表格的最后一行之后（即 "|---|---|" 表格块末尾）
    lines = text.splitlines(keepends=True)
    insert_at = None
    for i, line in enumerate(lines):
        if line.startswith("| ID | 名称 |"):
            j = i + 1
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            insert_at = j
            break
    if insert_at is None:
        return False
    lines.insert(insert_at, row)
    experiments_md.write_text("".join(lines), encoding="utf-8", newline="\n")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="创建 experiments/EXX_name 实验目录（含模板）")
    parser.add_argument("--id", required=True, help="实验编号，如 E10 或 E10_MOCK")
    parser.add_argument("--name", default=None, help="实验短名（用于目录后缀与 README 标题）")
    parser.add_argument("--owner", default="TODO", help="负责人")
    parser.add_argument("--date", default=date.today().isoformat(), help="开始日期 YYYY-MM-DD")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="experiments 根目录（默认本文件所在目录）")
    parser.add_argument("--register", action="store_true", help="同时在 EXPERIMENTS.md 总表追加一行")
    parser.add_argument("--force", action="store_true", help="目录已存在时覆盖模板文件（不覆盖 results.jsonl）")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - 非交互/老 Python 忽略
        pass

    if not _ID_RE.match(args.id):
        print("[error] --id 需形如 E10 或 E10_MOCK（E + 两位数字 + 可选 _后缀）", file=sys.stderr)
        return 2
    try:
        full_id = _full_id(args.id, args.name)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2
    display_name = args.name or full_id.split("_", 1)[-1]
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    exp_dir = root / full_id

    if exp_dir.exists() and not args.force:
        existing = sorted(p.name for p in exp_dir.iterdir())
        print("[error] 目录已存在：%s（内容：%s）\n        若要覆盖模板请加 --force；结果文件不会被覆盖。" % (exp_dir, existing))
        return 2

    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "runs").mkdir(exist_ok=True)
    (exp_dir / "runs" / ".gitkeep").touch()

    created = []
    for rel, text in (
        ("README.md", README_TMPL.format(full_id=full_id, name=display_name, created=args.date, owner=args.owner, status=_STATUS)),
        ("summary.md", SUMMARY_TMPL.format(full_id=full_id)),
    ):
        ok, _msg = _write_if_absent(exp_dir / rel, text, args.force)
        if ok:
            created.append(rel)
    # results.jsonl：append-only 结果流水，**永不覆盖**（只有不存在时才建空文件）
    results_path = exp_dir / "results.jsonl"
    if not results_path.exists():
        results_path.write_text("", encoding="utf-8", newline="\n")
        created.append("results.jsonl")

    if args.register:
        registered = _register(full_id, display_name, args.owner, args.date, root)
        if registered:
            created.append("EXPERIMENTS.md（已登记一行）")

    print("[ok] 已创建实验目录：%s" % exp_dir)
    if created:
        print("     文件：%s" % ", ".join(created))
    else:
        print("     （目录已存在，无可写文件）")
    print("     下一步：")
    print("       1) 编辑 %s/README.md 写协议（假设/条件/指标/步骤）" % full_id)
    print("       2) 复制 tasks/ 模板与执行脚本（参考 E09_MPSA/run_smoke.py）")
    print("       3) 运行后确认 results.jsonl / runs/*.md，并把结论写进 summary.md")
    if not args.register:
        print("       （可选：--register 时自动在 EXPERIMENTS.md 总表登记）")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
