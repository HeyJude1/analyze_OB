#!/usr/bin/env python3
"""Summarize paired functional comparisons against their full planned manifest."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re


METHODS = ("no_knowledge", "text_rag", "strategy_cards", "constraint_plan")
METHOD_LABELS = {
    "no_knowledge": "无外部知识",
    "text_rag": "文本 RAG",
    "strategy_cards": "独立策略卡",
    "constraint_plan": "约束规划",
}
PROFILE_LABELS = {"x86_avx2": "本机 x86", "k1_rvv256": "RISC-V QEMU"}
SPLIT_LABELS = {
    "build": "四个建图算子",
    "holdout": "独立留出",
    "independent_holdout": "独立留出",
    "near_transfer": "近迁移",
    "external_holdout": "外部留出",
}
STATUS_LABELS = {
    "passed": "通过",
    "compile_failed": "编译或链接失败",
    "wrong_answer": "数值或接口错误",
    "runtime_failed": "运行失败",
    "compile_timeout": "编译超时",
    "run_timeout": "运行超时",
    "generation_failed": "生成调用失败",
    "invalid_output": "未提取到有效 C 代码",
    "infra_error": "基础设施异常",
    "not_run": "未执行",
}
KEYS = ("operator", "profile", "method", "repetition")
STAGES = ("initial", "final")
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
SETTING_LABELS = {
    "context": "上下文上限（token）",
    "completion_cap": "每次生成或修复输出上限（token）",
    "temperature": "采样温度",
    "seed": "采样种子基数",
    "max_repairs": "每单元最多修复次数",
    "check_timeout_seconds": "验证超时上限（秒）",
    "riscv_cpu": "QEMU CPU 参数",
}
UNRUN = {"status": "not_run", "compile_ok": None, "correct": None, "failure_reason": None, "usage": None}


@dataclass
class Cell:
    case: dict
    result: dict | None
    location: str

    def stage(self, name):
        return self.result[name] if self.result is not None else UNRUN


def read_object(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: 应为 JSON 对象")
    return data


def cell_key(row):
    key = tuple(row.get(field) for field in KEYS)
    if (any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value)
            for value in key[:3])
            or type(key[3]) is not int or key[3] < 1):
        raise ValueError(f"无效单元标识 {key!r}")
    if key[2] not in METHODS:
        raise ValueError(f"未知方法 {key[2]!r}")
    return key


def validate_result(result, key, run):
    if cell_key(result) != key:
        raise ValueError(f"结果本体与路径所指单元不一致 {key}")
    for field in ("engine", "model"):
        if field not in result or result[field] != run[field]:
            raise ValueError(f"{key} 的 {field} 与 run.json 不一致")
    if type(result.get("repair_attempted")) is not bool:
        raise ValueError(f"{key} 缺少布尔 repair_attempted")
    for name in STAGES:
        stage = result.get(name)
        if not isinstance(stage, dict) or stage.get("status") not in STATUS_LABELS:
            raise ValueError(f"{key} 的 {name} 缺少有效状态")
        for field in ("compile_ok", "correct"):
            if stage.get(field) is not None and type(stage[field]) is not bool:
                raise ValueError(f"{key} 的 {name}.{field} 应为布尔值或 null")
        if stage.get("correct") is True and stage.get("compile_ok") is not True:
            raise ValueError(f"{key} 的 {name} 同时正确且未通过编译")
        if (stage["status"] == "passed") != (stage.get("correct") is True):
            raise ValueError(f"{key} 的 {name} 状态与 correct 不一致")
        reason = stage.get("failure_reason")
        if reason is not None and not isinstance(reason, str):
            raise ValueError(f"{key} 的 {name}.failure_reason 应为字符串或 null")
        usage = stage.get("usage")
        if usage is not None:
            if not isinstance(usage, dict) or any(
                type(value) is not int or value < 0
                for field in TOKEN_FIELDS
                if (value := usage.get(field)) is not None
            ):
                raise ValueError(f"{key} 的 {name}.usage 格式错误")


def load_run(prepared, run_dir):
    manifest_path = prepared / "manifest.json"
    manifest_hash = sha256(manifest_path.read_bytes()).hexdigest()
    manifest = read_object(manifest_path)
    run = read_object(run_dir / "run.json")
    if run.get("prepared_manifest_sha256") != manifest_hash:
        raise ValueError("run.json 中的准备清单哈希与所给清单不匹配")
    if not run.get("engine") or "model" not in run or not isinstance(run.get("settings"), dict):
        raise ValueError("run.json 缺少 engine、model 或 settings")
    cells = []
    expected_paths = set()
    for case in manifest["cases"]:
        key = cell_key(case)
        if not isinstance(case.get("split"), str) or not case["split"]:
            raise ValueError(f"{key} 缺少数据划分")
        op, profile, method, repetition = key
        relative = Path(op, profile, method, f"rep-{repetition}", "result.json")
        result_path = run_dir / relative
        if relative in expected_paths:
            raise ValueError(f"准备清单中有重复单元 {key}")
        expected_paths.add(relative)
        result = None
        if result_path.is_file():
            result = read_object(result_path)
            validate_result(result, key, run)
        cells.append(Cell(case, result, relative.as_posix()))
    if not cells:
        raise ValueError("准备清单没有单元")
    unexpected_paths = {
        path.relative_to(run_dir) for path in run_dir.glob("*/*/*/rep-*/result.json")
    } - expected_paths
    if unexpected_paths:
        raise ValueError(f"运行目录含清单之外的结果 {sorted(str(p) for p in unexpected_paths)[:3]}")
    return cells, run, manifest_hash, manifest.get("knowledge_byte_cap")


def key_for(cell, dimensions):
    return tuple(cell.case[name] for name in dimensions)


def stage_counts(items, name):
    stages = [cell.stage(name) for cell in items]
    return {
        "compile_observed": sum(stage.get("compile_ok") is not None for stage in stages),
        "compile_passed": sum(stage.get("compile_ok") is True for stage in stages),
        "correctness_observed": sum(stage.get("correct") is not None for stage in stages),
        "correct": sum(stage.get("correct") is True for stage in stages),
        "statuses": dict(sorted(Counter(stage["status"] for stage in stages).items())),
    }


def group_counts(items):
    results = [cell for cell in items if cell.result is not None]
    repairs = [cell for cell in results if cell.result["repair_attempted"]]
    attempts = [cell.stage("initial") for cell in results] + [cell.stage("final") for cell in repairs]
    usage_rows = [stage["usage"] for stage in attempts if isinstance(stage.get("usage"), dict)]
    tokens = {}
    for field in TOKEN_FIELDS:
        observed = [row[field] for row in usage_rows if row.get(field) is not None]
        tokens[field] = {"covered_attempts": len(observed), "sum": sum(observed) if observed else None}
    return {
        "planned": len(items),
        "result_files": len(results),
        "initial": stage_counts(items, "initial"),
        "final": stage_counts(items, "final"),
        "repair_attempted": len(repairs),
        "repair_rescued": sum(
            cell.stage("initial").get("correct") is not True and cell.stage("final").get("correct") is True
            for cell in repairs
        ),
        "regressed": sum(
            cell.stage("initial").get("correct") is True and cell.stage("final").get("correct") is not True
            for cell in repairs
        ),
        "tokens": tokens,
    }


def grouped(cells, dimensions):
    buckets = defaultdict(list)
    for cell in cells:
        buckets[key_for(cell, dimensions)].append(cell)

    def order(key):
        return tuple(METHODS.index(value) if name == "method" else value
                     for name, value in zip(dimensions, key))

    return [
        {**dict(zip(dimensions, key)), **group_counts(buckets[key])}
        for key in sorted(buckets, key=order)
    ]


def failure_rows(cells):
    failures = defaultdict(list)
    for cell in cells:
        for name in STAGES:
            stage = cell.stage(name)
            status = stage["status"]
            if status in {"passed", "not_run"}:
                continue
            profile, method = key_for(cell, ("profile", "method"))
            reason = stage.get("failure_reason") or status
            failures[(name, profile, method, status, reason)].append(cell.location)
    return [
        {"stage": name, "profile": profile, "method": method,
         "status": status, "reason": reason, "count": len(locations), "examples": locations[:3]}
        for (name, profile, method, status, reason), locations in sorted(
            failures.items(), key=lambda item: (STAGES.index(item[0][0]), -len(item[1]), item[0][1:]))
    ]


def report_data(cells, run, manifest_hash, knowledge_byte_cap):
    return {
        "run": run, "prepared_manifest_sha256": manifest_hash,
        "knowledge_byte_cap": knowledge_byte_cap,
        "overall": group_counts(cells),
        "by_profile_method": grouped(cells, ("profile", "method")),
        "by_split": grouped(cells, ("split", "profile", "method")),
        "by_operator": grouped(cells, ("split", "operator", "profile", "method")),
        "failures": failure_rows(cells),
    }


def table(lines, headers, rows):
    lines.extend([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("--" for _ in headers) + " |",
    ])
    for row in rows:
        values = [str(value).replace("|", r"\|").replace("\r", " ").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")


def metric(row, stage, name):
    count = row[stage][name]
    total = row["planned"]
    return f"{count}/{total}（{count / total:.1%}）"


def markdown(data):
    run, overall = data["run"], data["overall"]
    engine, model = run["engine"], run["model"]
    lines = [
        "# 四组方法本地功能比较", "",
        f"执行引擎为 `{engine}`，运行模型为 `{model or '未配置'}`。"
        f"清单共安排 {overall['planned']} 个单元，已保留 {overall['result_files']} 份结果。",
        "",
    ]
    if engine == "replay":
        lines.extend(["**离线回放只检验准备、编译与验证流程。本报告不代表模型生成效果。**", ""])
    settings = run["settings"]
    budget = [(label, settings[key]) for key, label in SETTING_LABELS.items()
              if settings.get(key) is not None]
    if data["knowledge_byte_cap"] is not None:
        budget.insert(0, ("检索知识上限（UTF-8 字节）", data["knowledge_byte_cap"]))
    if run.get("model_digest"):
        budget.insert(0, ("模型快照", run["model_digest"]))
    if budget:
        lines.extend(["各组共用以下运行设置。知识块的字节上限与模型实际输入 token 分别记录。", ""])
        table(lines, ("设置", "数值"), budget)
    lines.extend([
        "比较范围为目标编译、功能验证和一次修复结果。"
        "两轮通过率的分母均为清单计划单元，未执行单元保留在分母，失败原因表另列已观察到的异常。",
        "编译已判定表示编译状态为真或假。修复后正确为初次未通过且执行修复后通过，"
        "首轮已通过的单元直接沿用最终状态。成功率不以已编译子集为分母。真机性能测量另行开展。",
        "",
        "## 平台与方法", "",
    ])
    table(lines, (
        "验证环境", "方法", "计划/记录", "已判定编译 首轮/最终", "编译通过 首轮", "编译通过 最终",
        "正确 首轮", "正确 最终", "修复次数/转正确", "最终未执行",
    ), [
        (
            PROFILE_LABELS.get(row["profile"], row["profile"]), METHOD_LABELS[row["method"]],
            f"{row['planned']}/{row['result_files']}",
            f"{row['initial']['compile_observed']}/{row['final']['compile_observed']}",
            metric(row, "initial", "compile_passed"), metric(row, "final", "compile_passed"),
            metric(row, "initial", "correct"), metric(row, "final", "correct"),
            f"{row['repair_attempted']}/{row['repair_rescued']}",
            row["final"]["statuses"].get("not_run", 0),
        )
        for row in data["by_profile_method"]
    ])
    lines.extend(["## 数据划分", ""])
    table(lines, ("划分", "验证环境", "方法", "编译通过 首轮/最终", "正确 首轮/最终", "修复/转正确"), [
        (
            SPLIT_LABELS.get(row["split"], row["split"]), PROFILE_LABELS.get(row["profile"], row["profile"]),
            METHOD_LABELS[row["method"]],
            f"{row['initial']['compile_passed']} → {row['final']['compile_passed']} / {row['planned']}",
            f"{row['initial']['correct']} → {row['final']['correct']} / {row['planned']}",
            f"{row['repair_attempted']}/{row['repair_rescued']}",
        )
        for row in data["by_split"]
    ])
    lines.extend(["## 失败原因", ""])
    if data["failures"]:
        lines.extend(["同一失败若在两阶段持续出现，会在两个阶段各计一次。原始记录路径相对于运行目录。", ""])
        table(lines, ("阶段", "验证环境", "方法", "状态", "原因码", "单元数", "记录示例"), [
            (
                "首轮" if row["stage"] == "initial" else "最终",
                PROFILE_LABELS.get(row["profile"], row["profile"]), METHOD_LABELS[row["method"]],
                STATUS_LABELS[row["status"]], row["reason"], row["count"],
                f"`{row['examples'][0]}`",
            )
            for row in data["failures"]
        ])
    else:
        lines.extend(["运行记录中没有已观察到的失败。未执行情况见上表。", ""])
    lines.extend(["## 实际调用用量", ""])
    lines.extend(["用量仅累计来源返回的 token。未提供用量时显示未知，同一首轮结果不会因沿用到最终而重复计算。", ""])
    table(lines, ("验证环境", "方法", "有总量记录的调用", "输入 token", "输出 token", "总 token"), [
        (
            PROFILE_LABELS.get(row["profile"], row["profile"]), METHOD_LABELS[row["method"]],
            row["tokens"]["total_tokens"]["covered_attempts"],
            *(row["tokens"][field]["sum"] if row["tokens"][field]["sum"] is not None else "未知"
              for field in TOKEN_FIELDS),
        )
        for row in data["by_profile_method"]
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json", type=Path, dest="json_out")
    args = parser.parse_args()
    try:
        cells, run, manifest_hash, knowledge_byte_cap = load_run(args.prepared, args.run)
        data = report_data(cells, run, manifest_hash, knowledge_byte_cap)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown(data), encoding="utf-8")
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"汇总失败：{exc}\n")


if __name__ == "__main__":
    main()
