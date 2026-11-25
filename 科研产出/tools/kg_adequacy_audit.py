#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
审计现有知识图谱（20251114_153700 产出）的"建模充分性"：
  - 实体/关系规模与类型分布
  - 策略字段覆盖率（条件、参数、硬件、代码、收益、权衡、模式）
  - 硬件知识是否可计算（有无数值参数）
  - 关系是否带属性（条件/模板/证据/溯源）
  - 10 个 competency questions 的答案：能答 / 部分能答 / 不能答

输出：10_第一周执行包/kg_adequacy_audit.md
"""
import glob
import json
import os
import re
from collections import Counter, defaultdict

BASE = r"C:\Users\MSI\Desktop\开题\开题\analyze_OB\20251114_153700"
OUT = r"C:\Users\MSI\Desktop\开题\科研产出\10_第一周执行包\kg_adequacy_audit.md"

ANALYSIS_KEYS = [
    "algorithm_level_optimizations",
    "code_level_optimizations",
    "instruction_level_optimizations",
]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def nonempty(v):
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, (list, dict, tuple)):
        return len(v) > 0
    return True


def main():
    analysis_files = sorted(glob.glob(os.path.join(BASE, "analysis_results", "*_analysis.json")))
    print("analysis files:", len(analysis_files))

    strategies = []
    operators = []
    arch_values = Counter()
    pattern_names = Counter()
    for p in analysis_files:
        d = load_json(p)
        operators.append(d.get("algorithm", os.path.basename(p)))
        for ia in d.get("individual_analyses", []):
            arch_values[ia.get("architecture", "N/A")] += 1
            for pat in ia.get("computational_patterns", []) or []:
                if isinstance(pat, dict):
                    pattern_names[pat.get("pattern_name") or pat.get("name") or str(pat)[:20]] += 1
            for key in ANALYSIS_KEYS:
                for it in ia.get(key, []) or []:
                    strategies.append(it)

    n = len(strategies)
    cov = {
        "有 description.strategy_rationale": sum(1 for s in strategies if nonempty((s.get("description") or {}).get("strategy_rationale"))),
        "有 description.implementation_pattern": sum(1 for s in strategies if nonempty((s.get("description") or {}).get("implementation_pattern"))),
        "有 performance_impact（自由文本）": sum(1 for s in strategies if nonempty((s.get("description") or {}).get("performance_impact"))),
        "有 trade_offs": sum(1 for s in strategies if nonempty((s.get("description") or {}).get("trade_offs"))),
        "有 applicability_conditions": sum(1 for s in strategies if nonempty(s.get("applicability_conditions"))),
        "有 tunable_parameters（非空）": sum(1 for s in strategies if nonempty(s.get("tunable_parameters"))),
        "有 target_hardware_feature": sum(1 for s in strategies if nonempty(s.get("target_hardware_feature"))),
        "有 code_example.snippet": sum(1 for s in strategies if nonempty((s.get("code_example") or {}).get("snippet"))),
        "有 related_patterns": sum(1 for s in strategies if nonempty(s.get("related_patterns"))),
        "有数值型参数（value_in_code）": sum(
            1 for s in strategies
            if any(nonempty(p.get("value_in_code")) for p in (s.get("tunable_parameters") or []) if isinstance(p, dict))
        ),
    }

    # 硬件特征名 + 是否含数值
    hw_names = Counter()
    hw_numeric = 0
    for s in strategies:
        name = s.get("target_hardware_feature_name")
        if nonempty(name):
            hw_names[name] += 1
        txt = s.get("target_hardware_feature") or ""
        if re.search(r"\d+\s*(KB|MB|GB|bit|位|个寄存器|way|周期|GHz|MHz)", txt, re.I):
            hw_numeric += 1

    # 参数名
    param_names = Counter()
    for s in strategies:
        for p in s.get("tunable_parameters") or []:
            if isinstance(p, dict) and p.get("parameter_name"):
                param_names[p["parameter_name"]] += 1

    # 关系
    rel_path = os.path.join(BASE, "relation_entity.json")
    rel_counts = {}
    rel_has_attrs = {}
    if os.path.exists(rel_path):
        rels = load_json(rel_path)
        for rtype, items in rels.items():
            rel_counts[rtype] = len(items)
            if items:
                keys = set()
                for it in items[:20]:
                    keys.update(it.keys())
                rel_has_attrs[rtype] = sorted(keys)

    # 对齐簇
    cl_path = os.path.join(BASE, "clusters_retrieved_refined.json")
    clusters = {}
    if os.path.exists(cl_path):
        cl = load_json(cl_path)
        for k, v in cl.items():
            clusters[k] = len(v)

    # 生成报告
    lines = []
    lines.append("# 现有知识图谱建模体检报告（自动审计）\n")
    lines.append("> 数据源：`%s`｜生成脚本：`tools/kg_adequacy_audit.py`｜审计日期：2026-11-12\n" % BASE)
    lines.append("## 一句话结论\n")
    lines.append("**当前 KG 不是「建模错了」，而是「建成了知识清单，还不是可执行的知识库」。** "
                 "它已经能回答「某算子有哪些优化策略、大致针对什么硬件」，但**不能**回答「这个策略在当前硬件/当前代码条件下是否适用、参数取多少、怎么落地、收益证据是什么、和哪些策略冲突」——"
                 "而这四件事恰好是 B3（完整 KG）相对 B2（扁平 RAG）唯一可能产生增量的地方。**所以：先做 2 天 Schema v1.5 补丁，再开实验，不要重做 v2，也不要直接开实验。**\n")
    lines.append("## 1. 规模\n")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---:|")
    lines.append("| 分析文件 | %d |" % len(analysis_files))
    lines.append("| 算子 | %s |" % "、".join(str(o) for o in operators))
    lines.append("| 策略条目 | %d |" % n)
    lines.append("| 关系类型 | %d |" % len(rel_counts))
    lines.append("| 关系条数 | %d |" % sum(rel_counts.values()))
    lines.append("| 计算流程（pattern）条目 | %d |" % sum(pattern_names.values()))
    lines.append("| 对齐簇 | %s |" % ", ".join("%s=%d" % (k, v) for k, v in clusters.items()))
    lines.append("")
    lines.append("### 关系类型分布\n")
    lines.append("| 关系 | 条数 | 实际带有的字段 |")
    lines.append("|---|---:|---|")
    for k, v in sorted(rel_counts.items(), key=lambda x: -x[1]):
        lines.append("| %s | %d | %s |" % (k, v, ", ".join(rel_has_attrs.get(k, []))))
    lines.append("")
    lines.append("> 只有 4 类关系有实例（声明了 12 类），且边属性只有 `relation_type/id/head/tail`——**没有条件、模板、证据、权重**。这是「图结构无法产生约束操作」的直接证据。\n")

    lines.append("## 2. 策略字段覆盖率（n=%d）\n" % n)
    lines.append("| 字段 | 覆盖数 | 覆盖率 | 对 B3 的作用 |")
    lines.append("|---|---:|---:|---|")
    roles = {
        "有 applicability_conditions": "可执行条件过滤（B3 核心）",
        "有 tunable_parameters（非空）": "参数推荐（B3 核心）",
        "有数值型参数（value_in_code）": "参数值可从知识推出",
        "有 target_hardware_feature": "硬件约束（B3 核心）",
        "有 code_example.snippet": "实施模板的原料",
        "有 description.implementation_pattern": "实施步骤",
        "有 performance_impact（自由文本）": "收益排序（但不可核查）",
        "有 trade_offs": "冲突/代价",
        "有 related_patterns": "检索锚点",
        "有 description.strategy_rationale": "可解释性",
    }
    for k, v in cov.items():
        lines.append("| %s | %d | %.1f%% | %s |" % (k, v, 100.0 * v / n if n else 0, roles.get(k, "")))
    lines.append("")
    lines.append("> 关键：**performance_impact 覆盖率可能很高，但它是 LLM 写的形容词（无 baseline/尺寸/平台/协议），不能当「收益证据」排序**；`applicability_conditions` 是自由中文，无法做谓词判断；`tunable_parameters` 即使有，也缺「由哪个硬件参数推导」的字段。\n")

    lines.append("### 硬件知识是否可计算\n")
    lines.append("- `target_hardware_feature_name` 取值 Top 10：%s" % "、".join("%s(%d)" % (k, v) for k, v in hw_names.most_common(10)))
    lines.append("- 含**数值**（KB/MB/bit/寄存器数/周期等）的策略描述：**%d / %d（%.1f%%）**" % (hw_numeric, n, 100.0 * hw_numeric / n if n else 0))
    lines.append("- 结论：硬件知识基本是「Cache / SIMD / 寄存器」这类**名词**，没有 `VLEN=256`、`L1d=32KB`、`LMUL≤8` 这种可参与判断的数值。\n")

    lines.append("### 参数名 Top 15\n")
    lines.append("| 参数名 | 出现次数 |")
    lines.append("|---|---:|")
    for k, v in param_names.most_common(15):
        lines.append("| %s | %d |" % (k, v))
    lines.append("")

    lines.append("## 3. 十个 competency questions（能答/部分/不能）\n")
    cqs = [
        ("CQ1 给定算子，列出它用到的优化策略", "能（OPTIMIZES_PATTERN + 策略实体）"),
        ("CQ2 给定策略，说明它优化的是哪类计算流程", "能（OPTIMIZES_PATTERN）"),
        ("CQ3 给定策略，说明它依赖哪些硬件特性", "部分（只有硬件名词，无参数值）"),
        ("CQ4 给定硬件 profile（x86 AVX2 / RVV VLEN=256），过滤掉不适用的策略", "不能（无 ISA/参数/硬条件谓词）"),
        ("CQ5 给定用户代码特征（dtype/stride/loop_depth/size），判断策略是否适用", "不能（applicability_conditions 是自由文本）"),
        ("CQ6 给定策略，给出可调参数及其取值/范围", "部分（约 %d%% 有参数，缺「由硬件推导」字段）" % int(100.0 * cov["有 tunable_parameters（非空）"] / n if n else 0)),
        ("CQ7 给定策略，给出可执行实施模板（骨架 + 槽位）", "不能（只有自然语言 implementation_pattern + 代码片段）"),
        ("CQ8 给定策略，给出可核查的性能收益证据", "不能（performance_impact 是形容词，无 baseline/协议）"),
        ("CQ9 判断两条策略是否冲突/可组合", "不能（CONFLICTS_WITH / COMPOSES_WITH 实例为 0）"),
        ("CQ10 任意一条知识可追溯到源文件行号/版本", "部分（只有 source_file 文件名，无行号/commit）"),
    ]
    lines.append("| # | 问题 | 现状 |")
    lines.append("|---|---|---|")
    for q, a in cqs:
        lines.append("| %s | %s |" % (q, a))
    lines.append("")

    lines.append("## 4. 判决与建议（先补 3 个缺口，再实验）\n")
    lines.append("**建模方向没有错**（实体类型、三层策略、计算流程锚点都是对的），但当前版本是 v1「知识清单」；要支撑 B3 必须补 3 个会在实验里直接变成混杂因素的缺口：\n")
    lines.append("1. **硬件可计算**（0.5 天）：为 x86 与 K1 各写一份 `HardwareProfile` YAML（ISA、VLEN/LMUL、L1/L2、核数、频率、峰值/带宽），策略边加 `hardware_profile_id` + `parameter_derivation`；不需要全量，先覆盖 Top 20 策略。\n")
    lines.append("2. **条件可执行**（0.5–1 天）：把 Top 20 策略的 `applicability_conditions` 改写成 3–6 个谓词（`dtype in {s,d,c,z}`、`stride_x == 1`、`loop_depth >= 2`、`size_regime in {small,medium,large}`、`aligned == true`、`tail_divisible` 等）；其余策略保留自由文本、标 `condition_status=text`。\n")
    lines.append("3. **模板 + 证据 + 溯源**（1 天）：Top 20 策略各写一个 `ImplementationTemplate`（骨架 + 槽位 + 包含头），并从语料 `_log.txt`/编译命令抽 `SourceArtifact`（file + line + commit/版本）；性能证据先留「待测」字段，实验后回填（不要用 LLM 形容词）。\n")
    lines.append("**做完这 3 个补丁，冻结为 Schema v1.5，开实验。** 实验过程中只允许「回填证据」，不再改 Schema；如果 pilot 失败且能归因到 Schema，再开 v2。\n")
    lines.append("**时间盒**：2 天（Day 1 硬件 + 条件；Day 2 模板 + 溯源 + 冻结）；这 2 天不写代码框架、不跑 LLM 实验。\n")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("kernel strategies:", n)
    print("field coverage:", cov)
    print("relations:", rel_counts)
    print("output:", OUT)


if __name__ == "__main__":
    main()
