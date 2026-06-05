#!/usr/bin/env python3
"""
OpenBLAS 优化策略知识图谱 — 全流程管线
一键运行: 分析 → 抽取 → 对齐 → 检索推荐

用法:
  # 完整流程 (从源码分析开始)
  python run_pipeline.py --full --source-dir data/openblas-output/GENERIC/kernel

  # 仅检索推荐 (假设已有知识图谱)
  python run_pipeline.py --recommend --source-file your_code.c

  # 同时对接 Morph 代码生成
  python run_pipeline.py --recommend --source-file your_code.c --morph
"""

import os
import sys
import json
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
SRC_DIR = SCRIPT_DIR / "src"


def run_step(name: str, cmd: list, cwd=None) -> bool:
    """运行单个步骤，打印状态"""
    print(f"\n{'='*60}")
    print(f"▶  {name}")
    print(f"{'='*60}")
    print(f"  命令: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, cwd=cwd or SCRIPT_DIR,
                           capture_output=False, text=True)
    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"✅ {name} — 完成 ({elapsed:.1f}s)")
        return True
    else:
        print(f"❌ {name} — 失败 (exit={result.returncode}, {elapsed:.1f}s)")
        return False


def find_latest_results() -> str:
    """查找最新的分析结果目录"""
    results_dir = SCRIPT_DIR / "output" / "results"
    if not results_dir.exists():
        return ""
    dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name != "train"],
                  reverse=True)
    return str(dirs[0]) if dirs else ""


def run_full_pipeline(source_dir: str):
    """完整流程: 分析 → 抽取 → 对齐"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = SCRIPT_DIR / "output" / "results" / timestamp
    analysis_dir = output_dir / "analysis_results"
    os.makedirs(analysis_dir, exist_ok=True)

    print(f"📁 输出目录: {output_dir}")

    # Step 1: 优化策略分析 (使用现有的 analysis 模块)
    if not run_step("Step 1/4: 优化策略分析",
                    [sys.executable, "-c", f"""
import sys; sys.path.insert(0, '{SRC_DIR}')
from analysis.workflow import Workflow24
from analysis.optimizers import load_config

config = load_config()
workflow = Workflow24()

# 分析所有算子
import os, json
source_files = []
for f in sorted(os.listdir('{source_dir}')):
    if f.endswith('.c') or f.endswith('.cpp'):
        source_files.append(os.path.join('{source_dir}', f))

results = {{"algorithm": "openblas_kernels", "total_files": len(source_files),
            "analyzed_files": [], "individual_analyses": []}}

for sf in source_files:
    alg = os.path.splitext(os.path.basename(sf))[0]
    try:
        r = workflow.run_analysis(alg, [sf])
        results["analyzed_files"].append(sf)
        results["individual_analyses"].append({{
            "file_path": sf, "algorithm": alg,
            "architecture": "GENERIC",
            "computational_patterns": [],
            "algorithm_level_optimizations": r.get("algorithm_level_optimizations", []),
            "code_level_optimizations": r.get("code_level_optimizations", []),
            "instruction_level_optimizations": r.get("instruction_level_optimizations", [])
        }})
    except Exception as e:
        print(f"  ⚠️ 分析 {{sf}} 失败: {{e}}")

with open('{analysis_dir}/analysis.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"📄 分析结果已保存到 {analysis_dir}/analysis.json")
"""]):
        print("⚠️ 分析步骤未完成，但继续后续步骤...")

    # Step 2: 实体抽取
    config_path = SCRIPT_DIR / "config" / "kg_config.json"
    if not run_step("Step 2/4: 实体抽取",
                    [sys.executable, str(SRC_DIR / "kg" / "extractor.py"),
                     "--config", str(config_path),
                     "--data_dir", str(output_dir)]):
        print("❌ 实体抽取失败，终止")
        return

    # Step 3: 实体对齐
    if not run_step("Step 3/4: 实体对齐",
                    [sys.executable, str(SRC_DIR / "kg" / "alignment.py"),
                     "--config", str(config_path),
                     "--rounds", "3"]):
        print("⚠️ 实体对齐步骤有问题，但流程继续")

    print(f"\n🎉 全流程完成！输出目录: {output_dir}")
    print(f"   下一步: python run_pipeline.py --recommend --source-file <your_code.c>")


def run_recommendation(source_file: str, with_morph: bool = False):
    """检索推荐: 对输入代码推荐优化策略"""
    config_path = SCRIPT_DIR / "config" / "kg_config.json"
    output_dir = find_latest_results() or str(SCRIPT_DIR / "output" / "results" / "latest")

    # 确保输出目录存在
    os.makedirs(SCRIPT_DIR / "output" / "op_results", exist_ok=True)

    if not run_step("Step: 优化策略检索",
                    [sys.executable, str(SRC_DIR / "kg" / "retrieval.py"),
                     "--config", str(config_path),
                     "--source", source_file,
                     "--output_dir", str(SCRIPT_DIR / "output" / "op_results")]):
        print("❌ 检索失败")
        return

    # 找到输出结果
    source_name = os.path.splitext(os.path.basename(source_file))[0]
    result_dir = SCRIPT_DIR / "output" / "op_results" / source_name
    result_file = result_dir / f"{source_name}.json"

    if result_file.exists():
        with open(result_file) as f:
            data = json.load(f)
        strategies = data.get("final_strategies", [])
        print(f"\n📊 推荐结果: {len(strategies)} 个优化策略")
        for s in strategies[:5]:
            name = s.get("canonical_name", s.get("name", "?"))
            score = s.get("score", 0)
            print(f"   • {name} (评分: {score:.3f})")

    if with_morph and result_file.exists():
        print(f"\n🔗 对接 Morph 代码生成...")
        run_step("Morph: 代码生成",
                 [sys.executable, "-m", "src.llmgen",
                  "--input", str(SCRIPT_DIR / "Morph" / "prompts1.json"),
                  "--output", str(SCRIPT_DIR / "Morph" / "results" / "blas_code.json"),
                  "--strategy_dir", str(SCRIPT_DIR / "output" / "op_results")],
                 cwd=str(SCRIPT_DIR / "Morph"))


def main():
    parser = argparse.ArgumentParser(description="OpenBLAS 优化策略知识图谱 — 全流程管线")
    parser.add_argument("--full", action="store_true", help="运行完整流程 (分析→抽取→对齐)")
    parser.add_argument("--recommend", action="store_true", help="检索推荐优化策略")
    parser.add_argument("--source-dir", type=str, help="OpenBLAS 源码目录")
    parser.add_argument("--source-file", type=str, help="待分析的新代码文件")
    parser.add_argument("--morph", action="store_true", help="对接 Morph 代码生成")
    args = parser.parse_args()

    if args.full:
        source_dir = args.source_dir or str(SCRIPT_DIR / "data" / "openblas-output" / "GENERIC" / "kernel")
        run_full_pipeline(source_dir)
    elif args.recommend:
        if not args.source_file:
            print("❌ 请指定 --source-file")
            return
        run_recommendation(args.source_file, with_morph=args.morph)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
