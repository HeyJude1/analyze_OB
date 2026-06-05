#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实体对齐调度器v2.1 (默认3轮)
按轮次执行:
  1) cluster.py
  2) refine.py
  3) merger.py
并在每轮结束后备份输出文件。
"""

import os
import sys
import json
import shutil
import argparse
import datetime
import subprocess
from pathlib import Path

def get_base_dir_from_config(config_path: str) -> str:
    """从kg_config.json中读取analysis_results_dir"""
    if not os.path.exists(config_path):
        print(f"❌ 错误: 配置文件 '{config_path}' 不存在。")
        return ""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config.get("data_source", {}).get("analysis_results_dir", "")

def run_script(cmd: list, log_file: Path) -> subprocess.CompletedProcess:
    """运行子进程命令，实时打印输出并记录到日志文件"""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n▶️  执行命令: {' '.join(cmd)}")
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(f"\n▶️  执行命令: {' '.join(cmd)}\n")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                lf.write(output)
        
        rc = process.poll()
        return subprocess.CompletedProcess(cmd, rc)

def backup_file(src_path: Path, backup_dir: Path, new_name: str, log_file: Path):
    """备份文件到指定目录并重命名"""
    if not src_path.exists():
        log_text = f"    ⚠️ 未找到源文件进行备份: {src_path}"
        print(log_text)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_text + "\n")
        return
    
    backup_dir.mkdir(parents=True, exist_ok=True)
    dst_path = backup_dir / new_name
    shutil.copy2(str(src_path), str(dst_path))
    log_text = f"    💾 已备份: {src_path.name} -> {dst_path.relative_to(dst_path.parent.parent)}"
    print(log_text)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_text + "\n")

def main():
    parser = argparse.ArgumentParser(description="实体对齐调度器v2.1")
    # <<< MODIFIED: Changed default rounds from 1 to 3
    parser.add_argument("--rounds", type=int, default=3, help="执行轮次")
    parser.add_argument("--config", type=str, default="config/kg_config.json", help="KG配置文件路径")
    args = parser.parse_args()

    kg_dir = Path(__file__).parent.resolve()
    
    base_dir_str = get_base_dir_from_config(args.config)
    if not base_dir_str:
        print("❌ 错误: 未能在 kg_config.json 中找到 'analysis_results_dir'。")
        sys.exit(1)
        
    base_dir = Path(base_dir_str)
    if not base_dir.is_absolute():
        project_root = kg_dir.parent
        resolved_path = project_root / base_dir
        if not resolved_path.exists():
             project_folder_name = project_root.name
             if project_folder_name in base_dir_str:
                 try:
                     idx = base_dir_str.index(project_folder_name)
                     suffix = base_dir_str[idx:]
                     root_parent = project_root.parent
                     resolved_path = root_parent / suffix
                 except ValueError: pass
        base_dir = resolved_path.resolve()

    if not base_dir.exists():
        print(f"❌ 错误: 基准目录不存在: {base_dir}")
        sys.exit(1)

    log_file = base_dir / "entity_alignment.log"
    if log_file.exists():
        os.remove(log_file)

    header = f"""
============================================================
🚀 实体对齐调度器启动
  - 基准目录: {base_dir}
  - 执行轮次: {args.rounds}
  - 日志文件: {log_file}
============================================================
"""
    print(header)
    with open(log_file, "a", encoding="utf-8") as f: f.write(header)

    for r in range(1, args.rounds + 1):
        round_header = f"\n🔄 === 开始第 {r}/{args.rounds} 轮实体对齐流程 === 🔄\n"
        print(round_header)
        with open(log_file, "a", encoding="utf-8") as f: f.write(round_header)

        # --- 步骤 1: retrieve_clusters ---
        print("📝 步骤 1/3: 执行 retrieve_clusters...")
        proc = run_script(
            [sys.executable, str(kg_dir / "cluster.py"), "--config", args.config, "--data_dir", str(base_dir)],
            log_file
        )
        if proc.returncode != 0:
            error_msg = "    ❌ cluster.py 执行失败，终止流程。"
            print(error_msg)
            with open(log_file, "a", encoding="utf-8") as f: f.write(error_msg + "\n")
            break
        print("    ✅ cluster.py 执行成功。")
        
        backup_file(
            src_path=base_dir / "clusters_retrieved.json",
            backup_dir=base_dir / "clusters_retrieved",
            new_name=f"clusters_retrieved_{r}.json",
            log_file=log_file
        )

        # --- 步骤 2: refine_clusters ---
        print("\n📝 步骤 2/3: 执行 refine_clusters...")
        proc = run_script(
            [sys.executable, str(kg_dir / "refine.py"),
             "--config", args.config,
             "--data_dir", str(base_dir)],
            log_file
        )
        if proc.returncode != 0:
            error_msg = "    ❌ refine.py 执行失败，终止流程。"
            print(error_msg)
            with open(log_file, "a", encoding="utf-8") as f: f.write(error_msg + "\n")
            break
        print("    ✅ refine.py 执行成功。")

        backup_file(
            src_path=base_dir / "clusters_retrieved_refined.json",
            backup_dir=base_dir / "clusters_retrieved_refined",
            new_name=f"clusters_retrieved_refined_{r}.json",
            log_file=log_file
        )

        # --- 步骤 3: relation_merger ---
        print("\n📝 步骤 3/3: 执行 relation_merger...")
        proc = run_script(
            [sys.executable, str(kg_dir / "merger.py"),
             "--config", args.config,
             "--round", str(r)],
            log_file
        )
        if proc.returncode != 0:
            error_msg = "    ❌ merger.py 执行失败，终止流程。"
            print(error_msg)
            with open(log_file, "a", encoding="utf-8") as f: f.write(error_msg + "\n")
            break
        print("    ✅ merger.py 执行成功。")

        round_footer = f"\n✅ === 第 {r}/{args.rounds} 轮执行完成 === ✅\n"
        print(round_footer)
        with open(log_file, "a", encoding="utf-8") as f: f.write(round_footer)

    footer = f"""
============================================================
🎉 实体对齐流程全部执行完毕
  - 结果保存在: {base_dir}
============================================================
"""
    print(footer)
    with open(log_file, "a", encoding="utf-8") as f: f.write(footer)

if __name__ == "__main__":
    main()