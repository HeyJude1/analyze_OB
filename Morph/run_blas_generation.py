#!/usr/bin/env python3
"""
BLAS算子代码生成完整流程示例
演示从OpenBLAS源码到优化策略再到LLM代码生成的完整过程
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def run_command(cmd, cwd=None):
    """运行命令并返回结果"""
    print(f"🔄 执行命令: {cmd}")
    if cwd:
        print(f"📁 工作目录: {cwd}")
    
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ 命令执行成功")
        if result.stdout:
            print(f"📤 输出: {result.stdout}")
    else:
        print(f"❌ 命令执行失败 (退出码: {result.returncode})")
        if result.stderr:
            print(f"📤 错误: {result.stderr}")
    
    return result

def main():
    """主流程"""
    print("🚀 BLAS算子代码生成完整流程")
    print("=" * 60)
    
    # 设置路径
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    kg_dir = project_root / "KG"
    openblas_dir = script_dir / "openblas_output"
    
    print(f"📁 项目根目录: {project_root}")
    print(f"📁 KG目录: {kg_dir}")
    print(f"📁 OpenBLAS输出目录: {openblas_dir}")
    
    # 检查必要文件
    required_files = [
        kg_dir / "Operator_op2.py",
        kg_dir / "kg_config.json",
        script_dir / "functions" / "llmgenv4.py",
        script_dir / "prompts1.json"
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        print("❌ 缺少必要文件:")
        for f in missing_files:
            print(f"   - {f}")
        return
    
    print("✅ 所有必要文件检查通过")
    
    # 步骤1: 批量生成优化策略
    print("\n" + "="*60)
    print("步骤1: 批量生成优化策略")
    print("="*60)
    
    print(f"📂 OpenBLAS目录: {openblas_dir}")
    print(f"📁 输出目录: /home/dgc/mjs/project/analyze_OB/op_results")
    
    # 使用新的批量处理功能
    cmd = f"python Operator_op2.py --batch --openblas_dir {openblas_dir}"
    result = run_command(cmd, cwd=kg_dir)
    
    if result.returncode != 0:
        print("❌ 批量优化策略生成失败")
        return
    
    # 步骤2: 使用LLM生成优化代码
    print("\n" + "="*60)
    print("步骤2: LLM代码生成")
    print("="*60)
    
    # 运行 llmgenv4.py
    cmd = (f"python functions/llmgenv4.py "
           f"--input prompts1.json "
           f"--output results/blas_optimized_code_v4.json "
           f"--strategy_dir /home/dgc/mjs/project/analyze_OB/op_results "
           f"--config ../KG/kg_config.json")
    
    result = run_command(cmd, cwd=script_dir)
    
    if result.returncode == 0:
        print("✅ LLM代码生成完成")
    else:
        print("❌ LLM代码生成失败")
        return
    
    # 步骤3: 分析结果
    print("\n" + "="*60)
    print("步骤3: 结果分析")
    print("="*60)
    
    results_file = script_dir / "results" / "blas_optimized_code_v4.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print(f"📊 生成结果统计:")
        print(f"   - 总prompt数: {len(results)}")
        
        success_count = 0
        error_count = 0
        
        for result in results:
            if "outputs" in result and result["outputs"]:
                output = result["outputs"][0]
                if output.startswith("// Error"):
                    error_count += 1
                else:
                    success_count += 1
        
        print(f"   - 成功生成: {success_count}")
        print(f"   - 生成失败: {error_count}")
        print(f"   - 成功率: {success_count/len(results)*100:.1f}%")
        
        # 显示每个算子的结果
        print(f"\n📋 各算子生成结果:")
        for result in results:
            name = result.get("name", "unknown")
            model = result.get("parallelism_model", "unknown")
            status = "✅" if ("outputs" in result and result["outputs"] and not result["outputs"][0].startswith("// Error")) else "❌"
            print(f"   {status} {name} ({model})")
    
    else:
        print(f"❌ 结果文件不存在: {results_file}")
    
    print("\n🎉 完整流程执行完成！")
    print("\n📁 输出文件位置:")
    print(f"   - 优化策略: /home/dgc/mjs/project/analyze_OB/op_results/")
    print(f"   - 生成代码: {results_file}")

if __name__ == "__main__":
    main()
