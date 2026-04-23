"""
基于优化策略的代码生成器 v4
在 llmgenv3.py 基础上增加对 Operator_op2.py 生成的优化策略的支持
支持从Milvus查询实体详情以生成更丰富的优化策略说明
"""
"""
利用智能体，生成代码的运行过程在此处进行定义
"""
from dataclasses import dataclass, field
# std imports
from argparse import ArgumentParser
import json
import os
import re
import time
from typing import Optional, Dict, List
# tpl imports
from tqdm import tqdm
# langchain imports
from base_agents import AgentFactory
# 工具函数
import tiktoken
from utils import get_env_var, get_function_name, postprocess, cpu_info, gpu_info
# Milvus imports
from pymilvus import connections, Collection

PROMPT_TEMPLATE = """请补全 C++ 函数 {function_name}。仅编写函数 {function_name} 的函数体。

```cpp
{prompt}
```
目标硬件信息：
{hard_info}

来自分析的额外优化建议：
{optimization_strategies}
"""

class MilvusEntityQuerier:
    """Milvus实体查询器"""
    
    def __init__(self, config_path: str = "config.json"):
        """初始化Milvus连接"""
        self.config = self._load_config(config_path)
        self.connection_alias = "llmgen_connection"
        self._connect_to_milvus()
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        # 尝试多个可能的配置文件路径
        possible_paths = [
            config_path,
            os.path.join(os.path.dirname(__file__), config_path),
            "config.json",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        # 如果找不到配置文件，使用默认配置
        print("⚠️ 警告: 未找到配置文件，使用默认Milvus配置")
        return {
            "milvus": {
                "host": "localhost",
                "port": 19530,
                "database": "code_op"
            }
        }
    
    def _connect_to_milvus(self):
        """连接到Milvus"""
        try:
            milvus_config = self.config.get("milvus", {})
            host = milvus_config.get("host", "localhost")
            port = milvus_config.get("port", 19530)
            database = milvus_config.get("database", "code_op")
            
            connections.connect(
                alias=self.connection_alias,
                host=host,
                port=port,
                db_name=database
            )
            print(f"✅ 已连接到Milvus: {host}:{port}/{database}")
            
        except Exception as e:
            print(f"❌ 错误: 连接Milvus失败: {e}")
            self.connection_alias = None
    
    def query_entity_by_uid(self, collection_name: str, uid: str) -> Dict:
        """根据UID查询实体详情"""
        if not self.connection_alias:
            return {}
        
        try:
            collection = Collection(collection_name, using=self.connection_alias)
            
            # 查询实体
            results = collection.query(
                expr=f'uid == "{uid}"',
                output_fields=["*"],
                limit=1
            )
            
            if results:
                return results[0]
            else:
                print(f"⚠️ 警告: 未找到UID为 {uid} 的实体")
                return {}
                
        except Exception as e:
            print(f"❌ 错误: 查询实体 {uid} 失败: {e}")
            return {}
    
    def close_connection(self):
        """关闭Milvus连接"""
        if self.connection_alias:
            try:
                connections.disconnect(self.connection_alias)
                print("✅ Milvus连接已关闭")
            except Exception as e:
                print(f"⚠️ 警告: 关闭Milvus连接时出错: {e}")

def check_final_strategies_exist(strategy_file_path: str) -> bool:
    """
    检查优化策略文件中是否有非空的 final_strategies
    
    Args:
        strategy_file_path: 优化策略JSON文件路径
        
    Returns:
        True if final_strategies exists and is not empty, False otherwise
    """
    if not os.path.exists(strategy_file_path):
        return False
    
    try:
        with open(strategy_file_path, 'r', encoding='utf-8') as f:
            strategy_data = json.load(f)
        
        final_strategies = strategy_data.get('final_strategies', [])
        return len(final_strategies) > 0
        
    except Exception as e:
        print(f"❌ 错误: 无法检查策略文件 {strategy_file_path}: {e}")
        return False

def load_optimization_strategies(strategy_file_path: str, milvus_querier: Optional[MilvusEntityQuerier] = None) -> str:
    """
    加载优化策略文件并提取 final_strategies，从Milvus查询详细信息
    
    Args:
        strategy_file_path: 优化策略JSON文件路径
        milvus_querier: Milvus查询器实例
        
    Returns:
        格式化的优化策略字符串
    """
    if not os.path.exists(strategy_file_path):
        print(f"⚠️ 警告: 优化策略文件不存在: {strategy_file_path}")
        return "没有可用的特定优化策略。"
    
    try:
        with open(strategy_file_path, 'r', encoding='utf-8') as f:
            strategy_data = json.load(f)
        
        final_strategies = strategy_data.get('final_strategies', [])
        
        if not final_strategies:
            return "未找到最终优化策略。"
        
        # 格式化优化策略
        strategy_text = []
        strategy_text.append("推荐的优化策略：")
        
        for i, strategy in enumerate(final_strategies, 1):
            # final_strategies 中的字段结构
            strategy_name = strategy.get('canonical_name', strategy.get('name', '未知策略'))
            strategy_uid = strategy.get('strategy_uid', strategy.get('uid', ''))
            score = strategy.get('score', 0)
            
            strategy_text.append(f"\n{i}. **{strategy_name}** (得分: {score:.3f})")
            
            # 如果有Milvus查询器且有UID，查询详细信息
            detailed_info = {}
            if milvus_querier and strategy_uid:
                detailed_info = milvus_querier.query_entity_by_uid("optimization_strategy", strategy_uid)
            
            # 使用数据库中实际存在的字段构建策略描述
            if detailed_info:
                # 优化级别
                level = detailed_info.get('level', '')
                if level:
                    strategy_text.append(f"   - 优化级别: {level}")
                
                # 策略原理
                rationale = detailed_info.get('rationale', '')
                if rationale:
                    strategy_text.append(f"   - 原理: {rationale}")
                
                # 实现方法
                implementation = detailed_info.get('implementation', '')
                if implementation:
                    strategy_text.append(f"   - 实现方法: {implementation}")
                
                # 预期影响
                impact = detailed_info.get('impact', '')
                if impact:
                    strategy_text.append(f"   - 预期影响: {impact}")
                
                # 权衡考虑
                trade_offs = detailed_info.get('trade_offs', '')
                if trade_offs:
                    strategy_text.append(f"   - 权衡: {trade_offs}")
        
        return '\n'.join(strategy_text)
        
    except Exception as e:
        print(f"❌ 错误: 无法加载优化策略文件 {strategy_file_path}: {e}")
        return "加载优化策略时出错。"

def CodeGenv4(
    input_path: str = "../prompts1.json",
    output_path: str = "../results/prompts_code.json",
    strategy_dir: str = "/home/dgc/mjs/project/analyze_OB/op_results",
    config_path: str = "config.json",
    # 可选覆盖参数
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    
    dry: bool = False,
    overwrite: bool = False,
    num_samples_per_prompt: int = 1, 
    use_milvus: bool = True  # 是否使用Milvus查询详细信息
    )-> None:

    # 1. 初始化 Agent Factory
    print(f"🔧 加载配置: {config_path}")
    factory = AgentFactory(config_path)
    
    # 解析默认参数 (如果未传入，则使用Config中的值)
    model_config = factory.config.get("model", {})
    if model is None:
        model = model_config.get("name", "qwen-plus-2025-09-11")
    if temperature is None:
        temperature = model_config.get("temperature", 0.1)
    if top_p is None:
        top_p = model_config.get("top_p", 0.9)
    if max_new_tokens is None:
        max_new_tokens = model_config.get("max_tokens", 8192)

    # 2. 获取硬件信息
    print("🖥️ 获取硬件信息...")
    hard_info = cpu_info() + gpu_info()
    # 移除 PromptGenerate Agent 调用，直接使用字符串

    # 3. 初始化Milvus查询器
    milvus_querier = None
    if use_milvus:
        try:
            print("🔗 初始化Milvus连接...")
            milvus_querier = MilvusEntityQuerier(config_path)
        except Exception as e:
            print(f"⚠️ 警告: Milvus连接失败，将使用基础策略信息: {e}")
            use_milvus = False

    # 读取输入的prompts
    with open(input_path, "r") as prompts_json:
        prompts = json.load(prompts_json)
    
    # 步骤1: 确定本次要生成的算子种类
    operator_names = list(set([prompt["name"] for prompt in prompts]))
    print(f"📋 发现 {len(operator_names)} 种算子: {operator_names}")
    
    # 步骤2: 检查每个算子的 final_strategies 是否为空
    valid_operators = []
    for operator_name in operator_names:
        strategy_file = os.path.join(strategy_dir, operator_name, f"{operator_name}.json")
        if check_final_strategies_exist(strategy_file):
            valid_operators.append(operator_name)
            print(f"✅ {operator_name}: 有可用的优化策略")
        else:
            print(f"⚠️ {operator_name}: 无优化策略，将跳过")
    
    print(f"\n📊 统计: {len(valid_operators)}/{len(operator_names)} 个算子有可用策略")
    print(f"🎯 将生成代码的算子: {valid_operators}")
    
    if not valid_operators:
        print("❌ 没有任何算子有可用的优化策略，退出代码生成")
        if milvus_querier:
            milvus_querier.close_connection()
        return

    # 如果已有输出文件并且不覆盖，加载旧输出合并
    if not overwrite and os.path.exists(output_path):
        with open(output_path, 'r') as output_json:
            outputs = json.load(output_json)

        # 将已经存在的内容复制到结果中
        copy_count = 0
        for prompt in prompts:
            for o in outputs:
                if o["prompt"] == prompt["prompt"] and \
                   o["name"] == prompt["name"] and \
                   o["parallelism_model"] == prompt["parallelism_model"] and \
                   "outputs" in o and \
                   len(o["outputs"]) == num_samples_per_prompt:
                    for col in ["outputs"]:
                        prompt[col] = o[col]
                    copy_count += 1
                    break
        print(f"Copied {copy_count} existing outputs.")

    # 创建代码生成Agent (允许覆盖参数)
    code_agent = factory.create_code_generate_agent(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens
    )

    # 用于保存生成的完整prompt
    generated_prompts = []

    # 处理每个prompt
    for prompt in tqdm(prompts):
        # 如果已经有输出，跳过
        if "outputs" in prompt:
            continue

        # 提取算子名称，用于查找对应的优化策略
        operator_name = prompt["name"]  # 例如: "01_gemm"
        
        # 检查该算子是否在有效算子列表中
        if operator_name not in valid_operators:
            print(f"⏭️ 跳过算子 {operator_name} ({prompt['parallelism_model']}): 无可用优化策略")
            continue
        
        # 构建优化策略文件路径
        strategy_file = os.path.join(strategy_dir, operator_name, f"{operator_name}.json")
        
        # 加载优化策略（包含Milvus查询）
        optimization_strategies = load_optimization_strategies(strategy_file, milvus_querier)
        
        # 获取函数名（根据并行模型确定执行模型）
        execution_model = "cuda" if prompt["parallelism_model"] == "cuda" else "cpu"
        function_name = get_function_name(prompt["prompt"], execution_model)
        
        # 构建完整的prompt
        full_prompt = PROMPT_TEMPLATE.format(
            function_name=function_name,
            prompt=prompt["prompt"],
            hard_info=hard_info,
            optimization_strategies=optimization_strategies
        )
        
        # 保存完整的prompt到列表
        generated_prompts.append({
            "name": prompt["name"],
            "parallelism_model": prompt["parallelism_model"],
            "full_prompt": full_prompt
        })
        
        if dry:
            print(f"Dry run for {prompt['name']} ({prompt['parallelism_model']}):")
            print(f"Strategy file: {strategy_file}")
            print(f"Optimization strategies loaded: {len(optimization_strategies.split('**')) - 1} strategies")
            print("---")
            continue
        
        # 生成代码
        try:
            print(f"🔄 生成代码: {prompt['name']} ({prompt['parallelism_model']})")
            print(f"📁 策略文件: {strategy_file}")
            
            result = code_agent.invoke({"input": full_prompt})
            generated_code = result.content
            
            # 后处理生成的代码
            processed_code = postprocess(prompt["prompt"], generated_code)
            
            # 添加结果到结果json文件中的output字段
            prompt["outputs"] = [processed_code]
            
            print(f"✅ 完成: {prompt['name']} ({prompt['parallelism_model']})")
            
        except Exception as e:
            print(f"❌ 错误: 生成 {prompt['name']} ({prompt['parallelism_model']}) 时出错: {e}")
            prompt["outputs"] = [f"// Error generating code: {e}"]
        
        # 保存中间结果
        with open(output_path, 'w') as output_json:
            json.dump(prompts, output_json, indent=2)

    # 保存所有生成的prompt到prompts_gen.json
    prompts_gen_path = os.path.join(os.path.dirname(output_path), "prompts_gen.json")
    with open(prompts_gen_path, 'w', encoding='utf-8') as f:
        json.dump(generated_prompts, f, indent=2, ensure_ascii=False)
    print(f"💾 已保存生成的prompt到: {prompts_gen_path}")

    # 关闭Milvus连接
    if milvus_querier:
        milvus_querier.close_connection()

    print(f"🎉 代码生成完成！结果保存至: {output_path}")

def main():
    """主函数"""
    parser = ArgumentParser(description="基于优化策略的代码生成器 v4")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    parser.add_argument("--input", type=str, default="../prompts1.json", help="输入prompt文件")
    parser.add_argument("--output", type=str, default="../results/prompts_code.json", help="输出文件")
    parser.add_argument("--strategy_dir", type=str, default="/home/dgc/mjs/project/analyze_OB/op_results", help="优化策略目录")
    
    # 可选覆盖参数 (默认None，使用Config中的值)
    parser.add_argument("--model", type=str, default=None, help="模型名称 (覆盖Config)")
    parser.add_argument("--temperature", type=float, default=None, help="生成温度 (覆盖Config)")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p采样 (覆盖Config)")
    parser.add_argument("--max_tokens", type=int, default=None, help="最大生成token数 (覆盖Config)")
    
    parser.add_argument("--dry", action="store_true", help="干运行模式")
    parser.add_argument("--overwrite", action="store_true", help="覆盖现有输出")
    parser.add_argument("--no-milvus", action="store_true", help="不使用Milvus查询详细信息")
    
    args = parser.parse_args()
    
    print("🚀 基于优化策略的代码生成器 v4")
    print("=" * 50)
    print(f"⚙️ 配置文件: {args.config}")
    print(f"📄 输入文件: {args.input}")
    print(f"💾 输出文件: {args.output}")
    print("=" * 50)
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    CodeGenv4(
        config_path=args.config,
        input_path=args.input,
        output_path=args.output,
        strategy_dir=args.strategy_dir,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_tokens,
        dry=args.dry,
        overwrite=args.overwrite,
        use_milvus=not args.no_milvus
    )

if __name__ == "__main__":
    main()
