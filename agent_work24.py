#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenBLAS优化策略分析工作流 v24
集成格式验证和重试机制
"""

import os
import json
import time
from typing import Dict, Any, List, TypedDict
from pathlib import Path
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from agent24 import create_algorithm_optimizer, create_code_optimizer, create_instruction_optimizer, load_config

load_dotenv()


class WorkflowState(TypedDict):
    """工作流状态"""
    algorithm: str
    source_files: List[str]
    analysis_results: Dict[str, Any]
    current_stage: str
    errors: List[str]
    retry_count: int


class Workflow24:
    """OpenBLAS优化分析工作流v24"""
    
    def __init__(self):
        self.config = load_config()
        self.model_config = self.config.get("model", {})
        
        # 创建优化分析器
        self.algorithm_llm, self.algorithm_prompt = create_algorithm_optimizer(self.model_config)
        self.code_llm, self.code_prompt = create_code_optimizer(self.model_config)
        self.instruction_llm, self.instruction_prompt = create_instruction_optimizer(self.model_config)
        
        # 构建工作流
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """构建LangGraph工作流"""
        workflow = StateGraph(WorkflowState)
        
        # 添加节点
        workflow.add_node("algorithm_analysis", self.algorithm_analysis_node)
        workflow.add_node("code_analysis", self.code_analysis_node)
        workflow.add_node("instruction_analysis", self.instruction_analysis_node)
        workflow.add_node("finalize", self.finalize_node)
        
        # 设置边
        workflow.add_edge(START, "algorithm_analysis")
        workflow.add_edge("algorithm_analysis", "code_analysis")
        workflow.add_edge("code_analysis", "instruction_analysis")
        workflow.add_edge("instruction_analysis", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _validate_optimization_format(self, optimization: Dict[str, Any]) -> bool:
        """验证单个优化策略的格式"""
        required_fields = ["optimization_name", "level", "description", 
                          "applicability_conditions", "tunable_parameters", "related_patterns"]
        
        # 检查必需字段
        for field in required_fields:
            if field not in optimization:
                return False
        
        # 检查description字段结构
        description = optimization.get("description", {})
        if not isinstance(description, dict):
            return False
        
        required_desc_fields = ["strategy_rationale", "implementation_pattern", 
                               "performance_impact", "trade_offs"]
        
        for field in required_desc_fields:
            if field not in description:
                return False
        
        # 确保description中没有其他字段
        if len(description) != 4:
            return False
        
        return True
    
    def _validate_optimization_list(self, optimizations: List[Dict[str, Any]]) -> bool:
        """验证优化策略列表的格式"""
        if not isinstance(optimizations, list):
            return False
        
        for opt in optimizations:
            if not self._validate_optimization_format(opt):
                return False
        
        return True
    
    def analyzer_work_node(self, state: WorkflowState, level: str, llm, prompt: str) -> WorkflowState:
        """通用分析节点，带格式验证和重试"""
        algorithm = state["algorithm"]
        source_files = state.get("source_files", [])
        
        # 准备分析输入
        source_content = ""
        for file_path in source_files:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    source_content += f"\n=== {file_path} ===\n{content}\n"
        
        # 创建结构化输出解析器
        response_schemas = [
            ResponseSchema(name="optimization_name", description="优化策略名称"),
            ResponseSchema(name="level", description="优化层级"),
            ResponseSchema(name="description", description="包含4个子字段的详细描述对象"),
            ResponseSchema(name="applicability_conditions", description="适用条件"),
            ResponseSchema(name="tunable_parameters", description="可调参数"),
            ResponseSchema(name="related_patterns", description="相关计算流程")
        ]
        
        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = parser.get_format_instructions()
        
        # 构建完整提示
        full_prompt = f"""{prompt}

请分析以下{algorithm}算子的源代码，识别{level}层的优化策略：

{source_content}

{format_instructions}

请以JSON数组格式返回所有识别到的优化策略。"""
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 调用LLM
                response = llm.invoke(full_prompt)
                
                # 解析响应
                if hasattr(response, 'content'):
                    content = response.content
                else:
                    content = str(response)
                
                # 尝试解析JSON
                try:
                    # 提取JSON部分
                    if '```json' in content:
                        start = content.find('```json') + 7
                        end = content.find('```', start)
                        json_str = content[start:end].strip()
                    elif '```' in content:
                        start = content.find('```') + 3
                        end = content.rfind('```')
                        json_str = content[start:end].strip()
                    else:
                        json_str = content
                    
                    optimizations = json.loads(json_str)
                    
                    # 验证格式
                    if self._validate_optimization_list(optimizations):
                        # 格式正确，保存结果
                        state["analysis_results"][f"{level}_level_optimizations"] = optimizations
                        state["current_stage"] = f"{level}_completed"
                        print(f"✅ {level}层分析完成，识别到 {len(optimizations)} 个优化策略")
        return state
                    else:
                        raise ValueError("优化策略格式验证失败")
                
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSON解析失败: {e}")
            
        except Exception as e:
                retry_count += 1
                error_msg = f"{level}层分析失败 (尝试 {retry_count}/{max_retries}): {e}"
                print(f"⚠️ {error_msg}")
                
                if retry_count >= max_retries:
                    state["errors"].append(error_msg)
                    state["analysis_results"][f"{level}_level_optimizations"] = []
                    state["current_stage"] = f"{level}_failed"
                    break
                else:
                    print(f"🔄 重试{level}层分析...")
                    time.sleep(1)  # 短暂延迟后重试
        
        return state
    
    def algorithm_analysis_node(self, state: WorkflowState) -> WorkflowState:
        """算法层分析节点"""
        return self.analyzer_work_node(state, "algorithm", self.algorithm_llm, self.algorithm_prompt)
    
    def code_analysis_node(self, state: WorkflowState) -> WorkflowState:
        """代码层分析节点"""
        return self.analyzer_work_node(state, "code", self.code_llm, self.code_prompt)
    
    def instruction_analysis_node(self, state: WorkflowState) -> WorkflowState:
        """指令层分析节点"""
        return self.analyzer_work_node(state, "instruction", self.instruction_llm, self.instruction_prompt)
    
    def finalize_node(self, state: WorkflowState) -> WorkflowState:
        """最终化节点"""
        state["current_stage"] = "completed"
        
        # 统计结果
        total_optimizations = 0
        for level in ["algorithm", "code", "instruction"]:
            opts = state["analysis_results"].get(f"{level}_level_optimizations", [])
            total_optimizations += len(opts)
        
        print(f"🎉 分析完成！总共识别到 {total_optimizations} 个优化策略")
        
        if state["errors"]:
            print(f"⚠️ 分析过程中出现 {len(state['errors'])} 个错误")
        
        return state
    
    def run_analysis(self, algorithm: str, source_files: List[str]) -> Dict[str, Any]:
        """运行完整的分析流程"""
        print(f"🚀 开始分析 {algorithm} 算子")
        print(f"📁 源文件: {source_files}")
        
        # 初始化状态
        initial_state = WorkflowState(
            algorithm=algorithm,
            source_files=source_files,
            analysis_results={},
            current_stage="starting",
            errors=[],
            retry_count=0
        )
        
        # 运行工作流
        final_state = self.workflow.invoke(initial_state)
        
        return final_state["analysis_results"]


def main():
    """主函数"""
    workflow = Workflow24()
    
    # 示例用法
    algorithm = "gemm"
    source_files = [
        "OpenBLAS-develop/kernel/x86_64/gemm_kernel_4x4.c",
        "OpenBLAS-develop/kernel/generic/gemm_beta.c"
    ]
    
    results = workflow.run_analysis(algorithm, source_files)
    
    # 保存结果
    output_file = f"{algorithm}_analysis_v24.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"📄 结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
