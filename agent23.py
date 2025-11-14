#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenBLAS优化分析 - 工作流Agent工厂（agent23：按四阶段拆分的细粒度计算流程版）

说明：
- 基于 agent22.py 演进：将“计算流程识别”拆分为四个阶段（prep/transform/core/post），每个阶段独立提问与返回
- 计算流程对象字段遵循《完整的计算流程prompt模板.md》：pattern_type, name, description, code, data_object_features
- 暴露按阶段识别接口与仅基于已识别流程做优化策略分析的接口，便于工作流逐步保存
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()


# ===== 结构化任务定义（保持与 agent21/22 接口一致） =====
class AnalysisTask:
    """占位任务结构（兼容 agent2/agent21/22 工作流接口，不强制使用）。"""
    def __init__(self, algorithm: str, input_files: List[Dict[str, str]], output_file: str, report_folder: str):
        self.algorithm = algorithm
        self.input_files = input_files
        self.output_file = output_file
        self.report_folder = report_folder


# ===== 基础工具 =====
@tool
def read_source_file(file_path: str) -> str:
    """读取 openblas-output/GENERIC/kernel 下源代码（截断至15000字符）。"""
    try:
        full_path = os.path.join("openblas-output/GENERIC/kernel", file_path)
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(15000)
        return f"文件路径: {file_path}\n内容:\n{content}\n..."
    except Exception as e:
        return f"读取失败: {str(e)}"


@tool
def read_analysis_file(file_path: str) -> str:
    """读取已保存的分析结果文件（UTF-8）。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"读取失败: {str(e)}"


# ===== Agent工厂（四阶段流程版）=====
class AgentFactory:
    """Agent工厂 - 四类分析与两类总结Agent；将计算流程拆分为四阶段识别。"""

    def __init__(self):
        # 完全按照 agent2/agent21/22：仅从 config.json 读取模型配置
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
            model_config = config["model"]

        self.llm = ChatOpenAI(
            model=model_config["name"],
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"],
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    # ===== 计算流程识别（按阶段） =====
    def _create_pattern_parser(self) -> StructuredOutputParser:
        schemas = [
            ResponseSchema(name="computational_patterns", description=(
                "计算流程列表。每项包含: pattern_type(流程类型标签), name(流程中文名称), "
                "description(对流程的简要说明), code(该流程最相关的完整代码片段), "
                "data_object_features(对象，含 numeric_kind, numeric_precision, structural_properties, storage_layout 四键)"
            )),
        ]
        return StructuredOutputParser.from_response_schemas(schemas)

    # -- 四个阶段的专用创建函数（完整Prompt，便于修改） --
    def create_prep_pattern_agent(self) -> AgentExecutor:
        tools = [read_source_file]
        parser = self._create_pattern_parser()
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是"细粒度计算流程"识别专家。仅识别"阶段一：计算准备 (Computation Preparation)"中的流程，如果识别到以下某个计算流程，则需要并严格按模板输出。

【阶段一目录】
1) prep.parameter_validation（参数合法性校验）
   - description: 根据所给的实际代码生成具体描述，检查 n/m/k/inc_x/inc_y/lda 等边界并可提前退出，典型形态 if (...) return。
   - code: 贴出实现校验与早退的完整代码片段。
   - data_object_features:
     -numeric_kind=N/A
     -numeric_precision=N/A
     -structural_properties=N/A
     -storage_layout=N/A

2) prep.index_pointer_init（索引与指针初始化）
   - description: 根据所给的实际代码生成具体描述，说明初始化了哪些具体变量、变量的作用和初始值。识别标准，初始化循环变量/累加器/指针起点（如 i=0, sum=0, ptr=a 等）。
   - code: 贴出初始化相关的完整代码片段。
   - data_object_features: 
     -numeric_kind=实数/复数/不适用则用N/A（判断依据是检查累加器类型是否为`_Complex`或结构体，若仅为索引/指针，则为"N/A"）
     -numeric_precision= 单精度/双精度/不适用则用N/A（判断依据是检查累加器或指针的类型是`float`还是`double`，若仅为整数索引，则为"N/A"）
     -structural_properties=N/A
     -storage_layout=N/A

3) prep.loop_invariant_calc（循环不变量计算） 
   - description: 根据所给的实际代码生成具体描述，说明计算了哪些具体的不变量、计算公式和用途。识别标准，在循环外计算并在循环内复用的不变量（如 inc_x2=2*inc_x, lda2=2*lda 等）。
   - code: 贴出相关赋值片段。
   - data_object_features: 
     -numeric_kind=实数/复数（判断依据是如 2*inc_x 用于复数交错时标记复数）
     -numeric_precision=N/A
     -structural_properties=N/A
     -storage_layout=跨步

【输出要求】
- ⚠️ 重要：只有在代码中明确找到相应计算流程时才输出该流程！如果代码中没有某个计算流程，则完全不输出该流程的JSON对象。
- 严格输出 JSON 数组。数组中的每个元素都是一个对象，且必须包含以下五个字段：
  - pattern_type, name, description, code, data_object_features。
- 特别说明：
  - 对于 prep.parameter_validation：只有当代码中存在 if(...) return 形式的参数检查时才输出
  - 对于 prep.index_pointer_init：只有当代码中存在变量初始化语句时才输出  
  - 对于 prep.loop_invariant_calc：只有当代码中存在循环外的预计算赋值时才输出
- data_object_features 必须是对象，包含四键：numeric_kind, numeric_precision, structural_properties, storage_layout；值从上述说明中选择；不适用则用 "N/A"。
- "判断依据"仅供理解，JSON输出中data_object_features只包含具体的值，如"实数"、"单精度"等，不得包含"判断依据"。
- 不得发明新标签；只做流程识别。
- 如果某个阶段的所有计算流程都不存在，则返回空数组 []。

{format_instructions}
（提示：上面的格式说明描述了“单个流程对象”的字段，请将它作为数组元素的对象结构。）"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        formatted = prompt.partial(format_instructions=parser.get_format_instructions())
        agent = create_openai_tools_agent(self.llm, tools, formatted)
        return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=10)

    def create_transform_pattern_agent(self) -> AgentExecutor:
        tools = [read_source_file]
        parser = self._create_pattern_parser()
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是"细粒度计算流程"识别专家。仅识别"阶段二：数据转换与预处理 (Data Transformation & Pre-processing)"中的流程，并严格按模板输出。

【阶段二目录】
1) transform.packing（连续化拷贝/打包）
   - description: 根据实际代码生成具体描述，说明从哪里拷贝到哪里、使用了什么方法、源和目标的访问模式。识别标准，将跨步源数据复制到连续缓冲区，常见 memcpy 或显式循环，源索引含乘法(…*lda/inc_x)，目标简单递增。
   - code: 贴出拷贝/打包实现的完整代码片段。
   - data_object_features: 
     -numeric_kind=实数/复数
     -numeric_precision=单精度/双精度（判断依据是检查源/目标指针的变量类型）
     -structural_properties=通用
     -storage_layout=(源指针)跨步 -> (目标指针)连续（判断依据是检查源地址计算是否含"*lda"或"*inc_x"，目标地址是否为简单递增）

2) transform.unpacking_special（特殊结构解包/展开）
   - description: 根据实际代码生成具体描述，说明处理了哪种特殊结构、如何进行展开、涉及的分支逻辑。识别标准，将对称/厄米特/三角等特殊存储展开为通用布局，常含 uplo/diag 分支与复杂地址计算。
   - code: 贴出分支与解包逻辑的完整片段。
   - data_object_features: 
     -numeric_kind=实数/复数
     -numeric_precision=单/双精度
     -structural_properties=对称/厄米特/三角
     -storage_layout=打包 -> 连续 / 跨步 -> 连续

3) transform.transpose（数据转置）
   - description: 根据实际代码生成具体描述，说明如何实现转置、涉及的循环结构和索引变换。识别标准，行列互换以改变访存局部性，常见双层循环 B[j][i] = A[i][j]。
   - code: 贴出转置实现的完整代码片段。
   - data_object_features: 
     -numeric_kind=实数/复数
     -numeric_precision=单/双精度
     -structural_properties=通用
     -storage_layout=跨步（判断依据是布局发生变化）

【输出要求】
- ⚠️ 重要：只有在代码中明确找到相应计算流程时才输出该流程！如果代码中没有某个计算流程，则完全不输出该流程的JSON对象。
- 严格输出 JSON 数组。数组中的每个元素都是一个对象，且必须包含以下五个字段：
  - pattern_type, name, description, code, data_object_features。
- data_object_features 键为：numeric_kind, numeric_precision, structural_properties, storage_layout；不适用用 "N/A"。
- "判断依据"仅供理解，JSON输出中data_object_features只包含具体的值，如"实数"、"单精度"等，不得包含"判断依据"。
- 不得发明新标签；只做流程识别。
- 如果某个阶段的所有计算流程都不存在，则返回空数组 []。

{format_instructions}
（提示：上面的格式说明描述了“单个流程对象”的字段，请将它作为数组元素的对象结构。）"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        formatted = prompt.partial(format_instructions=parser.get_format_instructions())
        agent = create_openai_tools_agent(self.llm, tools, formatted)
        return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=10)

    def create_core_pattern_agent(self) -> AgentExecutor:
        tools = [read_source_file]
        parser = self._create_pattern_parser()
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是"细粒度计算流程"识别专家。仅识别"阶段三：核心计算 (Core Computation)"中的流程，并严格按模板输出。

【阶段三目录】
1) core.vector_reduction（向量归约）
   - description: 根据实际代码生成具体描述，说明进行了什么类型的归约操作、具体的计算逻辑和累积方式。识别标准，累加/极值/范数等；常见 sum+= / max/min，可能含 inc_x。
   - code: 贴出核心循环完整片段。
   - data_object_features: 
     -numeric_kind=实数/复数
     -numeric_precision=单/双精度
     -structural_properties=通用
     -storage_layout=连续/跨步

2) core.elementwise_update（元素级向量更新）
   - description: 根据实际代码生成具体描述，描述对向量进行逐元素计算和更新的操作，可能为复数乘加。
   - code: 贴出更新循环的完整片段。
   - data_object_features: 
     -numeric_kind=实数/复数
     -numeric_precision=单/双精度
     -structural_properties=通用
     -storage_layout=跨步

3) core.gemv_like（矩阵-向量乘）
   - description: 根据实际代码生成具体描述，说明矩阵和向量的乘法实现方式、循环结构和计算逻辑。识别标准，双层循环，内层点积 sum+=A[i,j]*x[j]。
   - code: 贴出 GEMV-like 实现的完整片段。
   - data_object_features: 
     -numeric_kind=实数/复数
     -numeric_precision=单/双精度
     -structural_properties=通用/带状/三角
     -storage_layout=跨步

4) core.rank1_update（秩-1 更新）
   - description: 根据实际代码生成具体描述，说明如何实现秩-1更新、涉及的向量外积计算和矩阵更新方式。识别标准，A+=alpha*x*y^T 外积更新。
   - code: 贴出双层循环完整片段。
   - data_object_features: 
     -numeric_kind=实数/复数
     -numeric_precision=单/双精度
     -structural_properties=通用/对称/厄米特
     -storage_layout=跨步

5) core.mm_microkernel（矩阵乘法微内核）
   - description: 根据实际代码生成具体描述，说明微内核的实现方式、寄存器使用和FMA指令序列。识别标准，固定尺寸寄存器级FMA展开；完全展开、寄存器累加器、规律性load/FMA。
   - code: 贴出微内核的完整片段（包含累加器与FMA序列）。
   - data_object_features: 
     -numeric_kind=实数/复数
     -numeric_precision=单/双精度
     -structural_properties=通用
     -storage_layout=连续

6) core.tiled_loop（分块矩阵处理循环）
   - description: 根据实际代码生成具体描述，说明分块循环的实现方式、块大小和循环嵌套结构。识别标准，外层遍历块、驱动微内核的三层 ijk 循环。
   - code: 贴出块循环框架完整片段。
   - data_object_features: 
     -numeric_kind=实数/复数
     -numeric_precision=单/双精度
     -structural_properties=通用
     -storage_layout=跨步（判断依据是操作原始大矩阵的子块）

7) core.triangular_solve（三角求解/回代）
   - description: 根据实际代码生成具体描述，说明三角求解的实现方式、求解顺序和回代更新逻辑。识别标准，小型三角系统求解与回代更新；常见递增/递减循环、除法与AXPY样更新。
   - code: 贴出相关循环完整片段。
   - data_object_features: 
     -numeric_kind=实数/复数
     -numeric_precision=单/双精度
     -structural_properties=三角
     -storage_layout=跨步

【输出要求】
- ⚠️ 重要：只有在代码中明确找到相应计算流程时才输出该流程！如果代码中没有某个计算流程，则完全不输出该流程的JSON对象。
- 严格输出 JSON 数组。数组中的每个元素都是一个对象，且必须包含以下五个字段：
  - pattern_type, name, description, code, data_object_features。
- data_object_features 必须是对象，包含四键：numeric_kind, numeric_precision, structural_properties, storage_layout；不适用用 "N/A"。
- "判断依据"仅供理解，JSON输出中data_object_features只包含具体的值，如"实数"、"单精度"等，不得包含"判断依据"。
- 不得发明新标签；只做流程识别。
- 如果某个阶段的所有计算流程都不存在，则返回空数组 []。

{format_instructions}
（提示：上面的格式说明描述了“单个流程对象”的字段，请将它作为数组元素的对象结构。）"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        formatted = prompt.partial(format_instructions=parser.get_format_instructions())
        agent = create_openai_tools_agent(self.llm, tools, formatted)
        return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=10)

    def create_post_pattern_agent(self) -> AgentExecutor:
        tools = [read_source_file]
        parser = self._create_pattern_parser()
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是"细粒度计算流程"识别专家。仅识别"阶段四：后处理与写回 (Post-processing & Write-back)"中的流程，并严格按模板输出。

【阶段四目录】
1) post.scale_accumulation（结果缩放与累加）
   - description: 根据实际代码生成具体描述，说明如何进行结果缩放和累加、涉及的系数处理和写回方式。识别标准，C=alpha*Temp + beta*C 写回；包含 alpha/beta 分支与目标内存写回。
   - code: 贴出写回逻辑的完整片段。
   - data_object_features: 
     -numeric_kind=实数/复数
     -numeric_precision=单精度/双精度
     -structural_properties=通用
     -storage_layout=跨步

【输出要求】
- ⚠️ 重要：只有在代码中明确找到相应计算流程时才输出该流程！如果代码中没有某个计算流程，则完全不输出该流程的JSON对象。
- 严格输出 JSON 数组。数组中的每个元素都是一个对象，且必须包含以下五个字段：
  - pattern_type, name, description, code, data_object_features。
- data_object_features 必须是对象，包含四键：numeric_kind, numeric_precision, structural_properties, storage_layout；不适用用 "N/A"。
- "判断依据"仅供理解，JSON输出中data_object_features只包含具体的值，如"实数"、"单精度"等，不得包含"判断依据"。
- 不得发明新标签；只做流程识别。
- 如果某个阶段的所有计算流程都不存在，则返回空数组 []。

{format_instructions}
（提示：上面的格式说明描述了“单个流程对象”的字段，请将它作为数组元素的对象结构。）"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        formatted = prompt.partial(format_instructions=parser.get_format_instructions())
        agent = create_openai_tools_agent(self.llm, tools, formatted)
        return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=10)

    # ===== 优化策略识别（与 agent22 相同） =====
    def create_algorithm_optimizer(self) -> AgentExecutor:
        tools = [read_source_file]
        algo_schemas = [
            ResponseSchema(name="algorithm_level_optimizations", description=(
                "算法层优化策略列表。每项包含: optimization_name, level='algorithm', description(含四子项), "
                "applicability_conditions, tunable_parameters[], target_hardware_feature_name, "
                "target_hardware_feature, code_example(snippet, explanation), related_patterns[]"
            )),
        ]
        algo_parser = StructuredOutputParser.from_response_schemas(algo_schemas)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是算法层优化策略识别专家。你的任务是识别代码中的算法设计层次优化策略。

🎯 **你的职责：**
1. 读取源文件和计算流程分析结果
2. **只识别算法设计层次的优化策略**
3. 生成JSON格式的算法层优化策略列表

📋 **算法设计层次优化策略识别：**
识别该文件中的算法层优化策略，每个策略包含以下完整结构：

**1. optimization_name**: 规范化策略名称，中文短语命名

**2. level**: 固定值 "algorithm"（算法层次）

**3. description**: 包含4个子字段的详细分析对象
  - strategy_rationale: 解释"为什么"要这么做的理论原理（基于计算机体系结构或算法理论）
  - implementation_pattern: 解释"怎么做"的代码实现模式（该优化在代码层面的典型表现）
  - performance_impact: 解释"有什么用"的性能提升（减少CPU周期、提高缓存命中率等）
  - trade_offs: 解释该优化的局限性或代价（如增加代码复杂度、额外内存开销等）

**4. applicability_conditions**: 适用条件（何时用？）
  - 描述必须满足什么样的代码条件，这个优化策略才是有效或安全的
  - 示例："输入数组必须是实部和虚部交错存储的浮点数组，且操作是标准的复数乘加运算"

**5. tunable_parameters**: 可调参数列表（怎么调？），每个参数包含：
  - parameter_name: 参数名称
  - description: 参数描述
  - value_in_code: OpenBLAS在此代码中选择的值
  - typical_range: 典型取值范围（数组）
  - impact: 不同取值的影响
  - 注意：如果该优化策略没有可调参数，设为空数组[]

**6. target_hardware_feature_name**: 目标硬件特性简短名称（为何做？）
  - 简短的硬件特性名称，用于实体标识，中文短名称
  - 示例："Cache"、"SIMD"、"寄存器文件"、"分支预测器"

**7. target_hardware_feature**: 目标硬件特性详细描述（为何做？）
  - 详细描述该优化利用了哪种底层硬件能力
  - 示例："CPU Cache Line架构和数据局部性原理"

**8. code_example**: 包含2个子字段的代码示例对象
  - snippet: 包含必要上下文的完整代码块（不是单行，要能自解释优化意图）
  - explanation: 自然语言解释代码块与优化策略的关联

**9. related_patterns**: 关联的计算流程类型列表（⭐新增字段）
  - 列出该优化策略通常应用于哪些计算流程类型
  - 从所给的计算流程中选择相关类型
  - 可以包含多个计算流程类型，因为一个优化策略可能同时优化多种计算流程

🔍 **分析要求：**
- 只关注算法设计层次的优化（如：分块、预计算、数据重用等）
- 完全基于代码内容发现优化策略
- 不要预设任何优化技术类型

{format_instructions}"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        formatted = prompt.partial(format_instructions=algo_parser.get_format_instructions())
        agent = create_openai_tools_agent(self.llm, tools, formatted)
        return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=15)

    def create_code_optimizer(self) -> AgentExecutor:
        tools = [read_source_file]
        code_schemas = [
            ResponseSchema(name="code_level_optimizations", description=(
                "代码层优化策略列表。每项包含: optimization_name, level='code', description(含四子项), "
                "applicability_conditions, tunable_parameters[], target_hardware_feature_name, "
                "target_hardware_feature, code_example(snippet, explanation), related_patterns[]"
            )),
        ]
        code_parser = StructuredOutputParser.from_response_schemas(code_schemas)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是代码层优化策略识别专家。你的任务是识别代码中的代码优化层次优化策略。

🎯 **你的职责：**
1. 读取源文件和计算流程分析结果
2. **只识别代码优化层次的优化策略**
3. 生成JSON格式的代码层优化策略列表

📋 **代码优化层次优化策略识别：**
识别该文件中的代码层优化策略，每个策略包含以下完整结构：

**1. optimization_name**: 规范化策略名称，中文短语命名

**2. level**: 固定值 "code"（代码层次）

**3. description**: 包含4个子字段的详细分析对象
  - strategy_rationale: 解释"为什么"要这么做的理论原理（基于计算机体系结构或算法理论）
  - implementation_pattern: 解释"怎么做"的代码实现模式（该优化在代码层面的典型表现）
  - performance_impact: 解释"有什么用"的性能提升（减少CPU周期、提高缓存命中率等）
  - trade_offs: 解释该优化的局限性或代价（如增加代码复杂度、额外内存开销等）

**4. applicability_conditions**: 适用条件（何时用？）
  - 示例："循环的迭代次数在进入循环前是已知的，且循环体内没有复杂的控制流（如break、continue）"

**5. tunable_parameters**: 可调参数列表（怎么调？），每个参数包含：
  - parameter_name: 参数名称
  - description: 参数描述
  - value_in_code: OpenBLAS在此代码中选择的值
  - typical_range: 典型取值范围（数组）
  - impact: 不同取值的影响
  - 注意：如果该优化策略没有可调参数，设为空数组[]

**6. target_hardware_feature_name**: 目标硬件特性简短名称（为何做？）
  - 简短的硬件特性名称，用于实体标识，中文短名称
  - 示例："指令流水线"、"寄存器"、"分支预测器"

**7. target_hardware_feature**: 目标硬件特性详细描述（为何做？）
  - 详细描述该优化利用了哪种底层硬件能力
  - 示例："CPU指令流水线和寄存器数量"

**8. code_example**: 包含2个子字段的代码示例对象
  - snippet: 包含必要上下文的完整代码块（不是单行，要能自解释优化意图）
  - explanation: 自然语言解释代码块与优化策略的关联

**9. related_patterns**: 关联的计算流程类型列表（⭐新增字段）
  - 列出该优化策略通常应用于哪些计算流程类型
  - 从所给的计算流程中选择相关类型
  - 可以包含多个计算流程类型，因为一个优化策略可能同时优化多种计算流程

🔍 **分析要求：**
- 只关注代码优化层次的优化（如：循环展开、指针优化、分支优化等）
- 完全基于代码内容发现优化策略
- 不要预设任何优化技术类型

{format_instructions}"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        formatted = prompt.partial(format_instructions=code_parser.get_format_instructions())
        agent = create_openai_tools_agent(self.llm, tools, formatted)
        return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=15)

    def create_instruction_optimizer(self) -> AgentExecutor:
        tools = [read_source_file]
        inst_schemas = [
            ResponseSchema(name="instruction_level_optimizations", description=(
                "指令层优化策略列表。每项包含: optimization_name, level='instruction', description(含四子项), "
                "applicability_conditions, tunable_parameters[], target_hardware_feature_name, target_hardware_feature, "
                "code_example(snippet, explanation), related_patterns[]"
            )),
            ResponseSchema(name="implementation_details", description="关键实现细节"),
            ResponseSchema(name="performance_insights", description="性能洞察"),
        ]
        inst_parser = StructuredOutputParser.from_response_schemas(inst_schemas)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是指令层优化策略识别专家。你的任务是识别代码中的指令级优化策略，并提供实现细节和性能洞察。

🎯 **你的职责：**
1. 读取源文件和计算流程分析结果
2. **识别特有指令层次的优化策略**
3. **总结关键实现细节**
4. **提供性能分析洞察**
5. 生成JSON格式的指令层优化策略列表及分析

📋 **特有指令层次优化策略识别：**
识别该文件中的指令层优化策略，每个策略包含以下完整结构：

**1. optimization_name**: 规范化策略名称，中文短语命名

**2. level**: 固定值 "instruction"（指令层次）

**3. description**: 包含4个子字段的详细分析对象
  - strategy_rationale: 解释"为什么"要这么做的理论原理（基于计算机体系结构或算法理论）
  - implementation_pattern: 解释"怎么做"的代码实现模式（该优化在代码层面的典型表现）
  - performance_impact: 解释"有什么用"的性能提升（减少CPU周期、提高缓存命中率等）
  - trade_offs: 解释该优化的局限性或代价（如增加代码复杂度、额外内存开销等）

**4. applicability_conditions**: 适用条件（何时用？）
  - 示例："数据类型支持SIMD指令，数组在内存中连续存储，无数据依赖冲突"

**5. tunable_parameters**: 可调参数列表（怎么调？），每个参数包含：
  - parameter_name: 参数名称
  - description: 参数描述
  - value_in_code: OpenBLAS在此代码中选择的值
  - typical_range: 典型取值范围（数组）
  - impact: 不同取值的影响
  - 注意：如果该优化策略没有可调参数，设为空数组[]

**6. target_hardware_feature_name**: 目标硬件特性简短名称（为何做？）
  - 简短的硬件特性名称，用于实体标识，中文短名称
  - 示例："SIMD"、"AVX2"、"NEON"、"SSE"

**7. target_hardware_feature**: 目标硬件特性详细描述（为何做？）
  - 详细描述该优化利用了哪种底层硬件能力
  - 示例："SIMD (Single Instruction, Multiple Data) execution units, such as SSE/AVX on x86 platforms"

**8. code_example**: 包含2个子字段的代码示例对象
  - snippet: 包含必要上下文的完整代码块（不是单行，要能自解释优化意图）
  - explanation: 自然语言解释代码块与优化策略的关联

**9. related_patterns**: 关联的计算流程类型列表（⭐新增字段）
  - 列出该优化策略通常应用于哪些计算流程类型
  - 从所给的计算流程中选择相关类型
  - 可以包含多个计算流程类型，因为一个优化策略可能同时优化多种计算流程

📋 **实现细节分析：**
- 总结该文件的关键实现细节
- 包括数据处理方式、控制流设计、特殊技巧等

📋 **性能洞察分析：**
- 分析该文件的性能特征
- 包括预期性能提升、性能瓶颈、优化效果等

🔍 **分析要求：**
- 只关注指令级优化（如：SIMD、向量化、内联汇编等）
- 完全基于代码内容发现优化策略
- 不要预设任何优化技术类型

{format_instructions}"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        formatted = prompt.partial(format_instructions=inst_parser.get_format_instructions())
        agent = create_openai_tools_agent(self.llm, tools, formatted)
        return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=15)

    # ===== 通用工具方法 =====
    def _extract_json_from_output(self, output: str) -> Optional[Dict]:
        if not output:
            return None
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            pass
        if "```json" in output:
            s = output.find("```json") + 7
            e = output.find("```", s)
            if e > s:
                try:
                    return json.loads(output[s:e].strip())
                except json.JSONDecodeError:
                    return None
        if "```" in output:
            s = output.find("```") + 3
            e = output.find("```", s)
            if e > s:
                try:
                    return json.loads(output[s:e].strip())
                except json.JSONDecodeError:
                    return None
        return None

    def _invoke_with_retry(self, agent: AgentExecutor, payload: Dict[str, Any], label: str,
                           retries: int = 3) -> Dict[str, Any]:
        attempt = 0
        delay_seq = [3, 6, 12]
        while True:
            try:
                return agent.invoke(payload)
            except Exception as e:
                if attempt >= retries:
                    raise e
                wait = delay_seq[attempt] if attempt < len(delay_seq) else delay_seq[-1]
                print(f"  - {label} 失败，第 {attempt+1} 次重试前等待 {wait}s：{e}")
                time.sleep(wait)
                attempt += 1

    # ===== 对外：按阶段识别计算流程 =====
    def analyze_patterns_stage(self, source_code: str, algorithm: str, stage: str) -> List[Dict[str, Any]]:
        if stage == "prep":
            stage_agent = self.create_prep_pattern_agent()
        elif stage == "transform":
            stage_agent = self.create_transform_pattern_agent()
        elif stage == "core":
            stage_agent = self.create_core_pattern_agent()
        elif stage == "post":
            stage_agent = self.create_post_pattern_agent()
        else:
            raise ValueError(f"未知的阶段: {stage}")
        stage_input = (
            f"请分析以下源码，识别‘{stage}’阶段的细粒度计算流程。算子名称: {algorithm}\n\n源码:\n{source_code}"
        )
        try:
            result = self._invoke_with_retry(stage_agent, {"input": stage_input}, f"计算流程({stage})")
            output_raw = self._extract_json_from_output(result.get("output", "")) or {}
            if isinstance(output_raw, list):
                output_raw = {"computational_patterns": output_raw}
            return output_raw.get("computational_patterns", []) if isinstance(output_raw, dict) else []
        except Exception as e:
            print(f"  - 计算流程({stage}) 失败: {e}")
            return []

    # ===== 摘要：适配 data_object_features =====
    @staticmethod
    def format_patterns_summary(patterns: List[Dict]) -> str:
        lines = []
        for p in patterns or []:
            if not isinstance(p, dict):
                lines.append(f"- {str(p)}")
                continue
            pt = p.get('pattern_type', '')
            name = p.get('name', '')
            desc = (p.get('description') or '').strip()
            dof = p.get('data_object_features') or {}
            nk = dof.get('numeric_kind')
            npv = dof.get('numeric_precision')
            sp = dof.get('structural_properties')
            sl = dof.get('storage_layout')
            parts = []
            if nk: parts.append(f"数值类型: {nk}")
            if npv: parts.append(f"数值精度: {npv}")
            if sp: parts.append(f"结构属性: {sp}")
            if sl: parts.append(f"存储布局: {sl}")
            dof_text = ("；".join(parts)) if parts else ""
            code = p.get('code', '')
            snippet = code if len(code) <= 240 else code[:240] + '…'
            lines.append(
                f"- pattern_type是{pt}，中文命名为{name}；描述：{desc}。数据对象特征：{dof_text}。相关代码：\n{snippet}"
            )
        return "\n".join(lines)

    # ===== 仅基于“已识别流程”做三层优化分析 =====
    def analyze_optimizations_only(self, source_code: str, algorithm: str, architecture: str,
                                   computational_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        def format_strategies_summary(strategies: List[Dict], level_name: str) -> str:
            lines = []
            for s in strategies or []:
                if not isinstance(s, dict):
                    lines.append(f"- {str(s)}")
                    continue
                name = s.get('optimization_name') or s.get('name') or '（未命名策略）'
                desc = s.get('description', {})
                rationale = desc.get('strategy_rationale', '') if isinstance(desc, dict) else str(desc)
                rationale_short = (rationale or '').strip().replace('\n', ' ')
                if len(rationale_short) > 160:
                    rationale_short = rationale_short[:160] + '…'
                
                # 添加代码片段信息作为"唯一码"
                code_example = s.get('code_example', {})
                snippet = code_example.get('snippet', '') if isinstance(code_example, dict) else ''
                snippet_short = snippet.strip().replace('\n', ' ') if snippet else ''
                if len(snippet_short) > 100:
                    snippet_short = snippet_short[:100] + '…'
                
                if snippet_short:
                    lines.append(f"- {name}: {rationale_short}\n  相关的代码片段: {snippet_short}")
                else:
                    lines.append(f"- {name}: {rationale_short}")
            header = f"所给算子代码的{level_name}层次的已识别优化策略摘要：" if lines else f"所给算子代码的{level_name}层次暂无已识别策略。"
            return header + ("\n" + "\n".join(lines) if lines else "")

        # 算法层
        algo_agent = self.create_algorithm_optimizer()
        algo_input = (
            f"请分析以下源码。\n\n算子: {algorithm}\n架构: {architecture}\n\n"
            f"源码:\n{source_code}\n\n"
            f"计算流程（摘要）:\n{self.format_patterns_summary(computational_patterns)}\n"
            f"请从算法设计层次进行分析：分析是否有更适合计算机计算逻辑或者以空间换时间、时间换空间的优化设计。不必局限示例。"
        )
        try:
            algo_result = self._invoke_with_retry(algo_agent, {"input": algo_input}, "分析 算法层")
            algo_raw = self._extract_json_from_output(algo_result.get("output", "")) or {}
            if isinstance(algo_raw, list):
                algo_raw = {"algorithm_level_optimizations": algo_raw}
            algo_output = algo_raw.get("algorithm_level_optimizations", []) if isinstance(algo_raw, dict) else []
        except Exception as e:
            print(f"  - 分析 算法层 失败: {e}")
            algo_output = []

        # 代码层
        code_agent = self.create_code_optimizer()
        code_input = (
            f"请分析以下源码。\n\n"
            f"源码:\n{source_code}\n\n"
            f"计算流程（摘要）:\n{self.format_patterns_summary(computational_patterns)}\n"
            f"\n算法层优化策略（摘要）:\n{format_strategies_summary(algo_output, '算法')}\n\n"
            f"请从代码优化层次进行分析：分析是否有做性能加速或者循环优化、代码顺序调整的优化设计，比如循环展开、指针优化、分支优化、内存对齐等。\n\n"
            f"⚠️ 重要提醒：避免与算法层重叠！专注于识别不同代码片段体现的代码层优化。不必局限示例。"
        )
        try:
            code_result = self._invoke_with_retry(code_agent, {"input": code_input}, "分析 代码层")
            code_raw = self._extract_json_from_output(code_result.get("output", "")) or {}
            if isinstance(code_raw, list):
                code_raw = {"code_level_optimizations": code_raw}
            code_output = code_raw.get("code_level_optimizations", []) if isinstance(code_raw, dict) else []
        except Exception as e:
            print(f"  - 分析 代码层 失败: {e}")
            code_output = []

        # 指令层
        inst_agent = self.create_instruction_optimizer()
        inst_input = (
            f"请分析以下源码。\n\n"
            f"源码:\n{source_code}\n\n"
            f"计算流程（摘要）:\n{self.format_patterns_summary(computational_patterns)}\n"
            f"\n算法层优化策略（摘要）:\n{format_strategies_summary(algo_output, '算法')}\n"
            f"\n代码层优化策略（摘要）:\n{format_strategies_summary(code_output, '代码')}\n\n"
            f"请从特有指令层次进行分析：SIMD向量化、内联汇编等；避免与其他层重叠，同时提供实现细节和性能洞察。\n\n"
            f"⚠️ 重要提醒：避免与算法层、代码层重叠！专注于识别不同代码片段体现的指令层优化。不必局限示例。"
        )
        try:
            inst_result = self._invoke_with_retry(inst_agent, {"input": inst_input}, "分析 指令层")
            inst_raw = self._extract_json_from_output(inst_result.get("output", "")) or {}
            if isinstance(inst_raw, list):
                inst_raw = {
                    "instruction_level_optimizations": inst_raw,
                    "implementation_details": "",
                    "performance_insights": "",
                }
            inst_output = {
                "instruction_level_optimizations": inst_raw.get("instruction_level_optimizations", []),
                "implementation_details": inst_raw.get("implementation_details", ""),
                "performance_insights": inst_raw.get("performance_insights", ""),
            }
        except Exception as e:
            print(f"  - 分析 指令层 失败: {e}")
            inst_output = {"instruction_level_optimizations": [], "implementation_details": "", "performance_insights": ""}

        return {
            "algorithm_level_optimizations": algo_output,
            "code_level_optimizations": code_output,
            "instruction_level_optimizations": inst_output.get("instruction_level_optimizations", []),
            "implementation_details": inst_output.get("implementation_details", ""),
            "performance_insights": inst_output.get("performance_insights", ""),
        }

    # ===== 兼容：整体 analyze_file（内部按阶段识别后合并） =====
    def analyze_file(self, source_code: str, file_path: str, algorithm: str, architecture: str = "通用") -> Dict:
        stages = ["prep", "transform", "core", "post"]
        all_patterns: List[Dict[str, Any]] = []
        for st in stages:
            pts = self.analyze_patterns_stage(source_code, algorithm, st)
            all_patterns.extend(pts)

        opt = self.analyze_optimizations_only(source_code, algorithm, architecture, all_patterns)

        return {
            "algorithm": algorithm,
            "file_path": file_path,
            "architecture": architecture,
            "computational_patterns": all_patterns,
            "algorithm_level_optimizations": opt.get("algorithm_level_optimizations", []),
            "code_level_optimizations": opt.get("code_level_optimizations", []),
            "instruction_level_optimizations": opt.get("instruction_level_optimizations", []),
            "implementation_details": opt.get("implementation_details", ""),
            "performance_insights": opt.get("performance_insights", ""),
        }

    # ===== Summarizers（与 agent22 保持一致） =====
    def create_individual_summarizer(self) -> AgentExecutor:
        tools = [read_analysis_file]
        individual_schemas = [
            ResponseSchema(name="algorithm", description="算子名称"),
            ResponseSchema(name="algorithm_level_optimizations", description="整合后的算法层策略列表"),
            ResponseSchema(name="code_level_optimizations", description="整合后的代码层策略列表"),
            ResponseSchema(name="instruction_level_optimizations", description="整合后的指令层策略列表"),
        ]
        individual_parser = StructuredOutputParser.from_response_schemas(individual_schemas)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是单算子增量整合专家。将新的分析结果整合为统一的策略列表。
{format_instructions}"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        formatted = prompt.partial(format_instructions=individual_parser.get_format_instructions())
        agent = create_openai_tools_agent(self.llm, tools, formatted)
        return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=15)

    def create_final_summarizer(self) -> AgentExecutor:
        tools = [read_analysis_file]
        final_schemas = [
            ResponseSchema(name="algorithm_level_optimizations", description="通用算法层策略库"),
            ResponseSchema(name="code_level_optimizations", description="通用代码层策略库"),
            ResponseSchema(name="instruction_level_optimizations", description="通用指令层策略库"),
        ]
        final_parser = StructuredOutputParser.from_response_schemas(final_schemas)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是跨算子总结专家。整合多算子策略为通用优化策略库。
{format_instructions}"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        formatted = prompt.partial(format_instructions=final_parser.get_format_instructions())
        agent = create_openai_tools_agent(self.llm, tools, formatted)
        return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=20)


# ===== 文件管理器（与 agent22 一致） =====
class FileManager:
    @staticmethod
    def ensure_directories(report_folder: str):
        Path(report_folder).mkdir(parents=True, exist_ok=True)
        Path(f"{report_folder}/discovery_results").mkdir(exist_ok=True)
        Path(f"{report_folder}/analysis_results").mkdir(exist_ok=True)
        Path(f"{report_folder}/strategy_reports").mkdir(exist_ok=True)

    @staticmethod
    def get_discovery_output_path(report_folder: str, algorithm: str) -> str:
        return f"{report_folder}/discovery_results/{algorithm}_discovery.json"

    @staticmethod
    def get_analysis_output_path(report_folder: str, algorithm: str) -> str:
        return f"{report_folder}/analysis_results/{algorithm}_analysis.json"

    @staticmethod
    def get_individual_summary_path(report_folder: str, algorithm: str) -> str:
        return f"{report_folder}/strategy_reports/{algorithm}_summary.json"

    @staticmethod
    def get_final_summary_path(report_folder: str) -> str:
        return f"{report_folder}/strategy_reports/final_optimization_summary.json"

    @staticmethod
    def save_content(file_path: str, content: str) -> bool:
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"保存文件失败 {file_path}: {e}")
            return False


__all__ = [
    'AgentFactory',
    'FileManager',
    'AnalysisTask',
]


