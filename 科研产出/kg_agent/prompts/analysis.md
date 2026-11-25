你是 BLAS 算子优化策略的「适用性分析器」。目标：判断下面这一条从知识图谱检索到的优化策略，在目标硬件上实现目标算子内核时是否适用、如何落地、有何风险。

【目标算子】{{OP}}（知识图谱规范名：{{OP_CANONICAL}}）
【目标硬件 profile】{{PROFILE}}
【查询代码特征】{{FEATURES_JSON}}
【任务函数签名（生成代码必须逐字一致）】{{SIGNATURE}}

【基线 generic 实现（仅供参考语义，不要求照抄）】
{{BASELINE}}

【待分析策略（StrategyPack 中的一项，JSON）】
{{STRATEGY_JSON}}

【分析要求】
1. 只做适用性分析，不要写 C 代码；
2. `applies` 表示该策略在 {{PROFILE}} 上实现 {{OP}} 时是否适用；`confidence` 是 0~1 置信度；
3. `evidence` 必须引用策略 JSON 中真实存在的字段（conditions / parameters / template / evidence / why_selected），不要编造；
4. `risks` 关注正确性（数值一致性、尾部/边界、任意步长、别名）与性能风险（寄存器压力、编译器/ISA 支持）；
5. 不得引入策略数据中没有的性能数字。

【输出要求】
只输出一个严格 JSON 对象（不要 markdown 代码围栏、不要解释文字、不要多余字段之外的散文），格式：
{
  "applies": true,
  "confidence": 0.8,
  "evidence": ["引用策略 conditions / parameters / evidence 的具体依据"],
  "risks": ["实现或正确性风险"],
  "parameters": {"参数名": "建议值或取值依据"},
  "conflicts": ["与 pack 中其它策略可能冲突的点；若无则空数组"]
}
