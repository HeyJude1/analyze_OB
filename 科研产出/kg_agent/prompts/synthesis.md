你是 BLAS 算子优化策略的「综合专家」。下面是同一算子、同一硬件上 top-k 策略的逐条适用性分析结果。请按 4 个标准综合，形成最终的代码生成指导。

【目标算子】{{OP}}（知识图谱规范名：{{OP_CANONICAL}}）
【目标硬件 profile】{{PROFILE}}
【任务函数签名】{{SIGNATURE}}
【查询代码特征】{{FEATURES_JSON}}

【4 个综合标准（必须逐一给出 0~1 分数与理由）】
{{CRITERIA_DEF}}

【逐条适用性分析（JSON 数组）】
{{ANALYSES_JSON}}

【综合要求】
1. 综合判断这些策略能否组合、在哪些阶段互补、组合后正确性与覆盖度是否足够；
2. `recommended` 只写 JSON 数组里真实出现的策略 id，按建议采用顺序排列；
3. `guidance` 要能直接指导代码生成（主循环怎么循环分块/向量化、尾部怎么处理、参数取什么量级、哪些策略不要用）；
4. 不要写 C 代码，不要编造未给出的性能数字。

【输出要求】
只输出一个严格 JSON 对象（不要 markdown 代码围栏、不要解释文字），格式：
{
  "criteria": {
    "supportive": {"score": 0.0, "reason": "策略之间是否相互支持、条件是否一致"},
    "complementarity": {"score": 0.0, "reason": "策略是否互补（向量化 / 展开 / 尾部 / 分块 / 参数调优）"},
    "accuracy": {"score": 0.0, "reason": "组合后正确性与数值一致性风险是否可控"},
    "completeness": {"score": 0.0, "reason": "是否覆盖主循环、尾部、任意步长、n<=0 等"}
  },
  "overall": "一句话综合结论",
  "recommended": ["策略 id，按优先级"],
  "guidance": "分条给出代码生成阶段的实现要点",
  "conflicts": ["生成时需要避免的冲突"]
}
