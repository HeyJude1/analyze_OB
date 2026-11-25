/* ============================================================================
 * sample_traces.js — 检索回放用示例 trace（不依赖 pipeline）
 * ----------------------------------------------------------------------------
 * 契约（SPEC §2.3）：window.KG_TRACES = { <trace_id>: {trace_id, query, steps[], result, created_at}, ... }
 * 硬要求：每个 step 必须有 name / description / candidates / selected；
 *         被淘汰的节点必须出现在 dropped[] 里并给出 reason。
 * 本文件两条 trace 均为**示例**，数值与结论仅用于演示回放交互。
 * ==========================================================================*/
window.KG_TRACES = {

  /* ---------------------------------------------------------------------
   * Trace 1：dgemm @ k1_rvv256（RISC-V 向量平台）→ 选 2 条策略
   * ------------------------------------------------------------------- */
  "trace:20261112T180000:dgemm:k1_rvv256": {
    "trace_id": "trace:20261112T180000:dgemm:k1_rvv256",
    "created_at": "2026-11-12T18:00:00+08:00",
    "query": {
      "operator": "dgemm",
      "operator_id": "op:gemm",
      "hardware_profile_id": "k1_rvv256",
      "code_features": {"dtype": "d", "loop_depth": 3, "stride_x": 1, "size_regime": "large", "aligned": true},
      "top_k": 2
    },
    "steps": [
      {
        "step": 1, "name": "pattern_match", "description": "算子计算流程匹配",
        "input": {"operator": "dgemm", "registry": "operator_pattern_index"},
        "candidates": ["pat:gemm:parameter_check", "pat:gemm:mm_microkernel", "pat:gemm:packing"],
        "selected": ["pat:gemm:mm_microkernel", "pat:gemm:packing"],
        "dropped": [{"id": "pat:gemm:parameter_check", "reason": "prep.parameter_check 层面暂无策略可召回，不计入检索上下文"}],
        "reason": "算子注册表命中 3 个计算流程，保留 core / transform 两类作为召回锚点"
      },
      {
        "step": 2, "name": "strategy_recall", "description": "沿 PATTERN_OPTIMIZED_BY 召回策略",
        "input": {"anchors": ["pat:gemm:mm_microkernel", "pat:gemm:packing"]},
        "candidates": [
          "stg:gemm:unroll_pointer", "stg:gemm:vector_regblock", "stg:gemm:vector_full_regblock",
          "stg:gemm:scalar_regblock", "stg:gemm:pack", "stg:gemm:prefetch"
        ],
        "selected": [
          "stg:gemm:unroll_pointer", "stg:gemm:vector_regblock", "stg:gemm:vector_full_regblock",
          "stg:gemm:scalar_regblock", "stg:gemm:pack", "stg:gemm:prefetch"
        ],
        "dropped": [],
        "reason": "沿 2 个计算流程召回 6 条策略（support 未加权，全部进入过滤阶段）"
      },
      {
        "step": 3, "name": "condition_filter", "description": "适用条件硬过滤",
        "input": {"code_features": {"dtype": "d", "loop_depth": 3, "stride_x": 1, "size_regime": "large", "aligned": true}},
        "candidates": [
          "stg:gemm:unroll_pointer", "stg:gemm:vector_regblock", "stg:gemm:vector_full_regblock",
          "stg:gemm:scalar_regblock", "stg:gemm:pack", "stg:gemm:prefetch"
        ],
        "selected": [
          "stg:gemm:unroll_pointer", "stg:gemm:vector_regblock", "stg:gemm:vector_full_regblock",
          "stg:gemm:pack", "stg:gemm:prefetch"
        ],
        "dropped": [{"id": "stg:gemm:scalar_regblock", "reason": "硬条件 dtype == 's' 不满足（代码 dtype='d'）"}],
        "reason": "谓词求值：5/6 通过硬条件（dtype 与 loop_depth 全部可判定）"
      },
      {
        "step": 4, "name": "hardware_filter", "description": "ISA / 微架构能力过滤",
        "input": {"hardware_profile_id": "k1_rvv256", "isa": ["rvv_fma_f64"], "vector_regs": 32, "vlen_bits": 256},
        "candidates": [
          "stg:gemm:unroll_pointer", "stg:gemm:vector_regblock", "stg:gemm:vector_full_regblock",
          "stg:gemm:pack", "stg:gemm:prefetch"
        ],
        "selected": ["stg:gemm:vector_regblock", "stg:gemm:vector_full_regblock", "stg:gemm:pack"],
        "dropped": [
          {"id": "stg:gemm:unroll_pointer", "reason": "STRATEGY_REQUIRES_ISA → isa:avx2_fma_f64，profile k1_rvv256 不支持 AVX2"},
          {"id": "stg:gemm:prefetch", "reason": "STRATEGY_REQUIRES_ISA → isa:avx512_prefetch，profile k1_rvv256 不支持 AVX512"}
        ],
        "reason": "ISA 能力过滤：3/5 与目标平台兼容"
      },
      {
        "step": 5, "name": "conflict_resolve", "description": "策略冲突消解 + 寄存器预算",
        "input": {"budget": {"vector_regs": 32, "max_use_ratio": 0.8}},
        "candidates": ["stg:gemm:vector_regblock", "stg:gemm:vector_full_regblock", "stg:gemm:pack"],
        "selected": ["stg:gemm:vector_regblock", "stg:gemm:pack"],
        "dropped": [{"id": "stg:gemm:vector_full_regblock", "reason": "与 stg:gemm:vector_regblock 存在 STRATEGY_CONFLICTS_WITH(resource=register, severity=high)：16 列累加 + 16 个 A 向量超出 32 个向量寄存器预算"}],
        "reason": "贪心保留高分项，冲突低分项淘汰；vector 与 pack 为 STRATEGY_COMPOSES_WITH，可共存"
      },
      {
        "step": 6, "name": "score_and_select", "description": "打分排序 + top_k 截断",
        "input": {"top_k": 2, "weights": {"pattern": 0.35, "condition": 0.20, "hardware": 0.15, "evidence": 0.20, "support": 0.10}},
        "candidates": ["stg:gemm:vector_regblock", "stg:gemm:pack"],
        "selected": ["stg:gemm:vector_regblock", "stg:gemm:pack"],
        "dropped": [],
        "scores": {"stg:gemm:vector_regblock": 0.81, "stg:gemm:pack": 0.66},
        "score_breakdown": {
          "stg:gemm:vector_regblock": {"pattern": 0.40, "condition": 0.18, "hardware": 0.15, "evidence": 0.08},
          "stg:gemm:pack": {"pattern": 0.30, "condition": 0.14, "hardware": 0.15, "evidence": 0.07}
        },
        "reason": "top_k=2：两条策略均保留；证据项因 QEMU-only 被降权（confidence=0.35）"
      },
      {
        "step": 7, "name": "pack", "description": "生成 StrategyPack（注入生成 prompt）",
        "input": {"pack_schema": "pack-v1"},
        "candidates": ["stg:gemm:vector_regblock", "stg:gemm:pack"],
        "selected": ["stg:gemm:vector_regblock", "stg:gemm:pack"],
        "dropped": [],
        "result": {
          "strategies": ["stg:gemm:vector_regblock", "stg:gemm:pack"],
          "templates": ["tpl:gemm:vector_regblock", "tpl:gemm:pack"],
          "order": ["stg:gemm:pack", "stg:gemm:vector_regblock"],
          "prompt_preview": "目标平台 k1_rvv256（VLEN=256, LMUL≤8）。先按 pack 模板把 A/B 子块写入连续缓冲，再套用 8×VLMAX 向量分块内核：vsetvl 提到最内层循环外，N 方向 8 列累加，写回使用掩码……"
        },
        "reason": "按 stage（transform→core）与 COMPOSES_WITH.order 排序后组装 StrategyPack"
      }
    ],
    "result": {
      "strategies": ["stg:gemm:vector_regblock", "stg:gemm:pack"],
      "pack": {
        "schema_version": "pack-v1",
        "strategies": [
          {"id": "stg:gemm:vector_regblock", "name": "寄存器分块向量化（8×VLMAX）", "score": 0.81, "template_id": "tpl:gemm:vector_regblock", "params": {"reg_block_n": 8, "k_unroll": 8, "lmul": 2, "sew": 64}},
          {"id": "stg:gemm:pack", "name": "打包到连续缓冲", "score": 0.66, "template_id": "tpl:gemm:pack", "params": {"MC": 256, "KC": 128}}
        ]
      },
      "prompt_preview": "目标平台 k1_rvv256（VLEN=256, LMUL≤8）。先按 pack 模板把 A/B 子块写入连续缓冲，再套用 8×VLMAX 向量分块内核……"
    }
  },

  /* ---------------------------------------------------------------------
   * Trace 2：dgemm @ x86_avx2（x86 标量/FMA 平台）→ 选 2 条策略
   * 用于演示"同一算子 + 不同硬件 profile"时回放路径的差异
   * ------------------------------------------------------------------- */
  "trace:20261112T181000:dgemm:x86_avx2": {
    "trace_id": "trace:20261112T181000:dgemm:x86_avx2",
    "created_at": "2026-11-12T18:10:00+08:00",
    "query": {
      "operator": "dgemm",
      "operator_id": "op:gemm",
      "hardware_profile_id": "x86_avx2",
      "code_features": {"dtype": "d", "loop_depth": 3, "stride_x": 1, "size_regime": "large", "aligned": true},
      "top_k": 2
    },
    "steps": [
      {
        "step": 1, "name": "pattern_match", "description": "算子计算流程匹配",
        "input": {"operator": "dgemm", "registry": "operator_pattern_index"},
        "candidates": ["pat:gemm:parameter_check", "pat:gemm:mm_microkernel", "pat:gemm:packing"],
        "selected": ["pat:gemm:mm_microkernel", "pat:gemm:packing"],
        "dropped": [{"id": "pat:gemm:parameter_check", "reason": "prep.parameter_check 层面暂无策略可召回，不计入检索上下文"}],
        "reason": "算子注册表命中 3 个计算流程，保留 core / transform 两类作为召回锚点"
      },
      {
        "step": 2, "name": "strategy_recall", "description": "沿 PATTERN_OPTIMIZED_BY 召回策略",
        "input": {"anchors": ["pat:gemm:mm_microkernel", "pat:gemm:packing"]},
        "candidates": [
          "stg:gemm:unroll_pointer", "stg:gemm:vector_regblock", "stg:gemm:vector_full_regblock",
          "stg:gemm:scalar_regblock", "stg:gemm:pack", "stg:gemm:prefetch"
        ],
        "selected": [
          "stg:gemm:unroll_pointer", "stg:gemm:vector_regblock", "stg:gemm:vector_full_regblock",
          "stg:gemm:scalar_regblock", "stg:gemm:pack", "stg:gemm:prefetch"
        ],
        "dropped": [],
        "reason": "沿 2 个计算流程召回 6 条策略"
      },
      {
        "step": 3, "name": "condition_filter", "description": "适用条件硬过滤",
        "input": {"code_features": {"dtype": "d", "loop_depth": 3, "stride_x": 1, "size_regime": "large", "aligned": true}},
        "candidates": [
          "stg:gemm:unroll_pointer", "stg:gemm:vector_regblock", "stg:gemm:vector_full_regblock",
          "stg:gemm:scalar_regblock", "stg:gemm:pack", "stg:gemm:prefetch"
        ],
        "selected": [
          "stg:gemm:unroll_pointer", "stg:gemm:vector_regblock", "stg:gemm:vector_full_regblock",
          "stg:gemm:pack", "stg:gemm:prefetch"
        ],
        "dropped": [{"id": "stg:gemm:scalar_regblock", "reason": "硬条件 dtype == 's' 不满足（代码 dtype='d'）"}],
        "reason": "谓词求值：5/6 通过硬条件"
      },
      {
        "step": 4, "name": "hardware_filter", "description": "ISA / 微架构能力过滤",
        "input": {"hardware_profile_id": "x86_avx2", "isa": ["avx2_fma_f64"], "vector_regs": 16, "vlen_bits": 256},
        "candidates": [
          "stg:gemm:unroll_pointer", "stg:gemm:vector_regblock", "stg:gemm:vector_full_regblock",
          "stg:gemm:pack", "stg:gemm:prefetch"
        ],
        "selected": ["stg:gemm:unroll_pointer", "stg:gemm:pack", "stg:gemm:prefetch"],
        "dropped": [
          {"id": "stg:gemm:vector_regblock", "reason": "STRATEGY_REQUIRES_ISA → isa:rvv_fma_f64，profile x86_avx2 不支持 RVV"},
          {"id": "stg:gemm:vector_full_regblock", "reason": "STRATEGY_REQUIRES_ISA → isa:rvv_fma_f64，profile x86_avx2 不支持 RVV"}
        ],
        "reason": "ISA 能力过滤：3/5 与目标平台兼容（跨架构知识被正确屏蔽）"
      },
      {
        "step": 5, "name": "conflict_resolve", "description": "策略冲突消解 + 寄存器预算",
        "input": {"budget": {"vector_regs": 16, "max_use_ratio": 0.8}},
        "candidates": ["stg:gemm:unroll_pointer", "stg:gemm:pack", "stg:gemm:prefetch"],
        "selected": ["stg:gemm:unroll_pointer", "stg:gemm:pack"],
        "dropped": [{"id": "stg:gemm:prefetch", "reason": "与 stg:gemm:unroll_pointer 存在 STRATEGY_CONFLICTS_WITH(resource=bandwidth, severity=medium)：展开已提高发射/寄存器压力，再叠加预取会挤占带宽"}],
        "reason": "贪心保留高分项；unroll 与 pack 不冲突，且 pack 在前"
      },
      {
        "step": 6, "name": "score_and_select", "description": "打分排序 + top_k 截断",
        "input": {"top_k": 2, "weights": {"pattern": 0.35, "condition": 0.20, "hardware": 0.15, "evidence": 0.20, "support": 0.10}},
        "candidates": ["stg:gemm:unroll_pointer", "stg:gemm:pack"],
        "selected": ["stg:gemm:unroll_pointer", "stg:gemm:pack"],
        "dropped": [],
        "scores": {"stg:gemm:unroll_pointer": 0.78, "stg:gemm:pack": 0.66},
        "score_breakdown": {
          "stg:gemm:unroll_pointer": {"pattern": 0.36, "condition": 0.18, "hardware": 0.15, "evidence": 0.09},
          "stg:gemm:pack": {"pattern": 0.30, "condition": 0.14, "hardware": 0.15, "evidence": 0.07}
        },
        "reason": "top_k=2：两条策略均保留"
      },
      {
        "step": 7, "name": "pack", "description": "生成 StrategyPack（注入生成 prompt）",
        "input": {"pack_schema": "pack-v1"},
        "candidates": ["stg:gemm:unroll_pointer", "stg:gemm:pack"],
        "selected": ["stg:gemm:unroll_pointer", "stg:gemm:pack"],
        "dropped": [],
        "result": {
          "strategies": ["stg:gemm:unroll_pointer", "stg:gemm:pack"],
          "templates": ["tpl:gemm:unroll_pointer", "tpl:gemm:pack"],
          "order": ["stg:gemm:pack", "stg:gemm:unroll_pointer"],
          "prompt_preview": "目标平台 x86_avx2（16 个 YMM, AVX2/FMA）。先按 pack 模板打包 A/B 子块，再对 k 循环展开 4 步并用指针递增替代索引乘法，保留尾部循环……"
        },
        "reason": "按 stage（transform→core）与 COMPOSES_WITH.order 排序后组装 StrategyPack"
      }
    ],
    "result": {
      "strategies": ["stg:gemm:unroll_pointer", "stg:gemm:pack"],
      "pack": {
        "schema_version": "pack-v1",
        "strategies": [
          {"id": "stg:gemm:unroll_pointer", "name": "循环展开与指针预计算", "score": 0.78, "template_id": "tpl:gemm:unroll_pointer", "params": {"unroll_factor": 4}},
          {"id": "stg:gemm:pack", "name": "打包到连续缓冲", "score": 0.66, "template_id": "tpl:gemm:pack", "params": {"MC": 256, "KC": 128}}
        ]
      },
      "prompt_preview": "目标平台 x86_avx2（16 个 YMM, AVX2/FMA）。先按 pack 模板打包 A/B 子块，再对 k 循环展开 4 步……"
    }
  }
};
