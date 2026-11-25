/* ============================================================================
 * sample_graph_data.js — 可视化开发用示例图谱（不依赖 pipeline）
 * ----------------------------------------------------------------------------
 * 契约（SPEC §2）：
 *   window.KG_DATA = {
 *     nodes: [{ id, type, name, properties:{...}, provenance:{...} }, ...],
 *     edges: [{ id, type, head, tail, properties:{...}, provenance:{...}, confidence }, ...],
 *     stats: {...}   // 可选，viz 也会自行统计
 *   }
 * 本文件只用于演示 / 前端开发；pipeline 完成后由 viz_export.py 生成的
 * graph_data.js 使用同名全局变量替换它。数据内容是**示例**，数值仅作展示，
 * 不代表真实测量结论（真实数据以 data/ 下 JSONL 为准）。
 * ==========================================================================*/
window.KG_DATA = {
  "stats": {
    "generated_at": "2026-11-12T18:00:00+08:00",
    "source": "sample/sample_graph_data.js (viz demo, synthetic)",
    "nodes": 32,
    "edges": 43
  },
  "nodes": [
    {
      "id": "op:gemm",
      "type": "Operator",
      "name": "GEMM 矩阵乘法",
      "properties": {
        "blas_level": 3,
        "op_class": "matrix",
        "dtype_variants": ["s", "d", "c", "z"],
        "flops_formula": "2*M*N*K",
        "description": "BLAS Level-3 通用矩阵乘法 C = alpha*A*B + beta*C，是算子优化知识最密集的入口。"
      },
      "provenance": {"source_file": "OpenBLAS/interface/gemm.c", "line_start": null, "line_end": null, "line_confidence": "approx", "commit": "OpenBLAS 0.3.27.dev", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "pat:gemm:parameter_check",
      "type": "ComputationPattern",
      "name": "参数合法性校验（prep）",
      "properties": {
        "stage": "prep",
        "category": "parameter_check",
        "description": "检查维度/步长/alpha/beta 的合法性，不合法时提前返回，避免无效计算。",
        "pattern_type": "prep.parameter_validation"
      },
      "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm.c", "line_start": 30, "line_end": 58, "commit": "OpenBLAS 0.3.27.dev", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "pat:gemm:mm_microkernel",
      "type": "ComputationPattern",
      "name": "矩阵乘法微内核（core）",
      "properties": {
        "stage": "core",
        "category": "mm_microkernel",
        "description": "在打包后的 A/B 块上做寄存器级外积累加，是最内层热点循环，决定算子的算力上限。",
        "pattern_type": "core.mm_microkernel",
        "hot_loop_depth": 3
      },
      "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k.c", "line_start": 19, "line_end": 120, "commit": "OpenBLAS 0.3.27.dev", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "pat:gemm:packing",
      "type": "ComputationPattern",
      "name": "连续化打包（transform）",
      "properties": {
        "stage": "transform",
        "category": "packing",
        "description": "把跨步存储的子块按 kernel 期望布局写入连续缓冲，消除内层跨步访问。",
        "pattern_type": "transform.packing"
      },
      "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_nn.c", "line_start": 40, "line_end": 190, "commit": "OpenBLAS 0.3.27.dev", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "ch:gemm:dtype_double",
      "type": "CodeCharacteristic",
      "name": "dtype = double",
      "properties": {"pred": "dtype", "value": "d", "extraction_method": "static", "confidence": 1.0, "evidence_span": "函数名 dgemm_ 前缀 d + sizeof(double)=8"},
      "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k.c", "line_start": 19, "line_end": 20, "commit": "OpenBLAS 0.3.27.dev", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "ch:gemm:loop_depth_3",
      "type": "CodeCharacteristic",
      "name": "loop_depth = 3",
      "properties": {"pred": "loop_depth", "value": 3, "extraction_method": "static", "confidence": 0.95, "evidence_span": "dgemm_k.c: j/i/k 三层循环"},
      "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k.c", "line_start": 30, "line_end": 60, "commit": "OpenBLAS 0.3.27.dev", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "ch:gemm:stride_x_1",
      "type": "CodeCharacteristic",
      "name": "stride_x = 1（连续）",
      "properties": {"pred": "stride_x", "value": 1, "extraction_method": "static", "confidence": 0.9, "evidence_span": "packed A 布局中 i 方向连续"},
      "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_nn.c", "line_start": 60, "line_end": 120, "commit": "OpenBLAS 0.3.27.dev", "extractor_version": "kgagent-v1.5"}
    },

    {
      "id": "stg:gemm:unroll_pointer",
      "type": "OptimizationStrategy",
      "name": "循环展开与指针预计算",
      "properties": {
        "family": "循环展开",
        "level": "code",
        "action": "unroll",
        "description": "把 k 循环按 4 步展开，并用指针递增代替索引乘法，减少分支与地址计算开销。",
        "rationale": "深度流水线上循环控制与地址计算占比不可忽略；展开 4 步可把每轮 1 次分支摊薄到 4 次计算，并让编译器做寄存器分配与指令重排。",
        "conditions": [
          {"pred": "loop_depth", "op": "ge", "value": 2, "hard": true},
          {"pred": "size_regime", "op": "in", "value": ["medium", "large"], "hard": false}
        ],
        "hardware_profile_ids": ["x86_avx2", "k1_rvv256"],
        "parameter_derivation": {"unroll_factor": "min(4, vector_regs/4)"},
        "template_id": "tpl:gemm:unroll_pointer",
        "status": "verified",
        "confidence": 0.78
      },
      "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k.c", "line_start": 55, "line_end": 96, "commit": "OpenBLAS 0.3.27.dev", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "stg:gemm:vector_regblock",
      "type": "OptimizationStrategy",
      "name": "寄存器分块向量化（8×VLMAX）",
      "properties": {
        "family": "SIMD向量指令",
        "level": "instruction",
        "action": "vectorize",
        "description": "沿 M 方向向量化、N 方向固定 8 列累加；每个 k 步加载 8 个 A 向量，用 8 个标量 B 广播，执行 64 条 vfmacc。",
        "rationale": "VLEN=256、LMUL=2、SEW=64 时 VLMAX = VLEN*LMUL/SEW = 8，一条向量加载正好取满 8 个 double；N 方向 8 个标量复用同一向量，把计算访存比从 1:2 提升到 8:1。",
        "conditions": [
          {"pred": "dtype", "op": "in", "value": ["s", "d"], "hard": true},
          {"pred": "loop_depth", "op": "ge", "value": 2, "hard": true},
          {"pred": "tail_divisible", "op": "eq", "value": 8, "hard": false}
        ],
        "hardware_profile_ids": ["k1_rvv256"],
        "parameter_derivation": {"reg_block_n": "min(vector_regs/2, N_tile) = 8", "reg_block_m": "VLEN*LMUL/SEW = 8", "k_unroll": "8"},
        "template_id": "tpl:gemm:vector_regblock",
        "status": "verified",
        "confidence": 0.82
      },
      "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k_rvv.c", "line_start": 46, "line_end": 213, "commit": "OpenBLAS 0.3.27.dev", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "stg:gemm:vector_full_regblock",
      "type": "OptimizationStrategy",
      "name": "寄存器分块向量化（16×VLMAX，激进版）",
      "properties": {
        "family": "SIMD向量指令",
        "level": "instruction",
        "action": "vectorize",
        "description": "N 方向 16 列累加，尝试进一步摊薄加载开销。",
        "rationale": "理论上每加载 1 个 A 向量可做 16 次 FMA；但 16 个累加向量 + 16 个 A 向量会超过可用向量寄存器数。",
        "conditions": [
          {"pred": "dtype", "op": "in", "value": ["s", "d"], "hard": true},
          {"pred": "loop_depth", "op": "ge", "value": 2, "hard": true}
        ],
        "hardware_profile_ids": ["k1_rvv256"],
        "parameter_derivation": {"reg_block_n": "16", "reg_block_m": "VLMAX = 8"},
        "template_id": "tpl:gemm:vector_regblock",
        "status": "raw",
        "confidence": 0.51
      },
      "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k_rvv.c", "line_start": 90, "line_end": 240, "commit": "OpenBLAS 0.3.27.dev", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "stg:gemm:scalar_regblock",
      "type": "OptimizationStrategy",
      "name": "标量 2×2 寄存器分块",
      "properties": {
        "family": "分块Tiling",
        "level": "code",
        "action": "tile",
        "description": "用 4 个标量累加器做 2×2 分块，k 循环展开 4 步。",
        "rationale": "寄存器中的累加器越多，每次访存参与的 FMA 越多；但标量寄存器数量有限，2×2 是通用内核的保守选择。",
        "conditions": [
          {"pred": "dtype", "op": "eq", "value": "s", "hard": true}
        ],
        "hardware_profile_ids": [],
        "parameter_derivation": {"reg_block_m": "2", "reg_block_n": "2", "k_unroll": "4"},
        "template_id": null,
        "status": "raw",
        "confidence": 0.55
      },
      "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k.c", "line_start": 19, "line_end": 120, "commit": "OpenBLAS 0.3.27.dev", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "stg:gemm:pack",
      "type": "OptimizationStrategy",
      "name": "打包到连续缓冲",
      "properties": {
        "family": "数据重排",
        "level": "algorithm",
        "action": "pack",
        "description": "把 A/B 子块复制到连续 pack 缓冲，使微内核可以按固定步长顺序读取。",
        "rationale": "跨步访问会浪费 cacheline；打包把跨步读换成一次连续写 + 多次连续读，打包成本可被多次复用摊薄。",
        "conditions": [
          {"pred": "stride_x", "op": "ge", "value": 16, "hard": true},
          {"pred": "aligned", "op": "eq", "value": true, "hard": false}
        ],
        "hardware_profile_ids": ["x86_avx2", "k1_rvv256"],
        "parameter_derivation": {"pack_buffer_kb": "MC*KC*8/1024 = 64"},
        "template_id": "tpl:gemm:pack",
        "status": "verified",
        "confidence": 0.74
      },
      "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_nn.c", "line_start": 40, "line_end": 190, "commit": "OpenBLAS 0.3.27.dev", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "stg:gemm:prefetch",
      "type": "OptimizationStrategy",
      "name": "软件预取",
      "properties": {
        "family": "预取",
        "level": "instruction",
        "action": "prefetch",
        "description": "在微内核中提前 2 个 cacheline 发起预取，隐藏 DRAM 延迟。",
        "rationale": "计算密度高时访存延迟可被预取掩盖；但预取距离过大挤占带宽，过小来不及。",
        "conditions": [
          {"pred": "size_regime", "op": "eq", "value": "large", "hard": false}
        ],
        "hardware_profile_ids": ["x86_avx512"],
        "parameter_derivation": {"prefetch_distance": "2*cacheline = 2"},
        "template_id": null,
        "status": "candidate",
        "confidence": 0.5
      },
      "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k.c", "line_start": 100, "line_end": 160, "commit": "OpenBLAS 0.3.27.dev", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "stg:gemm:reg_blocking_abstract",
      "type": "OptimizationStrategy",
      "name": "寄存器分块（抽象）",
      "properties": {
        "family": "分块Tiling",
        "level": "algorithm",
        "action": "tile",
        "description": "用寄存器/向量寄存器容纳输出块，使每个载入值参与尽可能多的 FMA。",
        "rationale": "跨实现通用的抽象原则：分块因子由可用寄存器数、向量宽度与缓存层级共同决定。",
        "conditions": [],
        "hardware_profile_ids": [],
        "parameter_derivation": {},
        "template_id": null,
        "status": "merged",
        "abstraction_level": 3,
        "confidence": 0.7
      },
      "provenance": {"source_file": "OpenBLAS/kernel/generic/gemmkernel_2x2.c", "line_start": 19, "line_end": 120, "line_confidence": "approx", "commit": "OpenBLAS 0.3.27.dev", "extractor_version": "kgagent-v1.5"}
    },

    {
      "id": "tpl:gemm:unroll_pointer",
      "type": "ImplementationTemplate",
      "name": "模板：展开 + 指针循环",
      "properties": {
        "language": "c",
        "slots": [{"name": "UNROLL", "type": "int", "default": 4, "source": "par:gemm:unroll_factor"}],
        "skeleton": "for (p = 0; p + {{UNROLL}} <= kc; p += {{UNROLL}}) {\n  for (u = 0; u < {{UNROLL}}; u++) acc += pa[p+u] * pb[p+u];\n}\n/* tail: for (; p < kc; p++) acc += pa[p] * pb[p]; */",
        "check_hints": ["尾部循环必须保留", "指针步长需与 pack 布局一致"],
        "status": "verified"
      },
      "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k.c", "line_start": 55, "line_end": 96, "commit": "OpenBLAS 0.3.27.dev", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "tpl:gemm:vector_regblock",
      "type": "ImplementationTemplate",
      "name": "模板：RVV 向量分块内核",
      "properties": {
        "language": "c",
        "includes": ["<riscv_vector.h>"],
        "slots": [
          {"name": "REG_BLOCK_N", "type": "int", "default": 8, "source": "par:gemm:reg_block_n"},
          {"name": "SEW", "type": "int", "default": 64},
          {"name": "LMUL", "type": "int", "default": 2}
        ],
        "skeleton": "for (j = bn/{{REG_BLOCK_N}}; j > 0; j--) {\n  /* 绑定 {{REG_BLOCK_N}} 个 C 列指针 */\n  for (i = bm; i > 0; i -= vl) {\n    vl = __riscv_vsetvl_e{{SEW}}m{{LMUL}}(i);\n    /* 每步：k 展开 + {{REG_BLOCK_N}} 个 vfmacc */\n  }\n}",
        "check_hints": ["vsetvl 必须在最内层循环外计算", "写回需用掩码避免越界"],
        "status": "verified"
      },
      "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k_rvv.c", "line_start": 46, "line_end": 213, "commit": "OpenBLAS 0.3.27.dev", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "tpl:gemm:pack",
      "type": "ImplementationTemplate",
      "name": "模板：pack 缓冲",
      "properties": {
        "language": "c",
        "slots": [{"name": "MC", "type": "int", "default": 256}, {"name": "KC", "type": "int", "default": 128}],
        "skeleton": "for (i = 0; i < mc; i++)\n  for (p = 0; p < kc; p++)\n    pack_a[p*mc + i] = a[(p0+p)*lda + i];",
        "check_hints": ["pack 布局必须与 kernel 读取顺序一致", "对齐到 cacheline"],
        "status": "verified"
      },
      "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_nn.c", "line_start": 40, "line_end": 190, "commit": "OpenBLAS 0.3.27.dev", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "par:gemm:unroll_factor",
      "type": "TunableParameter",
      "name": "unroll_factor",
      "properties": {"kind": "unroll", "unit": "iterations", "value_in_source": 4, "range": [2, 4, 8], "derived_from": "min(4, vector_regs/4)", "effect_direction": "大→控制开销↓、寄存器压力↑"},
      "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k.c", "line_start": 55, "line_end": 96, "commit": "OpenBLAS 0.3.27.dev", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "par:gemm:reg_block_n",
      "type": "TunableParameter",
      "name": "reg_block_n",
      "properties": {"kind": "blocking", "unit": "columns", "value_in_source": 8, "range": [4, 8, 16], "derived_from": "min(vector_regs/2, N_tile)", "effect_direction": "大→复用↑、向量寄存器压力↑"},
      "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k_rvv.c", "line_start": 46, "line_end": 90, "commit": "OpenBLAS 0.3.27.dev", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "par:gemm:prefetch_dist",
      "type": "TunableParameter",
      "name": "prefetch_distance",
      "properties": {"kind": "prefetch_dist", "unit": "cacheline", "value_in_source": 2, "range": [1, 2, 4], "derived_from": "2*cacheline", "effect_direction": "大→延迟隐藏↑、带宽挤占↑"},
      "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k.c", "line_start": 100, "line_end": 160, "commit": "OpenBLAS 0.3.27.dev", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "isa:rvv_fma_f64",
      "type": "ISACapability",
      "name": "RVV 1.0 FMA (f64)",
      "properties": {"isa": "RV_V1.0", "feature": "fma", "element_widths": [64], "instr_examples": ["vfmacc.vf", "vle64.v", "vsetvli"], "requires_flags": ["-march=rv64gcv_zvl256b", "-mabi=lp64d"]},
      "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/_command.txt", "line_start": null, "line_end": null, "line_confidence": "approx", "commit": "OpenBLAS 0.3.27.dev", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "isa:avx2_fma_f64",
      "type": "ISACapability",
      "name": "AVX2 FMA (f64)",
      "properties": {"isa": "AVX2", "feature": "fma", "element_widths": [64], "instr_examples": ["vfmadd213pd", "vmovupd"], "requires_flags": ["-mavx2", "-mfma"]},
      "provenance": {"source_file": "OpenBLAS/kernel/x86_64/dgemm_kernel_4x8_haswell.S", "line_start": null, "line_end": null, "line_confidence": "approx", "commit": "OpenBLAS 0.3.27.dev", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "isa:avx512_prefetch",
      "type": "ISACapability",
      "name": "AVX512 预取",
      "properties": {"isa": "AVX512", "feature": "prefetch", "element_widths": [64], "instr_examples": ["prefetcht0", "prefetchwt1"], "requires_flags": ["-mavx512f"]},
      "provenance": {"source_file": "OpenBLAS/kernel/x86_64/dgemm_kernel_16x4_skylakex.S", "line_start": null, "line_end": null, "line_confidence": "approx", "commit": "OpenBLAS 0.3.27.dev", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "uarch:rv64_k1",
      "type": "Microarchitecture",
      "name": "SpacemiT K1 (RVV 1.0)",
      "properties": {"vendor": "SpacemiT", "uarch": "K1", "vlen_bits": 256, "lmul_max": 8, "num_vector_regs": 32, "l1d_kb": 32, "l2_kb": 512, "cacheline_b": 64, "isa_ids": ["isa:rvv_fma_f64"], "note": "示例参数：以芯片手册/实测为准"},
      "provenance": {"source_file": "schema/hardware_profiles.yaml", "line_start": null, "line_end": null, "line_confidence": "approx", "commit": "kgagent-v1.5", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "uarch:xeon_skylake",
      "type": "Microarchitecture",
      "name": "Intel Xeon Skylake (AVX2)",
      "properties": {"vendor": "Intel", "uarch": "Skylake", "vlen_bits": 256, "lmul_max": null, "num_vector_regs": 16, "l1d_kb": 32, "l2_kb": 1024, "cacheline_b": 64, "isa_ids": ["isa:avx2_fma_f64"], "note": "示例参数：以 lscpu/uops.info 为准"},
      "provenance": {"source_file": "schema/hardware_profiles.yaml", "line_start": null, "line_end": null, "line_confidence": "approx", "commit": "kgagent-v1.5", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "hw:k1_rvv256",
      "type": "HardwareProfile",
      "name": "k1_rvv256（RISC-V）",
      "properties": {"profile_name": "k1_rvv256", "uarch_id": "uarch:rv64_k1", "cores": 8, "freq_ghz": 1.6, "ram_gb": 8, "compiler": "riscv64-linux-gcc 14.3.0", "measure_env": "qemu", "isa_ids": ["isa:rvv_fma_f64"]},
      "provenance": {"source_file": "schema/hardware_profiles.yaml", "line_start": null, "line_end": null, "line_confidence": "approx", "commit": "kgagent-v1.5", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "hw:x86_avx2",
      "type": "HardwareProfile",
      "name": "x86_avx2（服务器）",
      "properties": {"profile_name": "x86_avx2", "uarch_id": "uarch:xeon_skylake", "cores": 16, "freq_ghz": 2.4, "ram_gb": 64, "compiler": "gcc 11.4", "measure_env": "native", "isa_ids": ["isa:avx2_fma_f64"]},
      "provenance": {"source_file": "schema/hardware_profiles.yaml", "line_start": null, "line_end": null, "line_confidence": "approx", "commit": "kgagent-v1.5", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "ev:gemm_vector_geomean",
      "type": "PerformanceEvidence",
      "name": "dgemm 向量化加速几何平均",
      "properties": {"metric": "speedup", "value": 2.04, "aggregation": "geo_mean", "workload": {"op": "dgemm", "size_min": 64, "size_max": 200, "n_points": 137}, "baseline": "openblas riscv64 zvl256b rvv 0.3.27.dev", "protocol": "qemu-user-tcg / size 64-200 / 单次运行", "status": "qemu_only", "confidence": 0.35, "note": "示例聚合值，QEMU 计时不代表真实微架构"},
      "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/gemm_log.txt", "line_start": null, "line_end": null, "line_confidence": "approx", "commit": "OpenBLAS 0.3.27.dev", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "ev:gemm_unroll_speedup",
      "type": "PerformanceEvidence",
      "name": "循环展开收益（示例）",
      "properties": {"metric": "speedup", "value": 1.18, "aggregation": "median", "workload": {"op": "dgemm", "size": 1024}, "baseline": "未展开版本", "protocol": "待真实测量", "status": "sample", "confidence": 0.2, "note": "演示用示例值，不可引用"},
      "provenance": {"source_file": "data/riscv_log_summary.csv", "line_start": null, "line_end": null, "line_confidence": "approx", "commit": "kgagent-v1.5", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "ev:gemm_pack_bandwidth",
      "type": "PerformanceEvidence",
      "name": "打包后有效带宽（示例）",
      "properties": {"metric": "throughput", "value": 38.5, "unit": "GB/s", "aggregation": "median", "workload": {"op": "dgemm", "size": 2048}, "baseline": "未打包（跨步）", "protocol": "待真实测量", "status": "sample", "confidence": 0.2, "note": "演示用示例值，不可引用"},
      "provenance": {"source_file": "data/riscv_log_summary.csv", "line_start": null, "line_end": null, "line_confidence": "approx", "commit": "kgagent-v1.5", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "src:openblas_riscv64_kernel",
      "type": "SourceArtifact",
      "name": "OpenBLAS riscv64 gemmkernel_rvv_v1x8",
      "properties": {"repo": "OpenBLAS (rvv-OpenBLAS-develop)", "commit_or_tag": "0.3.27.dev", "path": "kernel/riscv64/gemmkernel_rvv_v1x8.c", "license": "BSD-3-Clause", "url": null},
      "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k_rvv.c", "line_start": 46, "line_end": 213, "commit": "OpenBLAS 0.3.27.dev", "extractor_version": "kgagent-v1.5"}
    },
    {
      "id": "src:openblas_generic_kernel",
      "type": "SourceArtifact",
      "name": "OpenBLAS generic gemmkernel_2x2",
      "properties": {"repo": "OpenBLAS", "commit_or_tag": "0.3.27.dev", "path": "kernel/generic/gemmkernel_2x2.c", "license": "BSD-3-Clause", "url": null},
      "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k.c", "line_start": 19, "line_end": 120, "commit": "OpenBLAS 0.3.27.dev", "extractor_version": "kgagent-v1.5"}
    }
  ],
  "edges": [
    {"id": "e:op~pat_check", "type": "OPERATOR_HAS_PATTERN", "head": "op:gemm", "tail": "pat:gemm:parameter_check", "properties": {"stage": "prep"}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm.c"}, "confidence": 1.0},
    {"id": "e:op~pat_mm", "type": "OPERATOR_HAS_PATTERN", "head": "op:gemm", "tail": "pat:gemm:mm_microkernel", "properties": {"stage": "core"}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k.c"}, "confidence": 1.0},
    {"id": "e:op~pat_pack", "type": "OPERATOR_HAS_PATTERN", "head": "op:gemm", "tail": "pat:gemm:packing", "properties": {"stage": "transform"}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_nn.c"}, "confidence": 1.0},
    {"id": "e:pat_mm~ch_dtype", "type": "PATTERN_HAS_CHARACTERISTIC", "head": "pat:gemm:mm_microkernel", "tail": "ch:gemm:dtype_double", "properties": {"role": "defining"}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k.c"}, "confidence": 1.0},
    {"id": "e:pat_mm~ch_depth", "type": "PATTERN_HAS_CHARACTERISTIC", "head": "pat:gemm:mm_microkernel", "tail": "ch:gemm:loop_depth_3", "properties": {"role": "defining"}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k.c"}, "confidence": 0.95},
    {"id": "e:pat_pack~ch_stride", "type": "PATTERN_HAS_CHARACTERISTIC", "head": "pat:gemm:packing", "tail": "ch:gemm:stride_x_1", "properties": {"role": "optional"}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_nn.c"}, "confidence": 0.9},

    {"id": "e:pat_mm~stg_unroll", "type": "PATTERN_OPTIMIZED_BY", "head": "pat:gemm:mm_microkernel", "tail": "stg:gemm:unroll_pointer", "properties": {"support": 2, "is_primary": true, "stage": "core"}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k.c", "line_start": 55, "line_end": 96}, "confidence": 0.78},
    {"id": "e:pat_mm~stg_vec", "type": "PATTERN_OPTIMIZED_BY", "head": "pat:gemm:mm_microkernel", "tail": "stg:gemm:vector_regblock", "properties": {"support": 3, "is_primary": true, "stage": "core"}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k_rvv.c", "line_start": 46, "line_end": 213}, "confidence": 0.82},
    {"id": "e:pat_mm~stg_vec_full", "type": "PATTERN_OPTIMIZED_BY", "head": "pat:gemm:mm_microkernel", "tail": "stg:gemm:vector_full_regblock", "properties": {"support": 1, "is_primary": false, "stage": "core"}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k_rvv.c"}, "confidence": 0.51},
    {"id": "e:pat_mm~stg_scalar", "type": "PATTERN_OPTIMIZED_BY", "head": "pat:gemm:mm_microkernel", "tail": "stg:gemm:scalar_regblock", "properties": {"support": 2, "is_primary": false, "stage": "core"}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k.c"}, "confidence": 0.55},
    {"id": "e:pat_mm~stg_prefetch", "type": "PATTERN_OPTIMIZED_BY", "head": "pat:gemm:mm_microkernel", "tail": "stg:gemm:prefetch", "properties": {"support": 1, "is_primary": false, "stage": "core"}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k.c"}, "confidence": 0.5},
    {"id": "e:pat_pack~stg_pack", "type": "PATTERN_OPTIMIZED_BY", "head": "pat:gemm:packing", "tail": "stg:gemm:pack", "properties": {"support": 3, "is_primary": true, "stage": "transform"}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_nn.c"}, "confidence": 0.74},

    {"id": "e:stg_unroll~tpl", "type": "STRATEGY_HAS_TEMPLATE", "head": "stg:gemm:unroll_pointer", "tail": "tpl:gemm:unroll_pointer", "properties": {"is_default": true}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k.c"}, "confidence": 0.8},
    {"id": "e:stg_vec~tpl", "type": "STRATEGY_HAS_TEMPLATE", "head": "stg:gemm:vector_regblock", "tail": "tpl:gemm:vector_regblock", "properties": {"is_default": true}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k_rvv.c"}, "confidence": 0.82},
    {"id": "e:stg_pack~tpl", "type": "STRATEGY_HAS_TEMPLATE", "head": "stg:gemm:pack", "tail": "tpl:gemm:pack", "properties": {"is_default": true}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_nn.c"}, "confidence": 0.74},

    {"id": "e:stg_unroll~par", "type": "STRATEGY_HAS_PARAMETER", "head": "stg:gemm:unroll_pointer", "tail": "par:gemm:unroll_factor", "properties": {"role": "tunable", "value_in_instance": 4}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k.c"}, "confidence": 0.78},
    {"id": "e:stg_vec~par", "type": "STRATEGY_HAS_PARAMETER", "head": "stg:gemm:vector_regblock", "tail": "par:gemm:reg_block_n", "properties": {"role": "tunable", "value_in_instance": 8}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k_rvv.c"}, "confidence": 0.82},
    {"id": "e:stg_prefetch~par", "type": "STRATEGY_HAS_PARAMETER", "head": "stg:gemm:prefetch", "tail": "par:gemm:prefetch_dist", "properties": {"role": "tunable", "value_in_instance": 2}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k.c"}, "confidence": 0.5},

    {"id": "e:stg_unroll~ch_depth", "type": "STRATEGY_APPLIES_WHEN", "head": "stg:gemm:unroll_pointer", "tail": "ch:gemm:loop_depth_3", "properties": {"op": "ge", "value": 2, "hard": true, "weight": 1.0}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k.c"}, "confidence": 0.78},
    {"id": "e:stg_vec~ch_dtype", "type": "STRATEGY_APPLIES_WHEN", "head": "stg:gemm:vector_regblock", "tail": "ch:gemm:dtype_double", "properties": {"op": "in", "value": ["s", "d"], "hard": true, "weight": 1.0}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k_rvv.c"}, "confidence": 0.82},
    {"id": "e:stg_pack~ch_stride", "type": "STRATEGY_APPLIES_WHEN", "head": "stg:gemm:pack", "tail": "ch:gemm:stride_x_1", "properties": {"op": "ge", "value": 16, "hard": true, "weight": 1.0}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_nn.c"}, "confidence": 0.74},
    {"id": "e:stg_prefetch~ch_stride", "type": "STRATEGY_APPLIES_WHEN", "head": "stg:gemm:prefetch", "tail": "ch:gemm:stride_x_1", "properties": {"op": "eq", "value": 1, "hard": false, "weight": 0.5}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k.c"}, "confidence": 0.5},

    {"id": "e:stg_vec~isa_rvv", "type": "STRATEGY_REQUIRES_ISA", "head": "stg:gemm:vector_regblock", "tail": "isa:rvv_fma_f64", "properties": {"hard": true, "min_width": 64}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/_command.txt"}, "confidence": 0.9},
    {"id": "e:stg_vec_full~isa_rvv", "type": "STRATEGY_REQUIRES_ISA", "head": "stg:gemm:vector_full_regblock", "tail": "isa:rvv_fma_f64", "properties": {"hard": true, "min_width": 64}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/_command.txt"}, "confidence": 0.6},
    {"id": "e:stg_unroll~isa_avx2", "type": "STRATEGY_REQUIRES_ISA", "head": "stg:gemm:unroll_pointer", "tail": "isa:avx2_fma_f64", "properties": {"hard": true, "min_width": 64}, "provenance": {"source_file": "OpenBLAS/kernel/x86_64/dgemm_kernel_4x8_haswell.S"}, "confidence": 0.7},
    {"id": "e:stg_prefetch~isa_avx512", "type": "STRATEGY_REQUIRES_ISA", "head": "stg:gemm:prefetch", "tail": "isa:avx512_prefetch", "properties": {"hard": true, "min_width": 64}, "provenance": {"source_file": "OpenBLAS/kernel/x86_64/dgemm_kernel_16x4_skylakex.S"}, "confidence": 0.55},

    {"id": "e:stg_vec~uarch_k1", "type": "STRATEGY_TARGETS_UARCH", "head": "stg:gemm:vector_regblock", "tail": "uarch:rv64_k1", "properties": {"fit": "good", "parameterization": {"reg_block_n": 8, "k_unroll": 8, "lmul": 2}}, "provenance": {"source_file": "schema/hardware_profiles.yaml"}, "confidence": 0.75},
    {"id": "e:stg_vec_full~uarch_k1", "type": "STRATEGY_TARGETS_UARCH", "head": "stg:gemm:vector_full_regblock", "tail": "uarch:rv64_k1", "properties": {"fit": "poor", "parameterization": {"reg_block_n": 16}}, "provenance": {"source_file": "schema/hardware_profiles.yaml"}, "confidence": 0.45},
    {"id": "e:stg_unroll~uarch_skx", "type": "STRATEGY_TARGETS_UARCH", "head": "stg:gemm:unroll_pointer", "tail": "uarch:xeon_skylake", "properties": {"fit": "good", "parameterization": {"unroll_factor": 4}}, "provenance": {"source_file": "schema/hardware_profiles.yaml"}, "confidence": 0.7},

    {"id": "e:stg_vec~ev_vec", "type": "STRATEGY_EVIDENCED_BY", "head": "stg:gemm:vector_regblock", "tail": "ev:gemm_vector_geomean", "properties": {"role": "supports", "scope": "operator"}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/gemm_log.txt"}, "confidence": 0.35},
    {"id": "e:stg_unroll~ev_unroll", "type": "STRATEGY_EVIDENCED_BY", "head": "stg:gemm:unroll_pointer", "tail": "ev:gemm_unroll_speedup", "properties": {"role": "supports", "scope": "strategy"}, "provenance": {"source_file": "data/riscv_log_summary.csv"}, "confidence": 0.2},
    {"id": "e:stg_pack~ev_pack", "type": "STRATEGY_EVIDENCED_BY", "head": "stg:gemm:pack", "tail": "ev:gemm_pack_bandwidth", "properties": {"role": "supports", "scope": "operator"}, "provenance": {"source_file": "data/riscv_log_summary.csv"}, "confidence": 0.2},

    {"id": "e:ev_vec~hw_k1", "type": "EVIDENCE_ON", "head": "ev:gemm_vector_geomean", "tail": "hw:k1_rvv256", "properties": {}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/gemm_log.txt"}, "confidence": 0.35},
    {"id": "e:ev_unroll~hw_x86", "type": "EVIDENCE_ON", "head": "ev:gemm_unroll_speedup", "tail": "hw:x86_avx2", "properties": {}, "provenance": {"source_file": "data/riscv_log_summary.csv"}, "confidence": 0.2},
    {"id": "e:ev_pack~hw_k1", "type": "EVIDENCE_ON", "head": "ev:gemm_pack_bandwidth", "tail": "hw:k1_rvv256", "properties": {}, "provenance": {"source_file": "data/riscv_log_summary.csv"}, "confidence": 0.2},

    {"id": "e:ev_vec~src_rvv", "type": "EVIDENCE_SOURCE", "head": "ev:gemm_vector_geomean", "tail": "src:openblas_riscv64_kernel", "properties": {}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/gemm_log.txt"}, "confidence": 0.9},
    {"id": "e:ev_unroll~src_gen", "type": "EVIDENCE_SOURCE", "head": "ev:gemm_unroll_speedup", "tail": "src:openblas_generic_kernel", "properties": {}, "provenance": {"source_file": "data/riscv_log_summary.csv"}, "confidence": 0.5},
    {"id": "e:ev_pack~src_rvv", "type": "EVIDENCE_SOURCE", "head": "ev:gemm_pack_bandwidth", "tail": "src:openblas_riscv64_kernel", "properties": {}, "provenance": {"source_file": "data/riscv_log_summary.csv"}, "confidence": 0.5},

    {"id": "e:stg_vec~stg_pack~compose", "type": "STRATEGY_COMPOSES_WITH", "head": "stg:gemm:vector_regblock", "tail": "stg:gemm:pack", "properties": {"order": -1, "synergy": 0.8, "condition": "pack 后布局必须与 kernel 读取一致"}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k_rvv.c"}, "confidence": 0.8},
    {"id": "e:stg_vec~stg_vecfull~conflict", "type": "STRATEGY_CONFLICTS_WITH", "head": "stg:gemm:vector_regblock", "tail": "stg:gemm:vector_full_regblock", "properties": {"resource": "register", "reason": "两者争用同一批向量寄存器：8 列累加已用满 VLMAX 预算，16 列会溢出", "severity": "high", "condition": "同一 kernel 内"}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k_rvv.c"}, "confidence": 0.7},
    {"id": "e:stg_unroll~stg_prefetch~conflict", "type": "STRATEGY_CONFLICTS_WITH", "head": "stg:gemm:unroll_pointer", "tail": "stg:gemm:prefetch", "properties": {"resource": "bandwidth", "reason": "深度展开已提高寄存器/发射压力，再叠加固定距离预取会挤占访存带宽且抬高寄存器压力", "severity": "medium", "condition": "L1 工作集接近临界"}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k.c"}, "confidence": 0.55},

    {"id": "e:stg_vec~stg_abs", "type": "STRATEGY_GENERALIZES", "head": "stg:gemm:vector_regblock", "tail": "stg:gemm:reg_blocking_abstract", "properties": {"abstraction_gain": 1, "conditions_removed": ["dtype", "isa"]}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k_rvv.c"}, "confidence": 0.7},
    {"id": "e:stg_scalar~stg_abs", "type": "STRATEGY_GENERALIZES", "head": "stg:gemm:scalar_regblock", "tail": "stg:gemm:reg_blocking_abstract", "properties": {"abstraction_gain": 1, "conditions_removed": ["dtype"]}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k.c"}, "confidence": 0.65},
    {"id": "e:stg_vecfull~stg_abs", "type": "STRATEGY_GENERALIZES", "head": "stg:gemm:vector_full_regblock", "tail": "stg:gemm:reg_blocking_abstract", "properties": {"abstraction_gain": 1, "conditions_removed": ["dtype"]}, "provenance": {"source_file": "operator_extract/openblas_newtest1/16_dgemm_op5/dgemm_k_rvv.c"}, "confidence": 0.45}
  ]
};
