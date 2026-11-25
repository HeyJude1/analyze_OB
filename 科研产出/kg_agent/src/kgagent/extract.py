"""S3 SemanticAgent：语义抽取（离线模式=规则；--llm 模式可接 LLM 命名与解释）。

三个来源（全部给出可核查事实，数字/行号/硬件参数都不由 LLM 编造）：
  A. 4 个算子的 LLM 三层分析 JSON（algorithm/code/instruction 策略）——只做词表映射与条件/模板补全；
  B. generic C kernel 的 S1 静态特征——规则产出基础策略（不需要 LLM）；
  C. generic vs RVV 的 S2 差分——每个 action 一条策略实例（真正的"为什么这样改"）。

输出 `data/strategies_raw.jsonl` + 计算流程（ComputationPattern）与算子元信息。
LLM 只允许改写 name/description/rationale 文本（且禁止数字），结构字段一律来自规则。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import corpus

__all__ = ["run", "load_analysis", "derive_patterns", "canonical_template", "family_action", "PATTERN_META"]

# ---------------------------------------------------------------- 词表映射

#: 策略家族/动作规则：optimization_name 关键词 -> (family, action)。顺序敏感，先匹配先用。
FAMILY_ACTION_RULES = [
    (r"向量化|SIMD|向量融合|向量指令|向量化归约", "SIMD向量指令", "vectorize"),
    (r"展开", "循环展开", "unroll"),
    (r"分块|blocking|寄存器块|微内核|block", "分块Tiling", "tile"),
    (r"打包|连续化|重排|拷贝|copy|pack|解包", "数据重排", "pack"),
    (r"预取|prefetch", "预取", "prefetch"),
    (r"尾部|余数|掩码|边界|tail", "尾部处理", "tail_handle"),
    (r"归约|累加|求和|reduction|splat", "循环拆分", "split"),
    (r"融合|fuse", "循环融合", "fuse"),
    (r"指针|索引|地址|寻址|指针算术", "指针/索引优化", "reorder"),
    (r"并行|线程|多核", "并行化计算", "split"),
    (r"精度|数值|复数|3M|恒等式|分解|缩放", "精度/数值变换", "specialize"),
    (r"循环|顺序", "循环重排", "reorder"),
]

_LEVEL_STAGE = {"algorithm": "transform", "code": "core", "instruction": "core"}

#: action -> 策略家族（闭集）
ACTION_TO_FAMILY = {
    "vectorize": "SIMD向量指令",
    "tile": "分块Tiling",
    "unroll": "循环展开",
    "reorder": "指针/索引优化",
    "pack": "数据重排",
    "prefetch": "预取",
    "tail_handle": "尾部处理",
    "fuse": "循环融合",
    "split": "循环拆分",
    "specialize": "指针/索引优化",
    "reduce_register_pressure": "循环拆分",
}

_PARAM_NAME_MAP = [
    (r"展开|unroll", "unroll_factor"),
    (r"blocking_factor_n|block_columns|列块", "block_size_n"),
    (r"blocking_factor_m|block_size_m|行块", "block_size_m"),
    (r"block|blocking|分块", "block_size"),
    (r"alignment|对齐", "alignment_boundary"),
    (r"vector_register|向量寄存器|寄存器宽度", "vector_register_width"),
    (r"accumulator|累加器", "accumulator_count"),
    (r"prefetch|预取", "prefetch_distance"),
    (r"inc_x|inc_y|步长", "inc_stride"),
]

#: 计算流程元信息：pattern_type -> (中文名, stage, category)
PATTERN_META = {
    "prep.parameter_validation": ("参数合法性校验", "prep", "parameter_check"),
    "prep.index_pointer_init": ("索引与指针初始化", "prep", "index_init"),
    "prep.loop_invariant_calc": ("循环不变量计算", "prep", "loop_invariant"),
    "transform.packing": ("连续化拷贝/打包", "transform", "packing"),
    "transform.unpacking_special": ("特殊解包", "transform", "packing"),
    "core.mm_microkernel": ("矩阵乘法微内核", "core", "mm_microkernel"),
    "core.tiled_loop": ("分块循环嵌套", "core", "mm_microkernel"),
    "core.gemv_like": ("类 GEMV 计算", "core", "mm_microkernel"),
    "core.elementwise_update": ("逐元素更新", "core", "elementwise_update"),
    "core.rank1_update": ("Rank-1 更新", "core", "elementwise_update"),
    "core.reduction": ("跨迭代归约", "core", "reduction"),
    "core.vector_reduction": ("向量归约", "core", "reduction"),
    "core.stride_access": ("跨步/单位步长访问", "core", "elementwise_update"),
    "post.scale_accumulation": ("缩放累加写回", "post", "scale_accumulation"),
    "post.tail_case": ("尾部/余数处理", "post", "tail_case"),
    "post.branch_elimination": ("分支消除", "post", "branch_elimination"),
}

#: 抽象策略（跨算子迁移层，abstraction_level=3）
ABSTRACT_STRATEGIES = [
    ("stg:abstract:vector_fma", "向量化（跨算子抽象）", "SIMD向量指令", "vectorize", "instruction",
     "沿独立或可合法向量归约的维度分组计算。仅当目标计算确有乘加且数值契约允许时采用 FMA。"),
    ("stg:abstract:register_blocking", "寄存器分块（以寄存器复用换访存）", "分块Tiling", "tile", "algorithm",
     "把输出块的重用距离控制在寄存器内，使每个装入值参与尽可能多的 FMA。"),
    ("stg:abstract:k_unroll", "循环展开（k 方向多步合并）", "循环展开", "unroll", "code",
     "一次迭代处理多个 k，减少循环控制开销并提高指令级并行。"),
    ("stg:abstract:tail_handling", "尾部/余数处理", "尾部处理", "tail_handle", "code",
     "用动态 vl/掩码或余数分支处理 n 不整除向量宽度/分块大小的尾部，避免越界与标量收尾开销。"),
    ("stg:abstract:pointer_precompute", "指针预计算", "指针/索引优化", "reorder", "code",
     "用递增指针替代 i*lda 形式的索引乘法，消去地址计算开销并利于编译器生成向量访存。"),
    ("stg:abstract:stride_dispatch", "步长分支派发", "指针/索引优化", "specialize", "code",
     "按 inc_x/inc_y 是否为 1 分流：单位步长走连续访存快速路径，跨步走 strided 访存。"),
]


# ---------------------------------------------------------------- 小工具


def dtype_from_name(name: str) -> str:
    """BLAS 命名前缀 -> dtype（x 前缀按复数单精度处理，并在 reason 里注明）。"""
    base = Path(name).name
    if not base:
        return "d"
    ch = base[0].lower()
    if ch in "sdcz":
        return ch
    if ch == "x":  # OpenBLAS 内部 complex 变体前缀
        return "c"
    return "d"


def family_action(name: str, level: str) -> tuple:
    for pattern, family, action in FAMILY_ACTION_RULES:
        if re.search(pattern, name):
            return family, action
    fallback = {
        "algorithm": ("分块Tiling", "tile"),
        "code": ("指针/索引优化", "reorder"),
        "instruction": ("SIMD向量指令", "specialize"),
    }
    return fallback.get(level, ("分块Tiling", "tile"))


def param_slug(name: str) -> str:
    for pattern, slug in _PARAM_NAME_MAP:
        if re.search(pattern, name, re.IGNORECASE):
            return slug
    en = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return en if en and en[0].isalpha() else "param_extra"


def _dedupe_conditions(conditions: list) -> list:
    out, seen = [], set()
    for cond in conditions:
        key = (cond["pred"], cond["op"], json.dumps(cond.get("value"), sort_keys=True), bool(cond.get("hard")))
        if key in seen:
            continue
        seen.add(key)
        out.append(cond)
    return out


def _pattern_prefix(pattern_types) -> str | None:
    for pt in pattern_types or []:
        stage = str(pt).split(".")[0]
        if stage in ("prep", "transform", "core", "post"):
            return stage
    return None


# ---------------------------------------------------------------- 模板库

_TEMPLATES = {
    ("vectorize", "rvv"): {
        "includes": ["<riscv_vector.h>"],
        "skeleton": (
            "for (BLASLONG i = 0; i < n; i += vl) {\n"
            "    size_t vl = __riscv_vsetvl_e{{SEW}}m{{LMUL}}(n - i);\n"
            "    vfloat{{SEW}}m{{LMUL}}_t vx = __riscv_vle{{SEW}}_v_f{{SEW}}m{{LMUL}}(x + i, vl);\n"
            "    vfloat{{SEW}}m{{LMUL}}_t vy = __riscv_vle{{SEW}}_v_f{{SEW}}m{{LMUL}}(y + i, vl);\n"
            "    vy = __riscv_vfmacc_vv_f{{SEW}}m{{LMUL}}(vy, vx, {{ALPHA}}, vl);\n"
            "    __riscv_vse{{SEW}}_v_f{{SEW}}m{{LMUL}}(y + i, vy, vl);\n"
            "}"
        ),
        "slots": [
            {"name": "SEW", "type": "int", "default": 64, "source": "par:sew"},
            {"name": "LMUL", "type": "int", "default": 8, "source": "par:lmul"},
            {"name": "ALPHA", "type": "expr", "default": "alpha", "source": "operator.alpha"},
        ],
        "hints": ["vsetvl 必须在循环内按剩余长度重新计算", "vl 与 pack 布局一致", "指针别名会阻止向量化"],
        "preconditions": ["RVV 1.0 向量扩展可用", "数据按元素连续访问"],
    },
    ("vectorize", "portable"): {
        "includes": ["<stdint.h>"],
        "skeleton": (
            "BLASLONG i = 0;\n"
            "for (; i + {{UNROLL_FACTOR}} <= n; i += {{UNROLL_FACTOR}}) {\n"
            "    for (int lane = 0; lane < {{UNROLL_FACTOR}}; ++lane) {\n"
            "        y[i + lane] = y[i + lane] + {{ALPHA}} * x[i + lane];\n"
            "    }\n"
            "}\n"
            "for (; i < n; ++i) { y[i] = y[i] + {{ALPHA}} * x[i]; }"
        ),
        "slots": [
            {"name": "UNROLL_FACTOR", "type": "int", "default": 4, "source": "par:unroll_factor"},
            {"name": "ALPHA", "type": "expr", "default": "alpha", "source": "operator.alpha"},
        ],
        "hints": ["用 restrict 或局部指针消除别名假设", "尾部循环避免越界"],
        "preconditions": ["数据连续（stride=1）"],
    },
    ("vectorize", "avx2"): {
        "includes": ["<immintrin.h>"],
        "skeleton": (
            "for (BLASLONG i = 0; i + {{VEC_LANES}} <= n; i += {{VEC_LANES}}) {\n"
            "    __m256d vx = _mm256_loadu_pd(x + i);\n"
            "    __m256d vy = _mm256_loadu_pd(y + i);\n"
            "    vy = _mm256_fmadd_pd(_mm256_set1_pd({{ALPHA}}), vx, vy);\n"
            "    _mm256_storeu_pd(y + i, vy);\n"
            "}\n"
            "for (; i < n; ++i) { y[i] += {{ALPHA}} * x[i]; }"
        ),
        "slots": [
            {"name": "VEC_LANES", "type": "int", "default": 4, "source": "par:vector_width"},
            {"name": "ALPHA", "type": "expr", "default": "alpha", "source": "operator.alpha"},
        ],
        "hints": ["AVX2 只有 16 个 ymm 寄存器，注意累加器预算", "未对齐地址用 loadu/storeu"],
        "preconditions": ["x86 AVX2 + FMA 可用", "数据连续"],
    },
    ("vectorize", "avx512"): {
        "includes": ["<immintrin.h>"],
        "skeleton": (
            "for (BLASLONG i = 0; i + {{VEC_LANES}} <= n; i += {{VEC_LANES}}) {\n"
            "    __m512d vx = _mm512_loadu_pd(x + i);\n"
            "    __m512d vy = _mm512_loadu_pd(y + i);\n"
            "    vy = _mm512_fmadd_pd(_mm512_set1_pd({{ALPHA}}), vx, vy);\n"
            "    _mm512_storeu_pd(y + i, vy);\n"
            "}\n"
            "for (; i < n; ++i) { y[i] += {{ALPHA}} * x[i]; }"
        ),
        "slots": [
            {"name": "VEC_LANES", "type": "int", "default": 8, "source": "par:vector_width"},
            {"name": "ALPHA", "type": "expr", "default": "alpha", "source": "operator.alpha"},
        ],
        "hints": ["AVX512 掩码可处理尾部，但要注意频率降频", "32 个 zmm 寄存器放宽了累加器预算"],
        "preconditions": ["x86 AVX512F 可用", "数据连续"],
    },
    ("vectorize", "neon"): {
        "includes": ["<arm_neon.h>"],
        "skeleton": (
            "for (BLASLONG i = 0; i + {{VEC_LANES}} <= n; i += {{VEC_LANES}}) {\n"
            "    float64x2_t vx = vld1q_f64(x + i);\n"
            "    float64x2_t vy = vld1q_f64(y + i);\n"
            "    vy = vfmaq_n_f64(vy, vx, {{ALPHA}});\n"
            "    vst1q_f64(y + i, vy);\n"
            "}\n"
            "for (; i < n; ++i) { y[i] += {{ALPHA}} * x[i]; }"
        ),
        "slots": [
            {"name": "VEC_LANES", "type": "int", "default": 2, "source": "par:vector_width"},
            {"name": "ALPHA", "type": "expr", "default": "alpha", "source": "operator.alpha"},
        ],
        "hints": ["NEON 固定 128-bit（2 个 double），靠多累加器展开提高 ILP", "尾部用标量循环"],
        "preconditions": ["ARMv8-A ASIMD 可用", "数据连续"],
    },
    ("vectorize", "sve"): {
        "includes": ["<arm_sve.h>"],
        "skeleton": (
            "svbool_t pg = svwhilelt_b64(0, n);\n"
            "for (BLASLONG i = 0; i < n; i += (BLASLONG)svcntd()) {\n"
            "    svfloat64_t vx = svld1_f64(pg, x + i);\n"
            "    svfloat64_t vy = svld1_f64(pg, y + i);\n"
            "    vy = svmla_n_f64_x(pg, vy, vx, {{ALPHA}});\n"
            "    svst1_f64(pg, y + i, vy);\n"
            "    pg = svwhilelt_b64((uint64_t)(i + {{VEC_LANES}}), (uint64_t)n);\n"
            "}"
        ),
        "slots": [
            {"name": "VEC_LANES", "type": "int", "default": 4, "source": "par:vector_width"},
            {"name": "ALPHA", "type": "expr", "default": "alpha", "source": "operator.alpha"},
        ],
        "hints": ["SVE 向量长度由硬件决定，循环步长用 svcntd()", "谓词寄存器承担尾部，无需标量收尾"],
        "preconditions": ["ARM SVE 可用", "数据连续"],
    },
    ("tile", "rvv"): {
        "includes": ["<riscv_vector.h>"],
        "skeleton": (
            "for (j = bn / {{REG_BLOCK_N}}; j > 0; j--) {\n"
            "    vfloat{{SEW}}m{{LMUL}}_t vres[{{REG_BLOCK_N}}];   /* {{REG_BLOCK_N}} 个向量累加器 */\n"
            "    for (k = bk / {{K_UNROLL}}; k > 0; k--) {\n"
            "        /* 每步 {{K_UNROLL}} 次 vle + {{REG_BLOCK_N}} 次 vfmacc，复用装入的 A */\n"
            "    }\n"
            "    /* 用掩码写回 {{REG_BLOCK_N}} 列 C */\n"
            "}"
        ),
        "slots": [
            {"name": "REG_BLOCK_N", "type": "int", "default": 8, "source": "par:reg_block_n"},
            {"name": "K_UNROLL", "type": "int", "default": 8, "source": "par:k_unroll"},
            {"name": "SEW", "type": "int", "default": 64, "source": "par:sew"},
            {"name": "LMUL", "type": "int", "default": 2, "source": "par:lmul"},
        ],
        "hints": ["寄存器预算：REG_BLOCK_N*K_UNROLL 不要超过可用向量寄存器数", "写回用掩码避免越界"],
        "preconditions": ["A/B 已打包成 kernel 期望布局"],
    },
    ("tile", "portable"): {
        "includes": [],
        "skeleton": (
            "for (j = 0; j < bn; j += {{REG_BLOCK_N}}) {\n"
            "    for (i = 0; i < bm; i += {{REG_BLOCK_M}}) {\n"
            "        double acc[{{REG_BLOCK_M}}][{{REG_BLOCK_N}}] = {{INIT}};\n"
            "        for (k = 0; k < bk; k++) { /* acc 中复用装入的 A/B */ }\n"
            "    }\n"
            "}"
        ),
        "slots": [
            {"name": "REG_BLOCK_M", "type": "int", "default": 2, "source": "par:block_size_m"},
            {"name": "REG_BLOCK_N", "type": "int", "default": 2, "source": "par:reg_block_n"},
            {"name": "INIT", "type": "code", "default": "0.0", "source": "operator.init"},
        ],
        "hints": ["累加器数量受标量寄存器限制", "注意 ldc 步长与 C 的列主序布局"],
        "preconditions": ["C 按列主序存储"],
    },
    ("tile", "avx2"): {
        "includes": ["<immintrin.h>"],
        "skeleton": (
            "/* 寄存器分块：{{REG_BLOCK_M}}x{{REG_BLOCK_N}} 个输出块常驻 ymm 累加器 */\n"
            "for (j = 0; j < bn; j += {{REG_BLOCK_N}}) {\n"
            "    __m256d acc[{{REG_BLOCK_M}}][{{REG_BLOCK_N}}/4] = {{INIT}};\n"
            "    for (k = 0; k < bk; k++) {\n"
            "        /* 每步装入 A 的 {{REG_BLOCK_M}} 个标量 + B 的 {{REG_BLOCK_N}} 个向量，做 {{REG_BLOCK_M}}x{{REG_BLOCK_N}} 次 FMA */\n"
            "    }\n"
            "    /* 写回 {{REG_BLOCK_N}} 列 C */\n"
            "}"
        ),
        "slots": [
            {"name": "REG_BLOCK_M", "type": "int", "default": 4, "source": "par:reg_block_m"},
            {"name": "REG_BLOCK_N", "type": "int", "default": 8, "source": "par:reg_block_n"},
            {"name": "INIT", "type": "code", "default": "_mm256_setzero_pd()", "source": "operator.init"},
        ],
        "hints": ["AVX2 只有 16 个 ymm：累加器 + A/B 向量不能超过它", "写回按列主序处理 ldc 步长"],
        "preconditions": ["x86 AVX2 可用", "A/B 已打包"],
    },
    ("tile", "avx512"): {
        "includes": ["<immintrin.h>"],
        "skeleton": (
            "/* 寄存器分块：{{REG_BLOCK_M}}x{{REG_BLOCK_N}}，zmm 累加器 */\n"
            "for (j = 0; j < bn; j += {{REG_BLOCK_N}}) {\n"
            "    __m512d acc[{{REG_BLOCK_M}}][{{REG_BLOCK_N}}/8] = {{INIT}};\n"
            "    for (k = 0; k < bk; k++) { /* {{REG_BLOCK_M}}x{{REG_BLOCK_N}} 次 FMA */ }\n"
            "}"
        ),
        "slots": [
            {"name": "REG_BLOCK_M", "type": "int", "default": 8, "source": "par:reg_block_m"},
            {"name": "REG_BLOCK_N", "type": "int", "default": 16, "source": "par:reg_block_n"},
            {"name": "INIT", "type": "code", "default": "_mm512_setzero_pd()", "source": "operator.init"},
        ],
        "hints": ["32 个 zmm 寄存器可支撑更大的寄存器块", "注意 AVX512 频率与掩码写回"],
        "preconditions": ["x86 AVX512F 可用"],
    },
    ("tile", "neon"): {
        "includes": ["<arm_neon.h>"],
        "skeleton": (
            "/* 寄存器分块：{{REG_BLOCK_M}}x{{REG_BLOCK_N}}（NEON 只有 128-bit） */\n"
            "for (j = 0; j < bn; j += {{REG_BLOCK_N}}) {\n"
            "    float64x2_t acc[{{REG_BLOCK_M}}][{{REG_BLOCK_N}}/2];\n"
            "    for (k = 0; k < bk; k++) { /* vfmaq_f64 累加 */ }\n"
            "}"
        ),
        "slots": [
            {"name": "REG_BLOCK_M", "type": "int", "default": 4, "source": "par:reg_block_m"},
            {"name": "REG_BLOCK_N", "type": "int", "default": 8, "source": "par:reg_block_n"},
        ],
        "hints": ["NEON 只有 32 个 128-bit 寄存器，块尺寸受其限制", "注意 .2d 通道布局"],
        "preconditions": ["ARMv8-A ASIMD 可用", "A/B 已打包"],
    },
    ("tile", "sve"): {
        "includes": ["<arm_sve.h>"],
        "skeleton": (
            "/* SVE 寄存器分块：M={{REG_BLOCK_M}} 行，N 方向按 svcntd() 向量化 */\n"
            "for (j = 0; j < bn; j += {{REG_BLOCK_N}}) {\n"
            "    svfloat64_t acc[{{REG_BLOCK_M}}];\n"
            "    for (k = 0; k < bk; k++) { /* svmla_f64_x 累加 */ }\n"
            "}"
        ),
        "slots": [
            {"name": "REG_BLOCK_M", "type": "int", "default": 4, "source": "par:reg_block_m"},
            {"name": "REG_BLOCK_N", "type": "int", "default": 8, "source": "par:reg_block_n"},
        ],
        "hints": ["SVE 只有 32 个 z 寄存器，块尺寸按 svcntd() 计算", "谓词处理尾部"],
        "preconditions": ["ARM SVE 可用"],
    },
    ("unroll", "rvv"): {
        "includes": ["<riscv_vector.h>"],
        "skeleton": (
            "for (k = bk / {{K_UNROLL}}; k > 0; k--) {\n"
            "    /* 一次迭代内 {{K_UNROLL}} 个独立 k 步：vle + vfmacc，摊薄循环控制开销 */\n"
            "}\n"
            "for (k = 0; k < (bk & {{K_MASK}}); k++) { /* 尾部余数 k 步 */ }"
        ),
        "slots": [
            {"name": "K_UNROLL", "type": "int", "default": 8, "source": "par:k_unroll"},
            {"name": "K_MASK", "type": "int", "default": 7, "source": "par:k_unroll"},
        ],
        "hints": ["K_UNROLL 与 K_MASK 必须一致（K_MASK = K_UNROLL-1）", "尾部余数单独处理"],
        "preconditions": ["bk 可以是任意正整数"],
    },
    ("unroll", "portable"): {
        "includes": [],
        "skeleton": (
            "for (k = 0; k + {{K_UNROLL}} <= bk; k += {{K_UNROLL}}) {\n"
            "    /* {{K_UNROLL}} 个独立加载/累加，减少循环计数器与分支 */\n"
            "}\n"
            "for (; k < bk; ++k) { /* 尾部 */ }"
        ),
        "slots": [{"name": "K_UNROLL", "type": "int", "default": 4, "source": "par:k_unroll"}],
        "hints": ["展开因子过大会造成寄存器溢出", "代码体积增大可能影响 I-cache"],
        "preconditions": ["循环体无跨迭代依赖"],
    },
    ("tail_handle", "rvv"): {
        "includes": ["<riscv_vector.h>"],
        "skeleton": (
            "for (BLASLONG i = 0; i < n; i += vl) {\n"
            "    size_t vl = __riscv_vsetvl_e{{SEW}}m{{LMUL}}(n - i);   /* 动态 vl 天然处理尾部 */\n"
            "    /* 主体：vle / 计算 / vse */\n"
            "}"
        ),
        "slots": [
            {"name": "SEW", "type": "int", "default": 64, "source": "par:sew"},
            {"name": "LMUL", "type": "int", "default": 8, "source": "par:lmul"},
        ],
        "hints": ["vl 是剩余长度与 VLMAX 的较小者", "不要假设 n 整除 VLMAX"],
        "preconditions": ["向量长度可变"],
    },
    ("tail_handle", "portable"): {
        "includes": [],
        "skeleton": (
            "for (i = 0; i + {{BLOCK_SIZE}} <= n; i += {{BLOCK_SIZE}}) { /* 主块 */ }\n"
            "for (; i < n; ++i) { /* 尾部余数逐元素处理，避免越界 */ }"
        ),
        "slots": [{"name": "BLOCK_SIZE", "type": "int", "default": 4, "source": "par:block_size"}],
        "hints": ["尾部循环必须覆盖所有余数", "避免尾部写越界"],
        "preconditions": ["n 可能不整除块大小"],
    },
    ("split", "rvv"): {
        "includes": ["<riscv_vector.h>"],
        "skeleton": (
            "vfloat{{SEW}}m{{LMUL}}_t vsum[{{SPLIT_FACTOR}}] = {0};\n"
            "for (i = 0; i < n; i += vl) {\n"
            "    /* {{SPLIT_FACTOR}} 条独立累加链，最后 vfredosum 归约 */\n"
            "}\n"
            "double s = /* 归约 vsum[0..{{SPLIT_FACTOR}}-1] */;"
        ),
        "slots": [
            {"name": "SPLIT_FACTOR", "type": "int", "default": 1, "source": "par:split_factor"},
            {"name": "SEW", "type": "int", "default": 64, "source": "par:sew"},
            {"name": "LMUL", "type": "int", "default": 8, "source": "par:lmul"},
        ],
        "hints": ["归约链是延迟瓶颈，需要多条独立链", "最终归约顺序不影响浮点结果时可用无序归约"],
        "preconditions": ["含跨迭代归约（dot/nrm2/asum）"],
    },
    ("split", "portable"): {
        "includes": [],
        "skeleton": (
            "double acc[{{SPLIT_FACTOR}}] = {0};\n"
            "for (i = 0; i < n; i += {{SPLIT_FACTOR}}) {\n"
            "    /* {{SPLIT_FACTOR}} 条标量累加链并行 */\n"
            "}\n"
            "double s = 0; for (int t = 0; t < {{SPLIT_FACTOR}}; t++) s += acc[t];"
        ),
        "slots": [{"name": "SPLIT_FACTOR", "type": "int", "default": 2, "source": "par:split_factor"}],
        "hints": ["归约语义（求和顺序）变化会影响浮点结果，需在文档中说明"],
        "preconditions": ["含跨迭代归约"],
    },
    ("specialize", "rvv"): {
        "includes": ["<riscv_vector.h>"],
        "skeleton": (
            "if (inc_x == 1 && inc_y == 1) {\n"
            "    /* 单位步长快速路径：vle{{SEW}}/vse{{SEW}} */\n"
            "} else if (inc_y == 1) {\n"
            "    /* 跨步 x：vlse{{SEW}} + 连续 y */\n"
            "} else {\n"
            "    /* 通用 strided 路径 */\n"
            "}"
        ),
        "slots": [
            {"name": "SEW", "type": "int", "default": 64, "source": "par:sew"},
            {"name": "LMUL", "type": "int", "default": 8, "source": "par:lmul"},
        ],
        "hints": ["把最常用路径（stride=1）放在分支最前", "多版本代码用宏展开，避免源码重复"],
        "preconditions": ["x/y 的步长可能不为 1"],
    },
    ("specialize", "portable"): {
        "includes": [],
        "skeleton": (
            "if (inc_x == 1) {\n"
            "    for (BLASLONG i = 0; i + {{BLOCK_SIZE}} <= n; i += {{BLOCK_SIZE}}) { /* 连续快速路径 */ }\n"
            "} else {\n"
            "    for (BLASLONG i = 0; i < n; i++) { /* 跨步路径：inc_x 逐次累加，消除索引乘法 */ }\n"
            "}"
        ),
        "slots": [{"name": "BLOCK_SIZE", "type": "int", "default": 4, "source": "par:block_size"}],
        "hints": ["分支预测友好的顺序", "消除跨步访问的索引乘法"],
        "preconditions": ["步长可能不为 1"],
    },
    ("reorder", "portable"): {
        "includes": [],
        "skeleton": (
            "const double *pa = a;\n"
            "double *pc = c;\n"
            "for (BLASLONG j = 0; j < n; j += {{BLOCK_SIZE}}) {\n"
            "    for (BLASLONG i = 0; i < m; i++) {\n"
            "        pc[i] += pa[i * lda];   /* 指针递增替代 i*lda 索引乘法 */\n"
            "    }\n"
            "    pa += {{BLOCK_SIZE}} * lda; pc += {{BLOCK_SIZE}};\n"
            "}"
        ),
        "slots": [{"name": "BLOCK_SIZE", "type": "int", "default": 1, "source": "par:block_size"}],
        "hints": ["不要创建别名指针导致编译器无法优化", "指针算术需与 ldc/lda 布局一致"],
        "preconditions": ["循环索引用于地址计算"],
    },
    ("reorder", "rvv"): {
        "includes": ["<riscv_vector.h>"],
        "skeleton": (
            "const double *pa = a;\n"
            "for (j = 0; j < n; j += {{BLOCK_SIZE}}) {\n"
            "    /* 指针递增替代索引乘法，配合 vle 连续访存 */\n"
            "}"
        ),
        "slots": [{"name": "BLOCK_SIZE", "type": "int", "default": 1, "source": "par:block_size"}],
        "hints": ["指针算术与 pack 布局一致"],
        "preconditions": ["循环索引用于地址计算"],
    },
    ("pack", "portable"): {
        "includes": ["<string.h>"],
        "skeleton": (
            "for (BLASLONG j = 0; j < n; j += {{PACK_BLOCK}}) {\n"
            "    for (BLASLONG i = 0; i < m; i++) {\n"
            "        double *dst = buf + i * {{PACK_BLOCK}};\n"
            "        for (BLASLONG t = 0; t < {{PACK_BLOCK}}; t++) dst[t] = a[i * lda + j + t];\n"
            "    }\n"
            "}"
        ),
        "slots": [{"name": "PACK_BLOCK", "type": "int", "default": 8, "source": "par:pack_block"}],
        "hints": ["打包缓冲大小 = m*PACK_BLOCK*sizeof(T)", "打包成本要被内层循环复用次数摊薄"],
        "preconditions": ["A/B 跨步访问导致内层循环不连续"],
    },
    ("pack", "rvv"): {
        "includes": ["<riscv_vector.h>"],
        "skeleton": (
            "for (BLASLONG j = 0; j < n; j += {{PACK_BLOCK}}) {\n"
            "    /* 按 kernel 期望布局把跨步 A/B 写入连续缓冲，再用 vlse/vle 读取 */\n"
            "}"
        ),
        "slots": [{"name": "PACK_BLOCK", "type": "int", "default": 8, "source": "par:pack_block"}],
        "hints": ["pack 后的布局必须与 kernel 读取一致", "缓冲对齐到 cacheline"],
        "preconditions": ["输入矩阵跨步存储"],
    },
    ("prefetch", "portable"): {
        "includes": [],
        "skeleton": (
            "for (BLASLONG i = 0; i < n; i++) {\n"
            "    __builtin_prefetch(&b[i + {{PREFETCH_DIST}}]);\n"
            "    c[i] += a[i] * b[i];\n"
            "}"
        ),
        "slots": [{"name": "PREFETCH_DIST", "type": "int", "default": 16, "source": "par:prefetch_distance"}],
        "hints": ["预取距离要匹配 L1/L2 延迟", "不要预取越界地址"],
        "preconditions": ["循环访问可预测的连续地址"],
    },
    ("fuse", "portable"): {
        "includes": [],
        "skeleton": (
            "for (BLASLONG i = 0; i < n; i += {{BLOCK_SIZE}}) {\n"
            "    c[i] = {{ALPHA}} * a[i] + b[i];   /* 把缩放融合进写回，避免额外遍历 */\n"
            "}"
        ),
        "slots": [
            {"name": "BLOCK_SIZE", "type": "int", "default": 4, "source": "par:block_size"},
            {"name": "ALPHA", "type": "expr", "default": "alpha", "source": "operator.alpha"},
        ],
        "hints": ["融合后临时变量复用", "注意广播语义"],
        "preconditions": ["同一数据被多次遍历"],
    },
    ("reduce_register_pressure", "portable"): {
        "includes": [],
        "skeleton": (
            "double acc[{{ACC_COUNT}}] = {0};\n"
            "for (BLASLONG i = 0; i < n; i += {{ACC_COUNT}}) { /* 累加器数量受寄存器限制 */ }\n"
            "double s = 0; for (int t = 0; t < {{ACC_COUNT}}; t++) s += acc[t];"
        ),
        "slots": [{"name": "ACC_COUNT", "type": "int", "default": 1, "source": "par:accumulator_count"}],
        "hints": ["累加器不要超过可用寄存器数", "归约顺序影响浮点结果"],
        "preconditions": ["含跨迭代归约"],
    },
}

_DEFAULT_SLOT_VALUES = {
    "SEW": 64,
    "LMUL": 8,
    "REG_BLOCK_N": 8,
    "K_UNROLL": 8,
    "REG_BLOCK_M": 2,
    "BLOCK_SIZE": 4,
    "PACK_BLOCK": 8,
    "PREFETCH_DIST": 16,
    "SPLIT_FACTOR": 2,
    "ACC_COUNT": 4,
    "UNROLL_FACTOR": 4,
}


def canonical_template(action: str, isa: str, params: dict = None) -> dict:
    """按 action + ISA 变体取模板骨架；槽位默认值用该策略的参数值覆盖。"""
    key = (action, isa)
    if key not in _TEMPLATES:
        key = (action, "portable")
    if key not in _TEMPLATES:
        key = ("tile", "portable")
    spec = _TEMPLATES[key]
    params = params or {}
    slots = []
    for slot in spec["slots"]:
        slot = dict(slot)
        if slot["name"] in params and params[slot["name"]] is not None:
            slot["default"] = params[slot["name"]]
        elif slot["default"] is None:
            slot["default"] = _DEFAULT_SLOT_VALUES.get(slot["name"], 0)
        slots.append(slot)
    return {
        "language": "c",
        "includes": list(spec["includes"]),
        "skeleton": spec["skeleton"],
        "slots": slots,
        "preconditions": list(spec["preconditions"]),
        "check_hints": list(spec["hints"]),
        "isa_variant": key[1],
    }


def _strip_perf_numbers(text: str) -> str:
    """v1.5 不变量：策略文本不得存收益数字；把 LLM 形容词里的百分比/倍数剥离为定性描述。"""
    text = re.sub(r"[约~]?\s*\d+(\.\d+)?\s*%\s*(到|至|-|~)?\s*\d*(\.\d+)?\s*%", "一定比例", text)
    text = re.sub(r"\d+(\.\d+)?\s*%\s*(到|至|-|~)?\s*\d*(\.\d+)?\s*%", "一定比例", text)
    text = re.sub(r"\d+(\.\d+)?\s*[x×]\s*(到|至|-|~)?\s*\d*(\.\d+)?\s*[x×]?", "数倍", text)
    text = re.sub(r"\d+(\.\d+)?\s*倍", "数倍", text)
    text = re.sub(r"(提升|加速|提速|降低)\s*(约|可达)?\s*数倍", r"\1显著", text)
    return text


# ---------------------------------------------------------------- 分析 JSON


def load_analysis(cfg) -> dict:
    """载入 4 个算子的分析 JSON，并计算每个 optimization 在 JSON 文件中的行号区间。"""
    out = {}
    for op in cfg.analysis_operators:
        artifact_id = "art:%s:analysis" % op
        rel_path = cfg.rel_workspace(cfg.analysis_dir / ("%s_analysis.json" % op))
        path = cfg.analysis_dir / ("%s_analysis.json" % op)
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        data = json.loads(text)
        spans = _optimization_spans(lines)
        out[op] = {"path": artifact_id, "rel_path": rel_path, "data": data, "lines": lines, "spans": spans}
    return out


def _optimization_spans(lines: list) -> dict:
    """name -> (line_start, line_end)：在分析 JSON 原文里定位每个 optimization 对象。"""
    spans = {}
    line_of = {}
    for idx, line in enumerate(lines):
        m = re.search(r'"optimization_name":\s*"(.*?)"\s*,?\s*$', line)
        if m:
            line_of[m.group(1)] = idx
    for name, idx in line_of.items():
        # 向前找最近的对象起始行（缩进较浅的 '{'）
        start = idx
        while start > 0 and "{" not in lines[start]:
            start -= 1
        depth = 0
        end = len(lines) - 1
        for j in range(start, len(lines)):
            depth += lines[j].count("{") - lines[j].count("}")
            if depth <= 0:
                end = j
                break
        spans[name] = (start + 1, end + 1)
    return spans


def _analysis_conditions(op: str, spec: dict, family: str, level: str, dtype: str) -> list:
    depth = 3 if spec.get("op_class") in ("matrix", "triangular") else 2
    conds = [
        {"pred": "dtype", "op": "in", "value": [dtype], "hard": True},
        {"pred": "loop_depth", "op": "ge", "value": depth, "hard": True},
        {"pred": "precision", "op": "eq", "value": "single" if dtype in ("s", "c") else "double", "hard": True},
    ]
    if spec.get("structural") and spec["structural"] != "none":
        conds.append({"pred": "structural", "op": "eq", "value": spec["structural"], "hard": True})
    if spec.get("reduction"):
        conds.append({"pred": "reduction", "op": "eq", "value": True, "hard": True})
    if level == "algorithm":
        conds.append({"pred": "size_regime", "op": "in", "value": ["medium", "large"], "hard": True})
    else:
        conds.append({"pred": "size_regime", "op": "in", "value": ["medium", "large"], "hard": False, "weight": 1.0})
    if family == "SIMD向量指令":
        conds.append({"pred": "stride_x", "op": "eq", "value": 1, "hard": True})
        conds.append({"pred": "aligned", "op": "eq", "value": True, "hard": False})
    return _dedupe_conditions(conds)


def _analysis_params(entry: dict) -> list:
    params = []
    for idx, p in enumerate(entry.get("tunable_parameters") or []):
        name = p.get("parameter_name") or ("param_%d" % idx)
        slug = param_slug(str(name))
        params.append({
            "name": slug,
            "label": str(name),
            "kind": "unroll" if "unroll" in slug else ("blocking" if "block" in slug else "blocking"),
            "unit": "iterations" if "unroll" in slug else "elements",
            "value_in_source": p.get("value_in_code"),
            "range": p.get("typical_range"),
            "derived_from": None,
            "effect_direction": str(p.get("impact") or p.get("description") or "")[:300] or None,
        })
    return params


def _provenance(base_rel: str, span, extractor_version: str, line_confidence="exact", origin_file=None):
    start, end = (span if span else (None, None))
    prov = {
        "source_file": base_rel,
        "line_start": start,
        "line_end": end,
        "line_confidence": line_confidence,
        "commit": "OpenBLAS 0.3.27.dev",
        "extractor_version": extractor_version,
    }
    if origin_file:
        prov["origin_file"] = origin_file
    return prov


def _analysis_strategies(cfg, analysis: dict) -> list:
    raw = []
    for op in cfg.analysis_operators:
        meta = analysis[op]
        spec = cfg.operators[op]
        grouped = {}
        for item in meta["data"]["individual_analyses"]:
            file_name = item.get("file_path", "")
            dtype = dtype_from_name(file_name)
            for layer_key, level in (
                ("algorithm_level_optimizations", "algorithm"),
                ("code_level_optimizations", "code"),
                ("instruction_level_optimizations", "instruction"),
            ):
                for entry in item.get(layer_key) or []:
                    name = entry.get("optimization_name") or ""
                    if not name:
                        continue
                    family, action = family_action(name, level)
                    key = (op, family, action, level)
                    desc = entry.get("description") or {}
                    pattern_stage = _pattern_prefix(entry.get("related_patterns")) or _LEVEL_STAGE.get(level, "core")
                    candidate = {
                        "raw_key": "analysis:%s:%s:%s:%s" % (op, family, action, level),
                        "op": op,
                        "source": "analysis",
                        "level": level,
                        "family": family,
                        "action": action,
                        "stage": pattern_stage,
                        "name": name,
                        "description": (desc.get("strategy_rationale") or "")[:1200],
                        "rationale": (desc.get("implementation_pattern") or "")[:1200],
                        "trade_offs": (desc.get("trade_offs") or "")[:600],
                        "conditions": _analysis_conditions(op, spec, family, level, dtype),
                        "params": _analysis_params(entry),
                        "template": canonical_template(action, "portable", {
                            p["name"]: p.get("value_in_source") for p in _analysis_params(entry)
                        }),
                        "hardware_profile_ids": ["x86_avx2", "x86_avx512", "k1_rvv256"],
                        "isa_requirements": (
                            [{"isa": "isa:avx2.fma.f64", "hard": False, "note": "SIMD 指令可加速（编译器自动向量化回退）"}]
                            if family == "SIMD向量指令" else []
                        ),
                        "origins": [{
                            "file_path": file_name,
                            "dtype": dtype,
                            "target_hardware": entry.get("target_hardware_feature_name"),
                            "code_example": ((entry.get("code_example") or {}).get("snippet") or "")[:600],
                            "related_patterns": entry.get("related_patterns") or [],
                        }],
                        "provenance": _provenance(
                            meta["rel_path"], meta["spans"].get(name),
                            cfg.extractor_version, origin_file=file_name,
                        ),
                        "support_count": 1,
                        "status": "raw",
                        "confidence": 0.75,
                        "pattern_stage": pattern_stage,
                    }
                    if key not in grouped:
                        grouped[key] = candidate
                    else:
                        existing = grouped[key]
                        existing["support_count"] += 1
                        existing["origins"].append(candidate["origins"][0])
                        if len(candidate["description"]) > len(existing["description"]):
                            existing["description"] = candidate["description"]
                            existing["rationale"] = candidate["rationale"]
                            existing["provenance"] = candidate["provenance"]
                    if key in grouped:
                        continue
        for key, candidate in grouped.items():
            if candidate["support_count"] > 1:
                candidate["status"] = "merged"
            raw.append(candidate)
    return raw


# ---------------------------------------------------------------- 静态/差分策略


def _static_conditions(op: str, spec: dict, family: str, action: str, impl: dict) -> list:
    depth = max(1, impl.get("loop_depth") or 1)
    conds = [
        {"pred": "dtype", "op": "in", "value": [spec.get("dtype", "d")], "hard": True},
        {"pred": "loop_depth", "op": "ge", "value": max(1, depth - 1 if depth > 2 else depth), "hard": True},
    ]
    if spec.get("structural") and spec["structural"] != "none":
        conds.append({"pred": "structural", "op": "eq", "value": spec["structural"], "hard": True})
    if spec.get("reduction"):
        conds.append({"pred": "reduction", "op": "eq", "value": True, "hard": True})
    conds.append({"pred": "size_regime", "op": "in", "value": ["medium", "large"], "hard": False, "weight": 1.0})
    if action == "tail_handle":
        conds.append({"pred": "tail_divisible", "op": "eq", "value": False, "hard": False})
    return _dedupe_conditions(conds)


def _static_params(action: str, impl: dict, spec: dict) -> list:
    params = []
    if action == "tile" and impl.get("acc_count"):
        params.append({
            "name": "reg_block_n", "label": "寄存器累加器个数", "kind": "blocking", "unit": "accumulators",
            "value_in_source": impl["acc_count"], "range": [2, 4, 8],
            "derived_from": "由标量/向量寄存器数量与 C 列块宽决定",
            "effect_direction": "累加器越多，装入值复用次数越高；超过寄存器数会溢出到栈",
        })
    if action == "unroll" and impl.get("k_den"):
        params.append({
            "name": "k_unroll", "label": "k 循环展开因子", "kind": "unroll", "unit": "iterations",
            "value_in_source": impl["k_den"], "range": [2, 4, 8, 16],
            "derived_from": "min(bk 可整除因子, 寄存器预算)",
            "effect_direction": "展开越大循环开销越低、寄存器压力越大、尾部处理越复杂",
        })
    if action == "reorder":
        params.append({
            "name": "block_size", "label": "指针步进块大小", "kind": "blocking", "unit": "elements",
            "value_in_source": 1, "range": [1, 2, 4], "derived_from": "cacheline_b / sizeof(T)",
            "effect_direction": "块越大地址计算越少，但寄存器/别名分析压力越大",
        })
    if action == "tail_handle":
        params.append({
            "name": "block_size", "label": "主块大小", "kind": "blocking", "unit": "elements",
            "value_in_source": None, "range": [1, 2, 4, 8], "derived_from": "vector_bits / (8 * sizeof(T))",
            "effect_direction": "块越大尾部越短，但余数分支组合越多",
        })
    if action == "reduce_register_pressure":
        params.append({
            "name": "accumulator_count", "label": "归约累加器个数", "kind": "blocking", "unit": "accumulators",
            "value_in_source": 1, "range": [1, 2, 4, 8], "derived_from": "寄存器预算 / 累加链延迟",
            "effect_direction": "多累加链提高指令级并行，最终归约增加一次开销",
        })
    return params


def _static_strategies(cfg, impls: dict, artifacts: list) -> list:
    index = {a["artifact_id"]: a for a in artifacts}
    raw = []
    for op, spec in cfg.operators.items():
        impl = impls[op]["generic"]
        dtype = spec.get("dtype", "d")
        artifact = index["art:%s:kernel_generic" % op]
        candidates = []
        if impl.get("acc_count", 0) >= 2:
            candidates.append(("tile", "固定 %d 累加器微内核分块" % impl["acc_count"],
                               "generic kernel 用 %d 个标量累加器把输出块留在寄存器里，减少 C 的重复访存。" % impl["acc_count"]))
        if (impl.get("k_den") or 0) >= 2:
            candidates.append(("unroll", "k 循环按 %d 手动展开" % impl["k_den"],
                               "内层 k 循环按 bk/%d 展开，单次迭代合并多步加载与乘加，摊薄循环控制开销。" % impl["k_den"]))
        if impl.get("pointer_arith"):
            candidates.append(("reorder", "指针预计算消除索引乘法",
                               "用 ptr/C0/C1 这类递增指针替代 i*lda 形式的地址计算，减少整数乘法与地址生成指令。"))
        if impl.get("has_tail_branch") or (impl.get("has_tail") and not impl.get("has_vector")):
            candidates.append(("tail_handle", "尾部余数分支单独处理",
                               "主循环按块处理，余数（如 bm&1、bk&3）用独立分支收尾，避免越界访问。"))
        if impl.get("reduction") or spec.get("reduction"):
            candidates.append(("reduce_register_pressure", "标量归约累加链",
                               "把跨迭代归约写成单条标量累加链，命中同一寄存器；这是后续拆多链/向量归约的基线。"))
        if impl.get("alpha_scale"):
            candidates.append(("fuse", "alpha 缩放融合到写回",
                               "在写回 C 前把 alpha 乘进累加器，避免对 C 的额外一遍读-改-写。"))
        if impl.get("uses_inc") and not impl.get("stride_dispatch"):
            candidates.append(("specialize", "任意步长 inc 累加访问",
                               "用 inc_x/inc_y 逐次累加访问 x/y，单版本支持任意步长，避免数据拷贝。"))
        if not candidates:
            candidates.append(("reorder", "参数校验与循环不变量前置",
                               "把 n<=0 / alpha==0 等早退判断放在循环外，减少热路径分支与重复计算。"))
        seen_actions = set()
        for action, name, desc in candidates:
            if action in seen_actions:
                continue
            seen_actions.add(action)
            params = _static_params(action, impl, spec)
            raw.append({
                "raw_key": "static:%s:%s" % (op, action),
                "op": op,
                "source": "static",
                "level": "code",
                "family": ACTION_TO_FAMILY.get(action, "指针/索引优化"),
                "action": action,
                "stage": _action_stage(action),
                "name": name,
                "description": desc,
                "rationale": "来自 generic kernel 的静态特征（第 %d-%d 行）：loop_depth=%d, acc_count=%d, k_den=%s, tail=%s。"
                             % (impl["line_start"], impl["line_end"], impl["loop_depth"], impl["acc_count"],
                                impl["k_den"], impl["has_tail"]),
                "trade_offs": None,
                "conditions": _static_conditions(op, spec, "", action, impl),
                "params": params,
                "template": canonical_template(action, "portable", {p["name"]: p.get("value_in_source") for p in params}),
                "hardware_profile_ids": profile_ids(cfg),
                "isa_requirements": [],
                "origins": [],
                "provenance": _provenance(
                    artifact["path"], (impl["line_start"], impl["line_end"]),
                    cfg.extractor_version, origin_file=artifact.get("repo_path"),
                ),
                "support_count": 1,
                "status": "verified",
                "confidence": 0.85,
                "pattern_stage": _action_stage(action),
            })
    return raw


_DIFF_GROUP = {
    "vectorize": ("SIMD向量指令", "vectorize", "RVV 向量化内核（vsetvl + vle/vfmacc/vse）"),
    "fma": ("SIMD向量指令", "vectorize", "RVV FMA 融合乘加"),
    "blocking": ("分块Tiling", "tile", "向量寄存器输出分块"),
    "unroll": ("循环展开", "unroll", "k 循环展开"),
    "lmul_tuning": ("SIMD向量指令", "specialize", "LMUL/SEW 向量寄存器分组调优"),
    "tail_handle": ("尾部处理", "tail_handle", "动态 vl / 掩码尾部处理"),
    "stride_dispatch": ("指针/索引优化", "specialize", "单位步长快速路径分支派发"),
    "reduction_split": ("循环拆分", "split", "向量归约累加链"),
    "prefetch": ("预取", "prefetch", "软件预取"),
}


def _action_stage(action: str) -> str:
    return {
        "pack": "transform",
        "prefetch": "prep",
        "tail_handle": "post",
        "reorder": "prep",
        "fuse": "post",
    }.get(action, "core")


def _diff_conditions(op: str, spec: dict, action: str, impl: dict) -> list:
    depth = max(1, impl.get("loop_depth") or 1)
    conds = [
        {"pred": "dtype", "op": "in", "value": [spec.get("dtype", "d")], "hard": True},
        {"pred": "loop_depth", "op": "ge", "value": max(1, depth - 1 if depth > 2 else depth), "hard": True},
    ]
    if spec.get("structural") and spec["structural"] != "none":
        conds.append({"pred": "structural", "op": "eq", "value": spec["structural"], "hard": True})
    if spec.get("reduction"):
        conds.append({"pred": "reduction", "op": "eq", "value": True, "hard": True})
    if action in ("vectorize", "stride_dispatch"):
        conds.append({"pred": "stride_x", "op": "eq", "value": 1, "hard": True})
        conds.append({"pred": "aligned", "op": "eq", "value": True, "hard": False})
    if action == "tail_handle":
        conds.append({"pred": "tail_divisible", "op": "eq", "value": False, "hard": False})
    conds.append({"pred": "size_regime", "op": "in", "value": ["medium", "large"], "hard": True})
    return _dedupe_conditions(conds)


def _diff_params(action: str, diff_params: dict, impl: dict) -> list:
    """只为该 action 真正需要的参数建模（避免把 tile 的寄存器预算错误地加到所有策略上）。"""
    allowed = {
        "vectorize": {"sew", "lmul", "vector_len"},
        "tile": {"reg_block_n", "k_unroll", "sew", "lmul", "vector_len"},
        "unroll": {"k_unroll"},
        "specialize": {"sew", "lmul"},
        "tail_handle": {"sew", "lmul"},
        "split": {"split_factor", "sew", "lmul"},
        "pack": {"block_size"},
        "prefetch": {"prefetch_distance"},
        "fuse": set(),
        "reorder": {"block_size"},
    }.get(action, set())
    params = []
    if "reg_block_n" in allowed and diff_params.get("reg_block_n"):
        params.append({
            "name": "reg_block_n", "label": "向量累加器列数", "kind": "blocking", "unit": "columns",
            "value_in_source": diff_params["reg_block_n"], "range": [2, 4, 8],
            "derived_from": "min(num_vector_regs / 2, N_tile)",
            "effect_direction": "列块越大装入 A 的复用次数越高；超过可用向量寄存器会溢出",
        })
    if "k_unroll" in allowed and diff_params.get("k_unroll"):
        params.append({
            "name": "k_unroll", "label": "k 循环展开因子", "kind": "unroll", "unit": "iterations",
            "value_in_source": diff_params["k_unroll"], "range": [2, 4, 8, 16],
            "derived_from": "寄存器预算内可并行发起的 vfmacc 数",
            "effect_direction": "展开越大循环开销越低、寄存器压力越大、尾部越复杂",
        })
    if "lmul" in allowed and diff_params.get("lmul"):
        params.append({
            "name": "lmul", "label": "向量寄存器分组 LMUL", "kind": "lmul", "unit": "regs",
            "value_in_source": diff_params["lmul"], "range": [1, 2, 4, 8],
            "derived_from": "num_vector_regs / 同时活跃向量数",
            "effect_direction": "LMUL 越大单指令吞吐越高、可用寄存器组越少",
        })
    if "sew" in allowed and diff_params.get("sew"):
        params.append({
            "name": "sew", "label": "元素宽度 SEW", "kind": "sew", "unit": "bits",
            "value_in_source": diff_params["sew"], "range": [32, 64], "derived_from": "dtype",
            "effect_direction": "由数据类型决定（double 固定 64）",
        })
    if "vector_len" in allowed and diff_params.get("vector_len"):
        params.append({
            "name": "vector_len", "label": "VLMAX（VLEN*LMUL/SEW）", "kind": "vector_len", "unit": "elements",
            "value_in_source": 256 * (diff_params.get("lmul") or 1) // (diff_params.get("sew") or 64),
            "range": [4, 8, 16, 32], "derived_from": diff_params["vector_len"],
            "effect_direction": "等于 VLMAX 时无尾部开销；VLEN 越大单指令处理元素越多",
        })
    if "split_factor" in allowed:
        params.append({
            "name": "split_factor", "label": "归约累加链数", "kind": "blocking", "unit": "chains",
            "value_in_source": 1, "range": [1, 2, 4],
            "derived_from": "num_vector_regs / 归约指令延迟",
            "effect_direction": "独立链越多归约延迟越低，最终归约多一次开销",
        })
    return params


def _diff_strategies(cfg, diffs: list, impls: dict) -> list:
    raw = []
    for diff in diffs:
        op = diff["op"]
        spec = cfg.operators[op]
        rvv = impls[op]["rvv"]
        actions = [a for a in diff["actions"] if a in _DIFF_GROUP]
        # fma 合并进 vectorize，避免同 (op, action) 重复
        if "fma" in actions and "vectorize" in actions:
            actions.remove("fma")
        groups = {}
        for action in actions:
            family, mapped_action, label = _DIFF_GROUP[action]
            groups.setdefault(mapped_action, []).append(action)
        for mapped_action, source_actions in groups.items():
            family, _, label = _DIFF_GROUP[source_actions[0]]
            params = _diff_params(mapped_action, diff["params"], rvv)
            desc = "；".join(diff["action_notes"]) if len(diff["action_notes"]) <= 3 else "；".join(diff["action_notes"][:3])
            isa = [{"isa": "isa:rvv1.0.vector", "hard": True, "note": "RVV 向量扩展（vsetvl/向量访存）"}]
            if "fma" in source_actions or mapped_action in ("vectorize", "tile"):
                isa.append({"isa": "isa:rvv1.0.fma.f64", "hard": True, "note": "vfmacc 融合乘加"})
            raw.append({
                "raw_key": "diff:%s:%s" % (op, mapped_action),
                "op": op,
                "source": "diff",
                "level": "instruction",
                "family": family,
                "action": mapped_action,
                "stage": _action_stage(mapped_action),
                "name": label,
                "description": "%s（%s -> %s，函数 %s）。%s" % (
                    desc, diff["anchors"]["before"]["file"], diff["anchors"]["after"]["file"],
                    diff["anchors"]["after"]["function"], "生成动作：%s" % "、".join(source_actions)),
                "rationale": "差分事实：generic 版 intrinsic=%d、累加器=%d(%s)、k_den=%s、tail=%s；"
                             "rvv 版 intrinsic=%d、累加器=%d(%s)、k_den=%s、tail=%s、sew=%s、lmul=%s。" % (
                                 diff["delta"]["intrinsic_count_old"], diff["delta"]["acc_old"]["n"],
                                 diff["delta"]["acc_old"]["kind"], diff["delta"]["k_unroll_old"],
                                 diff["delta"]["tail_old"], diff["delta"]["intrinsic_count_new"],
                                 diff["delta"]["acc_new"]["n"], diff["delta"]["acc_new"]["kind"],
                                 diff["delta"]["k_unroll_new"], diff["delta"]["tail_new"],
                                 diff["delta"]["sew_new"], diff["delta"]["lmul_new"]),
                "trade_offs": "RVV 版本只适用于支持 RVV 1.0 的 RISC-V 平台；LMUL/展开过大时会挤占向量寄存器。",
                "conditions": _diff_conditions(op, spec, source_actions[0], rvv),
                "params": params,
                "template": canonical_template(mapped_action, "rvv", {p["name"]: p.get("value_in_source") for p in params}),
                "hardware_profile_ids": ["k1_rvv256", "qemu-riscv64-vlen256"],
                "isa_requirements": isa,
                "origins": [{"diff_id": diff["diff_id"], "anchors": diff["anchors"]}],
                "provenance": _provenance(
                    diff["provenance"]["source_file"],
                    (diff["provenance"]["line_start"], diff["provenance"]["line_end"]),
                    cfg.extractor_version, origin_file=diff["anchors"]["after"]["file"],
                ),
                "support_count": 1,
                "status": "verified",
                "confidence": 0.9,
                "pattern_stage": _action_stage(mapped_action),
            })
    return raw


# ---------------------------------------------------------------- patterns / operators


def derive_patterns(cfg, analysis: dict, impls: dict, artifacts: list, ma_impls: dict = None) -> list:
    """计算流程节点：分析 JSON 的 pattern_type + 静态规则补齐（保证检索有 core 锚点）。

    多架构扩展：新增算子（asum/axpby）也用同一套 op_class 规则补 pattern；参考实现取多架构 C 内核。
    """
    index = {a["artifact_id"]: a for a in artifacts}
    patterns = {}

    def add(op, pattern_type, name, stage, category, description, provenance, support=1):
        pid = "pat:%s:%s" % (op, pattern_type)
        if pid in patterns:
            patterns[pid]["support_count"] += support
            return pid
        patterns[pid] = {
            "id": pid,
            "op": op,
            "pattern_type": pattern_type,
            "name": name,
            "stage": stage,
            "category": category,
            "description": description[:1200],
            "support_count": support,
            "provenance": provenance,
        }
        return pid

    for op in cfg.analysis_operators:
        meta = analysis[op]
        art = index[meta["path"]]
        for item in meta["data"]["individual_analyses"]:
            for pat in item.get("computational_patterns") or []:
                ptype = pat.get("pattern_type") or ""
                if ptype not in PATTERN_META:
                    stage = ptype.split(".")[0] if "." in ptype else "core"
                    category = {"prep": "parameter_check", "transform": "packing",
                                "core": "elementwise_update", "post": "tail_case"}.get(stage, "elementwise_update")
                    name = pat.get("name") or ptype
                else:
                    name, stage, category = PATTERN_META[ptype]
                prov = _provenance(meta["rel_path"], (1, len(meta["lines"])), cfg.extractor_version)
                add(op, ptype, pat.get("name") or name, stage, category,
                    pat.get("description") or name, prov)

    static_rules = {
        "matrix": ["prep.parameter_validation", "prep.index_pointer_init", "prep.loop_invariant_calc",
                   "core.mm_microkernel", "transform.packing", "post.tail_case"],
        "triangular": ["prep.parameter_validation", "prep.index_pointer_init", "core.mm_microkernel", "post.tail_case"],
        "vector": ["prep.parameter_validation", "prep.index_pointer_init", "core.elementwise_update", "post.tail_case"],
    }
    for op, spec in cfg.all_operator_specs.items():
        generic = (impls.get(op) or {}).get("generic")
        art = index.get("art:%s:kernel_generic" % op)
        if generic is None and ma_impls:
            for (ma_op, _arch), features in sorted(ma_impls.items()):
                if ma_op != op:
                    continue
                c_files = [f for f in (features.get("files") or []) if f.get("kind") == "c"]
                if c_files:
                    generic = c_files[0]
                    art = index.get(c_files[0].get("artifact_id"))
                    break
        if generic is None:
            generic = {"file": "config", "line_start": 1, "line_end": 1, "has_tail": False}
        if art is not None:
            prov = _provenance(art["path"], (generic.get("line_start") or 1, generic.get("line_end") or 1),
                               cfg.extractor_version, origin_file=art.get("repo_path"))
        else:
            prov = _prov_fallback(cfg)
        rules = list(static_rules.get(spec.get("op_class"), static_rules["vector"]))
        if spec.get("reduction"):
            rules = [r for r in rules if r != "core.elementwise_update"] + ["core.reduction"]
        if spec.get("op_class") == "matrix" and spec.get("structural") in ("symmetric", "hermitian"):
            rules = rules + ["core.gemv_like"]
        for ptype in rules:
            if ptype not in PATTERN_META:
                continue
            name, stage, category = PATTERN_META[ptype]
            desc = "静态规则识别：%s（参考实现 %s 第 %s-%s 行）" % (
                name, generic.get("file"), generic.get("line_start"), generic.get("line_end"))
            add(op, ptype, name, stage, category, desc, prov)
        if generic.get("has_tail"):
            name, stage, category = PATTERN_META["post.tail_case"]
            add(op, "post.tail_case", name, stage, category,
                "静态规则识别：主循环后的余数/边界处理分支", prov)
    return sorted(patterns.values(), key=lambda p: p["id"])


def _prov_fallback(cfg) -> dict:
    return {
        "source_file": cfg.rel_root(cfg.config_path),
        "line_start": None,
        "line_end": None,
        "line_confidence": "approx",
        "commit": None,
        "extractor_version": cfg.extractor_version,
        "note": "算子行号来自配置（多架构新增算子无 generic 基线时）",
    }


def derive_operators(cfg, analysis: dict, impls: dict, ma_impls: dict = None) -> dict:
    """算子元信息：dtype 变体（语料 + 分析文件 + 多架构）、接口签名、BLAS 级别、类别、公式。"""
    out = {}
    analysis_dtypes = {}
    for op in cfg.analysis_operators:
        dtypes = set()
        for item in analysis[op]["data"]["individual_analyses"]:
            dtypes.add(dtype_from_name(item.get("file_path", "")))
        analysis_dtypes[op] = sorted(dtypes)

    ma_interfaces = {}
    if ma_impls:
        for (op, _arch), features in sorted(ma_impls.items()):
            if op in ma_interfaces:
                continue
            for f in features.get("files") or []:
                if f.get("kind") == "c" and f.get("signature"):
                    ma_interfaces[op] = f["signature"]
                    break

    for op, spec in cfg.all_operator_specs.items():
        generic = (impls.get(op) or {}).get("generic") or {}
        dtypes = set(analysis_dtypes.get(op, [])) | {spec.get("dtype", "d")}
        for (ma_op, _arch) in (ma_impls or {}):
            if ma_op == op:
                dtypes.add(spec.get("dtype", "d"))
        out[op] = {
            "id": "op:%s" % op,
            "dtype_variants": sorted(dtypes),
            "blas_level": int(spec.get("blas_level", 1)),
            "op_class": spec.get("op_class", "vector"),
            "flops_formula": spec.get("flops_formula"),
            "bytes_formula": spec.get("bytes_formula"),
            "structural": spec.get("structural", "none"),
            "reduction": bool(spec.get("reduction", False)),
            "interface": generic.get("signature") or ma_interfaces.get(op) or "",
        }
    return out


def derive_abstract(patterns: dict, profile_ids: list = None) -> list:
    profile_ids = profile_ids or ["x86_avx2", "x86_avx512", "k1_rvv256", "qemu-riscv64-vlen256"]
    raw = []
    for sid, name, family, action, level, desc in ABSTRACT_STRATEGIES:
        raw.append({
            "raw_key": "abstract:%s" % sid.split(":")[-1],
            "op": None,
            "source": "abstract",
            "arch_scope": "arch_agnostic",
            "arch_family": "agnostic",
            "arch_evidence": [],
            "level": level,
            "family": family,
            "action": action,
            "stage": _action_stage(action),
            "name": name,
            "description": desc,
            "rationale": "由多个算子的具体策略归纳出的抽象策略（GENERALIZES 的目标端），用于跨算子迁移推荐。",
            "trade_offs": None,
            "conditions": _dedupe_conditions([
                {"pred": "dtype", "op": "in", "value": ["s", "d", "c", "z"], "hard": True},
                {"pred": "loop_depth", "op": "ge", "value": 1, "hard": True},
                {"pred": "size_regime", "op": "in", "value": ["medium", "large"], "hard": False},
            ]),
            "params": [],
            "template": canonical_template(action, "portable", {}),
            "hardware_profile_ids": sorted(profile_ids),
            "isa_requirements": [],
            "origins": [],
            "provenance": None,
            "support_count": 1,
            "status": "merged",
            "confidence": 0.7,
            "pattern_stage": _action_stage(action),
            "abstract_target": sid,
        })
    return raw


# ---------------------------------------------------------------- 多架构标注 / 跨架构合并


def profile_ids(cfg) -> list:
    """从 schema/hardware_profiles.yaml 读全部硬件 profile（含 arm_neon / arm_sve）。"""
    from .schema import load_schema

    return sorted(load_schema(cfg.schema_dir).hardware_profiles().keys())


def _annotate_arch(cfg, strategies: list) -> list:
    """给已有策略补 arch_scope：分析/静态/抽象=arch_agnostic；RVV 差分=arch_specific(rvv)。"""
    rvv_artifacts = {}
    for st in strategies:
        source = st.get("source")
        if source in ("analysis", "static", "abstract"):
            st.setdefault("arch_scope", "arch_agnostic")
            st.setdefault("arch_family", "agnostic")
            st.setdefault("arch_evidence", [])
            st.setdefault("arch_parameters", {})
        elif source == "diff":
            st["arch_scope"] = "arch_specific"
            st["arch"] = "rvv"
            st["arch_family"] = "rvv"
            anchors = ((st.get("origins") or [{}])[0].get("anchors") or {})
            after = anchors.get("after") or {}
            evidence = {
                "arch": "rvv",
                "arch_family": "rvv",
                "arch_label": "RISC-V RVV",
                "isa": "RVV 1.0",
                "artifact_id": after.get("artifact_id"),
                "file": after.get("file") or (st.get("provenance") or {}).get("source_file"),
                "repo_path": None,
                "role": "microk",
                "kind": "c",
                "line_start": (st.get("provenance") or {}).get("line_start"),
                "line_end": (st.get("provenance") or {}).get("line_end"),
                "method": "diff",
            }
            st["arch_evidence"] = [evidence]
            st["arch_parameters"] = {
                "rvv": {p.get("name"): p.get("value_in_source") for p in st.get("params") or []}
            }
            st["extraction_method"] = "diff"
    return strategies


def _cross_arch_strategies(cfg, strategies: list) -> list:
    """同一 (op, family, action) 在 >=2 个架构上出现时，合并为 parameterized 策略并保留各架构参数。"""
    groups = {}
    for st in strategies:
        if st.get("source") not in ("multiarch", "diff"):
            continue
        if st.get("arch_scope") != "arch_specific" or not st.get("arch"):
            continue
        key = (st.get("op"), st.get("family"), st.get("action"))
        groups.setdefault(key, []).append(st)
    out = []
    for key, members in sorted(groups.items()):
        op, family, action = key
        archs = sorted({m["arch"] for m in members})
        if len(archs) < 2:
            continue
        spec = cfg.all_operator_specs.get(op, {})
        first = sorted(members, key=lambda m: (0 if m["arch"].startswith("x86") else 1, m["arch"]))[0]
        arch_parameters = {}
        arch_evidence = []
        for member in sorted(members, key=lambda m: m["arch"]):
            arch_parameters[member["arch"]] = {
                p.get("name"): p.get("value_in_source") for p in member.get("params") or []
            }
            arch_evidence.extend(member.get("arch_evidence") or [])
        profiles = sorted({p for member in members for p in (member.get("hardware_profile_ids") or [])})
        depth = 3 if spec.get("op_class") in ("matrix", "triangular") else 2
        conditions = _dedupe_conditions([
            {"pred": "dtype", "op": "in", "value": [spec.get("dtype", "d")], "hard": True},
            {"pred": "loop_depth", "op": "ge", "value": max(1, depth - 1 if depth > 2 else depth), "hard": True},
            {"pred": "size_regime", "op": "in", "value": ["medium", "large"], "hard": True},
        ])
        if spec.get("structural") and spec["structural"] != "none":
            conditions.append({"pred": "structural", "op": "eq", "value": spec["structural"], "hard": True})
        suffix = first["name"].split(" ", 1)[-1] if " " in first["name"] else first["name"]
        out.append({
            "raw_key": "cross_arch:%s:%s:%s" % (op, family, action),
            "op": op,
            "source": "cross_arch",
            "arch_scope": "parameterized",
            "arch_family": "agnostic",
            "arch_evidence": arch_evidence,
            "arch_parameters": arch_parameters,
            "arches": archs,
            "level": "code",
            "family": family,
            "action": action,
            "stage": first.get("stage", "core"),
            "name": "%s（跨架构参数化）" % suffix,
            "description": "同一优化动作在 %d 个架构上的参数化实现：%s；各架构参数不同，统一为 parameterized 策略。"
                           % (len(archs), "、".join(archs)),
            "rationale": "跨架构共有动作 %s/%s；S4 合并时保留每个架构的 arch_parameters（unroll/向量宽度/寄存器块），"
                         "供生成端按目标 profile 取参数。" % (family, action),
            "trade_offs": "参数化策略本身不含具体架构实现，生成时必须按目标 profile 选择 arch_parameters。",
            "conditions": conditions,
            "params": [],
            "template": canonical_template(action, "portable", {}),
            "hardware_profile_ids": profiles,
            "isa_requirements": [],
            "origins": arch_evidence,
            "provenance": first.get("provenance"),
            "support_count": len(members),
            "status": "merged",
            "confidence": 0.8,
            "cross_arch_id": "stg:%s:%s_x" % (op, action),
            "pattern_stage": first.get("stage", "core"),
            "extraction_method": "static",
        })
    return out



# ---------------------------------------------------------------- 主入口


def run(cfg, artifacts: list, impls: dict, diffs: list, mode: str = "offline", llm_client=None,
        ma_artifacts: list = None, ma_impls: dict = None, ma_diffs: list = None) -> dict:
    """S3 主入口。返回 {strategies, patterns, operators, analysis_meta, mode, ma}。

    ma_* 为多架构层（multiarch.py）的三件产物：复制后的源工件、按 (op,arch) 合并的实现特征、架构差分记录。
    """
    ma_artifacts = list(ma_artifacts or [])
    all_artifacts = list(artifacts) + ma_artifacts
    analysis = load_analysis(cfg)
    patterns = derive_patterns(cfg, analysis, impls, all_artifacts, ma_impls=ma_impls)
    operators = derive_operators(cfg, analysis, impls, ma_impls=ma_impls)
    all_profiles = profile_ids(cfg)
    strategies = []
    strategies.extend(_analysis_strategies(cfg, analysis))
    strategies.extend(_static_strategies(cfg, impls, artifacts))
    strategies.extend(_diff_strategies(cfg, diffs, impls))
    strategies.extend(derive_abstract(patterns, profile_ids=all_profiles))
    if ma_impls is not None and ma_diffs is not None:
        from .multiarch import strategies as multiarch_strategies

        strategies.extend(multiarch_strategies(cfg, ma_diffs, ma_impls, ma_artifacts))
    _annotate_arch(cfg, strategies)
    strategies.extend(_cross_arch_strategies(cfg, strategies))
    if mode == "llm" and llm_client is not None:
        strategies = _enrich_with_llm(cfg, strategies, llm_client)
    # 不变量 3：策略文本不得含收益数字（MFlops/%/倍/加速 x 等）
    for strategy in strategies:
        for key in ("description", "rationale", "trade_offs"):
            if strategy.get(key):
                strategy[key] = _strip_perf_numbers(strategy[key])[:2000]
    # provenance 兜底：抽象/跨架构策略用其子策略里第一个带 provenance 的实例
    concrete_by_action = {}
    for st in strategies:
        if st.get("source") in ("multiarch", "diff") and st.get("provenance"):
            concrete_by_action.setdefault(st.get("action"), st["provenance"])
        if st.get("source") not in ("abstract", "cross_arch") and st.get("provenance"):
            concrete_by_action.setdefault("any", st["provenance"])
    for st in strategies:
        if st.get("source") in ("abstract", "cross_arch"):
            st["provenance"] = st.get("provenance") or concrete_by_action.get(
                st.get("action"), concrete_by_action.get("any"))
            st["status"] = "merged" if st.get("source") == "cross_arch" else st.get("status", "merged")
    return {
        "strategies": strategies,
        "patterns": patterns,
        "operators": operators,
        "analysis_meta": analysis,
        "mode": mode,
        "profile_ids": all_profiles,
        "multiarch": {
            "artifacts": ma_artifacts,
            "impls": ma_impls or {},
            "diffs": ma_diffs or [],
        },
    }


def _enrich_with_llm(cfg, strategies: list, llm_client) -> list:
    """LLM 只改文本（命名/解释），失败自动降级；结构字段保持规则来源。"""
    for st in strategies:
        prompt = (
            "你是 HPC 算子优化知识抽取器。根据下面的差分/静态事实，把策略名与解释改写得更专业；"
            "只输出 JSON 对象 {name, description, rationale}，禁止出现任何性能数字、百分比、倍数、未给出的文件名。"
            "事实：op=%s, action=%s, family=%s, 原始名=%s, 原始解释=%s"
        ) % (st.get("op"), st.get("action"), st.get("family"), st.get("name"), st.get("rationale"))
        try:
            result = llm_client.generate_json(prompt, purpose="s3_enrich")
        except Exception:  # LLM 不可用 -> 降级
            break
        if not isinstance(result, dict):
            continue
        for key in ("name", "description", "rationale"):
            value = result.get(key)
            if isinstance(value, str) and value.strip() and not _has_perf_number(value):
                st[key] = value.strip()[:1200]
    return strategies


def _has_perf_number(text: str) -> bool:
    return bool(re.search(r"\d+(\.\d+)?\s*(%|x|倍|MFlops|GFlops|GB/s)", text, re.IGNORECASE))
