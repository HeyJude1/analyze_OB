#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成论文架构图/流程图（SVG）。用 Chrome headless 转 PNG。
输出目录：科研产出/14_论文架构图/
"""
import json
import os
import re
import subprocess

OUT = r"C:\Users\MSI\Desktop\开题\科研产出\14_论文架构图"
COV = r"C:\Users\MSI\Desktop\开题\科研产出\kg_agent\data\arch_coverage.json"
os.makedirs(OUT, exist_ok=True)

TOKEN_RE = re.compile(r"[A-Za-z0-9_./@+\-×=]+|[\u4e00-\u9fff]|.", re.S)

FONT = "Microsoft YaHei, Noto Sans CJK SC, Segoe UI, sans-serif"
INK = "#1A202C"
MUTED = "#4A5568"
BG = "#F5F7FB"

PALETTE = {
    "blue":   ("#2B6CB0", "#EBF4FF"),
    "green":  ("#2F855A", "#E6FFFA"),
    "orange": ("#C05621", "#FFFAF0"),
    "purple": ("#6B46C1", "#F3E8FF"),
    "red":    ("#C53030", "#FFF5F5"),
    "gray":   ("#4A5568", "#EDF2F7"),
    "teal":   ("#2C7A7B", "#E6FFFA"),
}


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def text_width(s, size):
    w = 0.0
    for ch in str(s):
        if "\u4e00" <= ch <= "\u9fff" or ch in "，。；：、（）【】“”‘’！？—…":
            w += size
        else:
            w += size * 0.56
    return w


def wrap(s, max_px, size):
    """按 token 换行：英文单词/代码标识符不拆开；超长 token 才硬拆。"""
    tokens = TOKEN_RE.findall(str(s))
    out, cur, cw = [], "", 0.0
    for tok in tokens:
        w = text_width(tok, size)
        if cw + w <= max_px or not cur:
            if w > max_px and not cur:
                # 超长 token：按字符硬拆
                piece = ""
                for ch in tok:
                    cw2 = text_width(piece + ch, size)
                    if cw2 > max_px and piece:
                        out.append(piece)
                        piece = ch
                    else:
                        piece += ch
                cur, cw = piece, text_width(piece, size)
            else:
                cur += tok
                cw += w
        else:
            out.append(cur)
            cur, cw = tok, w
    if cur:
        out.append(cur)
    return out


def text(x, y, s, size=14, color=INK, bold=False, anchor="start", opacity=1.0):
    weight = ' font-weight="700"' if bold else ""
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'fill="{color}"{weight} text-anchor="{anchor}" opacity="{opacity}">{esc(s)}</text>')


def rect(x, y, w, h, color, fill="#FFFFFF", rx=12, sw=2, dashed=False):
    dash = ' stroke-dasharray="7 5"' if dashed else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{color}" stroke-width="{sw}"{dash}/>')


def box(x, y, w, h, title, lines, color="blue", title_size=17, line_size=13,
        fill=None, align="left", title_color=None):
    stroke, dfill = PALETTE[color]
    fill = fill or dfill
    svg = [rect(x, y, w, h, stroke, fill)]
    pad = 16
    ty = y + 30
    if align == "center":
        svg.append(text(x + w / 2, ty, title, title_size, title_color or stroke, True, "middle"))
    else:
        svg.append(text(x + pad, ty, title, title_size, title_color or stroke, True))
    cy = ty + 8
    for ln in lines:
        for seg in wrap(ln, w - 2 * pad, line_size):
            cy += line_size + 5
            if cy > y + h - 8:
                break
            if align == "center":
                svg.append(text(x + w / 2, cy, seg, line_size, MUTED, False, "middle"))
            else:
                svg.append(text(x + pad, cy, seg, line_size, MUTED))
        cy += 3
    return "\n".join(svg)


def chip(x, y, w, h, label, color="gray", size=12, bold=False):
    stroke, fill = PALETTE[color]
    return (rect(x, y, w, h, stroke, fill, rx=8, sw=1.5) +
            text(x + w / 2, y + h / 2 + size * 0.36, label, size, stroke, bold, "middle"))


MARKERS = "\n".join(
    f'<marker id="ar-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
    f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{v[0]}"/></marker>' for k, v in PALETTE.items()
)


def arrow(x1, y1, x2, y2, color="gray", dashed=False, width=2.2, label=None, label_pos=0.5, label_size=12):
    stroke = PALETTE[color][0]
    dash = ' stroke-dasharray="7 5"' if dashed else ""
    svg = (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
           f'stroke-width="{width}"{dash} marker-end="url(#ar-{color})"/>')
    if label:
        mx = x1 + (x2 - x1) * label_pos
        my = y1 + (y2 - y1) * label_pos - 8
        svg += (f'<rect x="{mx - text_width(label, label_size)/2 - 6}" y="{my - label_size}" '
                f'width="{text_width(label, label_size) + 12}" height="{label_size + 8}" rx="5" fill="{BG}" opacity="0.95"/>')
        svg += text(mx, my + 2, label, label_size, stroke, True, "middle")
    return svg


def svg_page(w, h, body, bg=BG):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<defs>{MARKERS}</defs>
<rect width="{w}" height="{h}" fill="{bg}"/>
{body}
</svg>'''


def save(name, svg):
    p = os.path.join(OUT, name + ".svg")
    with open(p, "w", encoding="utf-8") as f:
        f.write(svg)
    print("saved", p)


# ============================================================
# Fig 1 总架构
# ============================================================
def fig1():
    W, H = 1820, 1130
    b = []
    b.append(text(50, 48, "论文总架构：从多架构算子语料，到可执行知识图谱，再到双平台实验", 27, INK, True))
    b.append(text(50, 80, "输入 = 14 个 BLAS 算子 × 5 种架构实现 ｜ 处理 = S0–S6 七段流水线 ｜ 实验 = E1–E7 × x86 / RISC-V ｜ 输出 = 第 1–6 章", 15, MUTED))

    lanes = [("输入层（数据从哪来）", 40, 390, "blue"),
             ("知识层（KG 怎么建）", 470, 400, "green"),
             ("方法层（怎么查、怎么生成）", 910, 400, "orange"),
             ("实验层（怎么证明有用）", 1350, 430, "purple")]
    for name, x, w, c in lanes:
        b.append(text(x + w / 2, 125, name, 15, PALETTE[c][0], True, "middle"))
        b.append(f'<line x1="{x}" y1="136" x2="{x + w}" y2="136" stroke="{PALETTE[c][0]}" stroke-width="2" opacity="0.35"/>')

    # 输入层
    b.append(box(40, 160, 390, 130, "① 算子语料（14 个）",
                 ["L1：daxpy/dasum/ddot/dscal/dnrm2/daxpby", "L2：dgemv/dsymv/chemv", "L3：dgemm/dtrmm/dsymm",
                  "另有 copy/rot/swap/symm/trmm 等扩展"], "blue"))
    b.append(box(40, 310, 390, 140, "② 架构实现（5 类）",
                 ["x86 AVX2 / AVX512：OpenBLAS C/汇编内核", "ARM NEON / SVE：OpenBLAS C/汇编内核",
                  "RISC-V RVV 1.0（VLEN=256）：C 内核", "generic C：架构无关参考实现"], "blue"))
    b.append(box(40, 470, 390, 120, "③ 数据与证据",
                 ["QEMU 日志 44 个算子：只用于正确性", "真机/云实测：x86 + RISC-V（待跑）", "每个实现带文件 + 行号 + sha256"], "blue"))
    b.append(box(40, 610, 390, 120, "④ 评测任务（OpOptBench）",
                 ["12 个任务 × 3 个尺寸 × 2 个平台", "每个任务 = 通用 C + 接口契约 + 硬件画像", "输入给 LLM 的只有：代码 + 硬件 + 约束"], "blue"))

    # 知识层
    b.append(box(470, 160, 400, 150, "S0–S2 语料处理（规则，无需 LLM）",
                 ["S0 清单：112 个输入文件、sha256、arch/ISA", "S1 静态特征：intrinsic/汇编指令/循环/参数",
                  "S2 差分：同架构 generic vs 优化实现", "产出：corpus / characteristics / diffs"], "green"))
    b.append(box(470, 330, 400, 170, "S3–S4 策略抽取与对齐",
                 ["规则抽事实（结构/参数/行号）；LLM 只做命名与解释", "跨架构同名策略对齐：arch_scope = 无关/参数化/特定",
                  "参数按架构保留：vector_width 128/256/512、RVV LMUL/SEW", "产出：262 条策略 + 262 个实施模板"], "green"))
    b.append(box(470, 520, 400, 130, "S5–S6 性能证据与校验",
                 ["性能证据只来自日志/实测，标 qemu_only 或 measured", "Schema/不变量/边端/闭集校验，0 error",
                  "两次 build 字节级一致（可复现）"], "green"))
    b.append(box(470, 670, 400, 110, "OpKG v1.5 最终规模",
                 ["862 节点 / 3656 边 ｜ 262 策略 / 262 模板", "13/14 算子 ≥2 架构 ｜ 30 条跨架构共有策略",
                  "输出：nodes.jsonl / edges.jsonl / arch_coverage.json"], "green"))

    # 方法层
    b.append(box(910, 160, 400, 170, "7 步知识检索（可解释 trace）",
                 ["① 计算流程匹配 → ② 策略召回 → ③ 条件过滤", "④ 硬件/ISA 过滤 → ⑤ 冲突消解（同族/寄存器预算）",
                  "⑥ 打分排序 → ⑦ 组装 StrategyPack", "每步记录候选/选中/淘汰及原因（可回放）"], "orange"))
    b.append(box(910, 350, 400, 120, "结构化 Prompt（4 个条件）",
                 ["B0 零样本 ｜ B1 + 硬件信息", "B2 扁平文本 RAG（同 token 预算）", "B3 完整 KG（条件 + 模板 + 证据）"], "orange"))
    b.append(box(910, 490, 400, 160, "LLM 生成 + 验证闭环",
                 ["生成单个 C 文件（禁止调用 BLAS）", "编译 → D1–D4 正确性 → 20 次计时",
                  "失败原因回写 FailureCase，供下一轮检索", "输出代码 + 编译日志 + 性能 JSONL"], "orange"))

    # 实验层
    b.append(box(1350, 160, 430, 210, "实验矩阵 E1–E7",
                 ["E1 KG 质量：抽取/对齐 P/R/F1、120 条金标", "E2 策略推荐：P@5 / R@5 / nDCG / 可适用率",
                  "E3 x86 生成 ｜ E4 RISC-V 生成", "E5 跨架构迁移：Kx86 / K_ARM / K_RVV", "E6 消融 ｜ E7 案例与失败分析"], "purple"))
    b.append(box(1350, 390, 430, 140, "指标（每个实验都落到这些数）",
                 ["正确性：CR 编译率、FCR 四分布通过率", "性能：S_C0 / S_C1（主）/ S_P1、EGC、fast_p",
                  "知识：抽取 F1、对齐 P/R、溯源率", "成本：token、时延；统计：3 seeds + 配对检验"], "purple"))
    b.append(box(1350, 550, 430, 110, "三道门（到点执行，不达标就降级）",
                 ["G1 第 2 周：C1 与 P1 差距 ≥2×", "G2 第 5 周：B3/B2 ≥1.05（否则转 Plan B）",
                  "G3 第 8 周：主结果数据齐全，否则冻结范围"], "purple"))
    b.append(box(1350, 680, 430, 100, "论文第 5 章",
                 ["主表：12 算子 × 6 条件 × 2 平台", "图：性能箱线图、消融、迁移三层分类、案例"], "purple"))

    # 列间箭头
    b.append(arrow(430, 300, 470, 300, "gray"))
    b.append(arrow(430, 530, 470, 530, "gray"))
    b.append(arrow(870, 280, 910, 280, "gray"))
    b.append(arrow(870, 600, 910, 600, "gray"))
    b.append(arrow(1310, 280, 1350, 280, "gray"))
    b.append(arrow(1310, 600, 1350, 600, "gray"))
    # 反馈
    b.append(arrow(1110, 650, 1110, 800, "gray", dashed=True))
    b.append(arrow(1110, 800, 870, 780, "gray", dashed=True, label="失败案例回写", label_pos=0.55))
    b.append(text(1115, 816, "验证失败 → FailureCase → 检索/知识修正", 12, PALETTE["gray"][0]))

    # 论文六章
    y0 = 900
    b.append(text(50, y0 - 14, "论文六章与上图对应", 16, INK, True))
    chapters = [
        ("第 1 章 绪论", "背景 / 问题 / 贡献"),
        ("第 2 章 相关技术", "KG、RAG、LLM 代码生成、算子优化"),
        ("第 3 章 KG 构建", "S0–S6 + E1（知识质量）"),
        ("第 4 章 检索与生成", "7 步检索 + Prompt + 闭环 + E2"),
        ("第 5 章 实验与分析", "E3–E7 / 双平台 / 消融 / 迁移"),
        ("第 6 章 总结与展望", "局限、未来工作"),
    ]
    x = 50
    for i, (t1, t2) in enumerate(chapters):
        c = "blue" if i < 2 else ("green" if i == 2 else ("orange" if i == 3 else ("purple" if i == 4 else "gray")))
        b.append(box(x, y0, 270, 92, t1, [t2], c, title_size=15, line_size=12))
        if i < 5:
            b.append(arrow(x + 274, y0 + 46, x + 292, y0 + 46, "gray"))
        x += 292
    b.append(text(50, 1050, "读图顺序：① 左边输入是什么 → ② 中间怎么变成 KG → ③ 怎么检索和生成 → ④ 右边怎么证明有效 → ⑤ 底部对应论文哪一章",
                  13, MUTED))
    save("fig1_论文总架构", svg_page(W, H, "\n".join(b)))


# ============================================================
# Fig 2 输入地图
# ============================================================
def fig2():
    W, H = 1820, 1060
    cov = json.load(open(COV, encoding="utf-8"))
    b = []
    b.append(text(50, 48, "输入到底有什么：算子清单 × 架构覆盖 × 文件结构", 27, INK, True))
    b.append(text(50, 80, "当前输入不是“一个架构的几段代码”，而是 14 个 BLAS 算子在 5 类架构上的实现集合；每个实现都能回指到文件与行号。", 15, MUTED))

    # 左表：算子清单
    x, y, w = 40, 120, 700
    b.append(text(x, y - 10, "A. 算子清单（14 个，按 BLAS 层级）", 17, PALETTE["blue"][0], True))
    rows = [
        ("层级", "算子", "角色", "说明"),
        ("L1 向量", "daxpy dasum ddot dscal dnrm2 daxpby", "S1/S2", "向量-标量运算；编译器 -O3 已很强"),
        ("L1 扩展", "copy rot swap", "S1", "内存/交换类，适合看访存优化"),
        ("L2 矩阵-向量", "dgemv dsymv chemv", "S1/S2", "分块、pack、对称/复数访问"),
        ("L3 矩阵-矩阵", "dgemm dsymm trmm", "S1/S2", "微内核、寄存器分块、cache 分块"),
    ]
    colx = [x, x + 110, x + 430, x + 520]
    yy = y + 20
    for ri, row in enumerate(rows):
        hrow = 54 if ri == 0 else 72
        fill = "#DCE9F7" if ri == 0 else "#FFFFFF"
        b.append(f'<rect x="{x}" y="{yy}" width="{w}" height="{hrow}" rx="8" fill="{fill}" stroke="#CBD5E0"/>')
        for ci, cell in enumerate(row):
            cw = (colx[ci + 1] - colx[ci] - 8) if ci < 3 else (x + w - colx[ci] - 8)
            size = 12.5 if ci == 1 else 13
            weight = ci == 0 or ri == 0
            for k, seg in enumerate(wrap(cell, cw, size)):
                b.append(text(colx[ci] + 6, yy + 24 + k * 16, seg, size, INK if ri == 0 else MUTED, weight))
        yy += hrow
    b.append(text(x, yy + 24, "S1 = 进 KG（知识来源）；S2 = 留出（只用于评测，验证跨算子泛化）", 13, MUTED))

    # 右：架构矩阵
    mx, my = 790, 120
    b.append(text(mx, my - 10, "B. 架构 × 算子覆盖矩阵（单元格 = 实现文件数，颜色越深越多）", 17, PALETTE["green"][0], True))
    matrix = cov["arch_matrix"]
    archs = ["x86_avx2", "x86_avx512", "arm_neon", "arm_sve", "rvv"]
    ops = ["asum", "axpby", "axpy", "copy", "dot", "gemm", "gemv", "hemv", "nrm2", "rot", "scal", "swap", "symm", "symv", "trmm"]
    cw0 = 140
    cw = (W - mx - 60 - cw0) / len(ops)
    rh = 54
    # header
    for j, op in enumerate(ops):
        b.append(text(mx + cw0 + cw * j + cw / 2, my + 36, op, 12, INK, True, "middle"))
    def heat(n):
        return {0: "#FFFFFF", 1: "#C6F6D5", 2: "#9AE6B4", 3: "#68D391"}.get(n, "#48BB78")
    for i, arch in enumerate(archs):
        yrow = my + 50 + i * rh
        b.append(text(mx + 8, yrow + 32, arch, 13, PALETTE["green"][0], True))
        for j, op in enumerate(ops):
            n = matrix.get(arch, {}).get(op, 0)
            cx = mx + cw0 + cw * j
            b.append(f'<rect x="{cx}" y="{yrow}" width="{cw-3}" height="{rh-6}" rx="5" fill="{heat(n)}" stroke="#E2E8F0"/>')
            if n:
                b.append(text(cx + (cw - 3) / 2, yrow + 32, str(n), 14, INK if n < 3 else "#1C4532", True, "middle"))
    b.append(text(mx, my + 50 + 5 * rh + 24, "13/14 个算子有 ≥2 个架构实现（仅 hemv 只有 RVV）；30 条跨架构共有策略（同族不同参数）", 13, MUTED))

    # 下：文件结构示意
    fx, fy = 790, 520
    b.append(text(fx, fy - 10, "C. 一个算子目录长什么样（以 daxpy 为例）", 17, PALETTE["orange"][0], True))
    b.append(box(fx, fy + 6, 640, 230, "13_daxpy_op/",
                 ["daxpy.c          接口层（参数检查 / 分支）", "daxpy_k.c        通用 C 内核（未优化，输入）",
                  "daxpy_k_rvv.c    RVV 内核（优化实现，知识来源）", "zzz_*.i / *.c    预处理/模板文件（可追溯行号）",
                  "daxpy_log.txt    基准日志（QEMU：仅正确性）"], "orange", title_size=15, line_size=13))
    b.append(box(fx, fy + 250, 640, 150, "D. 输入到 KG 时记录什么",
                 ["每个实现：arch / isa / uarch / source_lang（C/汇编）", "每个文件：路径 + sha256 + 行号 + 版本（OpenBLAS 0.3.27）",
                  "每条策略：arch_scope + arch_evidence + 参数 + 证据状态"], "blue", title_size=15, line_size=13))

    # 左下：为什么这么选
    b.append(box(40, 545, 700, 260, "E. 为什么输入要这样设计（非 HPC 背景也能懂）",
                 ["① 算子按 L1/L2/L3 选：L1 看向量化，L2 看访存，L3 看分块，覆盖三类优化。",
                  "② 同一算子要有“通用版 vs 优化版”：两者的差异就是专家做了哪些优化决策。",
                  "③ 多架构：同一策略在 x86/ARM/RVV 上的实现与参数不同，才能回答“能否迁移”。",
                  "④ 证据只认日志/实测：LLM 写的“提升 20%”不算证据，必须能回指到文件与行号。",
                  "⑤ 留出 S2 算子：测试集不参与建图，才能证明“跨算子泛化”不是背答案。"], "blue", title_size=16, line_size=13))
    save("fig2_输入地图", svg_page(W, H, "\n".join(b)))


# ============================================================
# Fig 3 KG 构建流水线
# ============================================================
def fig3():
    W, H = 1820, 1000
    b = []
    b.append(text(50, 48, "知识图谱怎么建：S0–S6 七段流水线（每段的输入 / 方法 / 输出）", 27, INK, True))
    b.append(text(50, 80, "原则：结构事实全部由规则/日志产生；LLM 只负责命名与解释，禁止写性能数字。所有产物可离线重跑、字节级幂等。", 15, MUTED))

    stages = [
        ("S0 语料清单", "输入：OpenBLAS 源码 + 用户语料 + 日志",
         "方法：扫描 / 复制 / 记 sha256、arch、行号", "输出：corpus_manifest.jsonl\n（112 个输入文件）", "blue"),
        ("S1 静态特征", "输入：C / 汇编实现",
         "方法：正则 + 轻量 AST；抽 intrinsic、\nmnemonics、循环、unroll、寄存器块", "输出：characteristics.jsonl\n（371 条静态特征）", "blue"),
        ("S2 差分抽取", "输入：同架构 generic vs 优化实现",
         "方法：C 对做结构差分；汇编-only 抽特征\n（method=diff / static）", "输出：diffs.jsonl\n（18 条差分 + 21 条静态）", "green"),
        ("S3 策略抽取", "输入：差分 + 静态特征 + 算子上下文",
         "方法：规则命名/归类；LLM 只写解释\n（禁止数字，schema 校验+重试）", "输出：strategies_raw.jsonl\n（策略/条件/参数/模板草案）", "green"),
        ("S4 规范化对齐", "输入：原始策略 + 闭集词表",
         "方法：ID 稳定化、同义合并、跨架构对齐\narch_scope = 无关/参数化/特定", "输出：262 条策略 + 262 个模板", "orange"),
        ("S5 性能证据", "输入：QEMU 日志 + 真机实测",
         "方法：解析日志；标 qemu_only / measured；\n只出现在证据节点", "输出：88 条证据\n（含 idamax 0.18× 负例）", "orange"),
        ("S6 校验导出", "输入：全部 JSONL",
         "方法：Schema / 边端 / 闭集 / 冲突对称 /\n版本四元组 + 输入 sha256", "输出：nodes / edges /\nmanifest / arch_coverage", "purple"),
    ]
    x = 40
    w = 236
    gap = 22
    for i, (title, inp, method, out, c) in enumerate(stages):
        y = 130
        b.append(box(x, y, w, 430, title, [], c, title_size=16))
        b.append(text(x + 16, y + 66, "输入", 13, PALETTE[c][0], True))
        for k, seg in enumerate(wrap(inp.replace("输入：", ""), w - 32, 12)):
            b.append(text(x + 16, y + 88 + k * 18, seg, 12, MUTED))
        b.append(text(x + 16, y + 170, "方法", 13, PALETTE[c][0], True))
        for k, seg in enumerate(wrap(method.replace("方法：", ""), w - 32, 12)):
            b.append(text(x + 16, y + 192 + k * 18, seg, 12, MUTED))
        b.append(text(x + 16, y + 320, "输出", 13, PALETTE[c][0], True))
        for k, seg in enumerate(wrap(out.replace("输出：", "").replace("\n", " "), w - 32, 12)):
            b.append(text(x + 16, y + 342 + k * 18, seg, 12, MUTED))
        if i < len(stages) - 1:
            b.append(arrow(x + w + 2, y + 235, x + w + gap - 2, y + 235, "gray", width=2))
        x += w + gap

    # 实体/关系/文件
    b.append(text(40, 610, "Schema v1.5：12 类实体（节点）", 16, PALETTE["green"][0], True))
    entities = [("Operator", "算子"), ("ComputationPattern", "计算流程"), ("CodeCharacteristic", "代码特征"),
                ("OptimizationStrategy", "优化策略"), ("ImplementationTemplate", "实施模板"), ("TunableParameter", "可调参数"),
                ("ISACapability", "指令集能力"), ("Microarchitecture", "微架构"), ("HardwareProfile", "硬件实例"),
                ("PerformanceEvidence", "性能证据"), ("SourceArtifact", "来源"), ("FailureCase", "失败案例")]
    for i, (en, zh) in enumerate(entities):
        cx = 40 + (i % 6) * 290
        cy = 630 + (i // 6) * 52
        b.append(chip(cx, cy, 276, 40, f"{zh}  {en}", "green", 12))
    b.append(text(40, 760, "14 类关系（边）：算子—计算流程—策略—模板/参数/条件/ISA/硬件—证据—来源；另有组合/冲突/泛化/跨架构实现", 14, MUTED))
    b.append(text(40, 790, "输出文件：data/nodes.jsonl（862 节点）｜ data/edges.jsonl（3656 边）｜ data/manifest.json（版本+sha256）｜ data/arch_coverage.json", 14, MUTED))

    # 底部关键说明
    b.append(box(40, 820, 1740, 150, "为什么这样建（答辩可讲的三句话）",
                 ["① 差分而不是“让 LLM 读代码猜策略”：同一算子两版实现的差异可核查、带行号，幻觉空间最小。",
                  "② LLM 只做“起名和解释”，结构事实（dtype、循环深度、参数值、性能数字、行号）全部来自规则或日志。",
                  "③ 多架构对齐后，同一条策略能带不同架构参数（如 unroll=4 vs 8、vector_width=128/256/512），这是“可迁移性”分析的基础。"],
                 "gray", title_size=15, line_size=13))
    save("fig3_KG构建流水线", svg_page(W, H, "\n".join(b)))


# ============================================================
# Fig 4 检索与生成闭环
# ============================================================
def fig4():
    W, H = 1820, 1080
    b = []
    b.append(text(50, 48, "输入进去以后怎么处理：7 步检索 → 结构化 Prompt → 生成与验证闭环", 27, INK, True))
    b.append(text(50, 80, "用一个真实例子走一遍：dgemm，目标平台 x86_avx2。每一步都能在图谱可视化里回放（候选 / 选中 / 淘汰 + 原因）。", 15, MUTED))

    # 查询
    b.append(box(40, 120, 430, 120, "查询（Query）",
                 ["算子：dgemm（L3，double）", "目标硬件：x86_avx2（AVX2/FMA）", "代码特征：dtype=d、loop_depth=3、stride=1、size=large"], "blue"))
    # 7 步
    steps = [
        ("① 计算流程匹配", "算子 → 计算流程（注册表/指纹）", "命中：核心微内核 + pack + 尾处理"),
        ("② 策略召回", "沿 PATTERN_OPTIMIZED_BY 召回 + 泛化扩展", "候选：策略 A/B/C…（含跨架构抽象策略）"),
        ("③ 条件过滤", "对代码特征求值；不满足硬条件即淘汰", "例：dtype 不符 / stride≠1 / 尺寸档不符"),
        ("④ 硬件过滤", "ISA / 微架构 / 向量宽度是否满足", "例：需要 AVX512 的策略在 AVX2 上淘汰"),
        ("⑤ 冲突消解", "同族去重 + CONFLICTS_WITH + 寄存器预算", "例：两个分块策略不能共存；超预算淘汰"),
        ("⑥ 打分排序", "match / fit / hw / evidence / support − 迁移惩罚", "取 top-k（默认 6 条）"),
        ("⑦ 组装 StrategyPack", "策略卡 + 条件 + 参数 + 模板 + 证据 + 来源", "可直接注入 prompt 的结构化知识包"),
    ]
    y = 280
    for i, (t, d, ex) in enumerate(steps):
        c = "orange" if i < 3 else ("purple" if i < 6 else "green")
        b.append(box(40, y, 620, 88, t, [d, ex], c, title_size=15, line_size=12))
        if i < 6:
            b.append(arrow(350, y + 90, 350, y + 108, "gray"))
        y += 108

    # 右侧：prompt + 生成 + 验证
    b.append(box(720, 120, 500, 150, "4 种 Prompt 条件（同 LLM、同 token 预算）",
                 ["B0 零样本：只给代码 + 硬件描述", "B1 + 硬件信息：不给策略知识",
                  "B2 扁平文本 RAG：同内容拍平成文本", "B3 完整 KG：条件 + 模板 + 证据（主方法）"], "orange", title_size=16))
    b.append(arrow(500, 180, 720, 180, "gray"))
    b.append(box(720, 300, 500, 130, "组装 Prompt（B3 示例）",
                 ["系统：你是 BLAS kernel 优化专家", "知识块：策略 1/2/3（条件+参数+模板+来源）",
                  "任务：补全 C 函数，禁止调用 BLAS/内联汇编", "约束：单线程、内存 ≤ 参考实现 2×"], "orange", title_size=15))
    b.append(box(720, 460, 500, 170, "LLM 生成",
                 ["输入：generic.c + 接口契约 + 硬件画像 + StrategyPack", "输出：单个 C 文件（仅函数体）",
                  "同任务最多 3 个候选（pass@1 主指标）", "失败可回灌编译错误 1 轮（等预算）"], "orange", title_size=15))
    b.append(box(720, 660, 500, 170, "验证闭环（自动）",
                 ["编译：x86 gcc / RISC-V 交叉编译", "正确性：D1 随机 / D2 不规则尺寸 / D3 参数边界 / D4 数值极端",
                  "性能：5 warmup + 20 次中位数，固定线程/频率", "失败 → FailureCase 回写，供检索修正"], "purple", title_size=15))
    # 失败回写：绕到图下方，避免压住字
    b.append(f'<path d="M 970 832 L 970 892 L 750 892 L 750 664" fill="none" '
             f'stroke="{PALETTE["gray"][0]}" stroke-width="2.2" stroke-dasharray="7 5" '
             f'marker-end="url(#ar-gray)"/>')
    b.append(text(700, 912, "验证失败 → FailureCase → 检索/知识修正", 12, PALETTE["gray"][0]))

    # 右侧结果
    b.append(box(1280, 120, 500, 190, "产出（每次实验都落 JSONL）",
                 ["生成代码 + 编译日志 + 运行日志", "指标：CR / FCR / S_C0 / S_C1 / EGC / fast_p",
                  "token 成本 / 时延 / 失败类型", "可追溯到：任务、条件、seed、平台、日期"], "purple", title_size=16))
    b.append(box(1280, 340, 500, 160, "例：dgemm@x86_avx2 的 trace 实际数字",
                 ["条件过滤：15 条淘汰（dtype 不符等）", "硬件过滤：4 条淘汰（需要 AVX512/仅 RVV）",
                  "冲突消解：4 条淘汰（同族/寄存器预算）", "最终 top-6 策略进入 prompt"], "green", title_size=15))
    b.append(box(1280, 540, 500, 160, "同一套流程查 RISC-V 时",
                 ["profile 换成 k1_rvv256（RVV 1.0 / VLEN=256）", "过滤条件换成 SEW/LMUL/向量寄存器",
                  "策略参数换成 RVV 版本（如 LMUL=2/8）", "生成后 riscv64-gcc 交叉编译 + 板端计时"], "green", title_size=15))
    b.append(box(1280, 740, 500, 120, "为什么需要 7 步（而不是一次向量检索）",
                 ["向量检索只能“找相似文本”；条件/硬件/冲突是硬约束", "7 步把“知识选择”变成可审计、可回放的决策链"],
                 "gray", title_size=14))
    save("fig4_检索与生成闭环", svg_page(W, H, "\n".join(b)))


# ============================================================
# Fig 5 实验怎么做
# ============================================================
def fig5():
    W, H = 1820, 1140
    b = []
    b.append(text(50, 48, "实验怎么做：7 个实验 × 6 个条件 × 2 个平台（附单次实验步骤）", 27, INK, True))
    b.append(text(50, 80, "目标不是“超过 OpenBLAS”，而是回答：KG 知识是否提升生成代码的正确性/性能、相对扁平 RAG 的增量、以及能否跨架构迁移。", 15, MUTED))

    # 平台
    b.append(box(40, 120, 560, 150, "平台（性能结论只来自真机）",
                 ["x86：本机 / 服务器，-O3 -march=native，单线程 & 4 线程", "RISC-V：Bianbu Cloud K1 或自购 BPI-F3/RV2（RVV 1.0，VLEN=256）",
                  "QEMU：只做正确性前置，禁止写进性能表", "固定：编译器版本、频率、线程数、OpenBLAS 版本"], "blue", title_size=16))
    # 条件
    b.append(box(640, 120, 540, 150, "6 个主条件（每个实验都同 LLM/同 token/同尝试）",
                 ["C0 输入代码 -O2 ｜ C1 编译器 -O3（最重要下界）", "P1 OpenBLAS 专家实现（上界/EGC 分母）",
                  "B0 零样本 ｜ B1 +硬件信息", "B2 扁平文本 RAG ｜ B3 完整 KG（主方法）"], "orange", title_size=15))
    b.append(box(1220, 120, 560, 150, "指标（每个数字都要能追溯到 JSONL）",
                 ["正确性：CR 编译率、FCR 四分布全过率", "性能：S_C0、S_C1（主）、S_P1、EGC、fast_p",
                  "知识：抽取/对齐 P/R/F1、金标 κ、溯源率", "成本：token/任务、时延；统计：3 seeds + 配对检验"], "purple", title_size=16))

    # E1-E7 表
    b.append(text(40, 330, "A. 七个实验：每个实验回答一个问题、产出一张表/图", 17, PALETTE["purple"][0], True))
    rows = [
        ("实验", "回答什么问题", "怎么做（规模）", "产出", "失败时怎么办"),
        ("E1 KG 质量", "知识抽得准不准、可不可溯源？", "6 个 S1 算子，120 条人工金标，2 人标注", "抽取/对齐 P/R/F1、覆盖率、κ", "降到 60 条抽检"),
        ("E2 策略推荐", "给定代码+硬件，能否推荐对策略？", "12 算子 × 2 profile × 4 条件（不执行）", "P@5/R@5/nDCG/SAR/PIV", "只报 top-3 可适用率"),
        ("E3 x86 生成", "生成代码正确吗？比编译器快吗？", "12 算子 × 6 条件 × 3 尺寸 × 3 seeds", "CR/FCR/S_C1/EGC/成本", "缩到 8 算子"),
        ("E4 RISC-V 生成", "换成 RVV 平台还成立吗？", "12 算子 × 4 条件 × 2 尺寸（真机）", "同上 + RVV 指令使用率", "缩到 6 算子，QEMU 只验正确性"),
        ("E5 跨架构迁移", "哪些策略能迁移、边界在哪？", "6 算子 × 3 知识源（Kx86/KARM/KRVV）", "迁移保真度 + 三层分类表", "只用 E2 推荐结果做分析"),
        ("E6 消融", "增量到底来自哪一部分？", "B2 vs B3、去硬件、去条件、去模板、随机子图", "ΔGM 表", "只做 3 组核心消融"),
        ("E7 案例与失败", "为什么成功/失败？", "dgemm/daxpy/dtrmm + 20 个失败样本", "3 案例图 + 失败类型分布", "保留 1 个案例"),
    ]
    x0, y0 = 40, 350
    widths = [160, 330, 500, 380, 370]
    for ri, row in enumerate(rows):
        yy = y0 + ri * 66
        fill = "#DCE9F7" if ri == 0 else "#FFFFFF"
        b.append(f'<rect x="{x0}" y="{yy}" width="{sum(widths)}" height="66" rx="8" fill="{fill}" stroke="#CBD5E0"/>')
        cx = x0
        for ci, cell in enumerate(row):
            for k, seg in enumerate(wrap(cell, widths[ci] - 12, 12.5)):
                b.append(text(cx + 6, yy + 24 + k * 15, seg, 12.5, INK if ri == 0 else MUTED, ri == 0))
            cx += widths[ci]

    # 单次实验流程
    b.append(text(40, 900, "B. 单次实验怎么做（以 E3 一个任务为例）", 17, PALETTE["orange"][0], True))
    flow = [
        ("① 取任务", "OpOptBench 任务：generic.c + 接口契约 + 尺寸"),
        ("② 建 Prompt", "按 B0/B1/B2/B3 生成 4 份 prompt（同预算）"),
        ("③ 生成", "LLM 输出 C 文件（每条件 3 个候选）"),
        ("④ 编译", "x86 gcc / riscv64-gcc，记录编译日志"),
        ("⑤ 正确性", "D1–D4 四组隐藏测试，全部通过才算 FCR"),
        ("⑥ 计时", "5 warmup + 20 次中位数，固定线程/频率"),
        ("⑦ 落库", "写 JSONL：任务/条件/seed/平台/指标"),
    ]
    x = 40
    w = 225
    for i, (t1, t2) in enumerate(flow):
        b.append(box(x, 920, w, 110, t1, [t2], "orange", title_size=14, line_size=12))
        if i < len(flow) - 1:
            b.append(arrow(x + w + 2, 975, x + w + 18, 975, "gray"))
        x += w + 20

    b.append(box(40, 1050, 1740, 70, "统计与门禁",
                 ["每个条件 3 个 seed；主对比用配对 Wilcoxon + bootstrap 95% CI；消融用 Holm–Bonferroni。G1：C1 vs P1 差距 ≥2×；G2：B3/B2 ≥1.05；G3：数据齐全，否则冻结范围。"],
                 "gray", title_size=14, line_size=13))
    save("fig5_实验怎么做", svg_page(W, H, "\n".join(b)))


# ============================================================
# Fig 6 理论机制：KG 为什么可能提升性能
# ============================================================
def fig6():
    W, H = 1820, 980
    b = []
    b.append(text(50, 48, "理论机制：知识图谱为什么可能提升生成代码的性能（5 条机制 + 5 个可证伪假设）", 26, INK, True))
    b.append(text(50, 80, "不是“KG 自动提升性能”，而是 KG 把优化决策从“不可检索的经验”变成“可约束、可实例化、可核查的候选规划”。", 15, MUTED))

    # 左：问题
    b.append(box(40, 130, 380, 250, "问题：LLM 的优化决策空间",
                 ["给定代码 x 与硬件 h，可选策略 S × 参数 Θ × 组合顺序", "LLM 先验不知道：哪些策略可行、参数取多少、哪些互斥",
                  "扁平文本 RAG 只能“找相似段落”，不能精确判断可行性", "结果：编译失败、数值错误、性能回退"],
                 "red", title_size=16, line_size=13))
    # 中：KG 五条机制
    mech = [
        ("① 覆盖（Coverage）", "把长尾策略收进 KG：算子×架构×策略", "H1：覆盖↑ → 候选召回↑ → 命中率↑", "blue"),
        ("② 过滤（Feasibility）", "硬条件 + ISA/寄存器/缓存容量过滤", "H2：不可行策略↓ → 编译/正确率↑", "green"),
        ("③ 实例化（Parameterization）", "参数由硬件推导：VLEN/LMUL、寄存器数、块大小", "H3：参数错误↓ → 性能↑", "orange"),
        ("④ 组合（Composition）", "COMPOSES/CONFLICTS + 寄存器预算，避免负交互", "H4：坏组合↓ → 性能方差↓", "purple"),
        ("⑤ 迁移（Transfer）", "arch_scope：无关/参数化/特定 → 目标架构冷启动", "H5：迁移保真度可测 → 新架构增益", "teal"),
    ]
    x = 450
    for i, (t, d, h, c) in enumerate(mech):
        y = 128 + i * 88
        b.append(box(x, y, 760, 78, t, [d, h], c, title_size=15, line_size=12))
        b.append(arrow(420, 255, x - 2, y + 39, "gray", dashed=(i > 0), width=1.6))
    # 右：结果
    b.append(box(1250, 128, 530, 430, "结果：LLM 收到的是“可行候选规划”",
                 ["StrategyPack：6 条策略 × {条件, 参数, 模板, 证据, 来源}", "LLM 只负责“把规划写成 C 代码”，不再做可行性判断",
                  "生成代码 → 编译/正确性/计时 → 指标", "性能提升 = 候选更可行 + 参数更对 + 组合更稳 + 迁移更顺",
                  "", "Flat RAG：给的是“相似文本”，LLM 仍需自己判断可行/冲突",
                  "KG：给的是“已经过滤/实例化/排序过的规划”，判断成本转移给图"], "purple", title_size=16, line_size=13))
    b.append(arrow(1210, 340, 1250, 340, "gray"))

    # 下：形式化 + 指标映射
    b.append(box(40, 590, 760, 180, "形式化（写进论文第 3 章）",
                 ["任务：y = LLM(x, h, K)，K 为知识；候选空间 S×Θ，可行性 F = {s | C(s; x,h)=true}",
                  "KG 提供：候选 S_KG、硬约束 C、参数映射 θ(h)、组合/冲突关系、证据排序",
                  "提升条件：κ = |S_feasible ∩ S_KG| / |S_feasible| 足够大，且过滤/参数/组合机制有效",
                  "可证伪预测：L1（编译器已饱和）增益小；L2/L3（分块/寄存器）增益大——这本身就是边界结论"], "blue", title_size=15, line_size=13))
    b.append(box(830, 590, 950, 180, "假设 → 实验 → 指标（每条都能被数据推翻）",
                 ["H1 覆盖 → E2 策略推荐（R@5 / nDCG）；E3 生成命中率", "H2 过滤 → E6 A2/A3（去硬件/去条件）→ CR/FCR 变化",
                  "H3 参数 → E6 A4（去参数）→ S_C1 / EGC 变化", "H4 组合 → E6 A8（去冲突消解）→ 性能方差 / 回退率",
                  "H5 迁移 → E5 三层分类表 + 双平台 S_C1"], "green", title_size=15, line_size=13))

    b.append(text(50, 810, "一句话答辩：我不是声称“图一定比文本好”，而是提出 5 条可测机制，并用 E2/E5/E6 的消融逐条验证；如果某条不成立，就如实报告边界。",
                  14, INK, True))
    b.append(text(50, 840, "注：图中 H1–H5 为“机制假设”，不是定理；论文写“机制分析 + 可证伪预测 + 实验验证”，不写“证明 KG 必然提升性能”。",
                  13, MUTED))
    save("fig6_理论机制", svg_page(W, H, "\n".join(b)))


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    print("done")
