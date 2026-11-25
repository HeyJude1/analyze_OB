#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成"知识图谱构建流程"图（SVG + PNG）。

复用 14_论文架构图/gen_figs.py 的绘图工具与配色。
输出：科研产出/29_建图流程图/fig_建图流程.svg / .png
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "14_论文架构图"))
import gen_figs as G  # noqa: E402

OUT = HERE
W, H = 1690, 1235
b = []

RED = G.PALETTE["red"][0]


def save(name, svg):
    p = os.path.join(OUT, name + ".svg")
    with open(p, "w", encoding="utf-8") as f:
        f.write(svg)
    print("saved", p)


def section(y, no, title, sub, color):
    b.append(G.text(50, y, "%s %s" % (no, title), 17, G.PALETTE[color][0], True))
    b.append(G.text(50 + G.text_width("%s %s" % (no, title), 17) + 14, y, sub, 13, G.MUTED))
    b.append(f'<line x1="50" y1="{y + 10}" x2="{W - 50}" y2="{y + 10}" '
             f'stroke="{G.PALETTE[color][0]}" stroke-width="1.4" opacity="0.25"/>')


# ---------------------------------------------------------------- 标题
b.append(G.text(50, 50, "知识图谱构建流程：从多架构源码，到可执行、可追溯的策略图谱", 27, G.INK, True))
b.append(G.text(50, 82,
                "两条抽取通道（规则 / LLM 提议）产出同一种策略卡 ｜ 验证器只做否决 ｜ 通过才建图 ｜ 产物可直接页面展示",
                14, G.MUTED))

# ---------------------------------------------------------------- ① 输入
section(122, "①", "输入", "源码 / 硬件 / 证据 / 提示词——事实只能来自这四类", "blue")

b.append(G.box(50, 140, 300, 132, "源码：generic C", [
    "每个算子一份架构无关实现",
    "作为差分基准与被优化对象",
    "例：daxpy_k.c（14 个算子）",
], "blue", title_size=15, line_size=12))

b.append(G.box(368, 140, 330, 132, "源码：多架构优化实现", [
    "x86 AVX2 / AVX512、ARM NEON / SVE、RVV 1.0",
    "C 与内联汇编皆收；112 个输入带 sha256",
    "登记：arch / ISA / 起止行号",
], "blue", title_size=15, line_size=12))

b.append(G.box(716, 140, 300, 132, "硬件画像（能力约束）", [
    "6 个 profile：AVX2/AVX512/NEON/SVE/RVV/QEMU",
    "VLEN、寄存器、cache、编译器、测量环境",
    "不含性能结果——性能在右侧「证据源」",
], "blue", title_size=15, line_size=12))

b.append(G.box(1034, 140, 300, 132, "证据源", [
    "QEMU 日志 CSV（仅正确性/算子级线索）",
    "真机计时（x86 / K1）",
    "每条证据：metric/value/platform/protocol",
], "blue", title_size=15, line_size=12))

b.append(G.box(1352, 140, 300, 132, "提示词与配置", [
    "analysis/*.yaml：算法/代码/指令三级分析",
    "kg/*.yaml：原则/特征/消歧",
    "schema/词表：实体、关系、条件谓词",
], "blue", title_size=15, line_size=12))

# ---------------------------------------------------------------- ② 处理
section(300, "②", "处理（S0–S2）", "全部规则，不需要 LLM；产出可比对的结构化特征", "gray")

b.append(G.box(50, 318, 500, 150, "S0 / S0b 语料登记", [
    "扫描算子目录，登记文件、sha256、行号、arch/ISA",
    "多架构源自动发现并复制到自包含目录",
    "输出：corpus_manifest.jsonl（112 个输入）",
], "gray", title_size=15, line_size=12))

b.append(G.box(590, 318, 480, 150, "S1 静态特征", [
    "C：intrinsic、循环层数、展开因子、打包、预取、对齐",
    "汇编：mnemonic、向量宽度、FMA、寄存器分块线索",
    "输出：characteristics.jsonl（893 条，每条指回行号）",
], "gray", title_size=15, line_size=12))

b.append(G.box(1110, 318, 530, 150, "S2 差分", [
    "同算子：generic vs optimized 逐段对比",
    "记录：多了哪些指令、展开从几变几、尾部/布局怎么改",
    "输出：diffs.jsonl（51 条；C 差分 18 / 汇编静态 21）",
], "gray", title_size=15, line_size=12))

b.append(G.arrow(550, 393, 588, 393, "gray"))
b.append(G.arrow(1070, 393, 1108, 393, "gray"))

# ---------------------------------------------------------------- ③ 抽取
section(492, "③", "抽取", "两条通道并行，产出同一种「策略卡」；LLM 只提议，不做最终裁决", "orange")

b.append(G.box(50, 512, 760, 168, "通道 A：规则 / 差分（现状，201 条 = 77%）", [
    "static/diff 66 条：从 S1/S2 直接映射动作与参数",
    "multiarch 99 条：按架构实现对比，标 arch_scope",
    "cross_arch 30 条：跨架构同族合并（parameterized）",
    "abstract 6 条：手写抽象策略（INSTANCE_OF 挂到实例）",
    "优点：确定、可复现、零幻觉 ｜ 缺点：只认写死的模式",
], "green", title_size=15, line_size=12))

b.append(G.box(830, 512, 810, 168, "通道 B：LLM 提议（v1 61 条 → 新契约改造）", [
    "现状 v1：61 条（23%）已入图，但无行号、条件为散文——问题最集中",
    "改造输入：带行号片段 + generic/optimized diff（不再拼接全文）",
    "强制举证：evidence_quote + line_start/end；条件须附 justification_quote",
    "禁止性能数字；参数给 value_in_source；输出与通道 A 同构",
], "orange", title_size=15, line_size=12))
b.append(G.chip(1548, 522, 76, 22, "新增", "orange", 11, True))

# 统一数据契约（条带）
b.append(f'<rect x="50" y="692" width="1530" height="34" rx="8" fill="{G.PALETTE["gray"][1]}" '
         f'stroke="{G.PALETTE["gray"][0]}" stroke-width="1.2"/>')
b.append(G.text(64, 714,
                "统一数据契约：12 类实体 · 14 类关系 · 10 个条件谓词 ｜ 每条实体 = id + type + properties + provenance(source_file, line_start–end, line_confidence, version) + status",
                12, G.MUTED))

# ---------------------------------------------------------------- ④ 验证
section(762, "④", "验证（可证伪的检查）", "规则不判断\"好不好\"，只判断\"你的主张有没有被源码支持\"", "red")

b.append(G.box(50, 780, 780, 150, "验证器：逐条卡对照源码", [
    "R1 引文在行内：evidence_quote 必须出现在 line_start–line_end",
    "R2 行号合法；R3 精度一致（pd/ps ↔ dtype）；R4 对齐一致（vmovups ↔ aligned）",
    "R5 位宽一致（xmm/ymm/zmm ↔ 128/256/512）；R6 展开/步长一致（addq $16 ↔ 展开×每向量元素）",
    "其余：ISA 兼容 / 参数在声明域 / 类型闭集 / 同义合并",
], "red", title_size=15, line_size=12))
b.append(G.box(860, 780, 780, 150, "判定与去向", [
    "PASS → 进入规范化与建图；每条保留引用与行号，可回放",
    "FAIL → rejected.jsonl（带原因码），不静默丢弃",
    "拒绝集回灌通道 B：同一错误模式 ≥3 次即改提示词，再跑",
    "人工只抽检：50 条 / 批，算 precision；<0.8 停止交付",
], "red", title_size=15, line_size=12))

# 回灌虚线：从判定框右上角绕右边界回到 LLM 通道
b.append(f'<path d="M 1600 780 H 1662 V 600 H 1648" fill="none" stroke="{RED}" '
         f'stroke-width="2" stroke-dasharray="7 5" marker-end="url(#ar-red)"/>')

# ---------------------------------------------------------------- ⑤ 建图与产物
section(960, "⑤", "建图与产物", "通过验证的卡才允许进图；产物可重建、可展示、可回放", "green")

b.append(G.box(50, 978, 360, 136, "S4 规范化与合并（部分实现）", [
    "当前：ID 唯一化 + 参数合并 + 抽象归类",
    "缺失：alias 表 / merges.json / 合并 P-R",
    "目标：按 (op, action, level, arch) 合并",
], "green", title_size=15, line_size=12))

b.append(G.box(440, 978, 360, 136, "S5 证据挂接", [
    "证据节点绑定 metric/value/platform",
    "策略级 vs 算子级 scope 必须标注",
    "无证据的性能声明 → claim_status=unverified",
], "green", title_size=15, line_size=12))

b.append(G.box(830, 978, 360, 136, "S6 建图与校验", [
    "节点 / 边 / manifest（版本四元组）",
    "schema + 不变量 + 边端点 + 冲突对称",
    "两次构建字节级一致（可复现）",
], "green", title_size=15, line_size=12))

b.append(G.box(1220, 978, 420, 136, "产物", [
    "kg.json（实体 + 关系 + run 清单）",
    "quality.json（门 A–E 报告）",
    "graph.html（单文件、离线可开）",
    "traces/（每次检索的 7 步回放）",
], "green", title_size=15, line_size=12))

b.append(G.arrow(410, 1046, 438, 1046, "gray"))
b.append(G.arrow(800, 1046, 828, 1046, "gray"))
b.append(G.arrow(1190, 1046, 1218, 1046, "gray"))

# ---------------------------------------------------------------- 底部：规模 + 下游 + 图例
chips = [("14 算子", "blue"), ("6 硬件 profile", "blue"), ("262 条策略", "green"),
         ("862 节点 / 3656 边", "green"), ("129/262 有证据", "red"), ("multiarch 2/99", "red")]
x = 50
for label, c in chips:
    w = G.text_width(label, 12) + 26
    b.append(G.chip(x, 1128, w, 24, label, c, 12, False))
    x += w + 12

b.append(f'<rect x="50" y="1166" width="1590" height="34" rx="8" fill="{G.PALETTE["purple"][1]}" '
         f'stroke="{G.PALETTE["purple"][0]}" stroke-width="1.2"/>')
b.append(G.text(64, 1188,
                "下游使用：七步检索（约束选择）→ 策略包注入 → LLM 生成 → x86 / RISC-V 双平台实验（第 4–6 章）",
                12, G.PALETTE["purple"][0], True))
b.append(G.arrow(1430, 1114, 1430, 1164, "purple", dashed=True))

legend = [("输入", "blue"), ("规则处理", "gray"), ("LLM 提议", "orange"),
          ("验证/门", "red"), ("图与产物", "green"), ("下游", "purple")]
x = 50
b.append(G.text(x, 1222, "图例：", 12, G.MUTED))
x += 40
for label, c in legend:
    b.append(f'<rect x="{x}" y="1212" width="14" height="10" rx="3" '
             f'fill="{G.PALETTE[c][1]}" stroke="{G.PALETTE[c][0]}" stroke-width="1.2"/>')
    b.append(G.text(x + 20, 1222, label, 11, G.MUTED))
    x += 20 + G.text_width(label, 11) + 24

save("fig_建图流程", G.svg_page(W, H, "\n".join(b)))

# ---------------------------------------------------------------- 转 PNG
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
svg_path = os.path.join(OUT, "fig_建图流程.svg")
png_path = os.path.join(OUT, "fig_建图流程.png")
if os.path.exists(CHROME):
    url = "file:///" + svg_path.replace("\\", "/")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    "--force-device-scale-factor=1.6",
                    "--window-size=%d,%d" % (W, H),
                    "--screenshot=" + png_path, url], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("png:", png_path, os.path.getsize(png_path) if os.path.exists(png_path) else "MISSING")
