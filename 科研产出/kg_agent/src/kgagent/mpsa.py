"""MPSA（Multi-Perspective Strategy Analysis）：B3/B4 多视角策略分析与代码生成。

B3：
  把 `retrieve.retrieve(cfg, op, profile, top_k)` 返回的 top-k StrategyPack 直接拼进生成 prompt，
  一次 LLM 调用生成完整 C 函数。

B4：
  1) 对 top-`analysis_k` 个策略各做一次"适用性分析"（LLM，严格 JSON：applies/evidence/risks/parameters/conflicts）；
  2) 再按 4 个标准（supportive / complementarity / accuracy / completeness）综合出报告（LLM，严格 JSON）；
  3) 用「策略 pack + 逐条分析 + 综合报告」生成完整 C 函数（LLM）。

修复回路（`repair_code` / CLI `mpsa-repair`）：
  生成代码编译失败或运行结果错误时，把「原代码 + 编译器 stderr / QEMU maxdiff 输出 + 任务签名 + profile」
  回灌给 LLM 做 1 轮修复；prompt 明确要求「不确定的 RVV intrinsic 一律退回标量实现」，并给出 GCC 13
  实测可编译的 intrinsic 白名单与已证伪的幻觉名字黑名单。

LLM provider：
  - `dashscope`：DashScope OpenAI 兼容端点（默认模型 qwen-plus-2025-04-28），
    key 只从环境变量 `DASHSCOPE_API_KEY` 读取（**绝不写死、绝不落盘**），标准库 urllib.request 实现；
  - `mock`：完全离线、确定性输出（token 计 0，calls 记真实调用次数），可用于实验链路自测；
    `--mock-code FILE` 可指定返回的代码；否则 daxpy/ddot 内置正确标量实现，其它算子回退基线函数。

输出 `mpsa_result.json`：op/op_canonical/profile/mode/provider/model/retrieved/analysis/synthesis/
prompt/code/code_meta/usage/elapsed_s/created_at/artifacts；
`repair_result.json`：code/usage/code_meta/input_error/fix_summary/elapsed_s/...。

代码清洗：去 markdown 代码围栏 / 去掉函数外的解释文字 / 只保留目标 `_k` 函数（及前置 #include/typedef）/
若返回只有函数体则用任务签名包裹 / 补最小 `typedef long long BLASLONG;` 前导（使 generated.c 可独立编译）/
括号配平；保证 daxpy 产物可编译。

CLI 见 `python -m kgagent.cli mpsa --help` 与 `python -m kgagent.cli mpsa-repair --help`。
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from . import retrieve as retrieve_mod

__all__ = [
    "run_mpsa",
    "repair_code",
    "clean_code",
    "extract_signature",
    "resolve_signature",
    "load_prompt",
    "render_prompt",
    "MPSAError",
    "MPSAConfigError",
    "MissingAPIKeyError",
    "LLMError",
    "RetrievalError",
    "CodeCleanError",
    "MODES",
    "PROVIDERS",
    "CRITERIA",
    "DEFAULT_MODEL",
]

#: 4 个综合标准的闭集（synthesis 输出必须逐一给分）
CRITERIA = ("supportive", "complementarity", "accuracy", "completeness")

MODES = ("b3", "b4")
PROVIDERS = ("dashscope", "deepseek", "mock")

DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus-2025-04-28"
DEFAULT_API_KEY_ENV = "DASHSCOPE_API_KEY"

# DeepSeek OpenAI 兼容端点（用户 2026-11 指定；key 只从环境变量读）
DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEFAULT_TIMEOUT_S = 120
#: 输出 token 上限（显式给，避免 DashScope 默认值把生成代码截断）
MAX_TOKENS_ANALYSIS = 1536
MAX_TOKENS_SYNTHESIS = 1536
MAX_TOKENS_GENERATE = 4096
MAX_TOKENS_REPAIR = 4096

#: prompt 模板目录（kg_agent/prompts/）
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

#: 没有 --code 时的任务签名兜底（与 OpOptBench tasks/*/input/generic.c、driver 对齐）
BASELINE_SIGNATURES = {
    "daxpy": (
        "int daxpy_k(BLASLONG n, BLASLONG d0, BLASLONG d1, double da, double *x, "
        "BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG d2)"
    ),
    "axpy": (
        "int daxpy_k(BLASLONG n, BLASLONG d0, BLASLONG d1, double da, double *x, "
        "BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG d2)"
    ),
    # 现有 OpOptBench ddot generic.c 为 double 返回值签名（无输出指针）
    "ddot": "double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)",
    "dot": "double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)",
}

CRITERIA_DEF = """\
- supportive（相互支持性）：各策略的适用条件/参数是否一致、能否同时采用，是否会互相抵消；
- complementarity（互补性）：策略是否作用于不同阶段（向量化 / 循环展开 / 尾部处理 / 寄存器分块 / 步长特化等），组合后是否覆盖更多场景；
- accuracy（正确性/数值一致性）：组合实现是否保持 BLAS 语义与数值容差，n<=0 / 尾部 / 任意步长 / 别名等边界是否安全；
- completeness（覆盖完整性）：是否覆盖主循环、尾部、任意步长、n<=0 等分支，是否遗漏关键实现环节。"""

SYSTEM_ANALYSIS = (
    "你是 BLAS 算子优化策略的适用性分析专家。只输出严格 JSON 对象，"
    "不要输出 markdown 代码围栏、解释文字或 C 代码。"
)
SYSTEM_SYNTHESIS = (
    "你是 BLAS 算子优化策略的综合专家。只输出严格 JSON 对象，"
    "不要输出 markdown 代码围栏、解释文字或 C 代码。"
)
SYSTEM_GENERATE = (
    "你是 OpenBLAS 风格的高性能 BLAS 内核 C 语言实现工程师。只输出一个完整、可独立编译的 C 函数；"
    "不要输出 main、不要输出 BLAS 头文件、不要输出 markdown 代码围栏或任何解释文字。"
)
SYSTEM_REPAIR = (
    "你是高性能 BLAS 内核的 C 语言修复工程师。根据编译器/运行错误修复给定 C 函数：只输出一个完整、"
    "可独立编译的 C 函数；不要输出 main、不要输出 BLAS 头文件、不要输出 markdown 代码围栏或解释文字；"
    "不确定的向量 intrinsic 一律退回纯标量实现，正确性优先。"
)


# ---------------------------------------------------------------- 异常（带 CLI 退出码）


class MPSAError(RuntimeError):
    """MPSA 基类异常；exit_code 供 CLI 映射退出码。"""

    exit_code = 2


class MPSAConfigError(MPSAError):
    """参数/配置错误（mode/provider/文件不存在等）。"""

    exit_code = 2


class MissingAPIKeyError(MPSAError):
    """dashscope provider 缺少环境变量 API key。"""

    exit_code = 3


class LLMError(MPSAError):
    """LLM 调用失败 / 返回内容不合法。"""

    exit_code = 4


class RetrievalError(MPSAError):
    """检索失败（未知算子/profile、图数据为空、无策略）。"""

    exit_code = 5


class CodeCleanError(MPSAError):
    """生成的代码无法清洗成完整 C 函数。"""

    exit_code = 2


# ---------------------------------------------------------------- prompt 模板


_PLACEHOLDER_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")


def load_prompt(name: str) -> str:
    """读取 `kg_agent/prompts/<name>.md` 模板。"""
    path = PROMPTS_DIR / ("%s.md" % name)
    if not path.exists():
        raise MPSAConfigError("缺少 prompt 模板：%s" % path)
    return path.read_text(encoding="utf-8")


def render_prompt(name: str, **values) -> str:
    """把模板中的 `{{KEY}}` 替换为 values；缺参数时明确报错（避免生成空 prompt）。"""
    template = load_prompt(name)
    placeholders = set(_PLACEHOLDER_RE.findall(template))
    missing = sorted(placeholders - set(values))
    if missing:
        raise MPSAConfigError("prompt 模板 %s 缺少参数：%s" % (name, ", ".join(missing)))
    return _PLACEHOLDER_RE.sub(lambda m: str(values[m.group(1)]), template)


# ---------------------------------------------------------------- 通用小工具


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _as_str_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _as_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _strip_code_fences(text: str) -> tuple:
    """去掉 markdown 代码围栏；返回 (纯文本, 是否去掉了围栏)。"""
    if "```" not in text:
        return text, False
    match = re.search(r"```[A-Za-z0-9_+.\-]*[ \t]*\r?\n(.*?)```", text, re.S)
    if match:
        return match.group(1), True
    # 围栏不完整（只有开头）：逐行删掉围栏行，保留其余内容
    lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
    return "\n".join(lines), True


# ---------------------------------------------------------------- 函数签名


def extract_signature(source: str):
    """从 C 源码中提取第一个函数定义（优先 `_k` 函数）的签名，返回单行字符串或 None。"""
    if not source:
        return None
    text = source.replace("\r\n", "\n").replace("\r", "\n")
    sig = _extract_signature_with_re(text, require_k=True)
    if sig:
        return sig
    return _extract_signature_with_re(text, require_k=False)


_SIG_RE_K = re.compile(
    r"(?m)^[ \t]*(?P<ret>[A-Za-z_][A-Za-z0-9_ \t\*]*?)[ \t]+(?P<name>[A-Za-z_]\w*_k)[ \t]*"
    r"\((?P<params>[^;{}]*?)\)[ \t\r\n]*\{",
    re.S,
)
_SIG_RE_ANY = re.compile(
    r"(?m)^[ \t]*(?P<ret>[A-Za-z_][A-Za-z0-9_ \t\*]*?)[ \t]+(?P<name>[A-Za-z_]\w*)[ \t]*"
    r"\((?P<params>[^;{}]*?)\)[ \t\r\n]*\{",
    re.S,
)


def _extract_signature_with_re(text: str, require_k: bool):
    regex = _SIG_RE_K if require_k else _SIG_RE_ANY
    for match in regex.finditer(text):
        ret = _collapse(match.group("ret"))
        name = match.group("name")
        params = [_fmt_param(p) for p in _split_params(match.group("params"))]
        params = [p for p in params if p]
        if not ret or not name:
            continue
        return "%s %s(%s)" % (ret, name, ", ".join(params))
    return None


def _split_params(text: str) -> list:
    """按顶层逗号切分参数列表（忽略括号/方括号内的逗号）。"""
    params = []
    depth = 0
    current = []
    for ch in text or "":
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            params.append("".join(current))
            current = []
        else:
            current.append(ch)
    if "".join(current).strip():
        params.append("".join(current))
    return params


def _fmt_param(param: str) -> str:
    text = _collapse(param)
    if not text or text == "void":
        return "" if text == "void" else text
    text = re.sub(r"\s*\*\s*", " *", text)      # `double * x` -> `double *x`
    text = re.sub(r"\s*\[\s*", "[", text)
    text = re.sub(r"\s*\]", "]", text)
    return text


def parse_signature(signature: str) -> dict:
    """把单行签名解析成 {ret, name, params:[{type,stars,name,is_pointer}]}。"""
    text = _collapse(signature)
    match = re.match(r"^(?P<ret>.*?)\b(?P<name>[A-Za-z_]\w*)\s*\((?P<params>.*)\)\s*$", text)
    if not match:
        raise CodeCleanError("无法解析函数签名：%r" % (signature,))
    params = []
    for raw in _split_params(match.group("params")):
        parsed = _parse_param(raw)
        if parsed:
            params.append(parsed)
    return {
        "ret": _collapse(match.group("ret")),
        "name": match.group("name"),
        "params": params,
        "signature": text,
    }


_PARAM_RE = re.compile(
    r"^(?P<prefix>.*?)\s*(?P<name>[A-Za-z_]\w*)\s*(?P<array>\[[^\]]*\])?\s*$"
)


def _parse_param(raw: str):
    text = _collapse(raw)
    if not text or text == "void":
        return None
    match = _PARAM_RE.match(text)
    if not match:
        return {"type": text, "stars": 0, "name": "arg", "is_pointer": False, "decl": text}
    prefix = _collapse(match.group("prefix"))
    stars = prefix.count("*") + (1 if match.group("array") else 0)
    base = _collapse(prefix.replace("*", " "))
    return {
        "type": base,
        "stars": stars,
        "name": match.group("name"),
        "is_pointer": stars > 0,
        "decl": text,
    }


def resolve_signature(op: str, canonical: str | None = None, baseline_code: str | None = None):
    """任务签名来源优先级：--code 基线代码 > 内置签名表（按请求名/KG 名）。找不到返回 None。"""
    if baseline_code:
        signature = extract_signature(baseline_code)
        if signature:
            return signature
    for key in (op, canonical):
        key = (key or "").strip().lower()
        if key in BASELINE_SIGNATURES:
            return BASELINE_SIGNATURES[key]
    return None


# ---------------------------------------------------------------- C 代码清洗


def _code_indices(text: str):
    """产出不在注释/字符串里的字符下标（用于括号配平与函数边界扫描）。"""
    index = 0
    length = len(text)
    in_string = None
    in_line_comment = False
    in_block_comment = False
    while index < length:
        ch = text[index]
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
        elif in_block_comment:
            if ch == "*" and index + 1 < length and text[index + 1] == "/":
                in_block_comment = False
                index += 1
        elif in_string:
            if ch == "\\":
                index += 1
            elif ch == in_string:
                in_string = None
        else:
            if ch in "\"'":
                in_string = ch
            elif ch == "/" and index + 1 < length and text[index + 1] == "/":
                in_line_comment = True
                index += 1
            elif ch == "/" and index + 1 < length and text[index + 1] == "*":
                in_block_comment = True
                index += 1
            else:
                yield index
        index += 1


def _find_definition_brace(text: str, start: int):
    """从签名之后找到函数体左花括号（先遇到 ';' 说明只是声明，返回 None）。"""
    for index in _code_indices(text):
        if index < start:
            continue
        ch = text[index]
        if ch == "{":
            return index
        if ch == ";":
            return None
    return None


def _scan_matching_brace(text: str, open_index: int):
    depth = 0
    started = False
    for index in _code_indices(text):
        if not started:
            if index == open_index:
                started = True
                depth = 1
            continue
        ch = text[index]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _scan_matching_brace_backward(text: str, close_index: int):
    """从 '}' 往回找与它匹配的 '{'。"""
    depth = 0
    started = False
    for index in reversed(list(_code_indices(text))):
        if not started:
            if index == close_index:
                started = True
                depth = 1
            continue
        ch = text[index]
        if ch == "}":
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                return index
    return None


def _signature_start_before(text: str, open_index: int):
    """给定函数体 '{'，往回定位函数签名起点（上一个顶层 ';' / '}' / '{' 之后）。"""
    boundary = max(
        text.rfind(";", 0, open_index),
        text.rfind("}", 0, open_index),
        text.rfind("{", 0, open_index),
    )
    start = boundary + 1
    while start < open_index and text[start] in " \t\r\n":
        start += 1
    snippet = text[start:open_index + 1]
    if not re.search(r"[A-Za-z_]\w*\s*\([^;{}]*\)\s*\{", snippet, re.S):
        return None
    return start


def _gap_allows_join(segment: str) -> bool:
    """两个函数定义之间的间隙只允许空行/注释/预处理行，才把它们并进同一段代码。"""
    segment = (segment or "").strip()
    if not segment:
        return True
    for line in segment.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("#", "//", "/*", "*")) or stripped.endswith("*/"):
            continue
        return False
    return True


def _collect_code_region(text: str, target_span: tuple) -> tuple:
    """把目标函数前紧邻的 static 辅助函数也并进代码段（prompt 允许把 helper 放在目标函数之前）。"""
    start, end = target_span
    region_start = start
    while True:
        prefix = text[:region_start]
        trimmed = prefix.rstrip()
        if not trimmed.endswith("}"):
            break
        close_index = len(trimmed) - 1
        open_index = _scan_matching_brace_backward(text, close_index)
        if open_index is None:
            break
        helper_start = _signature_start_before(text, open_index)
        if helper_start is None:
            break
        if not _gap_allows_join(text[close_index + 1:region_start]):
            break
        region_start = helper_start
    return region_start, end


def _find_function_span(text: str, function_name: str):
    """定位函数定义 `(start, end)`（end 为右花括号后一位）；找不到返回 None。"""
    pattern = re.compile(
        r"(?m)^[ \t]*(?:[A-Za-z_][\w \t\*]*[ \t]+)?%s[ \t]*\(" % re.escape(function_name)
    )
    for match in pattern.finditer(text):
        if _line_is_comment(text, match.start()):
            continue
        open_index = _find_definition_brace(text, match.end())
        if open_index is None:
            continue
        close_index = _scan_matching_brace(text, open_index)
        if close_index is None:
            continue
        return match.start(), close_index + 1
    return None


def _line_is_comment(text: str, position: int) -> bool:
    """粗略判断 position 是否落在注释里（行注释 / /* */ 内）。"""
    line_start = text.rfind("\n", 0, position) + 1
    prefix = text[line_start:position]
    if "//" in prefix:
        return True
    block_open = prefix.rfind("/*")
    block_close = prefix.rfind("*/")
    return block_open > block_close


_FUNCTION_RE = re.compile(
    r"(?m)^[ \t]*(?P<ret>[A-Za-z_][A-Za-z0-9_ \t\*]*?)[ \t]+(?P<name>[A-Za-z_]\w*)[ \t]*"
    r"\((?P<params>[^;{}]*?)\)[ \t\r\n]*\{",
    re.S,
)


def _find_functions(text: str) -> list:
    """扫描所有函数定义，返回 [{name, arity, start, end, signature}]（按出现顺序）。"""
    functions = []
    seen = set()
    for match in _FUNCTION_RE.finditer(text):
        name = match.group("name")
        if name in ("if", "for", "while", "switch", "return", "sizeof"):
            continue
        open_index = match.end() - 1
        if open_index < 0 or text[open_index] != "{":
            continue
        close_index = _scan_matching_brace(text, open_index)
        if close_index is None:
            continue
        start = match.start()
        if start in seen:
            continue
        seen.add(start)
        params = [p for p in _split_params(match.group("params")) if p.strip()]
        ret = _collapse(match.group("ret"))
        functions.append({
            "name": name,
            "arity": len(params),
            "start": start,
            "end": close_index + 1,
            "signature": "%s %s(%s)" % (ret, name, ", ".join(_fmt_param(p) for p in params)),
        })
    return functions


def _pick_target_function(functions: list, expected_parsed: dict):
    """在模型返回的多个函数里挑目标函数（优先名字含算子词干，其次参数个数一致，取最后一个）。"""
    if not functions:
        return None
    expected_name = expected_parsed["name"]
    expected_arity = len(expected_parsed["params"])
    stem = (expected_name.split("_")[0] or "").lower()
    stem_matches = [item for item in functions if stem and stem in item["name"].lower()]
    if stem_matches:
        return stem_matches[-1]
    same_arity = [item for item in functions if item["arity"] == expected_arity]
    if same_arity:
        return same_arity[-1]
    if len(functions) == 1:
        return functions[0]
    return None


def _prelude_before(text: str, function_start: int) -> list:
    """收集函数定义前紧邻的 #include/#define/#pragma/typedef（及注释）行。"""
    lines = text[:function_start].split("\n")
    collected = []
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^(#include|#define|#pragma|typedef\b)", stripped) or stripped.startswith(
            ("//", "/*", "*")
        ):
            collected.append(line.rstrip())
            continue
        break
    return list(reversed(collected))


def _strip_prose(text: str) -> str:
    """从"只有函数体"的回复里去掉前面的解释文字：从第一行像 C 语句的行开始。"""
    lines = text.strip().split("\n")
    start = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("#", "/*", "//")):
            start = index
            break
        if re.search(r"[;{}]", stripped) or re.match(
            r"^(return|for|while|if|else|double|float|int|long|size_t|BLASLONG)\b", stripped
        ):
            start = index
            break
    else:
        return ""
    return "\n".join(lines[start:]).strip()


def _indent(text: str, spaces: int = 4) -> str:
    prefix = " " * spaces
    return "\n".join((prefix + line) if line.strip() else line for line in text.split("\n"))


def _brace_depth(text: str) -> tuple:
    """返回 (净深度, 是否出现过负深度)。"""
    depth = 0
    negative = False
    for index in _code_indices(text):
        ch = text[index]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                negative = True
    return depth, negative


def clean_code(raw_text: str, op: str | None = None, canonical: str | None = None,
               signature: str | None = None, baseline_code: str | None = None) -> dict:
    """把 LLM/文件返回清洗成一个完整、可独立编译的 C 函数。

    返回 dict：code / signature / function_name / wrapped_body / fences_removed /
    prologue_added / repairs / char_count。
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise CodeCleanError("待清洗的代码为空")
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text, fences_removed = _strip_code_fences(text)

    resolved = signature
    if not resolved:
        resolved = (
            extract_signature(text)
            or extract_signature(baseline_code or "")
            or resolve_signature(op, canonical)
        )
    if not resolved:
        raise CodeCleanError(
            "无法确定目标函数签名：请提供 --code（generic.c 基线）或 --mock-code，"
            "或使用内置签名的算子（daxpy/ddot）"
        )
    parsed = parse_signature(resolved)
    function_name = parsed["name"]

    code = text.strip()
    repairs = []
    wrapped_body = False

    span = _find_function_span(code, function_name)
    functions = _find_functions(code)
    if span is None and functions:
        # 容错：模型把目标函数改了名（如 daxpy_opt / daxpy_kernel）→ 按算子词干/参数个数识别并重命名
        candidate = _pick_target_function(functions, parsed)
        if candidate is not None and candidate["name"] != function_name:
            old_name = candidate["name"]
            code = re.sub(r"\b%s\b" % re.escape(old_name), function_name, code)
            repairs.append("renamed function %s -> %s" % (old_name, function_name))
            span = _find_function_span(code, function_name)
            functions = _find_functions(code)
    if span is None and functions:
        raise CodeCleanError(
            "模型返回了 %d 个函数（%s），但无法确定哪个是目标函数 %r：请检查/重试生成"
            % (len(functions), ", ".join(item["name"] for item in functions), function_name)
        )

    if span is not None:
        start, end = _collect_code_region(code, span)
        prelude = _prelude_before(code, start)
        function_text = code[start:end]
        code = "\n".join(prelude + [function_text]).strip()
    else:
        body = _strip_prose(code)
        if not body:
            raise CodeCleanError("无法从返回内容中定位完整 C 函数，也没有可包裹的函数体")
        if body.lstrip().startswith("{"):
            code = resolved + "\n" + body
        else:
            code = resolved + "\n{\n" + _indent(body) + "\n}"
        wrapped_body = True
        repairs.append("wrapped body with task signature")

    # 补最小 typedef，保证 generated.c 作为独立编译单元也能编过
    prologue_added = False
    if re.search(r"\bBLASLONG\b", code) and not re.search(
        r"(typedef[^;\n]*\bBLASLONG\b[^;\n]*;|#define\s+BLASLONG\b)", code
    ):
        code = "typedef long long BLASLONG;\n\n" + code
        prologue_added = True

    depth, negative = _brace_depth(code)
    if negative:
        raise CodeCleanError("生成的 C 代码括号不平衡（出现多余的 '}'）")
    if depth > 0:
        code = code.rstrip() + "\n" + "}" * depth
        repairs.append("appended %d closing brace(s)" % depth)
    if function_name not in code:
        raise CodeCleanError("清洗后的代码中找不到函数 %r" % function_name)

    code = code.rstrip() + "\n"
    return {
        "code": code,
        "signature": _collapse(resolved),
        "function_name": function_name,
        "wrapped_body": wrapped_body,
        "fences_removed": fences_removed,
        "prologue_added": prologue_added,
        "repairs": repairs,
        "char_count": len(code),
    }


# ---------------------------------------------------------------- mock 代码生成


def _mock_code_text(op: str, canonical: str | None, signature: str | None,
                    baseline_code: str | None = None,
                    mock_code_path: str | None = None) -> str:
    """mock provider 的"生成代码"来源（确定性，不联网）。

    优先级：--mock-code 文件 > daxpy/ddot 内置正确标量实现 > 基线 generic.c 原函数。
    """
    if mock_code_path:
        path = Path(mock_code_path)
        if not path.exists():
            raise MPSAConfigError("--mock-code 文件不存在：%s" % path)
        return path.read_text(encoding="utf-8", errors="replace")
    if signature is None:
        raise MPSAConfigError(
            "mock provider 需要 --code（generic.c）或 --mock-code 才能确定函数签名/实现"
        )
    parsed = parse_signature(signature)
    kind = _op_kind(op, canonical, parsed["name"])
    if kind in ("axpy", "dot"):
        return _builtin_mock_code(parsed, kind)
    if baseline_code:
        return baseline_code
    raise CodeCleanError(
        "mock provider 暂不为 op=%r 内置实现：请用 --mock-code 指定代码，或用 --code 提供 "
        "generic.c 让 mock 直接回退到基线函数" % op
    )


def _op_kind(op: str, canonical: str | None, function_name: str) -> str:
    haystack = " ".join(str(x or "") for x in (op, canonical, function_name)).lower()
    if "dot" in haystack:
        return "dot"
    if "axpy" in haystack:
        return "axpy"
    return "other"


def _find_inc(params: list, wanted: str, pointer_param: dict) -> dict:
    for param in params:
        if param["name"] == wanted:
            return param
    try:
        start = params.index(pointer_param)
    except ValueError:
        start = 0
    for param in params[start + 1:]:
        if not param["is_pointer"]:
            return param
    raise CodeCleanError("无法从签名中确定步长参数 %r：%s" % (wanted, pointer_param["name"]))


def _return_stmt(ret_type: str, value: str) -> str:
    if re.search(r"\bvoid\b", ret_type or ""):
        return "return;"
    return "return %s;" % value


def _builtin_mock_code(parsed: dict, kind: str) -> str:
    """为 daxpy/ddot 生成正确的标量实现（签名取自任务 generic.c，语义与基线一致）。"""
    signature = parsed["signature"]
    ret_type = parsed["ret"]
    params = parsed["params"]
    pointers = [p for p in params if p["is_pointer"]]
    if not pointers:
        raise CodeCleanError("内置 mock 实现要求至少一个指针参数：%s" % signature)
    n_param = next((p for p in params if p["name"] == "n"), params[0])
    n_name, n_type = n_param["name"], (n_param["type"] or "BLASLONG")
    x_param = pointers[0]
    y_param = pointers[1] if len(pointers) > 1 else None
    if y_param is None:
        raise CodeCleanError("内置 mock 实现要求两个向量指针参数：%s" % signature)
    inc_x = _find_inc(params, "inc_x", x_param)
    inc_y = _find_inc(params, "inc_y", y_param)
    ret0 = _return_stmt(ret_type, "0")

    if kind == "dot":
        out_param = None
        for candidate in pointers[2:]:
            if candidate["name"] in ("dot", "result", "sum", "res"):
                out_param = candidate
                break
        if out_param is None and len(pointers) > 2:
            out_param = pointers[2]
        lines = [
            signature,
            "{",
            "    %s i;" % n_type,
            "    %s ix = 0;" % n_type,
            "    %s iy = 0;" % n_type,
            "    double acc = 0.0;",
            "",
            "    for (i = 0; i < %s; i++) {" % n_name,
            "        acc += %s[ix] * %s[iy];" % (x_param["name"], y_param["name"]),
            "        ix += %s;" % inc_x["name"],
            "        iy += %s;" % inc_y["name"],
            "    }",
        ]
        if re.search(r"\b(double|float)\b", ret_type or ""):
            lines.append("    return acc;")
        else:
            target = out_param["name"] if out_param else None
            if target is None:
                raise CodeCleanError("int ddot_k 签名缺少输出指针：%s" % signature)
            lines.append("    *%s = acc;" % target)
            lines.append("    " + ret0)
        lines.append("}")
        return _normalize_function_text("\n".join(lines))

    # axpy
    da_param = None
    for param in params:
        if param["is_pointer"]:
            continue
        if re.search(r"\b(double|float)\b", param["type"] or "") and param["name"] != n_name:
            da_param = param
            break
    if da_param is None:
        raise CodeCleanError("内置 mock daxpy 实现找不到标量参数 da：%s" % signature)
    lines = [
        signature,
        "{",
        "    %s i;" % n_type,
        "    %s ix = 0;" % n_type,
        "    %s iy = 0;" % n_type,
        "",
        "    if (%s <= 0) %s" % (n_name, ret0),
        "    if (%s == 0.0) %s" % (da_param["name"], ret0),
        "",
        "    for (i = 0; i < %s; i++) {" % n_name,
        "        %s[iy] += %s * %s[ix];" % (y_param["name"], da_param["name"], x_param["name"]),
        "        ix += %s;" % inc_x["name"],
        "        iy += %s;" % inc_y["name"],
        "    }",
        "    " + ret0,
        "}",
    ]
    return _normalize_function_text("\n".join(lines))


def _normalize_function_text(text: str) -> str:
    text = text.replace("\r\n", "\n").strip()
    return text + "\n"


# ---------------------------------------------------------------- mock 分析/综合


def _strategy_conditions_text(strategy: dict) -> str:
    conditions = strategy.get("conditions") or []
    if not conditions:
        return "（无显式条件）"
    return "、".join("%s %s %r" % (c.get("pred"), c.get("op"), c.get("value")) for c in conditions)


def _mock_analysis(strategy: dict, index: int, profile: str) -> dict:
    """确定性 mock 适用性分析（内容无随机性；不依赖网络）。"""
    params = {}
    for param in strategy.get("parameters") or []:
        key = param.get("name") or (param.get("id") or "").split(":")[-1]
        value = param.get("value_in_instance")
        if value is None:
            value = param.get("value_in_source")
        if key and value is not None:
            params[key] = value
    if not params:
        params = {"unroll_factor": 4}
    score = _as_float(strategy.get("score"), 0.0) or 0.0
    return {
        "strategy_id": strategy["id"],
        "strategy_name": strategy.get("name"),
        "applies": True,
        "confidence": round(max(0.5, 0.9 - 0.05 * index), 4),
        "evidence": [
            "mock: 策略条件 %s 与目标 profile=%s 的默认 code_features 匹配（确定性占位证据，非真实测量）"
            % (_strategy_conditions_text(strategy), profile),
            "mock: 检索分数 score=%.4f，排序第 %d 位" % (score, index + 1),
        ],
        "risks": ["mock: 未在真实硬件/真实 API 下验证，收益与参数需实测校正"],
        "parameters": params,
        "conflicts": [],
        "raw": {"mock": True, "note": "provider=mock 的确定性占位分析"},
    }


def _mock_synthesis(analyses: list) -> dict:
    ids = [item["strategy_id"] for item in analyses]
    criteria = {
        "supportive": {"score": 0.82, "reason": "mock: 各策略条件一致、无硬冲突，可同时采用"},
        "complementarity": {"score": 0.76, "reason": "mock: 向量化/尾部处理/LMUL 调优作用于不同阶段，互补"},
        "accuracy": {"score": 0.90, "reason": "mock: 保留边界检查与标量语义，正确性风险可控"},
        "completeness": {"score": 0.72, "reason": "mock: 覆盖主循环/尾部/任意步长，未覆盖别名与对齐特化"},
    }
    return {
        "criteria": criteria,
        "criteria_scores": {key: criteria[key]["score"] for key in CRITERIA},
        "overall": "mock: 建议按检索顺序组合向量化内核 + 动态 vl 尾部处理 + LMUL/展开参数调优。",
        "recommended": ids,
        "guidance": (
            "mock: 1) 用 vsetvl 动态长度 + vle/vse 访存 + vfmacc 累加；"
            "2) 尾部用掩码处理；3) 保持 n<=0 与任意 inc 步长语义；4) 参数按 profile 实测调优。"
        ),
        "conflicts": [],
        "mock": True,
    }


# ---------------------------------------------------------------- 修复输入/结果辅助


#: 用于 fix_summary 统计修复后代码里出现的 RVV intrinsic
_INTRINSIC_RE = re.compile(r"__riscv_[A-Za-z0-9_]+")

#: BLAS 算子名（用于从 dgemm/daxpy 这类请求名推出 KG 规范名）
_KNOWN_OP_NAMES = {
    "asum", "axpy", "axpby", "copy", "dot", "gemm", "gemv", "hemv",
    "nrm2", "rot", "scal", "swap", "symm", "symv", "trmm",
}


def _canonical_op_name(op: str) -> str:
    """`daxpy` -> `axpy`、`dgemm` -> `gemm`；未知名字原样返回。"""
    name = (op or "").strip().lower()
    if len(name) > 1 and name[0] in "sdcz" and name[1:] in _KNOWN_OP_NAMES:
        return name[1:]
    return name


def _load_text_arg(value, label: str) -> dict:
    """CLI 的代码/错误参数：已存在文件则读文件，否则按内联文本处理。

    返回 {"text": str, "source": 路径或 "inline", "from_file": bool}。
    像路径但不存在时报清晰错误（避免把路径字符串当错误文本喂给 LLM）。
    """
    if value is None:
        return {"text": "", "source": "inline", "from_file": False}
    text = str(value)
    path = Path(text)
    try:
        is_file = path.is_file()
    except OSError:
        is_file = False
    if is_file:
        return {
            "text": path.read_text(encoding="utf-8", errors="replace"),
            "source": str(path),
            "from_file": True,
        }
    looks_like_path = (
        "\n" not in text
        and len(text) < 400
        and (
            "/" in text
            or "\\" in text
            or text.lower().endswith(
                (".log", ".txt", ".err", ".stderr", ".stdout", ".c", ".h", ".json")
            )
        )
    )
    if looks_like_path:
        raise MPSAConfigError("%s 文件不存在：%s" % (label, text))
    return {"text": text, "source": "inline", "from_file": False}


def _mock_repair_code(op: str, canonical: str | None, signature: str | None,
                      original_code: str) -> str:
    """mock provider 的确定性"修复"：daxpy/ddot 返回正确标量实现；其它算子原样返回。"""
    resolved = signature or extract_signature(original_code or "")
    if resolved:
        try:
            parsed = parse_signature(resolved)
        except CodeCleanError:
            parsed = None
        if parsed is not None:
            kind = _op_kind(op, canonical, parsed["name"])
            if kind in ("axpy", "dot"):
                return _builtin_mock_code(parsed, kind)
    return original_code or ""


def _fix_summary(original_code: str, repaired_code: str, provider: str) -> dict:
    """修复摘要（供实验侧记录：是否变化/长度/intrinsic 使用情况/mock 标记）。"""
    return {
        "changed": (original_code or "").strip() != (repaired_code or "").strip(),
        "original_chars": len(original_code or ""),
        "repaired_chars": len(repaired_code or ""),
        "intrinsics_used": sorted(set(_INTRINSIC_RE.findall(repaired_code or ""))),
        "mock": provider == "mock",
    }


# ---------------------------------------------------------------- 结果字段归一化


def normalize_analysis(payload, strategy: dict, index: int) -> dict:
    """把 LLM 返回的分析 JSON 归一化为固定字段（applies/evidence/risks/parameters/conflicts）。"""
    if not isinstance(payload, dict):
        raise LLMError("analysis 返回不是 JSON 对象：%r" % (payload,))
    applies = payload.get("applies", payload.get("applicable", True))
    if isinstance(applies, str):
        applies = applies.strip().lower() not in ("false", "no", "none", "0", "不适用", "否")
    return {
        "strategy_id": strategy["id"],
        "strategy_name": strategy.get("name"),
        "applies": bool(applies),
        "confidence": _as_float(payload.get("confidence"), None),
        "evidence": _as_str_list(payload.get("evidence")),
        "risks": _as_str_list(payload.get("risks")),
        "parameters": payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {},
        "conflicts": _as_str_list(payload.get("conflicts")),
        "raw": payload,
    }


def normalize_synthesis(payload) -> dict:
    """把 LLM 返回的综合报告归一化，强制 4 个标准齐全、分数落在 [0,1]。"""
    if not isinstance(payload, dict):
        raise LLMError("synthesis 返回不是 JSON 对象：%r" % (payload,))
    criteria_raw = payload.get("criteria")
    if not isinstance(criteria_raw, dict):
        criteria_raw = {key: payload.get(key) for key in CRITERIA if key in payload}
    criteria = {}
    for key in CRITERIA:
        entry = criteria_raw.get(key)
        if isinstance(entry, dict):
            score = _as_float(entry.get("score"), _as_float(entry.get("value"), None))
            reason = entry.get("reason") or entry.get("rationale") or entry.get("explanation") or ""
        elif isinstance(entry, (int, float)) and not isinstance(entry, bool):
            score = float(entry)
            reason = ""
        else:
            score = None
            reason = ""
        if score is None:
            raise LLMError("synthesis 缺少标准 %r 的 score（4 标准必须逐一给分）" % key)
        criteria[key] = {
            "score": round(min(1.0, max(0.0, float(score))), 4),
            "reason": str(reason),
        }
    return {
        "criteria": criteria,
        "criteria_scores": {key: criteria[key]["score"] for key in CRITERIA},
        "overall": str(payload.get("overall") or payload.get("summary") or ""),
        "recommended": _as_str_list(
            payload.get("recommended") or payload.get("recommended_strategies")
        ),
        "guidance": str(payload.get("guidance") or payload.get("implementation_guidance") or ""),
        "conflicts": _as_str_list(payload.get("conflicts")),
    }


def _analyses_for_prompt(analyses: list) -> list:
    """进 synthesis prompt 的分析结果去掉 raw，避免重复 token。"""
    return [{key: value for key, value in item.items() if key != "raw"} for item in analyses]


# ---------------------------------------------------------------- usage 统计


def _empty_usage() -> dict:
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "by_stage": {"analysis": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                  "total_tokens": 0},
                     "synthesis": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                   "total_tokens": 0},
                     "generate": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                  "total_tokens": 0}},
    }


def _add_usage(usage: dict, raw_usage: dict | None, stage: str) -> None:
    tokens = _normalize_usage(raw_usage or {})
    usage["calls"] += 1
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        usage[key] += tokens[key]
    bucket = usage["by_stage"].setdefault(
        stage, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    bucket["calls"] += 1
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        bucket[key] += tokens[key]


def _normalize_usage(raw: dict) -> dict:
    prompt = int(raw.get("prompt_tokens") or 0)
    completion = int(raw.get("completion_tokens") or 0)
    total = int(raw.get("total_tokens") or (prompt + completion))
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


# ---------------------------------------------------------------- LLM 客户端


class DashScopeClient:
    """DashScope OpenAI 兼容 chat/completions 客户端（标准库 urllib，无第三方依赖）。"""

    def __init__(self, model: str, api_base: str, api_key: str, timeout_s: int = DEFAULT_TIMEOUT_S):
        self.model = model
        self.endpoint = api_base.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.timeout_s = int(timeout_s)

    def chat(self, prompt: str, system: str, json_mode: bool = False,
             stage: str = "generate", temperature: float = 0.0,
             max_tokens: int | None = None) -> tuple:
        body = {
            "model": self.model,
            "temperature": float(temperature),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if max_tokens:
            body["max_tokens"] = int(max_tokens)
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer %s" % self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = json.loads(response.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:400]
            except Exception:  # noqa: BLE001 - 错误体读取失败不应掩盖 HTTP 错误
                detail = ""
            raise LLMError(
                "DashScope HTTP %s（stage=%s，model=%s）：%s"
                % (exc.code, stage, self.model, detail or exc.reason)
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise LLMError(
                "DashScope 请求失败（stage=%s，model=%s）：%s" % (stage, self.model, reason)
            ) from exc
        except json.JSONDecodeError as exc:
            raise LLMError("DashScope 响应不是合法 JSON（stage=%s）：%s" % (stage, exc)) from exc
        choices = raw.get("choices") or []
        if not choices:
            raise LLMError("DashScope 响应缺少 choices（stage=%s）：%s" % (
                stage, json.dumps(raw, ensure_ascii=False)[:300]))
        content = ((choices[0] or {}).get("message") or {}).get("content") or ""
        if not str(content).strip():
            raise LLMError("DashScope 返回空内容（stage=%s）" % stage)
        return str(content), _normalize_usage(raw.get("usage") or {})


def _parse_json_reply(text: str, stage: str) -> dict:
    cleaned, _ = _strip_code_fences(text)
    cleaned = cleaned.strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise LLMError("LLM 返回中没有 JSON 对象（stage=%s）：%r" % (stage, text[:200]))
        snippet = cleaned[start:end + 1]
        try:
            payload = json.loads(snippet)
        except json.JSONDecodeError as exc:
            raise LLMError(
                "LLM 返回不是合法 JSON（stage=%s）：%s；片段：%r" % (stage, exc, snippet[:200])
            ) from exc
    if not isinstance(payload, dict):
        raise LLMError("LLM 返回的 JSON 顶层不是对象（stage=%s）：%r" % (stage, payload))
    return payload


# ---------------------------------------------------------------- 内部：常量/上下文


def _mpsa_settings(cfg, provider: str = "dashscope") -> dict:
    mpsa_cfg = dict(cfg.raw.get("mpsa") or {})
    llm_cfg = dict(cfg.llm_cfg or {})
    if provider == "deepseek":
        return {
            "model": os.environ.get("MPSA_DEEPSEEK_MODEL") or DEEPSEEK_MODEL,
            "api_base": os.environ.get("MPSA_DEEPSEEK_API_BASE") or DEEPSEEK_API_BASE,
            "api_key_env": DEEPSEEK_API_KEY_ENV,
            "timeout_s": int(mpsa_cfg.get("timeout_s") or llm_cfg.get("timeout_s") or DEFAULT_TIMEOUT_S),
            "top_k": int(mpsa_cfg.get("top_k") or 0) or None,
            "analysis_k": int(mpsa_cfg.get("analysis_k") or 0) or None,
        }
    return {
        "model": mpsa_cfg.get("model") or DEFAULT_MODEL,
        "api_base": mpsa_cfg.get("api_base") or llm_cfg.get("api_base") or DEFAULT_API_BASE,
        "api_key_env": mpsa_cfg.get("api_key_env") or llm_cfg.get("api_key_env")
        or DEFAULT_API_KEY_ENV,
        "timeout_s": int(mpsa_cfg.get("timeout_s") or llm_cfg.get("timeout_s") or DEFAULT_TIMEOUT_S),
        "top_k": int(mpsa_cfg.get("top_k") or 0) or None,
        "analysis_k": int(mpsa_cfg.get("analysis_k") or 0) or None,
    }


def _strategy_json(strategy: dict) -> str:
    return json.dumps(strategy, ensure_ascii=False, indent=2, sort_keys=False)


def _strategies_json(strategies: list) -> str:
    return json.dumps(strategies, ensure_ascii=False, indent=2, sort_keys=False)


def _compact_strategy(strategy: dict) -> dict:
    return {
        "id": strategy.get("id"),
        "name": strategy.get("name"),
        "family": strategy.get("family"),
        "action": strategy.get("action"),
        "level": strategy.get("level"),
        "level_stage": strategy.get("stage"),
        "arch_scope": strategy.get("arch_scope"),
        "score": strategy.get("score"),
    }


def _prompt_context(op: str, canonical: str, profile: str, features: dict,
                    signature: str | None, baseline_code: str | None,
                    top_k: int, analysis_k: int) -> dict:
    baseline = (baseline_code or "").strip() or "（未提供基线代码；请保持 BLAS 算子等价语义）"
    return {
        "OP": op,
        "OP_CANONICAL": canonical,
        "PROFILE": profile,
        "FEATURES_JSON": json.dumps(features, ensure_ascii=False, indent=2, sort_keys=True),
        "SIGNATURE": signature or "（未提供基线签名；请自行给出该算子的完整 C 函数签名）",
        "BASELINE": baseline,
        "TOP_K": top_k,
        "ANALYSIS_K": analysis_k,
        "CRITERIA_DEF": CRITERIA_DEF,
    }


# ---------------------------------------------------------------- 写盘


def write_result(result: dict, out_dir) -> dict:
    """写 <out_dir>/mpsa_result.json 与 <out_dir>/generated.c，返回 artifacts 字典。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    generated_path = out / "generated.c"
    result_path = out / "mpsa_result.json"
    generated_path.write_text(result["code"], encoding="utf-8", newline="\n")
    artifacts = {
        "out_dir": str(out),
        "result_path": str(result_path),
        "generated_c_path": str(generated_path),
    }
    result["artifacts"] = artifacts
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8", newline="\n",
    )
    return artifacts


# ---------------------------------------------------------------- 主入口


def run_mpsa(cfg, op: str, profile: str, mode: str = "b3", provider: str = "dashscope",
             model: str | None = None, code_path=None, top_k: int | None = None,
             analysis_k: int | None = None, out_dir=None, mock_code_path=None,
             timeout_s: int | None = None, data_dir=None, write: bool = True,
             temperature: float = 0.0) -> dict:
    """跑一次 MPSA（B3 或 B4），返回结果 dict（write=True 且给定 out_dir 时落盘）。

    参数：
      cfg            kgagent.config.Config
      op             算子名（daxpy / ddot / dgemm ...）
      profile        硬件 profile（k1_rvv256 / x86_avx2 ...）
      mode           "b3" | "b4"
      provider       "dashscope" | "deepseek" | "mock"
      model          DashScope 模型（默认 qwen-plus-2025-04-28）
      code_path      基线 generic.c（提供任务签名 + prompt 语义参考）
      top_k          检索策略数（默认 config.mpsa.top_k / retrieval.top_k / 6）
      analysis_k     B4 逐条分析条数（默认 config.mpsa.analysis_k / 3）
      out_dir        输出目录（写 mpsa_result.json + generated.c）
      mock_code_path mock provider 指定返回代码文件
      timeout_s      LLM 超时（秒）
      data_dir       图数据目录（测试/实验隔离用）
      write          是否落盘（默认 True；需要 out_dir）
      temperature    LLM temperature（默认 0，保证可复现）
    """
    started = time.perf_counter()
    mode = (mode or "b3").strip().lower()
    provider = (provider or "dashscope").strip().lower()
    if mode not in MODES:
        raise MPSAConfigError("未知 mode=%r；可选：%s" % (mode, " / ".join(MODES)))
    if provider not in PROVIDERS:
        raise MPSAConfigError("未知 provider=%r；可选：%s" % (provider, " / ".join(PROVIDERS)))
    settings = _mpsa_settings(cfg, provider)
    model = (model or settings["model"] or DEFAULT_MODEL).strip()
    top_k = int(top_k or settings["top_k"] or cfg.retrieval_cfg.get("top_k") or 6)
    analysis_k = int(analysis_k or settings["analysis_k"] or 3)
    if top_k < 1:
        raise MPSAConfigError("--top-k 必须 >= 1")
    if analysis_k < 1:
        raise MPSAConfigError("--analysis-k 必须 >= 1")
    timeout_s = int(timeout_s or settings["timeout_s"])
    if write and out_dir is None:
        raise MPSAConfigError("write=True 时必须提供 out_dir")

    # ---- provider 凭据校验（mock 完全离线）----
    api_key = None
    if provider in ("dashscope", "deepseek"):
        key_env = settings["api_key_env"]
        api_key = os.environ.get(key_env, "")
        if not api_key:
            raise MissingAPIKeyError(
                "环境变量 %s 未设置：--provider %s 需要 API key（模型 %s）。"
                "请先 `$env:%s=\"...\"`；离线自测请用 --provider mock。" % (
                    key_env, provider, model, key_env)
            )

    # ---- 基线 generic.c ----
    baseline_code = None
    if code_path:
        baseline_path = Path(code_path)
        if not baseline_path.exists():
            raise MPSAConfigError("基线 generic 代码文件不存在：%s" % baseline_path)
        baseline_code = baseline_path.read_text(encoding="utf-8", errors="replace")

    # ---- 检索（7 步 trace；不在本模块落盘 trace）----
    try:
        trace = retrieve_mod.retrieve(
            cfg, op, profile=profile, top_k=top_k, data_dir=data_dir, write=False
        )
    except (ValueError, FileNotFoundError) as exc:
        raise RetrievalError("检索失败（op=%r profile=%r）：%s" % (op, profile, exc)) from exc
    query = trace.get("query") or {}
    canonical = query.get("operator") or op
    pack = (trace.get("result") or {}).get("pack") or {}
    strategies = pack.get("strategies") or []
    if not strategies:
        raise RetrievalError(
            "检索未返回任何策略（op=%r profile=%r）；无法生成代码" % (op, profile)
        )

    profile_name = query.get("hardware_profile_id") or profile
    features = query.get("code_features") or {}
    # 任务签名：--code 基线 > 内置签名表；都没有时由模型在生成时给出（clean_code 再从输出里提取）
    signature = resolve_signature(op, canonical, baseline_code)
    context = _prompt_context(op, canonical, profile_name, features, signature,
                              baseline_code, top_k, analysis_k)

    usage = _empty_usage()
    warnings = []
    if mock_code_path and provider != "mock":
        warnings.append("--mock-code 仅在 --provider mock 时生效；本次 provider=%s，已忽略" % provider)
    client = None
    if provider != "mock":
        client = DashScopeClient(model, settings["api_base"], api_key, timeout_s)

    analyses = []
    analysis_prompts = []
    synthesis = None
    synthesis_prompt = None

    # ---- B4：逐条适用性分析 + 综合 ----
    if mode == "b4":
        selected = strategies[: min(analysis_k, len(strategies))]
        if len(selected) < analysis_k:
            warnings.append(
                "analysis_k=%d 大于检索到的策略数 %d，按 %d 条分析"
                % (analysis_k, len(strategies), len(selected))
            )
        for index, strategy in enumerate(selected):
            analysis_prompt = render_prompt(
                "analysis", STRATEGY_JSON=_strategy_json(strategy), **context
            )
            analysis_prompts.append({
                "strategy_id": strategy["id"],
                "strategy_name": strategy.get("name"),
                "prompt": analysis_prompt,
            })
            if provider == "mock":
                analyses.append(_mock_analysis(strategy, index, profile_name))
                _add_usage(usage, {}, "analysis")
            else:
                reply, call_usage = client.chat(
                    analysis_prompt, SYSTEM_ANALYSIS, json_mode=True, stage="analysis",
                    temperature=temperature, max_tokens=MAX_TOKENS_ANALYSIS,
                )
                _add_usage(usage, call_usage, "analysis")
                analyses.append(
                    normalize_analysis(_parse_json_reply(reply, "analysis"), strategy, index)
                )
        synthesis_prompt = render_prompt(
            "synthesis",
            ANALYSES_JSON=json.dumps(_analyses_for_prompt(analyses), ensure_ascii=False,
                                     indent=2, sort_keys=False),
            **context,
        )
        if provider == "mock":
            synthesis = _mock_synthesis(analyses)
            _add_usage(usage, {}, "synthesis")
        else:
            reply, call_usage = client.chat(
                synthesis_prompt, SYSTEM_SYNTHESIS, json_mode=True, stage="synthesis",
                temperature=temperature, max_tokens=MAX_TOKENS_SYNTHESIS,
            )
            _add_usage(usage, call_usage, "synthesis")
            synthesis = normalize_synthesis(_parse_json_reply(reply, "synthesis"))
        generate_prompt = render_prompt(
            "generate_b4",
            STRATEGIES_JSON=_strategies_json(strategies),
            ANALYSES_JSON=json.dumps(_analyses_for_prompt(analyses), ensure_ascii=False,
                                     indent=2, sort_keys=False),
            SYNTHESIS_JSON=json.dumps(synthesis, ensure_ascii=False, indent=2, sort_keys=False),
            **context,
        )
    else:
        generate_prompt = render_prompt(
            "generate_b3", STRATEGIES_JSON=_strategies_json(strategies), **context
        )

    # ---- 生成 C 代码 ----
    if provider == "mock":
        raw_code = _mock_code_text(op, canonical, signature, baseline_code, mock_code_path)
        _add_usage(usage, {}, "generate")
    else:
        raw_code, call_usage = client.chat(
            generate_prompt, SYSTEM_GENERATE, json_mode=False, stage="generate",
            temperature=temperature, max_tokens=MAX_TOKENS_GENERATE,
        )
        _add_usage(usage, call_usage, "generate")

    cleaned = clean_code(
        raw_code, op=op, canonical=canonical, signature=signature, baseline_code=baseline_code
    )
    signature = cleaned["signature"]

    elapsed_s = round(time.perf_counter() - started, 3)
    now = datetime.now().astimezone()
    result = {
        "schema_version": "mpsa-v1",
        "op": op,
        "op_canonical": canonical,
        "profile": profile_name,
        "mode": mode,
        "provider": provider,
        "model": "mock" if provider == "mock" else model,
        "model_requested": model,
        "created_at": now.isoformat(timespec="seconds"),
        "retrieved": {
            "top_k": top_k,
            "count": len(strategies),
            "trace_id": trace.get("trace_id"),
            "query": query,
            "strategy_ids": [strategy.get("id") for strategy in strategies],
            "strategies": [_compact_strategy(strategy) for strategy in strategies],
            "prompt_preview": (trace.get("result") or {}).get("prompt_preview"),
        },
        "analysis": None if mode == "b3" else {
            "analysis_k": analysis_k,
            "count": len(analyses),
            "items": analyses,
        },
        "synthesis": synthesis,
        "prompt": {
            "analysis": [item["prompt"] for item in analysis_prompts],
            "synthesis": synthesis_prompt,
            "generate": generate_prompt,
        },
        "code": cleaned["code"],
        "code_meta": {
            "signature": cleaned["signature"],
            "function_name": cleaned["function_name"],
            "wrapped_body": cleaned["wrapped_body"],
            "fences_removed": cleaned["fences_removed"],
            "prologue_added": cleaned["prologue_added"],
            "repairs": cleaned["repairs"],
            "char_count": cleaned["char_count"],
        },
        "usage": usage,
        "elapsed_s": elapsed_s,
        "warnings": warnings,
        "artifacts": {},
    }
    if write:
        result["artifacts"] = write_result(result, out_dir)
    return result


# ---------------------------------------------------------------- 修复回路（B3/B4 生成后）


def _write_repair_result(result: dict, out_dir) -> dict:
    """写 <out_dir>/repair_result.json 与 <out_dir>/repaired.c，返回 artifacts。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    repaired_path = out / "repaired.c"
    result_path = out / "repair_result.json"
    repaired_path.write_text(result["code"], encoding="utf-8", newline="\n")
    artifacts = {
        "out_dir": str(out),
        "result_path": str(result_path),
        "repaired_c_path": str(repaired_path),
    }
    result["artifacts"] = artifacts
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8", newline="\n",
    )
    return artifacts


def repair_code(cfg, op: str, profile: str, code, error, mode: str = "b3",
                provider: str = "dashscope", model: str | None = None,
                signature: str | None = None, out_dir=None, mock_code_path=None,
                timeout_s: int | None = None, write: bool = True,
                temperature: float = 0.0) -> dict:
    """把「原代码 + 编译/运行错误 + 任务签名 + profile」回灌给 LLM，做 1 轮修复。

    参数：
      cfg            kgagent.config.Config
      op / profile   算子名 / 硬件 profile（仅用于 prompt 与结果记录，不检索图谱）
      code           待修复的原 C 代码：文件路径或内联文本
      error          编译器 stderr / QEMU 运行输出：文件路径或内联文本（含 maxdiff/reldiff）
      mode           "b3" | "b4"（记录来源条件，修复行为一致）
      provider       "dashscope" | "deepseek" | "mock"
      model          DashScope 模型（默认 qwen-plus-2025-04-28）
      signature      任务函数签名（可选；默认从 code 提取，再退回内置 daxpy/ddot 签名）
      out_dir        输出目录（写 repair_result.json + repaired.c）
      mock_code_path mock provider 指定"修复后代码"文件（优先于内置 mock 修复）
      timeout_s      LLM 超时（秒）
      write          是否落盘（默认 True；需要 out_dir）
      temperature    LLM temperature（默认 0）

    返回字段：schema_version/op/op_canonical/profile/mode/provider/model/input_error/error_source/
    calls/fix_summary/prompt/code/code_meta/usage/elapsed_s/created_at/warnings/artifacts。
    """
    started = time.perf_counter()
    mode = (mode or "b3").strip().lower()
    provider = (provider or "dashscope").strip().lower()
    if mode not in MODES:
        raise MPSAConfigError("未知 mode=%r；可选：%s" % (mode, " / ".join(MODES)))
    if provider not in PROVIDERS:
        raise MPSAConfigError("未知 provider=%r；可选：%s" % (provider, " / ".join(PROVIDERS)))
    settings = _mpsa_settings(cfg, provider)
    if code is None or (isinstance(code, str) and not code.strip()):
        raise MPSAConfigError("mpsa-repair 需要 --code（待修复的原 C 代码文件或内联文本）")
    if error is None or (isinstance(error, str) and not error.strip()):
        raise MPSAConfigError(
            "mpsa-repair 需要 --error（编译器 stderr / QEMU 运行输出文件或内联文本）"
        )
    model = (model or settings["model"] or DEFAULT_MODEL).strip()
    timeout_s = int(timeout_s or settings["timeout_s"])
    if write and out_dir is None:
        raise MPSAConfigError("write=True 时必须提供 out_dir")

    code_info = _load_text_arg(code, "--code")
    error_info = _load_text_arg(error, "--error")
    original_code = code_info["text"]
    error_text = error_info["text"].strip()
    if not original_code.strip():
        raise MPSAConfigError("--code 内容为空：没有可修复的 C 代码")
    if not error_text:
        raise MPSAConfigError("--error 内容为空：修复回路需要真实的编译/运行错误信息")

    # ---- provider 凭据校验（mock 完全离线）----
    api_key = None
    if provider in ("dashscope", "deepseek"):
        key_env = settings["api_key_env"]
        api_key = os.environ.get(key_env, "")
        if not api_key:
            raise MissingAPIKeyError(
                "环境变量 %s 未设置：--provider %s 需要 API key（模型 %s）。"
                "请先 `$env:%s=\"...\"`；离线自测请用 --provider mock。" % (
                    key_env, provider, model, key_env)
            )

    warnings = []
    if mock_code_path and provider != "mock":
        warnings.append("--mock-code 仅在 --provider mock 时生效；本次 provider=%s，已忽略" % provider)

    canonical = _canonical_op_name(op)
    # 签名优先级：显式 --signature > 原代码里的签名 > 内置 daxpy/ddot 签名
    resolved_signature = (
        (signature or "").strip()
        or extract_signature(original_code)
        or resolve_signature(op, canonical)
    )
    prompt = render_prompt(
        "repair",
        OP=op,
        OP_CANONICAL=canonical,
        PROFILE=profile,
        SIGNATURE=resolved_signature
        or "（未提供基线签名；请与原代码保持完全一致的函数名/返回类型/参数列表）",
        ORIGINAL_CODE=original_code.strip(),
        ERROR_TEXT=error_text,
    )

    usage = _empty_usage()
    client = None
    if provider != "mock":
        client = DashScopeClient(model, settings["api_base"], api_key, timeout_s)

    if provider == "mock":
        if mock_code_path:
            mock_path = Path(mock_code_path)
            if not mock_path.is_file():
                raise MPSAConfigError("--mock-code 文件不存在：%s" % mock_path)
            raw_repaired = mock_path.read_text(encoding="utf-8", errors="replace")
        else:
            raw_repaired = _mock_repair_code(op, canonical, resolved_signature, original_code)
        _add_usage(usage, {}, "repair")
    else:
        raw_repaired, call_usage = client.chat(
            prompt, SYSTEM_REPAIR, json_mode=False, stage="repair",
            temperature=temperature, max_tokens=MAX_TOKENS_REPAIR,
        )
        _add_usage(usage, call_usage, "repair")

    cleaned = clean_code(
        raw_repaired,
        op=op,
        canonical=canonical,
        signature=resolved_signature,
        baseline_code=original_code,
    )
    elapsed_s = round(time.perf_counter() - started, 3)
    now = datetime.now().astimezone()
    result = {
        "schema_version": "mpsa-repair-v1",
        "op": op,
        "op_canonical": canonical,
        "profile": profile,
        "mode": mode,
        "provider": provider,
        "model": "mock" if provider == "mock" else model,
        "model_requested": model,
        "created_at": now.isoformat(timespec="seconds"),
        "input_error": error_text,
        "error_source": error_info["source"],
        "calls": usage["calls"],
        "fix_summary": _fix_summary(original_code, cleaned["code"], provider),
        "prompt": {"repair": prompt, "chars": len(prompt)},
        "code": cleaned["code"],
        "code_meta": {
            "signature": cleaned["signature"],
            "function_name": cleaned["function_name"],
            "wrapped_body": cleaned["wrapped_body"],
            "fences_removed": cleaned["fences_removed"],
            "prologue_added": cleaned["prologue_added"],
            "repairs": cleaned["repairs"],
            "char_count": cleaned["char_count"],
        },
        "usage": usage,
        "elapsed_s": elapsed_s,
        "warnings": warnings,
        "artifacts": {},
    }
    if write:
        result["artifacts"] = _write_repair_result(result, out_dir)
    return result

