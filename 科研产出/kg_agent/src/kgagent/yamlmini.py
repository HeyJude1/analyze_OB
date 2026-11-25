"""极简 YAML 子集解析器（仅标准库）。

为什么不用 PyYAML：任务硬约束"只用 Python 标准库"。schema/*.yaml 与 config.yaml 都只用
一个受控子集，因此本模块实现该子集即可，且解析结果可被单元测试钉住：

支持：
  - 注释（整行 `#` 与行尾 ` #`，引号内的 # 不截断）
  - 缩进嵌套 map
  - `- scalar` / `- key: value` 列表（含后续同缩进 key 的列表项 map）
  - 行内 list `[a, b]`、行内 map `{k: v, k2: v2}`（可嵌套）
  - 引号字符串（单/双）、bool（true/false）、null（null/~/空）、int、float、plain string

不支持（也刻意不用）：锚点/别名、块标量 `|`/`>`、多文档、复杂 flow 语法。
"""

from __future__ import annotations

import re

__all__ = ["load", "loads", "YamlError"]


class YamlError(ValueError):
    """YAML 子集解析失败。"""


_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")


def _strip_comment(line: str) -> str:
    """去掉行尾注释：引号与 [] {} 内的 # 保留。"""
    quote = None
    depth = 0
    out = []
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
            elif ch == "\\" and quote == '"' and i + 1 < n:
                i += 1
                out.append(line[i])
        else:
            if ch in "\"'":
                quote = ch
                out.append(ch)
            elif ch in "[{":
                depth += 1
                out.append(ch)
            elif ch in "]}":
                depth = max(0, depth - 1)
                out.append(ch)
            elif ch == "#" and depth == 0 and (i == 0 or line[i - 1] in " \t"):
                break
            else:
                out.append(ch)
        i += 1
    return "".join(out)


def _split_top_level(text: str, seps: str = ",") -> list:
    """按顶层分隔符切分（忽略引号与 [] {} 内部）。"""
    parts = []
    buf = []
    quote = None
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            elif ch == "\\" and quote == '"' and i + 1 < n:
                i += 1
                buf.append(text[i])
        else:
            if ch in "\"'":
                quote = ch
                buf.append(ch)
            elif ch in "[{":
                depth += 1
                buf.append(ch)
            elif ch in "]}":
                depth = max(0, depth - 1)
                buf.append(ch)
            elif ch in seps and depth == 0:
                parts.append("".join(buf))
                buf = []
            else:
                buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


def _split_key(text: str):
    """把 `key: value` 拆成 (key, rest)；找不到顶层冒号返回 (None, text)。"""
    quote = None
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if quote:
            if ch == quote:
                quote = None
            elif ch == "\\" and quote == '"' and i + 1 < n:
                i += 1
        else:
            if ch in "\"'":
                quote = ch
            elif ch in "[{":
                depth += 1
            elif ch in "]}":
                depth = max(0, depth - 1)
            elif ch == ":" and depth == 0:
                if i + 1 == n or text[i + 1] in " \t":
                    return text[:i].strip(), text[i + 1:].strip()
        i += 1
    return None, text


def _unquote(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        inner = text[1:-1]
        if text[0] == '"':
            inner = inner.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")
        return inner
    return text


def _parse_scalar(token: str):
    token = token.strip()
    if token == "":
        return None
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return _unquote(token)
    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(p) for p in _split_top_level(inner) if p.strip() != ""]
    if token.startswith("{") and token.endswith("}"):
        inner = token[1:-1].strip()
        result = {}
        if not inner:
            return result
        for part in _split_top_level(inner):
            if part.strip() == "":
                continue
            key, rest = _split_key(part.strip())
            if key is None:
                raise YamlError("行内 map 缺少 ':' -> %r" % part)
            result[_unquote(key)] = _parse_scalar(rest)
        return result
    low = token.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "~"):
        return None
    if _INT_RE.match(token):
        return int(token)
    if _FLOAT_RE.match(token) and ("." in token or "e" in low):
        return float(token)
    return token


class _Line:
    __slots__ = ("indent", "content", "lineno")

    def __init__(self, indent, content, lineno):
        self.indent = indent
        self.content = content
        self.lineno = lineno

    def __repr__(self):  # pragma: no cover - 调试用
        return "<L%d i=%d %r>" % (self.lineno, self.indent, self.content)


def _tokenize(text: str) -> list:
    lines = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        stripped = _strip_comment(raw)
        if not stripped.strip():
            continue
        content = stripped.rstrip()
        indent = len(content) - len(content.lstrip(" "))
        if "\t" in content[:indent]:
            raise YamlError("第 %d 行：缩进不允许用 Tab" % lineno)
        lines.append(_Line(indent, content.strip(), lineno))
    return lines


def _is_list_item(content: str) -> bool:
    return content == "-" or content.startswith("- ")


def _parse_block(lines, i, indent):
    if _is_list_item(lines[i].content):
        return _parse_list(lines, i, indent)
    return _parse_map(lines, i, indent)


def _parse_list(lines, i, indent):
    out = []
    while i < len(lines) and lines[i].indent == indent and _is_list_item(lines[i].content):
        content = lines[i].content
        rest = content[2:].strip() if content.startswith("- ") else ""
        i += 1
        if not rest:
            if i < len(lines) and lines[i].indent > indent:
                value, i = _parse_block(lines, i, lines[i].indent)
            else:
                value = None
            out.append(value)
            continue
        key, val = _split_key(rest)
        if key is None:
            out.append(_parse_scalar(rest))
            continue
        # 列表项本身是 map：`- key: value`，后续同层 key 缩进更深
        item = {}
        item[key] = _parse_scalar(val) if val != "" else None
        # 若该 key 没有值且下一行更深，先当作嵌套块
        if val == "" and i < len(lines) and lines[i].indent > indent:
            value, i = _parse_block(lines, i, lines[i].indent)
            item[key] = value
        while i < len(lines) and lines[i].indent > indent:
            if _is_list_item(lines[i].content):
                raise YamlError("第 %d 行：列表项 map 中不允许嵌套列表" % lines[i].lineno)
            sub_indent = lines[i].indent
            sub_key, sub_val = _split_key(lines[i].content)
            if sub_key is None:
                raise YamlError("第 %d 行：无法解析 %r" % (lines[i].lineno, lines[i].content))
            i += 1
            if sub_val != "":
                item[sub_key] = _parse_scalar(sub_val)
            elif i < len(lines) and lines[i].indent > sub_indent:
                value, i = _parse_block(lines, i, lines[i].indent)
                item[sub_key] = value
            else:
                item[sub_key] = None
        out.append(item)
    if i < len(lines) and lines[i].indent == indent:
        raise YamlError("第 %d 行：列表缩进不一致" % lines[i].lineno)
    return out, i


def _parse_map(lines, i, indent):
    out = {}
    while i < len(lines) and lines[i].indent == indent and not _is_list_item(lines[i].content):
        key, rest = _split_key(lines[i].content)
        if key is None:
            raise YamlError("第 %d 行：缺少 ':' -> %r" % (lines[i].lineno, lines[i].content))
        key = _unquote(key)
        if key in out:
            raise YamlError("第 %d 行：重复的 key %r" % (lines[i].lineno, key))
        i += 1
        if rest != "":
            out[key] = _parse_scalar(rest)
        elif i < len(lines) and lines[i].indent > indent:
            value, i = _parse_block(lines, i, lines[i].indent)
            out[key] = value
        else:
            out[key] = None
    if i < len(lines) and lines[i].indent == indent:
        raise YamlError("第 %d 行：缩进/结构不一致 -> %r" % (lines[i].lineno, lines[i].content))
    return out, i


def loads(text: str):
    lines = _tokenize(text)
    if not lines:
        return {}
    value, i = _parse_block(lines, 0, lines[0].indent)
    if i != len(lines):
        raise YamlError("第 %d 行：多余的缩进内容 -> %r" % (lines[i].lineno, lines[i].content))
    return value


def load(path):
    from pathlib import Path

    p = Path(path)
    return loads(p.read_text(encoding="utf-8"))
