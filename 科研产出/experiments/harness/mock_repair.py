"""experiments/harness/mock_repair.py —— mpsa-repair CLI 未就绪时的本地 mock 修复兜底。

**只在 `--repair-backend auto`（默认）且 provider=mock 且 `python -m kgagent.cli mpsa-repair`
不存在时启用**（或显式 `--repair-backend stub`），用于把 E09 的"初始失败 → 修复 → 重编译/QEMU"
回路先跑通。真实 API（provider=dashscope）绝不走这个兜底。

产出对齐 mpsa-repair 契约：
  <out>/repair_result.json 与 <out>/generated.c
其中"修复后代码"默认取任务卡的 `input/generic.c`（即基线正确实现），
也可由 run_smoke `--repair-mock-code FILE` 指定。结果里 `stub=true`，不代表模型修复能力。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def run_stub(
    op: str,
    mode: str,
    code_path: str | Path,
    error_path: str | Path,
    out_dir: str | Path,
    profile: str = "k1_rvv256",
    *,
    fixed_code_path: str | Path | None = None,
    reason: str = "mpsa-repair CLI 不可用",
) -> dict:
    """写 repair_result.json + generated.c，返回 {exit, stdout, stderr, elapsed_s, backend}。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    code_path = Path(code_path)
    error_path = Path(error_path)
    fixed = Path(fixed_code_path) if fixed_code_path else None
    if fixed is not None and fixed.is_file():
        code = fixed.read_text(encoding="utf-8", errors="replace")
        source = str(fixed)
    else:
        # 兜底：拿不到指定代码时原样返回当前代码（修复无效，但链路不崩）
        code = code_path.read_text(encoding="utf-8", errors="replace") if code_path.is_file() else ""
        source = str(code_path)

    error_excerpt = ""
    try:
        error_excerpt = error_path.read_text(encoding="utf-8", errors="replace")[:1200]
    except OSError:
        pass

    gen_path = out_dir / "generated.c"
    gen_path.write_text(code, encoding="utf-8", newline="\n")
    result = {
        "schema_version": "repair-v1",
        "op": op,
        "op_canonical": op,
        "profile": profile,
        "mode": mode,
        "provider": "mock",
        "model": "stub",
        "code": code,
        "code_meta": {
            "signature": None,
            "wrapped_body": False,
            "fences_removed": False,
            "char_count": len(code),
            "source": source,
        },
        "usage": {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "by_stage": {"repair": None},
        },
        "error_input": {
            "path": str(error_path),
            "excerpt": error_excerpt,
        },
        "elapsed_s": 0.0,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "artifacts": {
            "result_path": str(out_dir / "repair_result.json"),
            "generated_c_path": str(gen_path),
        },
        "stub": True,
        "stub_reason": reason,
        "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
    }
    (out_dir / "repair_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    msg = "[stub] mpsa-repair 本地兜底：%s；修复后代码来自 %s（不代表模型修复能力）\n" % (reason, source)
    return {"exit": 0, "stdout": msg, "stderr": "", "elapsed_s": 0.0, "backend": "stub"}
