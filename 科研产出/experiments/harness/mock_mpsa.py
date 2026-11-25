"""experiments/harness/mock_mpsa.py —— mpsa CLI 尚未就绪时的本地 mock 兜底。

**只在 `--mpsa-backend auto`（默认）且 provider=mock 且 `python -m kgagent.cli mpsa`
不存在时启用**，用于把 E09 的"生成 → 编译 → QEMU"链路先跑通；结果里会明确标记
`mpsa_backend="stub"`、`retrieved.count=0`，**不代表方法效果，也不产生 KCS**。
真实实验（provider=dashscope）绝不走这个兜底，CLI 缺失会直接失败退出。

产出格式对齐 mpsa-engineer 的契约（mpsa-v1）：
  <out>/mpsa_result.json  和  <out>/generated.c
其中 generated.c 原样使用 `--code` 传入的 generic.c（即"生成=基线"），保证可编译、语义正确。
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
    out_dir: str | Path,
    profile: str = "k1_rvv256",
    *,
    reason: str = "mpsa CLI 不可用（invalid choice）",
) -> dict:
    """在 out_dir 写 mpsa_result.json + generated.c，返回与 CLI 一致的 (exit, stdout, stderr, elapsed)。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    code = Path(code_path).read_text(encoding="utf-8", errors="replace")
    gen_path = out_dir / "generated.c"
    gen_path.write_text(code, encoding="utf-8", newline="\n")

    result = {
        "schema_version": "mpsa-v1",
        "op": op,
        "op_canonical": op,
        "profile": profile,
        "mode": mode,
        "provider": "mock",
        "model": "stub",
        "model_requested": None,
        "retrieved": {
            "top_k": 0,
            "count": 0,
            "trace_id": None,
            "strategy_ids": [],
            "strategies": [],
            "query": {},
        },
        "analysis": None,
        "synthesis": None,
        "prompt": {"generate": "", "analysis": [], "synthesis": None},
        "code": code,
        "code_meta": {
            "signature": None,
            "wrapped_body": False,
            "fences_removed": False,
            "prologue_added": False,
            "char_count": len(code),
        },
        "usage": {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "by_stage": {"analysis": None, "synthesis": None, "generate": None},
        },
        "elapsed_s": 0.0,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "artifacts": {
            "result_path": str(out_dir / "mpsa_result.json"),
            "generated_c_path": str(gen_path),
        },
        "stub": True,
        "stub_reason": reason,
        "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
    }
    (out_dir / "mpsa_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    msg = "[stub] provider=mock 本地兜底：%s；generated.c 复用 generic.c（不产生 KG 检索/分析）\n" % reason
    return {"exit": 0, "stdout": msg, "stderr": "", "elapsed_s": 0.0}
