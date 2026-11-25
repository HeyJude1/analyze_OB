#!/usr/bin/env python3
"""Generate or replay one prepared comparison cell and preserve its provenance."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "kg_agent" / "src"))
from kgagent.mpsa import DashScopeClient, LLMError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engine", choices=("replay", "dashscope"), default="replay")
    parser.add_argument("--model")
    parser.add_argument("--completion-cap", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()
    if args.completion_cap <= 0 or not 0 <= args.temperature <= 2:
        parser.error("invalid completion cap or temperature")
    if args.engine == "dashscope":
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not args.model or not api_key:
            parser.error("dashscope requires --model and DASHSCOPE_API_KEY")

    manifest = json.loads((args.prepared / "manifest.json").read_text(encoding="utf-8"))
    desired = (args.task, args.profile, args.method, args.repetition)
    pending = [
        row for row in manifest["cases"]
        if (row["operator"], row["profile"], row["method"], row["repetition"]) == desired
    ]
    if len(pending) != 1:
        parser.error("specify one existing prepared cell")
    row = pending[0]
    prompt_path = (args.prepared / row["prompt"]).resolve()
    if not prompt_path.is_relative_to(args.prepared.resolve()):
        parser.error("prompt path is outside the prepared package")
    serialized = prompt_path.read_bytes()
    if hashlib.sha256(serialized).hexdigest() != row["prompt_sha256"]:
        parser.error("prompt hash mismatch")
    prompt = json.loads(serialized)

    args.out.mkdir(parents=True, exist_ok=False)
    result = {
        **row, "engine": args.engine, "model": args.model, "status": "engine_error",
        "completion_cap": args.completion_cap, "temperature": args.temperature,
    }
    started = time.monotonic()
    try:
        if args.engine == "replay":
            raw = json.loads(prompt["user"])["input_c"]
            result["usage"] = None
            result["status"] = "replayed_input"
        else:
            client = DashScopeClient(
                args.model, "https://dashscope.aliyuncs.com/compatible-mode/v1", api_key)
            raw, result["usage"] = client.chat(
                prompt["user"], prompt["system"], temperature=args.temperature,
                max_tokens=args.completion_cap)
            result["status"] = "generated_unverified"
        (args.out / "response.txt").write_text(raw, encoding="utf-8")
        fenced = re.fullmatch(r"\s*```(?:c|C)?[ \t]*\r?\n(.*?)\r?\n```\s*", raw, re.S)
        source = fenced.group(1) if fenced else raw
        source_path = args.out / "candidate.c"
        source_path.write_text(source.strip() + "\n", encoding="utf-8")
        result["candidate_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    except LLMError:
        # Preserve failure accounting without saving endpoint responses that may contain secrets.
        raise SystemExit("Model call failed; see the recorded cell status.")
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        (args.out / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(result["status"])


if __name__ == "__main__":
    main()
