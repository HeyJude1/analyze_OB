#!/usr/bin/env python3
"""Generate, compile, check and repair paired baseline cells. No performance measurement."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parents[1] / "算子输入"
sys.path.insert(0, str(HERE.parents[1] / "kg_agent" / "src"))
from kgagent.mpsa import DashScopeClient, LLMError

RVV_CPU = "rv64,v=true,vlen=256,elen=64,vext_spec=v1.0"
LOCAL_OPENER = build_opener(ProxyHandler({}))
ALLOWED_HEADERS = {
    "opbench.h", "immintrin.h", "riscv_vector.h", "stddef.h", "stdint.h",
    "stdbool.h", "limits.h", "float.h", "math.h", "string.h", "stdlib.h",
    "stdalign.h", "emmintrin.h", "xmmintrin.h", "assert.h", "stdio.h",
}
OUTSIDE_KERNEL = re.compile(
    r"\b(?:system|popen|_popen|execl|execvp|fork|fopen|freopen|open|socket|connect|"
    r"CreateProcess[AW]?|LoadLibrary[AW]?|ShellExecute[AW]?|unlink|remove|rename)\s*\("
)
KERNEL_INSTRUCTION = (
    " The candidate may use arithmetic, memory and vector primitives. "
    "It must not access files, launch processes, communicate or modify the test harness."
)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def ollama(path, port, body=None, timeout=20):
    data = None if body is None else json.dumps(body).encode()
    request = Request(f"http://127.0.0.1:{port}{path}", data,
                      headers={"Content-Type": "application/json"})
    with LOCAL_OPENER.open(request, timeout=timeout) as response:
        return json.load(response)


def wsl_path(path):
    path = path.resolve()
    return f"/mnt/{path.drive[0].lower()}/{path.as_posix()[3:]}" if path.drive else str(path)


def verifier(args, candidate, directory, op, profile):
    report = directory / "verification.json"
    binary_dir = directory / "bin"
    target = "x86" if profile == "x86_avx2" else "rvv"
    use_wsl = target == "rvv" and os.name == "nt"
    command = ["wsl.exe", "--exec", "python3"] if use_wsl else [sys.executable]
    path_arg = wsl_path if use_wsl else str
    command.extend([
        path_arg(CORPUS / "verify.py"), "--target", target, "--task", op,
        "--candidate", path_arg(candidate), "--build-dir", path_arg(binary_dir),
        "--report", path_arg(report), "--timeout", str(args.check_timeout),
    ])
    if target == "rvv":
        command.extend(["--cc", args.rvv_cc, "--cflag=-static", "--runner", "qemu-riscv64",
                        "--runner-arg=-cpu", f"--runner-arg={RVV_CPU}"])
    if report.exists():
        return load(report)["results"][0], ""
    try:
        result = subprocess.run(command, capture_output=True, timeout=2 * args.check_timeout + 30)
    except subprocess.TimeoutExpired:
        return None, "verifier_timeout"
    stderr = result.stderr.decode("utf-8", errors="replace").replace("\x00", "")
    stdout = result.stdout.decode("utf-8", errors="replace").replace("\x00", "")
    (directory / "verifier.log").write_text(stdout + "\n" + stderr, encoding="utf-8")
    if not report.exists():
        return None, "verifier_process_failed"
    return load(report)["results"][0], ""


def classify(row, reason):
    if row is None:
        return "infra_error", None, None, reason, ""
    status = row["status"]
    if status == "passed":
        return "passed", True, True, None, ""
    if status == "compile_timeout":
        return "compile_timeout", None, None, "compiler_timed_out", ""
    if status == "run_timeout":
        return "run_timeout", True, False, "candidate_timed_out", ""
    if status == "compile_error":
        diagnostics = row["compile"]["stderr"]
        if re.search(r"undefined reference|unresolved external", diagnostics, re.I):
            kind = "link_or_missing_symbol"
        elif re.search(r"implicit declaration.*(?:__riscv|_mm)|intrinsic|target specific option mismatch",
                       diagnostics, re.I):
            kind = "intrinsic_or_target_error"
        elif re.search(r"conflicting types|incompatible type|too (?:few|many) arguments", diagnostics, re.I):
            kind = "type_or_signature_error"
        elif re.search(r"no such file or directory", diagnostics, re.I):
            kind = "missing_header"
        else:
            kind = "other_compiler_error"
        return "compile_failed", False, None, kind, diagnostics
    if status == "check_failed":
        check = row["check"]
        details = check["stdout"] + "\n" + check["stderr"]
        probe = re.search(r"\b(\d+)\s+(?:failures|failed)\b", details)
        if probe and int(probe[1]):
            redacted = (
                f"{probe[1]} numerical or memory-contract assertions failed. "
                "Multiple assertions may fail in one case. Preserve the full public contract."
            )
            tag = "numerical_or_memory_contract"
            return "wrong_answer", True, False, tag, redacted
        return "runtime_failed", True, False, "execution_aborted", "Execution aborted or did not return the expected check status."
    return "infra_error", None, None, f"unexpected_{status}", ""


def candidate_text(raw):
    fences = re.findall(r"(?m)^[ \t]*```(?:[cC])?[ \t]*\r?\n([\s\S]*?)^[ \t]*```[ \t]*$", raw)
    source = fences[0] if len(fences) == 1 else raw
    if not source.strip() or "```" in source:
        return None
    headers = re.findall(r"(?m)^\s*#\s*include\s*([<\"][^>\"]+[>\"])", source)
    directives = re.findall(r"(?m)^\s*#\s*(?:include|include_next)\b", source)
    if len(headers) != len(directives) or any(h[1:-1] not in ALLOWED_HEADERS for h in headers):
        return None
    if OUTSIDE_KERNEL.search(source):
        return None
    if '"opbench.h"' not in headers:
        source = '#include "opbench.h"\n' + source
    return source.strip() + "\n"


def model_reply(args, prompt, seed):
    if args.engine == "replay":
        return json.loads(prompt["user"])["input_c"], None, "replay"
    if args.engine == "dashscope":
        client = DashScopeClient(args.model, "https://dashscope.aliyuncs.com/compatible-mode/v1",
                                  os.environ["DASHSCOPE_API_KEY"], timeout_s=args.model_timeout)
        raw, usage = client.chat(prompt["user"], prompt["system"] + KERNEL_INSTRUCTION,
                                 temperature=args.temperature, max_tokens=args.completion_cap)
        return raw, usage, None
    reply = ollama("/api/chat", args.port, {
        "model": args.model, "stream": False, "keep_alive": "10m",
        "messages": [{"role": "system", "content": prompt["system"] + KERNEL_INSTRUCTION},
                     {"role": "user", "content": prompt["user"]}],
        "options": {"num_ctx": args.context, "num_predict": args.completion_cap,
                    "temperature": args.temperature, "seed": seed, "top_p": 0.9},
    }, timeout=args.model_timeout)
    inputs = reply.get("prompt_eval_count", 0)
    outputs = reply.get("eval_count", 0)
    return reply["message"]["content"], {
        "prompt_tokens": inputs, "completion_tokens": outputs, "total_tokens": inputs + outputs,
    }, reply.get("done_reason")


def attempt(args, prompt, seed, directory, row):
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "attempt.json"
    if state_path.exists():
        return load(state_path)
    response_path = directory / "response.json"
    write(directory / "prompt.json", prompt)
    if response_path.exists():
        saved = load(response_path)
    else:
        try:
            raw, usage, stop_reason = model_reply(args, prompt, seed)
            saved = {"raw": raw, "usage": usage, "stop_reason": stop_reason}
            write(response_path, saved)
        except (LLMError, URLError, TimeoutError, OSError, KeyError, ValueError):
            result = {"status": "generation_failed", "compile_ok": None, "correct": None,
                      "failure_reason": "model_request_failed", "usage": None, "diagnostic": ""}
            write(state_path, result)
            return result
    binary_input = candidate_text(saved["raw"])
    if binary_input is None:
        reason = "output_budget_or_truncated_c" if saved["stop_reason"] == "length" else "output_format_or_kernel_scope"
        result = {"status": "invalid_output", "compile_ok": None, "correct": None,
                  "failure_reason": reason, "usage": saved["usage"],
                  "diagnostic": "Return one complete C translation unit using only local computation."}
    else:
        candidate = directory / "candidate.c"
        candidate.write_text(binary_input, encoding="utf-8")
        verification, error = verifier(args, candidate, directory, row["operator"], row["profile"])
        status, compiled, correct, reason, diagnostic = classify(verification, error)
        if saved["stop_reason"] == "length" and status == "compile_failed":
            reason = "output_budget_or_truncated_c"
        result = {"status": status, "compile_ok": compiled, "correct": correct,
                  "failure_reason": reason, "usage": saved["usage"], "diagnostic": diagnostic}
        result["candidate_sha256"] = digest(candidate.read_bytes())
    write(state_path, result)
    return result


def repair_prompt(prompt, initial, directory):
    task = json.loads(prompt["user"])
    task.pop("knowledge", None)
    previous = directory / "candidate.c"
    if previous.exists():
        task["previous_candidate"] = previous.read_text(encoding="utf-8")
    else:
        task["previous_candidate"] = load(directory / "response.json")["raw"][:5000]
    # Normalize file diagnostics and limit the amount of feedback identically for every arm.
    diagnostic = initial["diagnostic"].replace("\\", "/")
    diagnostic = re.sub(r"[^\s:]*?/candidate\.c", "candidate.c", diagnostic)
    task["diagnostic"] = (diagnostic or initial["status"])[:1800]
    return {"system": prompt["system"] + " Repair the previous candidate using the supplied diagnostic.",
            "user": json.dumps(task, ensure_ascii=False)}


def run_cell(args, row):
    directory = args.out / row["operator"] / row["profile"] / row["method"] / f"rep-{row['repetition']}"
    result_path = directory / "result.json"
    if result_path.exists():
        return
    prompt_input = (args.prepared / row["prompt"]).resolve()
    if not prompt_input.is_relative_to(args.prepared.resolve()):
        raise ValueError("prompt outside prepared package")
    raw = prompt_input.read_bytes()
    if digest(raw) != row["prompt_sha256"]:
        raise ValueError(f"prompt changed: {prompt_input}")
    prompt = json.loads(raw)
    seed = args.seed + row["repetition"]
    initial_dir = directory / "initial"
    initial = attempt(args, prompt, seed, initial_dir, row)
    eligible = initial["status"] in {
        "compile_failed", "wrong_answer", "runtime_failed", "invalid_output", "run_timeout",
    }
    repair = eligible and args.engine != "replay"
    final = initial
    if repair:
        final = attempt(args, repair_prompt(prompt, initial, initial_dir), seed + 1_000_000,
                        directory / "repair", row)
    write(result_path, {
        **row, "engine": args.engine, "model": args.model, "repair_attempted": repair,
        "initial": initial, "final": final,
    })
    print(f"{row['operator']} {row['profile']} {row['method']} "
          f"{initial['status']} -> {final['status']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engine", choices=["ollama", "dashscope", "replay"], default="ollama")
    parser.add_argument("--model", default="qwen2.5-coder:7b-instruct-q4_K_M")
    parser.add_argument("--port", type=int, default=11437)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--completion-cap", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--model-timeout", type=int, default=180)
    parser.add_argument("--check-timeout", type=int, default=20)
    parser.add_argument("--rvv-cc", help="RVV compiler path or name on the Linux PATH")
    parser.add_argument("--task", action="append")
    parser.add_argument("--profile", action="append")
    args = parser.parse_args()

    manifest_bytes = (args.prepared / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    selected = [row for row in manifest["cases"]
                if (not args.task or row["operator"] in args.task)
                and (not args.profile or row["profile"] in args.profile)]
    if not selected:
        parser.error("no selected workload")
    if any(row["profile"] == "k1_rvv256" for row in selected) and not args.rvv_cc:
        parser.error("pass --rvv-cc with the Linux RVV compiler executable")
    if args.engine == "dashscope" and not os.environ.get("DASHSCOPE_API_KEY"):
        parser.error("DASHSCOPE_API_KEY is missing")
    model_snapshot = None
    version = None
    if args.engine == "ollama":
        models = ollama("/api/tags", args.port)["models"]
        model_snapshot = next((m["digest"] for m in models if m["name"] == args.model), None)
        if not model_snapshot:
            parser.error("model has not been pulled")
        version = ollama("/api/version", args.port)["version"]

    metadata = {"engine": args.engine, "model": args.model, "model_digest": model_snapshot,
                "prepared_manifest_sha256": digest(manifest_bytes),
                "settings": {"context": args.context if args.engine == "ollama" else None,
                             "completion_cap": args.completion_cap, "temperature": args.temperature,
                             "seed": args.seed if args.engine == "ollama" else None,
                             "max_repairs": int(args.engine != "replay"),
                             "check_timeout_seconds": args.check_timeout,
                             "candidate_scope": "C operator only", "ollama_version": version,
                             "riscv_compiler": args.rvv_cc, "riscv_cpu": RVV_CPU}}
    args.out.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / "run.json"
    if meta_path.exists() and load(meta_path) != metadata:
        parser.error("run parameters changed; use a fresh output directory")
    write(meta_path, metadata)
    for row in selected:
        run_cell(args, row)


if __name__ == "__main__":
    main()
