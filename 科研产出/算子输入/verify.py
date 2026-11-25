"""Compile and check OpOptBench candidates without collecting timing data."""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import sysconfig


ROOT = Path(__file__).resolve().parent
TARGETS = {"x86": "x86_avx2", "rvv": "riscv_rvv"}
FLAGS = {"x86": ["-mavx2", "-mfma"], "rvv": ["-march=rv64gcv", "-mabi=lp64d"]}


def local_arg(path):
    """Keep compiler paths relative when possible, including on MinGW."""
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return str(path)


def invoke(args, timeout):
    try:
        result = subprocess.run(args, cwd=ROOT, text=True, encoding="utf-8", errors="replace",
                                capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or b"").decode("utf-8", errors="replace")
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
        return {"argv": args, "exit_code": None, "stdout": stdout,
                "stderr": stderr + f"\nProcess timed out after {timeout:g} seconds.\n",
                "timed_out": True}
    return {"argv": args, "exit_code": result.returncode, "stdout": result.stdout,
            "stderr": result.stderr}


def run_task(name, task, args, cc, compile_only, version):
    variant = "candidate" if args.candidate else args.variant
    target_key = TARGETS[args.target]
    if args.candidate:
        source = args.candidate.resolve(strict=True)
    else:
        path = task["generic"] if variant == "generic" else task["implementations"][target_key]
        source = ROOT / path

    suffix = ".exe" if os.name == "nt" and args.target == "x86" else ""
    bin_path = args.build_dir / f"{name}-{args.target}-{variant}{suffix}"
    flag = f"-DOPBENCH_TASK_{name.upper()}"
    command = [cc, "-std=c11", "-O3", "-Wall", "-Wextra", *FLAGS[args.target],
               *args.cflag, "-Iinclude", flag, "driver/check.c", local_arg(source),
               "-o", local_arg(bin_path), "-lm"]
    result = {"task": name, "target": args.target, "variant": variant, "compiler_version": version,
              "source": local_arg(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "status": "compile_error"}
    result["compile"] = invoke(command, args.timeout)
    if result["compile"].get("timed_out"):
        result["status"] = "compile_timeout"
        return result
    if result["compile"]["exit_code"] != 0:
        return result
    result["binary_sha256"] = hashlib.sha256(bin_path.read_bytes()).hexdigest()
    result["status"] = "compiled_only"
    if compile_only:
        return result

    command = [str(bin_path)]
    if args.runner:
        command = [args.runner, *args.runner_arg, str(bin_path)]
    result["check"] = invoke(command, args.timeout)
    result["environment"] = "external_runner" if args.runner else "native"
    if result["check"].get("timed_out"):
        result["status"] = "run_timeout"
    else:
        result["status"] = "passed" if result["check"]["exit_code"] == 0 else "check_failed"
    return result


def main():
    tasks = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))["tasks"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[*tasks, "all"], default="all")
    parser.add_argument("--target", choices=TARGETS, default="x86")
    parser.add_argument("--variant", choices=["generic", "arch"], default="arch")
    parser.add_argument("--candidate", type=Path, help="replace the selected task's C source")
    parser.add_argument("--cc", help="GCC-compatible compiler executable")
    parser.add_argument("--cflag", action="append", default=[], help="extra flag; pass as --cflag=-flag")
    parser.add_argument("--runner", help="optional executable such as qemu-riscv64")
    parser.add_argument("--runner-arg", action="append", default=[], help="one external runner argument")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--build-dir", type=Path, default=ROOT / ".build")
    parser.add_argument("--report", type=Path, help="write a machine-readable result JSON")
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if args.candidate and args.task == "all":
        parser.error("--candidate requires one --task")
    if args.runner_arg and not args.runner:
        parser.error("--runner-arg requires --runner")
    if args.compile_only and args.runner:
        parser.error("specify either --compile-only or --runner")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    host = platform.machine().casefold() or sysconfig.get_platform().casefold().rsplit("-", 1)[-1]
    native_target = host in {"amd64", "x86_64"} if args.target == "x86" else host == "riscv64"
    compile_only = args.compile_only or (not native_target and not args.runner)
    cc = args.cc or ("riscv64-linux-gnu-gcc" if args.target == "rvv" and not native_target else "gcc")
    args.build_dir = args.build_dir.resolve()
    args.build_dir.mkdir(parents=True, exist_ok=True)
    try:
        compiler = invoke([cc, "--version"], args.timeout)
        if compiler["exit_code"] != 0:
            parser.exit(2, compiler["stderr"])
        version = compiler["stdout"].splitlines()[0]
        selected = tasks if args.task == "all" else {args.task: tasks[args.task]}
        results = [run_task(name, task, args, cc, compile_only, version) for name, task in selected.items()]
    except OSError as exc:
        parser.exit(2, f"tool invocation failed: {exc}\n")

    for row in results:
        command = row.get("check", row["compile"])
        print(f"{row['task']} {row['target']} {row['variant']}: {row['status']} {command['stdout'].strip()}")
        if command["stderr"]:
            print(command["stderr"], end="", file=sys.stderr)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        payload = {"kind": "compile_and_correctness", "host": host, "results": results}
        args.report.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return int(any(row["status"] not in {"passed", "compiled_only"} for row in results))


if __name__ == "__main__":
    sys.exit(main())
