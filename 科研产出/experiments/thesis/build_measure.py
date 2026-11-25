#!/usr/bin/env python3
"""Compile a paired Linux timing executable without executing it."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
CORPUS = HERE.parents[1] / "算子输入"
TASKS = ("daxpy", "ddot", "dgemv", "dgemm", "conv1d")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--cc", default="gcc")
    parser.add_argument("--cflag", action="append", default=[])
    parser.add_argument("--ldflag", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.candidate.is_file():
        parser.error("candidate does not exist")
    baseline = CORPUS / "tasks" / args.task / "generic.c"
    cflags = [args.cc, "-std=c11", "-O3", "-fno-lto", *args.cflag,
              "-I", str(CORPUS / "include")]
    args.out.mkdir(parents=True, exist_ok=False)
    base_obj = args.out / "base.o"
    candidate_obj = args.out / "candidate.o"
    driver_obj = args.out / "measure.o"
    compiler_target = subprocess.check_output([args.cc, "-dumpmachine"], text=True).strip()
    binary = args.out / ("measure.exe" if "mingw" in compiler_target or "w64" in compiler_target
                         else "measure")
    symbol = f"op_{args.task}"
    commands = [
        [*cflags, f"-D{symbol}=base_op", "-c", str(baseline), "-o", str(base_obj)],
        [*cflags, f"-D{symbol}=test_op", f"-DOPBENCH_TASK_{args.task.upper()}",
         "-c", str(args.candidate), "-o", str(candidate_obj)],
        [*cflags, f"-DOPBENCH_TASK_{args.task.upper()}", "-c",
         str(HERE / "measure.c"), "-o", str(driver_obj)],
        [args.cc, *args.cflag, "-fno-lto", str(driver_obj), str(base_obj),
         str(candidate_obj), *args.ldflag, "-lm", "-o", str(binary)],
    ]
    compiler = subprocess.check_output([args.cc, "--version"], text=True)
    provenance = {
        "task": args.task, "compiler": compiler, "compiler_target": compiler_target,
        "build_host": os.name, "commands": commands,
        "inputs": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (baseline, args.candidate, HERE / "measure.c", CORPUS / "include" / "opbench.h")
        },
        "status": "compiling",
    }
    log = args.out / "build.log"
    try:
        with log.open("w", encoding="utf-8") as handle:
            for command in commands:
                subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
        provenance["status"] = "compiled"
        provenance["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    except subprocess.CalledProcessError as exc:
        provenance["status"] = "compile_failed"
        raise SystemExit(f"Compilation failed with exit status {exc.returncode}; see {log}.") from exc
    finally:
        (args.out / "build.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(binary)


if __name__ == "__main__":
    main()
