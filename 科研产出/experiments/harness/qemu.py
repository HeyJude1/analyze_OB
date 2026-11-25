"""experiments/harness/qemu.py —— Windows 侧 → WSL 交叉编译 + QEMU 用户态运行 harness。

设计要点
--------
1. **只做"可编译 + 数值正确"**：QEMU 计时数字**不能**当性能结论（见 `08b_RISC-V真机实验协议.md` §5.1）；
   本模块照常解析 driver 的 `best=`，但结果里一律带 `qemu_only` 标记，论文性能表必须用 K1 真机。
2. 脚本落盘为 **UTF-8 无 BOM + LF**：WSL 下中文路径/`set -e` 才能正常工作。
3. 路径换到 WSL：`C:\\...` → `/mnt/c/...`；Windows 侧只调
   `wsl.exe -d Ubuntu -e bash <script.sh>`。
4. **日志分流**：WSL 自身把 NAT 网络警告用 UTF-16 写到 stderr，会与子进程 UTF-8 日志混流产生乱码。
   因此 build.sh / run.sh 把 gcc / qemu 的 stdout、stderr 分别重定向到 out_dir 下的
   `build.stdout.log` / `build.stderr.log` / `qemu.stdout.log` / `qemu.stderr.log`，
   Windows 侧直接按 UTF-8 读文件——既没有编码问题，又能保留完整诊断信息。
5. 超时/失败可诊断：构建与运行分开计时，脚本内 `timeout` + Windows 侧外层超时双保险。
6. 无第三方依赖（Windows 自带 Python 即可跑）；driver 输出约定：
   `<op> n=<N> maxdiff=<f> [reldiff=<f>] best=<f> ms ...`，退出码 0=正确、非 0=不正确。
   若 driver 只有 `maxdiff`/`best` 也能用；`maxdiff`/`reldiff` 缺失时只按退出码判定。

公开接口
--------
- `compile_run(op_dir, generated_c, out_dir, *, n=None, reps=None, ...) -> dict`
- `read_task_meta(task_dir) -> dict`
- `parse_driver_output(stdout) -> dict`
- `win_to_wsl(path) -> str`
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# 环境常量（可用环境变量覆盖，便于换 WSL 发行版 / 工具链）
# ---------------------------------------------------------------------------

WSL_DISTRO = os.environ.get("EXP_WSL_DISTRO", "Ubuntu")
CROSS_CC = os.environ.get("EXP_CROSS_CC", "riscv64-linux-gnu-gcc")
CFLAGS = [
    "-O3",
    "-march=rv64gcv_zvl256b_zba_zbb_zbc_zbs",
    "-mabi=lp64d",
    "-static",
]
QEMU = os.environ.get("EXP_QEMU", "qemu-riscv64")
QEMU_CPU = os.environ.get(
    "EXP_QEMU_CPU", "rv64,v=true,vlen=256,zba=true,zbb=true,zbc=true,zbs=true"
)

DEFAULT_N = 16384          # 与 OpOptBench 向量类算子中档一致
DEFAULT_REPS = 10          # QEMU 上只求"能跑完"，不是测性能
DEFAULT_COMPILE_TIMEOUT_S = 300.0
DEFAULT_RUN_TIMEOUT_S = 300.0
DEFAULT_TOL_ABS = 1e-12    # 未在 task.yaml 写容差时的兜底

# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def win_to_wsl(path: str | os.PathLike) -> str:
    """Windows 路径 → WSL 路径：`C:\\Users\\x` → `/mnt/c/Users/x`。

    非盘符路径（已含 `/`）原样返回（正斜杠化），便于手工传 WSL 路径。
    """
    s = str(path).replace("\\", "/")
    m = re.match(r"^([A-Za-z]):/(.*)$", s)
    if m:
        return "/mnt/%s/%s" % (m.group(1).lower(), m.group(2))
    return s


def _decode(raw: bytes) -> str:
    """wsl.exe 的 stderr 是 UTF-16LE，子进程日志通常是 UTF-8——混合时挑更可信的编码。"""
    if not raw:
        return ""
    candidates = []
    for enc in ("utf-8", "utf-16-le"):
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError:
            text = raw.decode(enc, errors="replace")
        score = sum(1 for ch in text if ch.isprintable() or ch in "\r\n\t")
        score -= 10 * text.count("\ufffd")
        # 顺序敏感：分数相同（或少）时优先 UTF-8（子进程日志是 UTF-8）
        candidates.append((score + (1 if enc == "utf-8" else 0), enc, text))
    candidates.sort(key=lambda x: (x[0], x[1] == "utf-8"), reverse=True)
    return candidates[0][2]


_WSL_NOISE = re.compile(r"(localhost|代理配置|NAT 模式)", re.IGNORECASE)


def _filter_wsl_noise(text: str) -> str:
    """去掉 wsl.exe 自己往 stderr 打的网络代理警告（NAT 模式常见），只留程序输出。"""
    out = []
    for line in (text or "").splitlines():
        low = line.lower()
        if "wsl:" in low and _WSL_NOISE.search(line):
            continue
        if "localhost" in low and ("wsl" in low or "代理" in line):
            continue
        out.append(line)
    return "\n".join(out)


def _write_script(path: Path, body: str) -> None:
    """写 bash 脚本：UTF-8 无 BOM + LF（Windows 上必须显式指定，否则 WSL 会坏）。"""
    path.write_text(body.rstrip("\n") + "\n", encoding="utf-8", newline="\n")


def _read_log(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _run_wsl(script: Path, timeout_s: float) -> dict:
    """调用 `wsl.exe -d <distro> -e bash <script>` 并捕获输出（子进程日志另走文件）。"""
    wsl_script = win_to_wsl(script)
    cmd = ["wsl.exe", "-d", WSL_DISTRO, "-e", "bash", wsl_script]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s, check=False
        )
        return {
            "ok": True,
            "timeout": False,
            "returncode": proc.returncode,
            "stdout": _decode(proc.stdout),
            "stderr": _filter_wsl_noise(_decode(proc.stderr)),
            "stderr_raw": _decode(proc.stderr),
            "elapsed_s": round(time.time() - t0, 3),
            "cmd": cmd,
            "error": None,
        }
    except FileNotFoundError:
        return {
            "ok": False,
            "timeout": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "stderr_raw": "",
            "elapsed_s": round(time.time() - t0, 3),
            "cmd": cmd,
            "error": "wsl_not_found：找不到 wsl.exe（或在非 Windows 主机上跑）；"
            "请确认 WSL 发行版 %r 可用（可用环境变量 EXP_WSL_DISTRO 覆盖）" % WSL_DISTRO,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "timeout": True,
            "returncode": None,
            "stdout": _decode(exc.stdout or b""),
            "stderr": _filter_wsl_noise(_decode(exc.stderr or b"")),
            "stderr_raw": _decode(exc.stderr or b""),
            "elapsed_s": round(time.time() - t0, 3),
            "cmd": cmd,
            "error": "timeout：wsl.exe 外层超时 %.0fs（脚本内 timeout 未生效？）" % timeout_s,
        }


# ---------------------------------------------------------------------------
# task.yaml 轻量解析（不依赖 PyYAML；Windows Python 通常没装 pyyaml）
# ---------------------------------------------------------------------------


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def read_task_meta(task_dir: str | os.PathLike) -> dict:
    """从 OpOptBench 风格 task.yaml 里读 op / driver / sizes / 容差（容错，缺字段给默认值）。"""
    task_dir = Path(task_dir)
    yaml_path = task_dir / "task.yaml"
    text = _read_text(yaml_path)

    def _search(pattern, default=None):
        m = re.search(pattern, text, re.MULTILINE)
        if not m:
            return default
        return m.group(1).strip().strip("\"'")

    op = _search(r"^\s*op:\s*(\S+)", "op")
    driver = _search(r"^\s*driver:\s*([^\s#]+)")
    sizes_raw = _search(r"^\s*sizes:\s*\[([^\]]*)\]")
    sizes = []
    if sizes_raw:
        for tok in sizes_raw.split(","):
            try:
                sizes.append(int(tok.strip()))
            except ValueError:
                pass
    tol_abs = _search(r"abs_maxdiff:\s*([0-9.eE+-]+)")
    tol_rel = _search(r"rel_maxdiff:\s*([0-9.eE+-]+)")
    return {
        "op": op,
        "driver": driver,
        "sizes": sizes,
        "tol_abs": float(tol_abs) if tol_abs else DEFAULT_TOL_ABS,
        "tol_rel": float(tol_rel) if tol_rel else None,
        "task_yaml": str(yaml_path),
    }


def _resolve_driver(task_dir: Path, op: str, meta: dict) -> Path | None:
    candidates = []
    if meta.get("driver"):
        candidates.append(task_dir / meta["driver"])
    candidates.append(task_dir / "driver" / ("test_%s.c" % op))
    candidates.extend(sorted((task_dir / "driver").glob("test_*.c")))
    for c in candidates:
        if c.is_file():
            return c
    return None


# ---------------------------------------------------------------------------
# driver 输出解析
# ---------------------------------------------------------------------------

_RE_MAXDIFF = re.compile(r"maxdiff\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
_RE_RELDIFF = re.compile(r"reldiff\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
_RE_BEST = re.compile(r"best\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*ms")
_RE_N = re.compile(r"\bn\s*=\s*(\d+)")


def parse_driver_output(stdout: str) -> dict:
    """解析 `<op> n=... maxdiff=... reldiff=... best=... ms`；缺失字段为 None。"""
    raw = stdout or ""
    maxdiff = re.search(_RE_MAXDIFF, raw)
    reldiff = re.search(_RE_RELDIFF, raw)
    best = re.search(_RE_BEST, raw)
    n = re.search(_RE_N, raw)
    return {
        "maxdiff": float(maxdiff.group(1)) if maxdiff else None,
        "reldiff": float(reldiff.group(1)) if reldiff else None,
        "best_ms": float(best.group(1)) if best else None,
        "n": int(n.group(1)) if n else None,
        "line": next((ln.strip() for ln in raw.splitlines() if "maxdiff" in ln), None),
    }


# ---------------------------------------------------------------------------
# 脚本模板
# ---------------------------------------------------------------------------


def _build_script(
    wsl_out: str,
    wsl_driver: str,
    wsl_gen: str,
    wsl_bin: str,
    cflags: list[str],
    compile_timeout_s: float,
) -> str:
    return """#!/usr/bin/env bash
# 自动生成：experiments/harness/qemu.py（Windows→WSL 交叉编译）
# 本文件必须 UTF-8 无 BOM + LF；手改请保持该编码。
# 子进程 stdout/stderr 落盘到 build.stdout.log / build.stderr.log（避免 WSL UTF-16 警告混流）。
set -uo pipefail
cd "%s" || { echo "[build] FAIL: cd 失败"; exit 8; }
echo "[build] cc=%s"
echo "[build] driver=%s"
echo "[build] generated=%s"
timeout %ds %s %s "%s" "%s" -lm -o "%s" > build.stdout.log 2> build.stderr.log
rc=$?
echo "[build] gcc_exit=$rc"
if [ $rc -ne 0 ]; then
  echo "[build] FAIL（见 build.stderr.log）"
  exit $rc
fi
if [ ! -x "%s" ]; then
  echo "[build] FAIL：未生成可执行文件 %s"
  exit 9
fi
echo "[build] ok bin=%s bytes=$(stat -c %%s "%s" 2>/dev/null || echo '?')"
exit 0
""" % (
        wsl_out,
        CROSS_CC,
        wsl_driver,
        wsl_gen,
        int(compile_timeout_s),
        CROSS_CC,
        " ".join(cflags),
        wsl_driver,
        wsl_gen,
        wsl_bin,
        wsl_bin,
        wsl_bin,
        wsl_bin,
        wsl_bin,
    )


def _run_script(wsl_out: str, wsl_bin: str, n: int, reps: int, run_timeout_s: float) -> str:
    return """#!/usr/bin/env bash
# 自动生成：experiments/harness/qemu.py（Windows→QEMU 用户态运行）
# 本文件必须 UTF-8 无 BOM + LF；手改请保持该编码。
# QEMU 的 stdout/stderr 落盘到 qemu.stdout.log / qemu.stderr.log（harness 直接读文件解析）。
set -uo pipefail
cd "%s" || { echo "[run] FAIL: cd 失败"; exit 8; }
echo "[run] %s -cpu %s ./bin %d %d"
timeout --signal=TERM --kill-after=5s %ds %s -cpu "%s" "%s" %d %d > qemu.stdout.log 2> qemu.stderr.log
rc=$?
echo "[run] qemu_exit=$rc"
echo "[run] stdout_bytes=$(stat -c %%s qemu.stdout.log 2>/dev/null || echo '?')"
exit $rc
""" % (
        wsl_out,
        QEMU,
        QEMU_CPU,
        int(n),
        int(reps),
        int(run_timeout_s),
        QEMU,
        QEMU_CPU,
        wsl_bin,
        int(n),
        int(reps),
    )


# ---------------------------------------------------------------------------
# 主接口
# ---------------------------------------------------------------------------


def compile_run(
    op_dir: str | os.PathLike,
    generated_c: str | os.PathLike,
    out_dir: str | os.PathLike,
    *,
    n: int | None = None,
    reps: int | None = None,
    compile_timeout_s: float = DEFAULT_COMPILE_TIMEOUT_S,
    run_timeout_s: float = DEFAULT_RUN_TIMEOUT_S,
    extra_cflags: list[str] | None = None,
) -> dict:
    """交叉编译 `driver + generated.c`，QEMU 运行并解析正确性/计时。

    参数
    ----
    op_dir : 任务目录，需含 `task.yaml`；driver 取 task.yaml 的 `driver:` 或 `driver/test_<op>.c`
    generated_c : 待测实现；会被复制为 `<out_dir>/generated.c` 存档
    out_dir : 本次编译/运行产物目录（build.sh / run.sh / bin / *.log / result.json 都写这里）
    n, reps : driver 的规模与重复次数；默认 n 取 task.yaml sizes 的中位数、reps=10
    extra_cflags : 追加编译选项（如 `-ffast-math`），慎用（会改变语义对比）

    返回
    ----
    dict：compile_ok / correct / maxdiff / qemu_exit / stdout / stderr / timeout
          （另有 best_ms / reldiff / commands / elapsed / error 等诊断字段）
    """
    t_start = time.time()
    op_dir = Path(op_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = read_task_meta(op_dir)
    op = meta["op"]
    driver = _resolve_driver(op_dir, op, meta)
    gen_src = Path(generated_c).resolve()

    result = {
        "ok": False,
        "compile_ok": False,
        "correct": False,
        "maxdiff": None,
        "reldiff": None,
        "qemu_exit": None,
        "stdout": "",
        "stderr": "",
        "timeout": False,
        "best_ms": None,
        "qemu_only": True,  # 提醒：QEMU 计时不是性能结论
        "n": None,
        "reps": None,
        "tolerance": meta["tol_abs"],
        "op": op,
        "task_dir": str(op_dir),
        "driver": str(driver) if driver else None,
        "generated_c": str(out_dir / "generated.c"),
        "out_dir": str(out_dir),
        "scripts": {"build": str(out_dir / "build.sh"), "run": str(out_dir / "run.sh")},
        "logs": {
            "build_stdout": str(out_dir / "build.stdout.log"),
            "build_stderr": str(out_dir / "build.stderr.log"),
            "qemu_stdout": str(out_dir / "qemu.stdout.log"),
            "qemu_stderr": str(out_dir / "qemu.stderr.log"),
        },
        "commands": {},
        "compile_timeout": False,
        "elapsed_build_s": None,
        "elapsed_run_s": None,
        "elapsed_s": None,
        "error": None,
        "env": {
            "wsl_distro": WSL_DISTRO,
            "cc": CROSS_CC,
            "cflags": CFLAGS + list(extra_cflags or []),
            "qemu": QEMU,
            "qemu_cpu": QEMU_CPU,
            "flags_source": "experiments/harness/qemu.py",
        },
    }

    # --- 输入检查 ---------------------------------------------------------
    if driver is None:
        result["error"] = "task_dir 下找不到 driver（需要 driver/test_<op>.c 或 task.yaml 的 driver:）"
        _finish(result, out_dir, t_start)
        return result
    if not gen_src.is_file():
        result["error"] = "generated_c 不存在：%s" % gen_src
        _finish(result, out_dir, t_start)
        return result

    # 存档一份 generated.c，保证 out_dir 自包含
    gen_dst = out_dir / "generated.c"
    if gen_src != gen_dst:
        shutil.copyfile(gen_src, gen_dst)

    # 规模/重复次数：命令行 > task.yaml 中位数 > 默认
    sizes = meta.get("sizes") or []
    if n is None:
        n = sorted(sizes)[len(sizes) // 2] if sizes else DEFAULT_N
    reps = DEFAULT_REPS if reps is None else int(reps)
    result["n"], result["reps"] = int(n), int(reps)

    # --- 写 build.sh / run.sh --------------------------------------------
    bin_name = "%s_test" % (op or "op")
    wsl_out = win_to_wsl(out_dir)
    wsl_driver = win_to_wsl(driver)
    wsl_gen = win_to_wsl(gen_dst)
    wsl_bin = wsl_out + "/" + bin_name
    cflags = CFLAGS + list(extra_cflags or [])

    build_path = out_dir / "build.sh"
    run_path = out_dir / "run.sh"
    _write_script(
        build_path,
        _build_script(wsl_out, wsl_driver, wsl_gen, wsl_bin, cflags, compile_timeout_s),
    )
    _write_script(run_path, _run_script(wsl_out, wsl_bin, int(n), int(reps), run_timeout_s))

    # --- 编译 -------------------------------------------------------------
    build = _run_wsl(build_path, timeout_s=compile_timeout_s + 60)
    result["commands"]["build"] = build["cmd"]
    result["elapsed_build_s"] = build["elapsed_s"]
    build_stdout = _read_log(out_dir / "build.stdout.log")
    build_stderr = _read_log(out_dir / "build.stderr.log")
    result["compile_timeout"] = bool(build["timeout"]) or build["returncode"] == 124
    if build["error"]:
        result["error"] = build["error"]
    compile_ok = (
        build["ok"]
        and not build["timeout"]
        and build["returncode"] == 0
        and "[build] ok" in (build["stdout"] or "")
    )
    result["compile_ok"] = compile_ok
    if not compile_ok:
        if not result["error"]:
            if result["compile_timeout"]:
                result["error"] = "compile_timeout：交叉编译超过 %.0fs" % compile_timeout_s
            else:
                tail = (build_stderr or build["stderr"] or build["stdout"] or "").strip()
                tail_txt = " | ".join(tail.splitlines()[-8:]) if tail else "(无输出)"
                result["error"] = "compile_failed：%s（见 %s）" % (
                    tail_txt,
                    out_dir / "build.stderr.log",
                )
        result["stderr"] = build_stderr or build["stderr"]
        result["stdout"] = build_stdout or build["stdout"]
        _finish(result, out_dir, t_start)
        return result

    # --- 运行 -------------------------------------------------------------
    run = _run_wsl(run_path, timeout_s=run_timeout_s + 60)
    result["commands"]["run"] = run["cmd"]
    result["elapsed_run_s"] = run["elapsed_s"]
    qemu_stdout = _read_log(out_dir / "qemu.stdout.log")
    qemu_stderr = _read_log(out_dir / "qemu.stderr.log")
    result["stdout"] = qemu_stdout
    result["stderr"] = qemu_stderr
    if run["error"]:
        result["error"] = run["error"]
    parsed = parse_driver_output(qemu_stdout)
    result.update(parsed)
    qemu_exit = run["returncode"]
    result["qemu_exit"] = qemu_exit

    # 超时：脚本内 `timeout` 返回 124/137；外层 TimeoutExpired 也已标记
    timeout = bool(run["timeout"]) or qemu_exit in (124, 137)
    result["timeout"] = timeout
    if timeout:
        result["error"] = "run_timeout：QEMU 运行超过 %.0fs（程序可能死循环/OOM）" % run_timeout_s
        result["correct"] = False
    else:
        # 正确性：退出码 0 为准；能解析出 maxdiff 时再做数值复核
        correct = qemu_exit == 0
        if parsed["maxdiff"] is not None and parsed["maxdiff"] > result["tolerance"]:
            correct = False
            if not result["error"]:
                result["error"] = "maxdiff %.3e 超过容差 %.3e" % (
                    parsed["maxdiff"],
                    result["tolerance"],
                )
        result["correct"] = bool(correct)
        if not result["correct"] and not result["error"]:
            result["error"] = "wrong_answer：qemu_exit=%r（见 %s）" % (
                qemu_exit,
                out_dir / "qemu.stdout.log",
            )

    result["ok"] = bool(result["compile_ok"] and result["correct"] and not result["timeout"])
    _finish(result, out_dir, t_start)
    return result


def _finish(result: dict, out_dir: Path, t_start: float) -> None:
    result["elapsed_s"] = round(time.time() - t_start, 3)
    try:
        (out_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError:
        pass


# ---------------------------------------------------------------------------
# 命令行自检（开发用）：python experiments/harness/qemu.py ...
# ---------------------------------------------------------------------------


def _main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="qemu.py 自检：用 generic.c 当候选跑通编译+QEMU")
    parser.add_argument(
        "--task",
        default=str(Path(__file__).resolve().parents[1] / "E09_MPSA" / "tasks" / "daxpy"),
        help="任务目录（含 task.yaml）",
    )
    parser.add_argument("--generated", default=None, help="候选 C（默认 tasks/<op>/input/generic.c）")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / ".cache" / "qemu_selftest"),
        help="产物根目录（默认 experiments/.cache/qemu_selftest）",
    )
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - 老 Python/非交互环境忽略
        pass
    task = Path(args.task)
    meta = read_task_meta(task)
    gen = Path(args.generated) if args.generated else task / "input" / "generic.c"
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = Path(args.out) / ("%s_%s" % (meta.get("op", "op"), stamp))
    res = compile_run(task, gen, out, n=args.n, reps=args.reps)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if res["ok"]:
        print(
            "\n[SELFTEST OK] op=%s maxdiff=%s reldiff=%s qemu_exit=%s 产物=%s"
            % (res["op"], res["maxdiff"], res["reldiff"], res["qemu_exit"], out)
        )
        return 0
    print("\n[SELFTEST FAIL] %s 产物=%s" % (res["error"], out))
    return 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
