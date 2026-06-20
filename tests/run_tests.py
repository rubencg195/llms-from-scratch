"""Standalone lab test runner — no pytest required.

Detects the host system, picks online/offline mode automatically, then runs
every lab (each in an isolated subprocess with a hard timeout) and prints a
pass/fail report. Works identically on the MacBook Air M5 and the Windows
i7 + RTX 3080 box:

    python tests/run_tests.py                 # auto-detect everything
    python tests/run_tests.py --quick         # skip slow labs
    python tests/run_tests.py --offline        # force synthetic data (air-gapped)
    python tests/run_tests.py --online         # force real dataset downloads
    python tests/run_tests.py --phase 1        # only phase 1 labs
    python tests/run_tests.py --only tokenization   # substring filter
    python tests/run_tests.py --jobs 4         # run N labs in parallel
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import system as sysinfo  # noqa: E402
from manifest import discover  # noqa: E402

DEFAULT_TIMEOUT = 300
SLOW_TIMEOUT = 1800

GREEN, RED, YELLOW, GREY, BOLD, RESET = (
    "\033[32m",
    "\033[31m",
    "\033[33m",
    "\033[90m",
    "\033[1m",
    "\033[0m",
)


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(text: str, color: str) -> str:
    return f"{color}{text}{RESET}" if _supports_color() else text


def network_available(host: str = "huggingface.co", port: int = 443, timeout: float = 3.0) -> bool:
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


def run_one(spec, online: bool, timeout: int):
    cmd = [sys.executable, os.path.join(_HERE, "exec_lab.py"), str(spec.path), "--rel", spec.rel]
    if online:
        cmd.append("--online")
    start = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {"rel": spec.rel, "status": "timeout", "elapsed": time.time() - start,
                "error_message": f"exceeded {timeout}s"}

    elapsed = time.time() - start
    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith("__LABTEST_RESULT__"):
            payload = json.loads(line[len("__LABTEST_RESULT__"):])
            break
    if payload is None:
        return {"rel": spec.rel, "status": "error", "elapsed": elapsed,
                "error_message": (proc.stderr or proc.stdout)[-600:] or "no result emitted"}
    payload["elapsed"] = elapsed
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Run all LLMs-From-Scratch lab notebooks as tests.")
    ap.add_argument("--quick", action="store_true", help="skip labs marked slow")
    ap.add_argument("--offline", action="store_true", help="force synthetic data")
    ap.add_argument("--online", action="store_true", help="force real dataset downloads")
    ap.add_argument("--phase", type=str, default=None, help="only run labs from this phase number/name")
    ap.add_argument("--only", type=str, default=None, help="substring filter on lab path")
    ap.add_argument("--jobs", type=int, default=1, help="parallel labs (use 1 on a single GPU)")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="per-lab timeout (s)")
    args = ap.parse_args()

    info = sysinfo.detect()
    print(sysinfo.format_report(info))

    if args.online and args.offline:
        print("error: pass only one of --online / --offline")
        return 2
    if args.online:
        online = True
    elif args.offline:
        online = False
    else:
        online = network_available()
    mode = "ONLINE (real downloads)" if online else "OFFLINE (synthetic data)"
    print(f"  Data mode   : {mode}")
    print("=" * 64)

    specs = discover()
    if args.phase:
        key = args.phase if args.phase.startswith("phase-") else f"phase-{int(args.phase):02d}"
        specs = [s for s in specs if s.phase.startswith(key) or s.phase == key
                 or s.phase.split("-")[1] == args.phase.zfill(2)]
    if args.only:
        specs = [s for s in specs if args.only in s.rel]
    if args.quick:
        specs = [s for s in specs if not s.slow]

    if not specs:
        print("No labs matched the given filters.")
        return 1

    print(f"Running {len(specs)} lab(s)  [jobs={args.jobs}]\n")

    results = {}

    def timeout_for(spec):
        return max(args.timeout, SLOW_TIMEOUT) if spec.slow else args.timeout

    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(run_one, s, online, timeout_for(s)): s for s in specs}
            for fut in as_completed(futs):
                s = futs[fut]
                results[s.rel] = fut.result()
                _print_line(results[s.rel])
    else:
        for s in specs:
            res = run_one(s, online, timeout_for(s))
            results[s.rel] = res
            _print_line(res)

    return _summary(results)


def _print_line(res):
    status = res["status"]
    elapsed = res.get("elapsed", 0.0)
    if status == "pass":
        tag = _c("PASS ", GREEN)
    elif status == "timeout":
        tag = _c("TIME ", YELLOW)
    else:
        tag = _c("FAIL ", RED)
    extra = ""
    if status != "pass":
        if res.get("failed_cell") is not None:
            extra = f"  cell #{res['failed_cell']}: {res.get('error_type')}: {res.get('error_message')}"
        elif res.get("error_message"):
            extra = f"  {res['error_message']}"
    print(f"  {tag} {res['rel']:58s} {elapsed:6.1f}s{extra}")


def _summary(results) -> int:
    passed = [r for r in results.values() if r["status"] == "pass"]
    failed = [r for r in results.values() if r["status"] not in ("pass",)]
    total_time = sum(r.get("elapsed", 0.0) for r in results.values())
    print("\n" + "=" * 64)
    print(f"  {len(passed)}/{len(results)} labs passed   ({total_time:.0f}s total)")
    if failed:
        print(_c(f"  {len(failed)} failing:", RED))
        for r in failed:
            print(f"    - {r['rel']}  [{r['status']}]")
    print("=" * 64)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
