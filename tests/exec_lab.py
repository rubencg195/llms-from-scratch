"""Subprocess entrypoint: run ONE lab in a fresh interpreter and emit JSON.

Invoked by ``run_tests.py`` as::

    python tests/exec_lab.py <lab.md> --rel <rel> [--offline|--online] [--workdir DIR]

Running each lab in its own process gives true namespace isolation (mirrors a
fresh notebook kernel) and lets the parent enforce a hard, cross-platform
timeout by killing this process.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from core import run_lab  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lab")
    ap.add_argument("--rel", default=None)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--online", action="store_true")
    args = ap.parse_args()

    workdir = args.workdir or tempfile.mkdtemp(prefix="labtest_")
    result = run_lab(
        args.lab,
        rel=args.rel,
        offline=not args.online,
        workdir=workdir,
        capture_output=True,
    )

    payload = {
        "rel": result.rel,
        "status": result.status,
        "n_cells": result.n_cells,
        "failed_cell": result.failed_cell,
        "error_type": result.error_type,
        "error_message": result.error_message,
        "traceback": result.traceback,
    }
    # Sentinel-wrapped JSON so the parent can find it even amid lab stdout.
    print("__LABTEST_RESULT__" + json.dumps(payload))
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
