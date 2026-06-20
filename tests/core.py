"""Execute a lab end-to-end, in order, the way a student's kernel would.

A lab is treated as a single ordered notebook: all code cells share one
namespace and run top-to-bottom (later cells depend on earlier ones). The
runner:

  1. (offline mode) installs synthetic dataset / tokenizer stubs
  2. runs the shared preamble (headless plots, seeds)
  3. chdir's into an isolated working dir so checkpoints / PNGs / WAVs that the
     labs write land in a throwaway folder, not the repo
  4. executes each cell, raising a precise :class:`LabError` on the first failure

Timeouts are enforced by the *caller* (the CLI runner runs each lab in its own
subprocess so it can hard-kill a hang cross-platform).
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from extract import extract_cells  # noqa: E402
from preamble import PREAMBLE  # noqa: E402


class LabError(Exception):
    """Raised when a lab cell fails. Carries the cell index and source."""

    def __init__(self, rel: str, cell_index: int, cell_source: str, original: BaseException):
        self.rel = rel
        self.cell_index = cell_index
        self.cell_source = cell_source
        self.original = original
        self.tb = "".join(
            traceback.format_exception(type(original), original, original.__traceback__)
        )
        super().__init__(
            f"{rel}: cell #{cell_index} raised {type(original).__name__}: {original}"
        )


@dataclass
class LabResult:
    rel: str
    status: str  # "pass" | "fail"
    n_cells: int
    failed_cell: Optional[int] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    cell_source: Optional[str] = None


def run_lab(
    md_path,
    *,
    rel: Optional[str] = None,
    offline: bool = True,
    workdir: Optional[str] = None,
    capture_output: bool = True,
) -> LabResult:
    """Run one lab. Returns a LabResult; never raises for *lab* errors."""
    md_path = Path(md_path)
    rel = rel or md_path.name

    if offline:
        from stubs import install_offline_stubs

        install_offline_stubs()

    cells = extract_cells(md_path)
    ns = {"__name__": "__main__", "__file__": str(md_path)}

    prev_cwd = os.getcwd()
    if workdir:
        os.makedirs(workdir, exist_ok=True)
        os.chdir(workdir)

    sink = io.StringIO()
    cm = contextlib.redirect_stdout(sink) if capture_output else contextlib.nullcontext()
    try:
        with cm:
            exec(compile(PREAMBLE, "<preamble>", "exec"), ns)
            for i, cell in enumerate(cells):
                try:
                    exec(compile(cell, f"{rel}#cell{i}", "exec"), ns)
                except BaseException as exc:  # noqa: BLE001 - report everything
                    err = LabError(rel, i, cell, exc)
                    return LabResult(
                        rel=rel,
                        status="fail",
                        n_cells=len(cells),
                        failed_cell=i,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        traceback=err.tb,
                        cell_source=cell,
                    )
    finally:
        os.chdir(prev_cwd)

    return LabResult(rel=rel, status="pass", n_cells=len(cells))
