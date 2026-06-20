"""Pytest entry: one test per lab notebook, run in order, in a temp workdir.

    pytest tests/                       # all labs, synthetic data
    pytest tests/ -m "not slow"         # skip heavy training labs
    pytest tests/ -m "not network"      # skip dataset-loading labs
    pytest tests/ --online              # use real dataset downloads
    pytest tests/ -k tokenization       # single lab by name

Each lab runs in-process (fast feedback, rich tracebacks). The standalone
``run_tests.py`` runner is preferred for full isolation + hard timeouts.
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from core import run_lab  # noqa: E402
from manifest import discover  # noqa: E402

_SPECS = discover()


def _params():
    params = []
    for spec in _SPECS:
        marks = []
        if spec.slow:
            marks.append(pytest.mark.slow)
        if spec.network:
            marks.append(pytest.mark.network)
        if spec.extra:
            marks.append(pytest.mark.extra_deps)
        params.append(pytest.param(spec, id=spec.test_id, marks=marks))
    return params


@pytest.mark.parametrize("spec", _params())
def test_lab_runs(spec, tmp_path, request):
    online = request.config.getoption("--online")
    result = run_lab(
        spec.path,
        rel=spec.rel,
        offline=not online,
        workdir=str(tmp_path),
        capture_output=True,
    )
    if result.status != "pass":
        pytest.fail(
            f"\n{spec.rel} failed at cell #{result.failed_cell} "
            f"({result.error_type}: {result.error_message})\n\n"
            f"--- cell source ---\n{result.cell_source}\n\n"
            f"--- traceback ---\n{result.traceback}",
            pytrace=False,
        )
