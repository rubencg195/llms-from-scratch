"""Pytest configuration for the lab suite.

Adds ``--online`` (default offline/synthetic data), registers markers, and
prints the detected system once at session start so a CI log shows exactly
which machine / accelerator ran the suite.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import system as sysinfo  # noqa: E402


def pytest_addoption(parser):
    parser.addoption(
        "--online",
        action="store_true",
        default=False,
        help="Run labs against real dataset downloads instead of synthetic stubs.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: heavy training / long-running lab")
    config.addinivalue_line("markers", "network: lab downloads a dataset")
    config.addinivalue_line("markers", "extra_deps: needs packages beyond requirements.txt")


def pytest_report_header(config):
    return sysinfo.format_report(sysinfo.detect())


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    info = sysinfo.detect()
    terminalreporter.write_line(
        f"\nLabs executed on: {info.accelerator.upper()} "
        f"(device={info.lab_device}, mode={'online' if config.getoption('--online') else 'offline'})"
    )
