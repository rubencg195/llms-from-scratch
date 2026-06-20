"""System and accelerator detection for the lab test harness.

Auto-detects the host machine and the compute device the labs will use so the
same test command works on both target systems:

  * MacBook Air M5 (Apple Silicon) -> Metal (MPS) available, labs run on CPU
  * Windows i7 + RTX 3080          -> CUDA available, labs run on GPU

The lab code itself selects its device with the pattern
``device = "cuda" if torch.cuda.is_available() else "cpu"`` (no MPS), so this
module reports what the labs *will actually use* (``lab_device``) as well as
everything that is physically *available* on the box.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SystemInfo:
    os_name: str
    os_release: str
    machine: str
    processor: str
    python_version: str
    torch_version: Optional[str] = None
    cuda_available: bool = False
    cuda_device_name: Optional[str] = None
    cuda_total_gb: Optional[float] = None
    mps_available: bool = False
    lab_device: str = "cpu"
    accelerator: str = "cpu"  # human label: cuda / mps / cpu
    notes: list = field(default_factory=list)


def detect() -> SystemInfo:
    """Probe the current machine and torch installation."""
    info = SystemInfo(
        os_name=platform.system(),
        os_release=platform.release(),
        machine=platform.machine(),
        processor=platform.processor() or "unknown",
        python_version=sys.version.split()[0],
    )

    try:
        import torch

        info.torch_version = torch.__version__

        # CUDA (Windows + RTX 3080)
        if torch.cuda.is_available():
            info.cuda_available = True
            try:
                props = torch.cuda.get_device_properties(0)
                info.cuda_device_name = props.name
                info.cuda_total_gb = round(props.total_memory / (1024**3), 1)
            except Exception as exc:  # pragma: no cover - defensive
                info.notes.append(f"CUDA props read failed: {exc}")

        # MPS (Apple Silicon / M5)
        try:
            if torch.backends.mps.is_available():
                info.mps_available = True
        except Exception:
            pass

    except ImportError:
        info.notes.append("PyTorch is not installed; lab execution will fail.")

    # The labs only branch on cuda-vs-cpu, so that is what they will run on.
    info.lab_device = "cuda" if info.cuda_available else "cpu"

    if info.cuda_available:
        info.accelerator = "cuda"
    elif info.mps_available:
        info.accelerator = "mps"
    else:
        info.accelerator = "cpu"

    if info.mps_available and not info.cuda_available:
        info.notes.append(
            "Apple MPS is available but the course labs intentionally use CPU "
            "on non-CUDA machines (the device pattern only checks cuda)."
        )

    return info


def format_report(info: SystemInfo) -> str:
    """Human-readable summary block printed before a test run."""
    lines = [
        "=" * 64,
        " LLMs-From-Scratch lab test harness — detected environment",
        "=" * 64,
        f"  OS          : {info.os_name} {info.os_release} ({info.machine})",
        f"  Processor   : {info.processor}",
        f"  Python      : {info.python_version}",
        f"  PyTorch     : {info.torch_version or 'NOT INSTALLED'}",
        f"  CUDA        : {'yes' if info.cuda_available else 'no'}"
        + (
            f" — {info.cuda_device_name} ({info.cuda_total_gb} GB)"
            if info.cuda_available
            else ""
        ),
        f"  Apple MPS   : {'yes' if info.mps_available else 'no'}",
        f"  Accelerator : {info.accelerator.upper()}",
        f"  Labs run on : {info.lab_device.upper()}",
    ]
    for note in info.notes:
        lines.append(f"  note        : {note}")
    lines.append("=" * 64)
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_report(detect()))
