"""Per-lab metadata: discovery + classification flags.

Flags drive test selection (markers in pytest, filters in the CLI runner):
  * network : the lab calls ``load_dataset`` (downloads on first run)
  * slow    : the lab trains a sizeable model or runs long/nested loops
  * extra   : extra pip packages beyond requirements.txt (e.g. scikit-learn)

``network`` and ``extra`` are auto-detected by scanning file contents so they
stay correct as labs change. ``slow`` is a curated set (loop cost is not
reliably inferable from text).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

LABS_DIR = Path(__file__).resolve().parent.parent / "course" / "labs"

# Labs with heavy training loops / large models / O(n^2) stress tests.
# Relative to course/labs/. These run slowly on CPU (fine on RTX 3080 / M5 GPU).
SLOW = {
    "phase-00-bridging-the-gap/section-0-3-autograd.md",
    "phase-00-bridging-the-gap/section-0-4-first-layer.md",
    "phase-01-dense-core/section-1-5-training-loop.md",
    "phase-02-instruction-tuning/section-2-3-gsm8k-training.md",
    "phase-03-qat/section-3-3-straight-through-estimator.md",
    "phase-03-qat/section-3-4-qat-finetune.md",
    "phase-04-moe/section-4-3-load-balancing.md",
    "phase-04-moe/section-4-4-specialization.md",
    "phase-05-turboquant/section-5-3-35bit-compression.md",
    "phase-06-multimodal/section-6-3-spatial-coords.md",
    "phase-08-titans/section-8-2-neural-memory.md",
    "phase-08-titans/section-8-4-momentum-decay.md",
    "phase-08-titans/section-8-6-vram-profiling.md",
}


@dataclass
class LabSpec:
    path: Path
    rel: str
    phase: str
    network: bool
    slow: bool
    extra: List[str]

    @property
    def test_id(self) -> str:
        return self.rel.replace("/", "::").replace(".md", "")


def _classify(path: Path, rel: str) -> LabSpec:
    text = path.read_text(encoding="utf-8")
    network = "load_dataset(" in text
    extra: List[str] = []
    if "sklearn" in text:
        extra.append("scikit-learn")
    if "import soundfile" in text or "soundfile" in text:
        extra.append("soundfile")
    return LabSpec(
        path=path,
        rel=rel,
        phase=rel.split("/")[0],
        network=network,
        slow=rel in SLOW,
        extra=sorted(set(extra)),
    )


def discover(labs_dir: Path = LABS_DIR) -> List[LabSpec]:
    """Return every lab section file, sorted, with classification flags."""
    specs = []
    for md in sorted(labs_dir.glob("phase-*/section-*.md")):
        rel = md.relative_to(labs_dir).as_posix()
        specs.append(_classify(md, rel))
    return specs


if __name__ == "__main__":
    for s in discover():
        tags = []
        if s.network:
            tags.append("network")
        if s.slow:
            tags.append("slow")
        tags += s.extra
        print(f"{s.rel:60s} {','.join(tags)}")
