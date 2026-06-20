"""Extract executable Python cells from Jupytext light-format lab markdown.

The labs are ``.md`` files with fenced ```python code blocks (Jupytext light
format). When converted to ``.ipynb`` each fence becomes one notebook cell.
This module reproduces that cell split so the tests run *exactly the code a
student runs*, in order.

Two backends:
  1. jupytext (preferred) — reads the file the same way the export pipeline
     does, guaranteeing identical cell boundaries to the shipped notebooks.
  2. regex fallback — used when jupytext is not installed, parses fenced
     ```python ... ``` blocks directly.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

_FENCE_RE = re.compile(
    r"^[ \t]*```[ \t]*python[ \t]*\n(.*?)^[ \t]*```[ \t]*$",
    re.DOTALL | re.MULTILINE,
)
_FRONT_MATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)


def _strip_front_matter(text: str) -> str:
    return _FRONT_MATTER_RE.sub("", text, count=1)


def extract_cells_regex(md_path: Path) -> List[str]:
    text = md_path.read_text(encoding="utf-8")
    text = _strip_front_matter(text)
    return [m.group(1).rstrip("\n") for m in _FENCE_RE.finditer(text)]


def extract_cells_jupytext(md_path: Path) -> List[str]:
    import jupytext  # type: ignore

    nb = jupytext.read(str(md_path))
    return [
        cell["source"]
        for cell in nb.cells
        if cell.get("cell_type") == "code" and cell.get("source", "").strip()
    ]


def extract_cells(md_path) -> List[str]:
    """Return the ordered list of code-cell sources for a lab file."""
    md_path = Path(md_path)
    try:
        cells = extract_cells_jupytext(md_path)
        if cells:
            return cells
    except Exception:
        pass
    return extract_cells_regex(md_path)


if __name__ == "__main__":
    import sys

    for i, c in enumerate(extract_cells(sys.argv[1])):
        print(f"\n----- cell {i} -----")
        print(c)
