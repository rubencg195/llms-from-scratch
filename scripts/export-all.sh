#!/usr/bin/env bash
# Export all course markdown to Word, PDF, PowerPoint, and Jupyter notebooks.
set -euo pipefail
set +H

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPORT="${ROOT}/exports"
LECTURES="${ROOT}/course/lectures"
LABS="${ROOT}/course/labs"

# ── Pre-flight checks ──────────────────────────────────────────────────────
if ! command -v pandoc &>/dev/null; then
  echo "ERROR: pandoc not found. Install from https://pandoc.org/installing.html"
  exit 1
fi
if ! command -v jupytext &>/dev/null; then
  echo "ERROR: jupytext not found. Run: pip install jupytext"
  exit 1
fi
echo "pandoc $(pandoc --version | head -1)"
echo "jupytext $(jupytext --version)"

# ── Create output dirs ─────────────────────────────────────────────────────
mkdir -p "${EXPORT}/lectures/pptx" "${EXPORT}/lectures/pdf" "${EXPORT}/lectures/docx"
mkdir -p "${EXPORT}/labs/ipynb" "${EXPORT}/labs/pdf"

# ── Lectures → PPTX, PDF (beamer), DOCX ───────────────────────────────────
echo ""
echo "==> Exporting lectures (Pandoc → PPTX, PDF, DOCX)"
for md in "${LECTURES}"/phase-*.md; do
  base="$(basename "${md}" .md)"

  pandoc "${md}" \
    -o "${EXPORT}/lectures/pptx/${base}.pptx" \
    --slide-level=2 \
    -t pptx

  # PDF via beamer (requires LaTeX); skip gracefully if unavailable
  if command -v pdflatex &>/dev/null || command -v xelatex &>/dev/null; then
    pandoc "${md}" \
      -o "${EXPORT}/lectures/pdf/${base}.pdf" \
      --slide-level=2 \
      -V geometry:margin=1in \
      -t beamer 2>/dev/null || echo "    WARNING: beamer PDF failed for ${base} (LaTeX issue)"
  else
    echo "    SKIP PDF: no LaTeX found"
  fi

  pandoc "${md}" \
    -o "${EXPORT}/lectures/docx/${base}.docx"

  echo "    ${base}"
done

# ── Labs → IPYNB ───────────────────────────────────────────────────────────
echo ""
echo "==> Exporting labs (Jupytext → IPYNB)"
find "${LABS}" -name 'section-*.md' | sort | while read -r md; do
  rel="${md#"${LABS}"/}"
  phase_dir="$(dirname "${rel}")"
  base="$(basename "${md}" .md)"
  out_dir="${EXPORT}/labs/ipynb/${phase_dir}"
  mkdir -p "${out_dir}"
  jupytext --to ipynb "${md}" -o "${out_dir}/${base}.ipynb"
  echo "    ${phase_dir}/${base}.ipynb"
done

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
PPTX_COUNT=$(find "${EXPORT}/lectures/pptx" -name '*.pptx' 2>/dev/null | wc -l | tr -d ' ')
DOCX_COUNT=$(find "${EXPORT}/lectures/docx" -name '*.docx' 2>/dev/null | wc -l | tr -d ' ')
IPYNB_COUNT=$(find "${EXPORT}/labs/ipynb" -name '*.ipynb' 2>/dev/null | wc -l | tr -d ' ')
echo "==> Done. ${PPTX_COUNT} PPTX, ${DOCX_COUNT} DOCX, ${IPYNB_COUNT} IPYNB in ${EXPORT}/"
