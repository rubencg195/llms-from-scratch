# LLMs From Scratch — Master Curriculum

PhD-level course: demystify modern LLMs by building an ~80M-parameter stack on an RTX 3080 (10 GB VRAM), from $y = mx + b$ through Google's Titans architecture.

**Source of truth:** Markdown in `course/lectures/` and `course/labs/`. Export to Word, PDF, PowerPoint, and JupyterLab 4 notebooks via the toolchain below.

---

## Architecture

### High-level course pipeline

```mermaid
flowchart LR
    subgraph Authoring["Authoring (Markdown)"]
        L["course/lectures/*.md"]
        B["course/labs/**/section-*.md"]
    end

    subgraph Export["Export Toolchain"]
        P["Pandoc"]
        J["Jupytext"]
        N["nbconvert"]
    end

    subgraph Deliverables["Deliverables"]
        PPTX["PowerPoint (.pptx)"]
        DOCX["Word (.docx)"]
        PDF["PDF"]
        IPYNB["JupyterLab 4 (.ipynb)"]
    end

    L --> P
    P --> PPTX
    P --> DOCX
    P --> PDF
    B --> J
    J --> IPYNB
    IPYNB --> N
    N --> PDF
```

### Detailed component map

```mermaid
flowchart TB
    subgraph Phase0["Phase 0 — Math Primer"]
        P0L["Lecture: Tensors, Dot Product, Autograd"]
        P0B["Labs 0.1–0.4"]
    end

    subgraph Phase1["Phase 1 — Dense Transformer"]
        P1L["Lecture: Tokenize → RoPE → Attention"]
        P1B["Labs 1.1–1.5"]
        TS["Dataset: TinyStories"]
    end

    subgraph Phase2["Phase 2 — Instruction Tuning"]
        P2L["Lecture: Chat templates, masked loss"]
        P2B["Labs 2.1–2.4"]
        GSM["GSM8K + Glaive JSON"]
    end

    subgraph Phase3["Phase 3 — QAT"]
        P3L["Lecture: Min-max, STE"]
        P3B["Labs 3.1–3.4"]
    end

    subgraph Phase4["Phase 4 — MoE"]
        P4L["Lecture: Router + experts"]
        P4B["Labs 4.1–4.4"]
        OWT["OpenWebText + Glaive"]
    end

    subgraph Phase5["Phase 5 — TurboQuant"]
        P5L["Lecture: KV cache compression"]
        P5B["Labs 5.1–5.4"]
        NIAH["Needle-in-a-Haystack"]
    end

    subgraph Phase6["Phase 6 — Multimodal"]
        P6L["Lecture: Patch projector"]
        P6B["Labs 6.1–6.4"]
        TIS["TinyImage-Stories"]
    end

    subgraph Phase7["Phase 7 — Full-Duplex Audio"]
        P7L["Lecture: Codec tokens, barge-in"]
        P7B["Labs 7.1–7.4"]
        SI["SpeechInstruct + codec"]
    end

    subgraph Phase8["Phase 8 — Titans TTT"]
        P8L["Lecture: Neural memory, surprise"]
        P8B["Labs 8.1–8.6"]
        ECFA["Extended Complex Facts"]
    end

    P0B --> P1B
    P1B --> TS
    P1B --> P2B
    P2B --> GSM
    P2B --> P3B
    P3B --> P4B
    P4B --> OWT
    P4B --> P5B
    P5B --> NIAH
    P5B --> P6B
    P6B --> TIS
    P6B --> P7B
    P7B --> SI
    P7B --> P8B
    P8B --> ECFA

    subgraph Runtime["Student Runtime"]
        GPU["RTX 3080 — 10 GB VRAM"]
        PT["PyTorch + CUDA"]
        JL["JupyterLab 4"]
    end

    P0B --> GPU
    P8B --> GPU
    GPU --> PT
    JL --> GPU
```

### Repository layout

```
llms-from-scratch/
├── README.md                          ← This file (syllabus + instructor guide)
├── requirements.txt
├── requirements-test.txt              ← Extra deps for the lab test harness
├── pytest.ini
├── jupytext.toml
├── scripts/export-all.sh
├── .vscode/extensions.json
├── tests/                             ← Lab test harness (see "Testing the labs")
│   ├── run_tests.py                   ← Standalone runner (detects machine, runs all labs)
│   ├── test_labs.py                   ← Pytest entry (one test per lab)
│   ├── conftest.py / pytest config
│   ├── system.py                      ← CPU / CUDA / MPS detection
│   ├── extract.py                     ← Markdown → ordered code cells
│   ├── core.py / exec_lab.py          ← Isolated, ordered cell execution
│   ├── stubs.py                       ← Offline dataset / tokenizer fakes
│   ├── preamble.py                    ← Headless plots + RNG seeding
│   └── manifest.py                    ← Lab discovery + slow/network flags
└── course/
    ├── lectures/                      ← Slide markdown (Pandoc → PPTX/PDF/DOCX)
    │   ├── phase-00-bridging-the-gap.md
    │   ├── phase-01-dense-core.md
    │   └── … phase-08-titans.md
    ├── labs/                          ← Jupytext light markdown (→ .ipynb)
    │   ├── phase-00-bridging-the-gap/
    │   │   ├── section-0-1-tensors.md
    │   │   └── …
    │   └── phase-08-titans/
    │       └── section-8-6-vram-profiling.md
    └── interactive/                   ← Gamified browser app (React + Tailwind + Three.js)
        ├── package.json
        ├── index.html
        └── src/
            ├── data/curriculum.ts     ← 9 phases × interactive modules
            ├── content.ts             ← loads all lectures + labs from ../lectures & ../labs
            ├── store/progress.ts      ← XP, levels, streaks, achievements, reads
            ├── components/Markdown.tsx ← markdown + math + code renderer
            ├── pages/                 ← journey map, phase, lecture, lab, module, trophies
            └── modules/phaseXX/*.tsx   ← one playground per concept
```

---

## Syllabus

| Phase | Title | Objective | Dataset |
|-------|-------|-----------|---------|
| 0 | Bridging the Gap | Python + algebra → tensors, autograd | Synthetic arrays |
| 1 | The Dense Core | 80M Transformer from scratch | TinyStories |
| 2 | Instruction Tuning | Chat format, tool JSON | GSM8K + Glaive |
| 3 | QAT | Quantization-aware training | TinyStories + Glaive mix |
| 4 | Mixture of Experts | Router + 4 experts, same VRAM | OpenWebText + Glaive |
| 5 | TurboQuant | 3.5-bit KV cache | Needle-in-a-Haystack |
| 6 | Encoder-Free Multimodal | Gemma 4-style patches | TinyImage-Stories |
| 7 | Full-Duplex Audio | Streaming + interruptions | SpeechInstruct + codec |
| 8 | Titans | Test-time training memory | Extended Complex Facts |

**Hardware target:** NVIDIA RTX 3080, 10 GB VRAM, ~80M parameters for the core text model.

---

## How to use this course (read this first)

This course is **self-contained**. If you can write a basic Python `for` loop and you remember the line equation $y = mx + b$ from algebra, you have everything you need to start. Every new idea is introduced from scratch, in order, and each phase only relies on what earlier phases already taught.

**The golden rule:** do the phases in order. Each phase ends with a *Bridge to the Next Phase* slide, and each lab ends with a *Where This Leads Next* note, so you always know how today's idea feeds into tomorrow's.

**What each phase gives you:**

- A **lecture** (`course/lectures/phase-XX-*.md`) — the concepts, analogies, and math, exported to slides.
- A set of **labs** (`course/labs/phase-XX-*/section-*.md`) — hands-on notebooks where you write and run the code.

**How to study a single phase:**

1. Read (or attend) the lecture first — it builds the mental model.
2. Open the labs in order and run every cell. Change numbers and break things on purpose; intuition comes from experimenting.
3. Each lab opens with **What You Need to Know First** (a quick recap of prerequisites, all taught earlier) and closes with **Key Takeaway** + **Where This Leads Next**.
4. The **Further Reading (Optional)** section at the end of every lecture and lab lists the original research papers. **These are strictly optional enrichment — you never need to read them to continue.**

**If you get stuck:** re-read the "What You Need to Know First" box for that section; it names the exact earlier section where the concept was introduced.

---

## Prerequisites

**Required (that's all):**

- Basic Python: variables, lists, loops, functions, and `import`.
- High-school algebra: the line equation $y = mx + b$, and what "multiply and add" means.

**Explicitly NOT required** (we teach these as we go):

- Calculus — PyTorch's autograd computes all derivatives for you (Phase 0 explains the idea).
- Linear algebra beyond multiply-and-add — dot products and matrices are introduced in Phase 0.
- Any prior machine learning, deep learning, or PyTorch experience.

**Software:** Python 3.11+, and the packages in `requirements.txt`. A CUDA GPU (RTX 3080 target) is ideal, but every lab also runs on CPU (just slower).

---

## Instructor setup

### 1. System prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| Python 3.11+ | Labs & export scripts | `pyenv` or system Python |
| Pandoc 3.x | Lectures → PPTX, PDF, DOCX | [pandoc.org/installing](https://pandoc.org/installing.html) |
| LaTeX (optional) | Higher-quality PDF | MacTeX / TeX Live |
| CUDA 12.x + cuDNN | Student GPU training | NVIDIA driver + toolkit |

### 2. Python environment

```bash
cd llms-from-scratch
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
chmod +x scripts/export-all.sh
```

**Verify the labs run** before teaching (see [Testing the labs](#testing-the-labs)):

```bash
pip install -r requirements-test.txt
python tests/run_tests.py --quick   # fast smoke test; drop --quick for the full run
```

### 3. VS Code / Cursor extensions

Install the recommended set (Cursor/VS Code will prompt via `.vscode/extensions.json`):

| Extension | ID | Role |
|-----------|-----|------|
| Python | `ms-python.python` | Interpreter, linting |
| Jupyter | `ms-toolsai.jupyter` | Open `.ipynb`, run cells |
| Markdown All in One | `yzhang.markdown-all-in-one` | TOC, preview |
| Marp for VS Code | `marp-team.marp-vscode` | Optional slide preview |
| Pandoc Citer | `chrischinchilla.vscode-pandoc` | Single-file Pandoc export from editor |

**Command Palette → “Extensions: Show Recommended Extensions” → Install All.**

### 4. JupyterLab 4 extensions (student & instructor)

```bash
pip install jupyterlab>=4.0 jupytext
jupyter labextension list
```

Recommended JupyterLab 4 extensions:

```bash
pip install jupyterlab-git
jupyter labextension enable jupyterlab-jupytext   # if packaged separately
```

Configure Jupytext pairing globally (once per machine):

```bash
jupyter labextension enable jupytext
# Or in JupyterLab: Settings → Advanced Settings Editor → Jupytext → add "light" for .md
```

The repo root `jupytext.toml` sets default kernel metadata for all paired notebooks.

---

## Export workflows

Generated files go to `exports/` (gitignored). **Always edit markdown source**, then re-export.

### Lectures → PowerPoint (primary)

Lectures use Pandoc slide breaks (`---` on its own line). Level-2 headings (`##`) start new slides.

**Single lecture:**

```bash
pandoc course/lectures/phase-01-dense-core.md \
  -o exports/lectures/pptx/phase-01-dense-core.pptx \
  --slide-level=2 \
  -t pptx
```

**All lectures:**

```bash
./scripts/export-all.sh
```

**Tips for better PPTX:**

- Keep one main idea per slide; use `##` not `#` for slide titles after the title slide.
- Put `$...$` math inline; Pandoc converts to Office Math.
- Optional branded template: add `--reference-doc=instructor/templates/reference.pptx` once you create a master deck.

**Lectures → Word / PDF:**

```bash
pandoc course/lectures/phase-01-dense-core.md -o exports/lectures/docx/phase-01-dense-core.docx

pandoc course/lectures/phase-01-dense-core.md \
  -o exports/lectures/pdf/phase-01-dense-core.pdf \
  --slide-level=2 -t beamer
```

### Labs → JupyterLab 4 `.ipynb` (primary)

Labs use [Jupytext "light" format](https://jupytext.readthedocs.io/en/latest/formats.html#light-format): YAML front matter + fenced `python` blocks become code cells; other markdown becomes markdown cells.

**Single lab:**

```bash
jupytext --to ipynb \
  course/labs/phase-00-bridging-the-gap/section-0-1-tensors.md \
  -o exports/labs/ipynb/phase-00-bridging-the-gap/section-0-1-tensors.ipynb
```

**Open paired notebook in JupyterLab 4:**

```bash
jupyter lab course/labs/
# Jupytext auto-pairs .md ↔ .ipynb when configured; or export first and open exports/labs/ipynb/
```

**Round-trip (student edits ipynb → sync back to markdown):**

```bash
jupytext --sync exports/labs/ipynb/phase-00-bridging-the-gap/section-0-1-tensors.ipynb
```

**Lab → PDF (after export to ipynb):**

```bash
jupyter nbconvert --to pdf exports/labs/ipynb/phase-00-bridging-the-gap/section-0-1-tensors.ipynb
```

### Batch export everything

```bash
./scripts/export-all.sh
```

Outputs:

- `exports/lectures/pptx/*.pptx`
- `exports/lectures/docx/*.docx`
- `exports/lectures/pdf/*.pdf`
- `exports/labs/ipynb/**/*.ipynb`

---

## Testing the labs

Every lab is a runnable notebook, so "is the course still correct?" is answered by
**actually executing the real code in each `.md` file**. The harness in `tests/`
does exactly that — it extracts the Python cells from the markdown (the same cell
split Jupytext uses to build the `.ipynb`), runs them in order in an isolated
namespace, and reports any failure with the offending cell and traceback.

It is designed to run unchanged on both target machines and to auto-detect which
one it is on:

| Machine | Accelerator detected | Device labs run on |
|---------|----------------------|--------------------|
| MacBook Air M5 (24 GB) | Apple MPS (reported) | CPU* |
| Windows i7 + RTX 3080 (10 GB) | CUDA | CUDA (GPU) |

\* The lab code selects its device with `device = "cuda" if torch.cuda.is_available() else "cpu"`,
so on Apple Silicon it runs on CPU by design. The harness still detects and reports MPS.

### Quick start

```bash
# from the repo root, in your course virtualenv
pip install -r requirements.txt -r requirements-test.txt

# Option A — standalone runner (recommended): detects the machine, runs every lab
python tests/run_tests.py

# Option B — pytest: one test per lab, rich tracebacks
pytest tests/
```

Both auto-detect the system and print it before running. The standalone runner
also auto-picks **online** mode (real dataset downloads) when the network is
reachable and **offline** mode (synthetic data) when it is not.

### Common commands

```bash
python tests/run_tests.py --quick         # skip the slow training labs (fast smoke test)
python tests/run_tests.py --offline        # force synthetic data (air-gapped / no downloads)
python tests/run_tests.py --online         # force real Hugging Face downloads
python tests/run_tests.py --phase 1        # only Phase 1 labs
python tests/run_tests.py --only attention # substring filter on the lab path
python tests/run_tests.py --jobs 4         # run 4 labs in parallel (use --jobs 1 on a single GPU)

pytest tests/ -m "not slow"               # skip heavy training labs
pytest tests/ -m "not network"            # skip dataset-loading labs
pytest tests/ --online                     # use real downloads
pytest tests/ -k tokenization             # single lab by name
```

> On a single-GPU box (RTX 3080) keep `--jobs 1` so labs don't contend for the
> 10 GB of VRAM. Parallel jobs are useful on the M5 (CPU) for faster smoke runs.

### How it works

```mermaid
flowchart LR
    A["course/labs/**/section-*.md"] --> B["extract.py<br/>jupytext cell split<br/>(regex fallback)"]
    B --> C["core.py<br/>ordered exec in one namespace"]
    P["preamble.py<br/>Agg backend · seed RNGs · no-op plt.show"] --> C
    S["stubs.py<br/>offline dataset / tiktoken fakes"] -. offline mode .-> C
    SY["system.py<br/>CPU / CUDA / MPS"] --> R
    C --> R["run_tests.py / test_labs.py<br/>PASS / FAIL / TIMEOUT report"]
```

Key design decisions:

- **Real code, real order.** A lab is treated as one notebook: later cells depend
  on earlier ones, so all cells share a single namespace and run top-to-bottom.
  Tests fail exactly where a student's kernel would.
- **System auto-detection** (`system.py`) reports OS, PyTorch, CUDA device + VRAM,
  and Apple MPS, and computes the device the labs will actually use.
- **Two data modes.** *Online* exercises the real `load_dataset` downloads that
  students hit. *Offline* installs synthetic stand-ins (`stubs.py`) that mirror the
  Hugging Face `Dataset` indexing contract exactly — so air-gapped runs still catch
  API-misuse bugs instead of hiding them.
- **Isolation + timeouts.** `run_tests.py` runs each lab in its own subprocess, so
  state never leaks between labs and a hung lab is killed by a per-lab timeout
  (longer for labs tagged `slow`). Working directories are throwaway temp folders,
  so checkpoints / PNGs / WAVs never touch the repo.
- **Headless + reproducible** (`preamble.py`): forces the matplotlib `Agg` backend,
  neutralises `plt.show()`, and seeds Python/NumPy/Torch before each lab.
- **Markers** (`network`, `slow`, `extra_deps`) are derived per lab in `manifest.py`
  so you can include/exclude categories.

### Detailed harness architecture

```mermaid
flowchart TD
    subgraph Detect["1 · Detect environment"]
        SYS["system.detect()<br/>OS · torch · CUDA name+VRAM · MPS<br/>→ lab_device = cuda|cpu"]
        NET{"network<br/>reachable?"}
    end

    subgraph Discover["2 · Discover & classify labs"]
        DISC["manifest.discover()<br/>glob course/labs/phase-*/section-*.md"]
        FLAGS["per-lab flags<br/>network = contains load_dataset<br/>extra = sklearn / soundfile<br/>slow = curated set"]
        DISC --> FLAGS
    end

    subgraph Mode["3 · Choose data mode"]
        ON["ONLINE<br/>real datasets + tiktoken"]
        OFF["OFFLINE<br/>stubs.install_offline_stubs()<br/>FakeDataset · byte-level tokenizer"]
    end

    subgraph Exec["4 · Execute one lab (per subprocess)"]
        EX["exec_lab.py → core.run_lab()"]
        PRE["exec preamble<br/>Agg · seeds · plt.show=noop"]
        CELLS["extract.extract_cells()<br/>ordered python code cells"]
        LOOP["for cell in cells:<br/>exec in shared namespace<br/>chdir → temp workdir"]
        EX --> PRE --> CELLS --> LOOP
    end

    subgraph Report["5 · Aggregate"]
        RES["LabResult<br/>pass | fail(cell#, traceback) | timeout"]
        SUM["summary table + exit code<br/>(non-zero if any fail)"]
        RES --> SUM
    end

    SYS --> NET
    NET -- yes --> ON
    NET -- no --> OFF
    FLAGS --> EX
    ON --> EX
    OFF --> EX
    LOOP --> RES

    PYTEST["pytest test_labs.py<br/>(in-process, markers)"] -. alternative front-end .-> EX
```

### Interpreting results

- `PASS` — every cell in the lab executed without raising.
- `FAIL` — a cell raised; the report shows the cell index, exception type/message,
  and (in pytest) the cell source plus traceback.
- `TIME` — the lab exceeded its timeout (raise `--timeout`, or it may indicate a hang).

For full-fidelity validation before teaching, run on the RTX 3080 box in online mode:
`python tests/run_tests.py --online --jobs 1`. Use `--quick` on the laptop for a fast
sanity check that skips the heavy training labs (which are slow on CPU but fast on GPU).

---

## Interactive playground (gamified)

`course/interactive/` is a browser app that contains **the entire course** — all **9
lectures**, all **39 lab sections**, and **23 playable, gamified mini-apps** — in one place.
It is designed especially for students who learn better by poking at something than by
reading. No JupyterLab, no GPU, no install of the Python stack: it runs entirely
client-side and every concept builds intuition *before* the student opens the matching lab.

The lecture and lab markdown is read straight from `course/lectures/` and `course/labs/`
(bundled at build time via Vite's `import.meta.glob`), so the website is always in sync with
the source material — there is no duplicated content to maintain:

- **Lectures** render as a keyboard-navigable **slide reader** (← / → arrows, progress bar,
  collapsible speaker notes) — one deck per phase.
- **Labs** render as fully formatted reading pages with LaTeX math (KaTeX), GitHub tables,
  and syntax-styled Python code cells — the exact code students run in JupyterLab.
- **Playgrounds** are the interactive "expand" layer that makes each concept tangible.

Every phase page lays out the suggested flow: **watch the lecture → play the concepts → do
the labs.**

It is built with **React + TypeScript + Vite**, styled with **Tailwind CSS v4**, animated
with **Framer Motion**, renders content with **react-markdown + remark-gfm + KaTeX**, and
uses **Three.js** (via React Three Fiber) *only* where 3D genuinely helps (the
embedding-space module) — everything else is fast 2D SVG/canvas.

### Run it

```bash
cd course/interactive
npm install
npm run dev        # http://localhost:5173
npm test           # Vitest unit + component tests
npm run test:e2e   # Playwright browser interaction tests
npm run test:all   # both suites
npm run test:watch # re-run Vitest on file changes
npm run test:e2e:ui # Playwright interactive UI mode
# or: npm run build && npm run preview   (static build in dist/)
```

### Deploy

Production URL: **https://rubenchevez.com/llms-from-scratch/**

The build uses base path `/llms-from-scratch/` so assets resolve correctly under the parent site. From the **repo root**:

```bash
npm install          # once — installs firebase-tools
npm run build        # production build → course/interactive/dist/
npm run deploy       # upload dist/ to Firebase Hosting (if using Firebase)
```

Copy `course/interactive/dist/` into your parent site's hosting at `/llms-from-scratch/` on [rubenchevez.com](https://rubenchevez.com/). The app uses hash routing (`#/…`), so no SPA rewrite rules are required.

For local preview of the production base path:

```bash
cd course/interactive
npm run build && npm run preview
# open http://localhost:4173/llms-from-scratch/
```

### Testing the interactive app

**Vitest** (unit + component) and **Playwright** (end-to-end browser interactions) cover the gamified app:

```bash
cd course/interactive
npm test              # Vitest — fast, isolated
npm run test:e2e      # Playwright — builds app, runs browser tests
npm run test:all      # both
npm run test:e2e:ui   # Playwright interactive UI mode
```

Playwright starts a production preview server automatically, clears `localStorage` before each test, and exercises real clicks, keyboard navigation, route transitions, XP/trophy persistence, and all 24 playground loads.

| Area | Vitest | Playwright E2E |
|------|--------|----------------|
| `src/store/progress.ts` | XP, streaks, achievements, journey stats, persist/rehydrate | XP after lecture/lab/playground, reset, trophy unlocks |
| `src/content.ts` | All 9 lectures + 39 labs load from markdown | Lecture slides, lab reader, markdown render |
| `src/data/curriculum.ts` | Phase/module structure, XP totals | All 9 phase pages, 24 playground smoke loads |
| `src/modules/registry.tsx` | Every module has a component | Mark complete, next/finish phase navigation |
| Routing & layout | `App.tsx`, Home, Phase | Navbar, breadcrumbs, invalid routes, phase cards |
| Gamification UI | — | Trophies, journey counters, sound toggle, full phase-0 flow |

E2E specs live in `course/interactive/e2e/` (`navigation`, `home`, `progress`, `phase`, `lecture`, `lab`, `playground`, `trophies`, `journey-flow`).

Progress is stored under the key `llms-interactive-progress` in `localStorage`; Vitest uses an
in-memory storage shim so runs are isolated and do not touch your browser profile.

The production build is fully static (relative `base`), so `dist/` can be dropped on any
static host or opened behind a simple file server.

### Gamification

- **XP + levels** — playgrounds award XP the first time you reach their "aha" interaction; reading a lecture (+20) or opening a lab (+10) also grants XP. Levels scale up a curve.
- **Achievements & streaks** — unlock trophies (first module, finish a phase, 3-day streak, open a 3D scene, reach level 5, complete everything) with confetti + chime.
- **Journey map** — the home screen is a phase-by-phase map showing each phase's lecture, lab count, playground count, and progress ring.
- **Persistence** — all progress is saved to `localStorage`; nothing leaves the device.
- **Animated transitions** between phases, sections, and modules for a smooth, game-like flow.

### The 24 playgrounds (mapped to the course)

| Phase | Interactive modules |
|-------|---------------------|
| 0 — Bridging the Gap | Tensor Explorer · Dot-Product Radar · Gradient-Descent Arcade · Fit-the-Line |
| 1 — Dense Core | Tokenizer Lab · **Embedding Galaxy (3D)** · RoPE Rotator · Attention Spotlight |
| 2 — Instruction Tuning | Chat-Template Builder · Masked-Loss Painter · Tool-Call Simulator |
| 3 — QAT | Bit Crusher (quantization grid) · Straight-Through Estimator |
| 4 — MoE | Token Router · Load-Balance Juggler |
| 5 — TurboQuant | KV-Cache Meter · PolarQuant Spinner |
| 6 — Multimodal | Patchify Studio · Modality Mixer |
| 7 — Audio | Waveform → Tokens · Barge-In Booth |
| 8 — Titans | Surprise-o-Meter · Neural Memory Vault · O(1) vs Linear |

Each module mirrors a specific lab section (shown in-app) and ends with a plain-language
"Key insight" tying the interaction back to the concept.

### Architecture

```mermaid
flowchart TD
    DATA["data/curriculum.ts<br/>9 phases × modules (+ XP, colors, icons)"]
    CONTENT["content.ts<br/>import.meta.glob → all lectures + labs<br/>(front matter, slides, titles parsed)"]
    SRC["course/lectures/*.md<br/>course/labs/**/section-*.md"]
    STORE["store/progress.ts (Zustand + localStorage)<br/>XP · levels · streaks · achievements · reads"]

    subgraph Pages["Routed pages (HashRouter + Framer Motion)"]
        HOME["Home — journey map"]
        PHASE["Phase — lecture + labs + playgrounds"]
        LEC["LecturePage — slide reader"]
        LAB["LabPage — rendered lab + code"]
        MODPAGE["ModulePage — hosts one playground"]
        TROPHY["Trophies — achievements"]
    end

    MD["components/Markdown.tsx<br/>react-markdown + remark-gfm + KaTeX"]
    REG["modules/registry.tsx<br/>lazy-loads each playground (code-split)"]
    MODS["modules/phaseXX/*.tsx<br/>self-contained activities → onDiscover()"]
    THREE["Three.js / R3F<br/>(only the 3D Embedding Galaxy chunk)"]

    SRC --> CONTENT
    DATA --> HOME --> PHASE
    CONTENT --> PHASE
    PHASE --> LEC
    PHASE --> LAB
    PHASE --> MODPAGE
    CONTENT --> LEC & LAB
    LEC & LAB --> MD
    DATA --> REG --> MODPAGE
    REG --> MODS
    MODS -. "onDiscover()" .-> MODPAGE
    MODPAGE -- "award XP + confetti" --> STORE
    LEC & LAB -- "markRead() + XP" --> STORE
    STORE --> TROPHY
    MODS -. "3D only" .-> THREE
```

Modules never touch the store directly — they emit a single `onDiscover()` when the student
hits the key insight, and the host page converts that into XP, a confetti burst, and a
celebratory chime. Lecture and lab pages call `markRead()` to award reading XP once. This
keeps each piece small, self-contained, and easy to extend: add an entry to `curriculum.ts`
and a component to `modules/`, and it appears on the map — while lecture/lab content flows
in automatically from the markdown sources.

---

## Authoring conventions

### Lectures (`course/lectures/`)

- YAML metadata block: `title`, `subtitle`, `author`.
- `---` horizontal rule = new slide.
- `## Slide Title` for content slides.
- Speaker notes: HTML comment `<!-- notes: ... -->` (visible in Marp; ignored by Pandoc PPTX unless using `--reference-doc` with notes master).

### Labs (`course/labs/phase-XX-*/section-*.md`)

- Jupytext front matter with `jupytext` + `kernelspec` keys (see any lab file).
- One concept per section; executable Python in fenced blocks.
- Use `# %%` only if you switch to percent format; this course standardizes on **light** format.
- Keep VRAM-safe defaults: `batch_size=8`, `block_size=256` until Phase 5+.

---

## Teaching notes

1. **Phase 0** is mandatory for students weak on linear algebra; skip only if they already use PyTorch daily.
2. **Phase 1** checkpoint is the backbone reused in Phases 2–8; save `checkpoints/phase1_80m.pt`.
3. **VRAM budget:** log `torch.cuda.max_memory_allocated()` at the end of every lab (Phase 8.6 formalizes this).
4. Datasets are downloaded inside labs via `datasets` / custom loaders; large files stay out of git.

---

## Glossary (plain-language)

Quick definitions for terms used throughout the course. Each is explained in full where it first appears.

| Term | Plain meaning |
|------|---------------|
| **Tensor** | A multi-dimensional array of numbers (a number, list, table, or cube of numbers). |
| **Weight / bias** | The $m$ and $b$ in $y = mx + b$ — the numbers the model learns. |
| **Parameter** | Any learnable number in the model (weights and biases). |
| **Dot product** | Multiply matching entries of two lists and add them up; measures similarity. |
| **Embedding** | A learned vector (list of numbers) that represents a token as a point in space. |
| **Token** | A chunk of text (word or sub-word) turned into an integer ID. |
| **Logits** | The raw, pre-probability scores a model outputs before softmax. |
| **Softmax** | Turns a list of scores into probabilities that sum to 1. |
| **Loss** | A single number measuring how wrong the model is; training shrinks it. |
| **Gradient** | The direction/size of a nudge to each weight that reduces the loss. |
| **Autograd** | PyTorch's automatic gradient calculator — no hand calculus needed. |
| **Attention** | A mechanism that lets each token look at others via dot products. |
| **RoPE** | Rotary Position Embedding — encodes word order by rotating vectors. |
| **Quantization** | Storing numbers with fewer bits (e.g., 4-bit ints) to save memory. |
| **STE** | Straight-Through Estimator — lets gradients flow through rounding. |
| **MoE** | Mixture of Experts — route each token to one of several sub-networks. |
| **KV cache** | Stored Keys/Values of past tokens so the model needn't recompute them. |
| **Codec** | A model that turns audio waveforms into discrete integer tokens. |
| **TTT** | Test-Time Training — updating a small memory network during inference. |
| **VRAM** | The GPU's memory; our hard limit is 10 GB (RTX 3080). |

---

## Master bibliography (all optional)

Every paper referenced in the course, gathered here for convenience. **None of these are required reading** — the course is fully self-contained. They are listed for students who want to go deeper into a topic.

**Foundations (Phase 0)**

- Rumelhart, Hinton, & Williams (1986). *Learning representations by back-propagating errors*. Nature.
- Cybenko (1989). *Approximation by Superpositions of a Sigmoidal Function*.
- Goodfellow, Bengio, & Courville (2016). *Deep Learning*. MIT Press (free at deeplearningbook.org).
- Paszke et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS.
- Kingma & Ba (2015). *Adam: A Method for Stochastic Optimization*. ICLR.

**Transformers & language modeling (Phase 1)**

- Vaswani et al. (2017). *Attention Is All You Need*. NeurIPS.
- Sennrich, Haddow, & Birch (2016). *Neural Machine Translation of Rare Words with Subword Units (BPE)*. ACL.
- Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv:2104.09864.
- Ba, Kiros, & Hinton (2016). *Layer Normalization*. arXiv:1607.06450.
- Radford et al. (2019). *Language Models are Unsupervised Multitask Learners (GPT-2)*. OpenAI.
- Eldan & Li (2023). *TinyStories*. arXiv:2305.07759.
- Dao et al. (2022). *FlashAttention*. NeurIPS.

**Instruction tuning & tools (Phase 2)**

- Ouyang et al. (2022). *Training language models to follow instructions with human feedback (InstructGPT)*. NeurIPS.
- Wei et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in LLMs*. NeurIPS.
- Cobbe et al. (2021). *Training Verifiers to Solve Math Word Problems (GSM8K)*. arXiv:2110.14168.
- Schick et al. (2023). *Toolformer*. NeurIPS.
- DeepSeek-AI (2025). *DeepSeek-R1*. arXiv:2501.12948.

**Quantization (Phases 3 & 5)**

- Bengio, Léonard, & Courville (2013). *Estimating or Propagating Gradients Through Stochastic Neurons (STE)*. arXiv:1308.3432.
- Jacob et al. (2018). *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*. CVPR.
- Dettmers et al. (2022). *LLM.int8()*. NeurIPS.
- Frantar et al. (2022). *GPTQ*. arXiv:2210.17323.
- Chee et al. (2023). *QuIP*. NeurIPS.
- Tseng et al. (2024). *QuIP#*. ICML.
- Ashkboos et al. (2024). *QuaRot*. arXiv:2404.00456.
- Liu et al. (2024). *KIVI*. ICML.

**Mixture of Experts (Phase 4)**

- Shazeer et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer*. ICLR.
- Lepikhin et al. (2020). *GShard*. arXiv:2006.16668.
- Fedus, Zoph, & Shazeer (2021). *Switch Transformers*. JMLR.
- Jiang et al. (2024). *Mixtral of Experts*. arXiv:2401.04088.

**Multimodal (Phase 6)**

- Dosovitskiy et al. (2020). *An Image is Worth 16x16 Words (ViT)*. ICLR.
- Radford et al. (2021). *CLIP*. ICML.
- Alayrac et al. (2022). *Flamingo*. NeurIPS.
- Bavishi et al. (2023). *Fuyu-8B*. Adept AI.
- Liu et al. (2023). *Visual Instruction Tuning (LLaVA)*. NeurIPS.

**Audio (Phase 7)**

- van den Oord, Vinyals, & Kavukcuoglu (2017). *Neural Discrete Representation Learning (VQ-VAE)*. NeurIPS.
- Zeghidour et al. (2021). *SoundStream*. IEEE/ACM TASLP.
- Défossez et al. (2022). *High Fidelity Neural Audio Compression (EnCodec)*. arXiv:2210.13438.
- Wang et al. (2023). *VALL-E*. arXiv:2301.02111.
- Défossez et al. (2024). *Moshi*. arXiv:2410.00037.

**Memory & test-time training (Phase 8)**

- Graves, Wayne, & Danihelka (2014). *Neural Turing Machines*. arXiv:1410.5401.
- Finn, Abbeel, & Levine (2017). *Model-Agnostic Meta-Learning (MAML)*. ICML.
- Katharopoulos et al. (2020). *Transformers are RNNs (Linear Attention)*. ICML.
- Gu & Dao (2023). *Mamba*. arXiv:2312.00752.
- Sun et al. (2024). *Learning to (Learn at Test Time)*. arXiv:2407.04620.
- Behrouz, Zhong, & Mirrokni (2024). *Titans: Learning to Memorize at Test Time*. arXiv:2501.00663.

> Naming note: *TurboQuant* and *PolarQuant* are teaching names used in this course for the rotation-then-quantize family of methods pioneered by QuIP#/QuaRot.

---

## License

MIT — adapt freely for classroom use.
