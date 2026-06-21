/**
 * Curriculum model that drives the whole app: the journey map, phase pages,
 * module routing, and XP accounting. Each "module" is one interactive activity
 * tied to a lecture/lab concept. Module ids map to lazy components in
 * `src/modules/registry.ts`.
 */

export interface ModuleDef {
  id: string;
  title: string;
  blurb: string;
  /** Which lab section(s) in the course this mirrors / expands. */
  mirrors: string;
  /** XP awarded the first time the student completes it. */
  xp: number;
  /** Marks modules that render a 3D (three.js) scene. */
  threeD?: boolean;
}

export interface PhaseDef {
  id: number;
  slug: string;
  title: string;
  tagline: string;
  /** emoji used as the phase "badge" on the map. */
  icon: string;
  /** tailwind-friendly gradient stops for theming this phase. */
  color: { from: string; to: string; ring: string };
  modules: ModuleDef[];
}

export const PHASES: PhaseDef[] = [
  {
    id: 0,
    slug: "bridging-the-gap",
    title: "Bridging the Gap",
    tagline: "From y = mx + b to tensors & autograd",
    icon: "🧮",
    color: { from: "#22d3ee", to: "#0ea5e9", ring: "#38bdf8" },
    modules: [
      {
        id: "tensor-explorer",
        title: "Tensor Explorer",
        blurb: "Morph a number into a vector, matrix, and cube. Feel what a shape is.",
        mirrors: "Lab 0.1 — Tensors",
        xp: 40,
      },
      {
        id: "dot-product",
        title: "Dot-Product Radar",
        blurb: "Drag two arrows and watch similarity rise and fall in real time.",
        mirrors: "Lab 0.2 — Dot Product",
        xp: 50,
      },
      {
        id: "micrograd",
        title: "Micrograd Chain",
        blurb: "Trace the chain rule backward through a tiny computation graph.",
        mirrors: "Lab 0.25 — Micrograd",
        xp: 55,
      },
      {
        id: "gradient-descent",
        title: "Gradient Descent Arcade",
        blurb: "Roll a ball down a loss curve. Tune the learning rate without crashing.",
        mirrors: "Lab 0.3 — Autograd",
        xp: 60,
      },
      {
        id: "line-fit",
        title: "Fit the Line",
        blurb: "Become the optimizer: drag slope & intercept to minimize the loss.",
        mirrors: "Lab 0.4 — First Layer",
        xp: 60,
      },
    ],
  },
  {
    id: 1,
    slug: "dense-core",
    title: "The Dense Core",
    tagline: "Tokenize → embed → rotate → attend",
    icon: "🔤",
    color: { from: "#a78bfa", to: "#7c3aed", ring: "#a78bfa" },
    modules: [
      {
        id: "tokenizer",
        title: "Tokenizer Lab",
        blurb: "Type anything and watch BPE chop it into colorful tokens & IDs.",
        mirrors: "Lab 1.1 — Tokenization",
        xp: 50,
      },
      {
        id: "bigram-lm",
        title: "Bigram Predictor",
        blurb: "Pick the next word from a count table — the simplest language model.",
        mirrors: "Lab 1.18 — Bigram LM",
        xp: 45,
      },
      {
        id: "embeddings-3d",
        title: "Embedding Galaxy",
        blurb: "Fly through a 3D word-space where similar meanings cluster.",
        mirrors: "Lab 1.2 — Embeddings",
        xp: 70,
        threeD: true,
      },
      {
        id: "rope",
        title: "RoPE Rotator",
        blurb: "See how position is baked in by spinning vectors by their index.",
        mirrors: "Lab 1.3 — RoPE",
        xp: 60,
      },
      {
        id: "attention",
        title: "Attention Spotlight",
        blurb: "Pick a word and watch the heads light up what it pays attention to.",
        mirrors: "Lab 1.4 — Attention",
        xp: 80,
      },
    ],
  },
  {
    id: 2,
    slug: "instruction-tuning",
    title: "Instruction Tuning",
    tagline: "Chat templates, masked loss & tools",
    icon: "💬",
    color: { from: "#34d399", to: "#10b981", ring: "#34d399" },
    modules: [
      {
        id: "chat-template",
        title: "Chat Template Builder",
        blurb: "Snap together user / assistant / tool turns and see the special tokens.",
        mirrors: "Lab 2.1 — Chat Templates",
        xp: 55,
      },
      {
        id: "masked-loss",
        title: "Masked-Loss Painter",
        blurb: "Paint which tokens the model learns from — only the assistant should count.",
        mirrors: "Lab 2.2 — Masked Loss",
        xp: 65,
      },
      {
        id: "tool-call",
        title: "Tool-Call Simulator",
        blurb: "Watch the model emit JSON, call a function, and read the result back.",
        mirrors: "Lab 2.4 — JSON Tool Use",
        xp: 65,
      },
      {
        id: "preference-pairs",
        title: "Preference Picker",
        blurb: "Rank two answers — see how DPO alignment picks the helpful response.",
        mirrors: "Lab 2.5 — Preference Alignment",
        xp: 60,
      },
    ],
  },
  {
    id: 3,
    slug: "qat",
    title: "Quantization-Aware Training",
    tagline: "Shrink weights without losing your mind",
    icon: "🗜️",
    color: { from: "#fbbf24", to: "#f59e0b", ring: "#fbbf24" },
    modules: [
      {
        id: "quantize-grid",
        title: "Bit Crusher",
        blurb: "Drag a bit-width slider and watch smooth weights snap to a grid.",
        mirrors: "Lab 3.1–3.2 — Compression & Fake-Quant",
        xp: 60,
      },
      {
        id: "ste",
        title: "Straight-Through Trick",
        blurb: "See why rounding kills gradients — and how the STE sneaks them through.",
        mirrors: "Lab 3.3 — STE",
        xp: 70,
      },
    ],
  },
  {
    id: 4,
    slug: "moe",
    title: "Mixture of Experts",
    tagline: "Route each token to the right specialist",
    icon: "🧠",
    color: { from: "#f472b6", to: "#db2777", ring: "#f472b6" },
    modules: [
      {
        id: "router",
        title: "Token Router",
        blurb: "Send tokens through a gate and watch them flow to top-k experts.",
        mirrors: "Lab 4.1 — Router",
        xp: 65,
      },
      {
        id: "load-balance",
        title: "Load-Balance Juggler",
        blurb: "Tune the balance loss so no expert gets overloaded (or ignored).",
        mirrors: "Lab 4.3 — Load Balancing",
        xp: 70,
      },
    ],
  },
  {
    id: 5,
    slug: "turboquant",
    title: "TurboQuant KV Cache",
    tagline: "3.5-bit memory for endless context",
    icon: "🚀",
    color: { from: "#60a5fa", to: "#2563eb", ring: "#60a5fa" },
    modules: [
      {
        id: "kv-cache",
        title: "KV-Cache Meter",
        blurb: "Slide context length and watch VRAM explode — then compress it.",
        mirrors: "Lab 5.1 — VRAM Problem",
        xp: 65,
      },
      {
        id: "polarquant",
        title: "PolarQuant Spinner",
        blurb: "Rotate vectors into polar buckets to pack them into a few bits.",
        mirrors: "Lab 5.2 — PolarQuant",
        xp: 70,
      },
    ],
  },
  {
    id: 6,
    slug: "multimodal",
    title: "Encoder-Free Multimodal",
    tagline: "Pixels become patches become tokens",
    icon: "🖼️",
    color: { from: "#2dd4bf", to: "#0d9488", ring: "#2dd4bf" },
    modules: [
      {
        id: "patchify",
        title: "Patchify Studio",
        blurb: "Slice an image into patches and flatten them into the token stream.",
        mirrors: "Lab 6.2 — Patch Projector",
        xp: 65,
      },
      {
        id: "modality-mix",
        title: "Modality Mixer",
        blurb: "Interleave image patches and text and watch one sequence form.",
        mirrors: "Lab 6.4 — Modality Mixing",
        xp: 65,
      },
    ],
  },
  {
    id: 7,
    slug: "audio",
    title: "Full-Duplex Audio",
    tagline: "Listen and speak at the same time",
    icon: "🎧",
    color: { from: "#fb923c", to: "#ea580c", ring: "#fb923c" },
    modules: [
      {
        id: "audio-tokens",
        title: "Waveform → Tokens",
        blurb: "Watch a sound wave get quantized into discrete codec tokens.",
        mirrors: "Lab 7.2 — Audio Tokens",
        xp: 65,
      },
      {
        id: "barge-in",
        title: "Barge-In Booth",
        blurb: "Interrupt the model mid-sentence and see full-duplex turn-taking.",
        mirrors: "Lab 7.4 — Barge-In Loop",
        xp: 70,
      },
    ],
  },
  {
    id: 8,
    slug: "titans",
    title: "Titans: Memory at Test Time",
    tagline: "A model that keeps learning while it runs",
    icon: "🏛️",
    color: { from: "#c084fc", to: "#9333ea", ring: "#c084fc" },
    modules: [
      {
        id: "surprise",
        title: "Surprise-o-Meter",
        blurb: "Feed facts to the memory and watch 'surprise' decide what sticks.",
        mirrors: "Lab 8.3 — Surprise Metric",
        xp: 75,
      },
      {
        id: "neural-memory",
        title: "Neural Memory Vault",
        blurb: "Write facts at test time, then query them — no context window needed.",
        mirrors: "Lab 8.2 — Neural Memory",
        xp: 80,
      },
      {
        id: "memory-vram",
        title: "O(1) vs Linear",
        blurb: "Race static KV cache against constant Titans memory as context grows.",
        mirrors: "Lab 8.6 — VRAM Profiling",
        xp: 80,
      },
    ],
  },
];

export const ALL_MODULES = PHASES.flatMap((p) =>
  p.modules.map((m) => ({ ...m, phaseId: p.id, phaseSlug: p.slug })),
);

export const TOTAL_XP = ALL_MODULES.reduce((sum, m) => sum + m.xp, 0);
export const TOTAL_MODULES = ALL_MODULES.length;

export function getPhase(slug: string): PhaseDef | undefined {
  return PHASES.find((p) => p.slug === slug);
}

export function moduleKey(phaseId: number, moduleId: string): string {
  return `${phaseId}:${moduleId}`;
}
