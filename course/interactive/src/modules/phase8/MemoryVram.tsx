import { useState } from "react";
import { Insight, Panel, Slider, Stat } from "@/components/ui/primitives";
import { fmtBytes } from "@/lib/math";
import type { ModuleProps } from "../types";

const W = 460;
const H = 240;
const LAYERS = 8;
const HEADS = 8;
const HEAD_DIM = 64;
const BUDGET = 10 * 1024 ** 3;

// log-scale context axis from 512 to ~1M tokens
const MIN_LOG = Math.log2(512);
const MAX_LOG = Math.log2(1_048_576);

export default function MemoryVram({ onDiscover }: ModuleProps) {
  const [ctxLog, setCtxLog] = useState(13); // 8192
  const ctx = Math.round(2 ** ctxLog);
  const [sawCrossover, setSawCrossover] = useState(false);

  const kvBytes = (n: number) => 2 * LAYERS * n * HEADS * HEAD_DIM * 2; // fp16
  const titansBytes = LAYERS * (256 * 256) * 2; // fixed memory matrices

  const kv = kvBytes(ctx);
  const overBudget = kv > BUDGET;
  if (overBudget && !sawCrossover) {
    setSawCrossover(true);
    onDiscover();
  }

  const px = (logv: number) => 36 + ((logv - MIN_LOG) / (MAX_LOG - MIN_LOG)) * (W - 56);
  const maxBytes = kvBytes(1_048_576);
  const py = (b: number) => H - 26 - (Math.min(b, maxBytes) / maxBytes) * (H - 50);

  const kvCurve: string[] = [];
  const titanCurve: string[] = [];
  for (let l = MIN_LOG; l <= MAX_LOG; l += 0.2) {
    const n = 2 ** l;
    kvCurve.push(`${px(l)},${py(kvBytes(n))}`);
    titanCurve.push(`${px(l)},${py(titansBytes)}`);
  }
  const budgetY = py(BUDGET);

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_300px]">
      <Panel title="Memory vs context length" hint="static KV (red) vs Titans (green)">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
          <line x1={30} y1={budgetY} x2={W - 10} y2={budgetY} stroke="#fff" strokeDasharray="5 4" opacity={0.5} />
          <text x={34} y={budgetY - 5} fontSize={10} fill="#fff" opacity={0.6}>
            10 GB budget
          </text>

          <polyline points={kvCurve.join(" ")} fill="none" stroke="#f87171" strokeWidth={2.5} />
          <polyline points={titanCurve.join(" ")} fill="none" stroke="#34d399" strokeWidth={2.5} />

          {/* current position marker */}
          <line x1={px(ctxLog)} y1={20} x2={px(ctxLog)} y2={H - 26} stroke="#c084fc55" />
          <circle cx={px(ctxLog)} cy={py(kv)} r={5} fill="#f87171" />
          <circle cx={px(ctxLog)} cy={py(titansBytes)} r={5} fill="#34d399" />
        </svg>
        <div className="flex gap-4 text-xs">
          <span className="text-red-300">— static KV cache (linear)</span>
          <span className="text-emerald-300">— Titans memory (constant)</span>
        </div>
      </Panel>

      <div className="space-y-4">
        <Slider label="Context length" value={ctxLog} min={MIN_LOG} max={MAX_LOG} step={0.1} onChange={setCtxLog} display={ctx.toLocaleString()} accent="#c084fc" />
        <div className="grid grid-cols-1 gap-3">
          <Stat label="Static KV cache" value={fmtBytes(kv)} accent={overBudget ? "#f87171" : "#60a5fa"} />
          <Stat label="Titans memory" value={fmtBytes(titansBytes)} accent="#34d399" />
        </div>

        {overBudget && (
          <div className="rounded-xl border border-emerald-400/30 bg-emerald-500/10 p-3 text-sm text-emerald-200">
            🏛️ Static KV just blew the 10 GB budget — but Titans memory hasn't moved. That's the
            whole point.
          </div>
        )}

        <Insight>
          The standard KV cache grows <b>linearly</b> forever — double the context, double the
          memory — so it eventually exhausts any GPU. Titans stores history in a{" "}
          <b>fixed-size</b> memory matrix that's <b>O(1)</b> in context length: 1,000 tokens or
          1,000,000, the memory cost is identical. Slide the context right until the red line
          crosses the budget to bank your XP and finish the journey.
        </Insight>
      </div>
    </div>
  );
}
