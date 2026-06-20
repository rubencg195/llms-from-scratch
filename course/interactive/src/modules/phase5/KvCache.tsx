import { useState } from "react";
import { Insight, Panel, Slider, Stat } from "@/components/ui/primitives";
import { fmtBytes } from "@/lib/math";
import type { ModuleProps } from "../types";

const LAYERS = 8;
const HEADS = 8;
const HEAD_DIM = 64;
const BUDGET = 10 * 1024 ** 3; // 10 GB

export default function KvCache({ onDiscover }: ModuleProps) {
  const [ctx, setCtx] = useState(2048);
  const [bits, setBits] = useState(16);
  const [sawOver, setSawOver] = useState(false);

  // KV bytes = 2 (K and V) * layers * ctx * heads * head_dim * bytes_per_elem
  const bytesPerElem = bits / 8;
  const kvBytes = 2 * LAYERS * ctx * HEADS * HEAD_DIM * bytesPerElem;
  const pct = Math.min(100, (kvBytes / BUDGET) * 100);
  const over = kvBytes > BUDGET;

  if (over && !sawOver) {
    setSawOver(true);
  }
  const compressedOk = sawOver && !over && bits <= 4;
  if (compressedOk) onDiscover();

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_320px]">
      <Panel title="KV-cache memory vs the 10 GB budget" hint="RTX 3080">
        <div className="relative h-72 w-full overflow-hidden rounded-2xl bg-black/40 p-4">
          <div className="absolute inset-x-4 top-4 text-xs text-white/40">10 GB ceiling</div>
          <div className="absolute inset-x-4 bottom-4 flex items-end" style={{ top: "2rem" }}>
            <div
              className="w-full rounded-t-xl transition-all duration-300"
              style={{
                height: `${pct}%`,
                background: over ? "#ef4444" : "#3b82f6",
              }}
            />
          </div>
          {over && (
            <div className="absolute inset-x-0 top-1/2 text-center text-2xl font-bold text-red-300">
              💥 OUT OF MEMORY
            </div>
          )}
        </div>
      </Panel>

      <div className="space-y-4">
        <Slider label="Context length (tokens)" value={ctx} min={512} max={262144} step={512} onChange={setCtx} display={ctx.toLocaleString()} accent="#60a5fa" />
        <Slider label="Bits per KV value" value={bits} min={2} max={16} step={2} onChange={setBits} accent="#22d3ee" />

        <div className="grid grid-cols-2 gap-3">
          <Stat label="KV cache" value={fmtBytes(kvBytes)} accent={over ? "#f87171" : "#60a5fa"} />
          <Stat label="of budget" value={`${pct.toFixed(0)}%`} accent={over ? "#f87171" : "#34d399"} />
        </div>

        {compressedOk && (
          <div className="rounded-xl border border-emerald-400/30 bg-emerald-500/10 p-3 text-sm text-emerald-200">
            🚀 You blew the budget, then compressed to ≤4-bit to fit again. That's TurboQuant.
          </div>
        )}

        <Insight>
          The KV cache stores the keys & values of <i>every past token</i>, so it grows{" "}
          <b>linearly</b> with context length — it's the real memory hog for long conversations.
          First push the context up until you hit <span className="text-red-300">OOM</span>, then
          drop the bits to ≤4 to squeeze it back under 10 GB. Quantizing the cache is how we reach
          long context on one GPU. (Phase 8 removes the linear growth entirely.)
        </Insight>
      </div>
    </div>
  );
}
