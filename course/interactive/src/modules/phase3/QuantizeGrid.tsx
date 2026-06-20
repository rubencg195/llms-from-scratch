import { useMemo, useState } from "react";
import { Insight, Panel, Slider, Stat } from "@/components/ui/primitives";
import { seeded } from "@/lib/math";
import type { ModuleProps } from "../types";

const W = 460;
const H = 240;
const N = 40;

export default function QuantizeGrid({ onDiscover }: ModuleProps) {
  const [bits, setBits] = useState(8);
  const [reached, setReached] = useState(false);

  const weights = useMemo(() => {
    const rnd = seeded(7);
    return Array.from({ length: N }, () => (rnd() * 2 - 1) * 0.95);
  }, []);

  const levels = 2 ** bits;
  const step = 2 / (levels - 1);
  const quant = (w: number) => Math.round((w + 1) / step) * step - 1;

  const err =
    weights.reduce((s, w) => s + Math.abs(w - quant(w)), 0) / weights.length;

  if (bits <= 2 && !reached) {
    setReached(true);
    onDiscover();
  }

  const sx = (i: number) => 30 + (i / (N - 1)) * (W - 50);
  const sy = (v: number) => H / 2 - v * (H / 2 - 20);

  // grid lines for quant levels (cap how many we draw)
  const gridVals: number[] = [];
  if (levels <= 64) for (let i = 0; i < levels; i++) gridVals.push(-1 + i * step);

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_300px]">
      <Panel title="Weights snapping to a grid" hint="cyan = original · amber = quantized">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
          {gridVals.map((g, i) => (
            <line key={i} x1={20} y1={sy(g)} x2={W - 10} y2={sy(g)} stroke="#fbbf2422" />
          ))}
          <line x1={20} y1={sy(0)} x2={W - 10} y2={sy(0)} stroke="#26304f" />
          {weights.map((w, i) => (
            <g key={i}>
              <line x1={sx(i)} y1={sy(w)} x2={sx(i)} y2={sy(quant(w))} stroke="#f8717155" />
              <circle cx={sx(i)} cy={sy(w)} r={3} fill="#22d3ee" />
              <circle cx={sx(i)} cy={sy(quant(w))} r={4} fill="#fbbf24" />
            </g>
          ))}
        </svg>
      </Panel>

      <div className="space-y-4">
        <Slider label="Bits per weight" value={bits} min={1} max={8} step={1} onChange={setBits} accent="#fbbf24" />
        <div className="grid grid-cols-2 gap-3">
          <Stat label="Levels" value={levels} accent="#fbbf24" />
          <Stat label="Avg error" value={err.toFixed(3)} accent={err > 0.1 ? "#f472b6" : "#34d399"} />
        </div>
        <Stat label="Memory vs FP32" value={`${((bits / 32) * 100).toFixed(0)}%`} accent="#22d3ee" />

        <Insight>
          A 32-bit weight can hold billions of values; quantization forces every weight onto one of
          just <b>{levels}</b> rungs (2<sup>{bits}</sup>). Fewer bits = far less memory (great for
          the 10 GB RTX 3080) but bigger rounding error. Drag down to 2 bits to feel how brutal the
          snapping gets — that's the problem QAT and TurboQuant exist to solve. Reaching 2 bits banks
          your XP.
        </Insight>
      </div>
    </div>
  );
}
