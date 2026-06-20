import { useMemo, useState } from "react";
import { Insight, Panel, Slider, Stat } from "@/components/ui/primitives";
import { seeded } from "@/lib/math";
import type { ModuleProps } from "../types";

const SIZE = 300;
const C = SIZE / 2;
const R = 120;

export default function PolarQuant({ onDiscover }: ModuleProps) {
  const [buckets, setBuckets] = useState(16);
  const [low, setLow] = useState(false);

  const points = useMemo(() => {
    const rnd = seeded(21);
    return Array.from({ length: 40 }, () => {
      const a = rnd() * Math.PI * 2;
      const r = 0.4 + rnd() * 0.6;
      return { a, r };
    });
  }, []);

  const step = (Math.PI * 2) / buckets;
  const snap = (a: number) => Math.round(a / step) * step;

  const avgErr =
    points.reduce((s, p) => {
      const da = Math.abs(((p.a - snap(p.a) + Math.PI) % (Math.PI * 2)) - Math.PI);
      return s + da;
    }, 0) / points.length;

  if (buckets <= 8 && !low) {
    setLow(true);
    onDiscover();
  }

  const bitsNeeded = Math.ceil(Math.log2(buckets));

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_300px]">
      <Panel title="Vectors snapped to angular buckets" hint="polar quantization">
        <svg viewBox={`0 0 ${SIZE} ${SIZE}`} className="mx-auto w-full max-w-[320px]">
          <circle cx={C} cy={C} r={R} fill="none" stroke="#26304f" />
          {Array.from({ length: buckets }, (_, i) => {
            const a = i * step;
            return (
              <line
                key={i}
                x1={C}
                y1={C}
                x2={C + R * Math.cos(a)}
                y2={C - R * Math.sin(a)}
                stroke="#60a5fa22"
              />
            );
          })}
          {points.map((p, i) => {
            const sa = snap(p.a);
            const ox = C + p.r * R * Math.cos(p.a);
            const oy = C - p.r * R * Math.sin(p.a);
            const qx = C + p.r * R * Math.cos(sa);
            const qy = C - p.r * R * Math.sin(sa);
            return (
              <g key={i}>
                <line x1={ox} y1={oy} x2={qx} y2={qy} stroke="#f8717155" />
                <circle cx={ox} cy={oy} r={2.5} fill="#22d3ee" />
                <circle cx={qx} cy={qy} r={3.5} fill="#60a5fa" />
              </g>
            );
          })}
        </svg>
      </Panel>

      <div className="space-y-4">
        <Slider label="Angular buckets" value={buckets} min={4} max={64} step={4} onChange={setBuckets} accent="#60a5fa" />
        <div className="grid grid-cols-2 gap-3">
          <Stat label="Bits / angle" value={bitsNeeded} accent="#60a5fa" />
          <Stat label="Avg angle error" value={`${(avgErr * (180 / Math.PI)).toFixed(0)}°`} accent={avgErr > 0.3 ? "#f472b6" : "#34d399"} />
        </div>

        <Insight>
          Instead of storing each vector's raw coordinates, PolarQuant stores its <b>angle</b> (and
          a shared magnitude). Angles only need to land in one of a few <b>buckets</b>, so a vector
          collapses to a handful of bits. Fewer buckets = fewer bits = more error. Drop to ≤8
          buckets to feel the trade-off and bank your XP — this is how the KV cache gets to ~3.5
          bits per value.
        </Insight>
      </div>
    </div>
  );
}
