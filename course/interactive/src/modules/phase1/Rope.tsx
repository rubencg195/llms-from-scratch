import { useRef, useState } from "react";
import { Insight, Panel, Slider, Stat } from "@/components/ui/primitives";
import type { ModuleProps } from "../types";

const SIZE = 300;
const C = SIZE / 2;
const R = 110;

export default function Rope({ onDiscover }: ModuleProps) {
  const [pos, setPos] = useState(0);
  const [freq, setFreq] = useState(1);
  const maxPos = useRef(0);

  // RoPE rotates the (x,y) pair of a query/key by an angle proportional to its
  // position in the sequence. theta = pos * base_freq.
  const baseAngle = 0.6; // the vector's "content" direction
  const theta = pos * freq * 0.4;
  const angle = baseAngle + theta;

  const setP = (p: number) => {
    setPos(p);
    maxPos.current = Math.max(maxPos.current, p);
    if (maxPos.current >= 7) onDiscover();
  };

  const x = C + R * Math.cos(angle);
  const y = C - R * Math.sin(angle);
  const ox = C + R * Math.cos(baseAngle);
  const oy = C - R * Math.sin(baseAngle);

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_320px]">
      <Panel title="A token vector, rotated by its position" hint="ghost = position 0">
        <svg viewBox={`0 0 ${SIZE} ${SIZE}`} className="mx-auto w-full max-w-[340px]">
          <circle cx={C} cy={C} r={R} fill="none" stroke="#26304f" />
          <line x1={C - R} y1={C} x2={C + R} y2={C} stroke="#1c2742" />
          <line x1={C} y1={C - R} x2={C} y2={C + R} stroke="#1c2742" />

          {/* original */}
          <line x1={C} y1={C} x2={ox} y2={oy} stroke="#475569" strokeWidth={2} strokeDasharray="4 4" />
          <circle cx={ox} cy={oy} r={6} fill="#475569" />

          {/* rotated by position */}
          <line x1={C} y1={C} x2={x} y2={y} stroke="#a78bfa" strokeWidth={3} />
          <circle cx={x} cy={y} r={9} fill="#a78bfa" />

          {/* arc showing rotation */}
          <path
            d={describeArc(C, C, R * 0.4, baseAngle, angle)}
            fill="none"
            stroke="#fbbf24"
            strokeWidth={2}
          />
        </svg>
      </Panel>

      <div className="space-y-4">
        <Slider label="Token position in sequence" value={pos} min={0} max={12} step={1} onChange={setP} accent="#a78bfa" />
        <Slider label="Frequency (which dimension pair)" value={freq} min={0.5} max={3} step={0.1} onChange={setFreq} display={freq.toFixed(1)} accent="#22d3ee" />

        <div className="grid grid-cols-2 gap-3">
          <Stat label="Rotation angle" value={`${(theta * (180 / Math.PI)).toFixed(0)}°`} accent="#fbbf24" />
          <Stat label="Position" value={pos} accent="#a78bfa" />
        </div>

        <Insight>
          RoPE (Rotary Position Embedding) encodes <i>where</i> a token sits by literally{" "}
          <b>rotating</b> its vector by an angle proportional to its position. Two tokens that are{" "}
          <i>N</i> steps apart always differ by the same rotation, so attention can read relative
          distance for free — and it generalizes to sequences longer than those seen in training.
          Slide the position past 7 to bank your XP.
        </Insight>
      </div>
    </div>
  );
}

function polar(cx: number, cy: number, r: number, a: number) {
  return { x: cx + r * Math.cos(a), y: cy - r * Math.sin(a) };
}
function describeArc(cx: number, cy: number, r: number, a0: number, a1: number) {
  const s = polar(cx, cy, r, a0);
  const e = polar(cx, cy, r, a1);
  const large = Math.abs(a1 - a0) > Math.PI ? 1 : 0;
  const sweep = a1 > a0 ? 0 : 1;
  return `M ${s.x} ${s.y} A ${r} ${r} 0 ${large} ${sweep} ${e.x} ${e.y}`;
}
