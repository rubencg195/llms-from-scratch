import { useRef, useState } from "react";
import { Insight, Panel, Stat } from "@/components/ui/primitives";
import { cosine, dot } from "@/lib/math";
import type { ModuleProps } from "../types";

const SIZE = 320;
const C = SIZE / 2;
const SCALE = 110;

export default function DotProduct({ onDiscover }: ModuleProps) {
  const [a, setA] = useState<[number, number]>([1, 0.4]);
  const [b, setB] = useState<[number, number]>([0.3, 1]);
  const [drag, setDrag] = useState<null | "a" | "b">(null);
  const [wentNeg, setWentNeg] = useState(false);
  const svgRef = useRef<SVGSVGElement>(null);

  const d = dot(a, b);
  const cos = cosine(a, b);
  if (d < -0.05 && !wentNeg) {
    setWentNeg(true);
    onDiscover();
  }

  const toScreen = (v: [number, number]) => [C + v[0] * SCALE, C - v[1] * SCALE];

  const onMove = (e: React.PointerEvent) => {
    if (!drag || !svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left - C) / SCALE;
    const y = -(e.clientY - rect.top - C) / SCALE;
    const v: [number, number] = [
      Math.max(-1.4, Math.min(1.4, x)),
      Math.max(-1.4, Math.min(1.4, y)),
    ];
    drag === "a" ? setA(v) : setB(v);
  };

  const [ax, ay] = toScreen(a);
  const [bx, by] = toScreen(b);

  const verdict =
    cos > 0.7 ? "Pointing together → big positive score"
    : cos < -0.3 ? "Pointing apart → negative score"
    : "Roughly perpendicular → near-zero score";

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_320px]">
      <Panel title="Drag the two arrows" hint="similarity = how aligned they are">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${SIZE} ${SIZE}`}
          className="mx-auto w-full max-w-[360px] touch-none select-none"
          onPointerMove={onMove}
          onPointerUp={() => setDrag(null)}
          onPointerLeave={() => setDrag(null)}
        >
          {[-1, -0.5, 0.5, 1].map((g) => (
            <g key={g}>
              <line x1={C + g * SCALE} y1={0} x2={C + g * SCALE} y2={SIZE} stroke="#1c2742" />
              <line x1={0} y1={C + g * SCALE} x2={SIZE} y2={C + g * SCALE} stroke="#1c2742" />
            </g>
          ))}
          <line x1={0} y1={C} x2={SIZE} y2={C} stroke="#34406b" />
          <line x1={C} y1={0} x2={C} y2={SIZE} stroke="#34406b" />

          <Arrow x={ax} y={ay} color="#22d3ee" onDown={() => setDrag("a")} label="A" />
          <Arrow x={bx} y={by} color="#f472b6" onDown={() => setDrag("b")} label="B" />
        </svg>
      </Panel>

      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-3">
          <Stat label="Dot product" value={d.toFixed(2)} accent="#22d3ee" />
          <Stat label="Cosine sim" value={cos.toFixed(2)} accent="#f472b6" />
        </div>

        <Panel title="What you're seeing">
          <p className="text-sm text-white/70">{verdict}</p>
          <div className="mt-3 font-mono text-xs text-white/60">
            A·B = {a[0].toFixed(2)}×{b[0].toFixed(2)} + {a[1].toFixed(2)}×{b[1].toFixed(2)} ={" "}
            {d.toFixed(2)}
          </div>
        </Panel>

        <Insight>
          The dot product is the engine of attention. When two vectors point the same way the score
          is large and positive; opposite directions give a negative score. Drag the arrows so they
          point <b>away</b> from each other (negative score) to unlock your XP — that "score how
          similar are these?" question is exactly what every attention head asks.
        </Insight>
      </div>
    </div>
  );
}

function Arrow({
  x,
  y,
  color,
  label,
  onDown,
}: {
  x: number;
  y: number;
  color: string;
  label: string;
  onDown: () => void;
}) {
  return (
    <g>
      <line x1={C} y1={C} x2={x} y2={y} stroke={color} strokeWidth={3} />
      <circle
        cx={x}
        cy={y}
        r={11}
        fill={color}
        className="cursor-grab active:cursor-grabbing"
        onPointerDown={onDown}
      />
      <text x={x} y={y + 4} textAnchor="middle" fontSize={11} fontWeight={700} fill="#04121f">
        {label}
      </text>
    </g>
  );
}
