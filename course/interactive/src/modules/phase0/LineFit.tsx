import { useMemo, useState } from "react";
import { Insight, Panel, Slider, Stat, GameButton } from "@/components/ui/primitives";
import { seeded } from "@/lib/math";
import type { ModuleProps } from "../types";

const W = 460;
const H = 300;
const PAD = 36;

// True line the noisy points were sampled from.
const TRUE_M = 1.6;
const TRUE_B = -0.3;

export default function LineFit({ onDiscover }: ModuleProps) {
  const [m, setM] = useState(0);
  const [b, setB] = useState(1.5);
  const [solved, setSolved] = useState(false);

  const points = useMemo(() => {
    const rnd = seeded(99);
    return Array.from({ length: 14 }, () => {
      const x = rnd() * 2 - 1;
      const y = TRUE_M * x + TRUE_B + (rnd() - 0.5) * 0.5;
      return { x, y };
    });
  }, []);

  const mse = useMemo(() => {
    const e = points.reduce((s, p) => s + (m * p.x + b - p.y) ** 2, 0) / points.length;
    return e;
  }, [m, b, points]);

  if (mse < 0.08 && !solved) {
    setSolved(true);
    onDiscover();
  }

  const sx = (x: number) => PAD + ((x + 1.4) / 2.8) * (W - 2 * PAD);
  const sy = (y: number) => H - PAD - ((y + 2.5) / 5) * (H - 2 * PAD);

  const autoFit = () => {
    // closed-form least squares, animated in a few hops for feel
    const n = points.length;
    const sxv = points.reduce((s, p) => s + p.x, 0);
    const syv = points.reduce((s, p) => s + p.y, 0);
    const sxy = points.reduce((s, p) => s + p.x * p.y, 0);
    const sxx = points.reduce((s, p) => s + p.x * p.x, 0);
    const mm = (n * sxy - sxv * syv) / (n * sxx - sxv * sxv);
    const bb = (syv - mm * sxv) / n;
    setM(Number(mm.toFixed(2)));
    setB(Number(bb.toFixed(2)));
  };

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_320px]">
      <Panel title="Fit the line: y = m·x + b" hint="residuals shown in red">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
          <line x1={PAD} y1={sy(0)} x2={W - PAD} y2={sy(0)} stroke="#26304f" />
          <line x1={sx(0)} y1={PAD} x2={sx(0)} y2={H - PAD} stroke="#26304f" />

          {points.map((p, i) => (
            <g key={i}>
              <line x1={sx(p.x)} y1={sy(p.y)} x2={sx(p.x)} y2={sy(m * p.x + b)} stroke="#f8717188" strokeWidth={1.5} />
              <circle cx={sx(p.x)} cy={sy(p.y)} r={5} fill="#22d3ee" />
            </g>
          ))}

          <line
            x1={sx(-1.4)}
            y1={sy(m * -1.4 + b)}
            x2={sx(1.4)}
            y2={sy(m * 1.4 + b)}
            stroke="#fbbf24"
            strokeWidth={3}
          />
        </svg>
      </Panel>

      <div className="space-y-4">
        <Slider label="Slope m" value={m} min={-3} max={3} step={0.05} onChange={setM} display={m.toFixed(2)} accent="#fbbf24" />
        <Slider label="Intercept b" value={b} min={-2.5} max={2.5} step={0.05} onChange={setB} display={b.toFixed(2)} accent="#fbbf24" />

        <Stat label="Mean Squared Error" value={mse.toFixed(3)} accent={mse < 0.08 ? "#34d399" : "#f472b6"} />

        <div className="h-3 overflow-hidden rounded-full bg-white/10">
          <div
            className="h-full rounded-full transition-all"
            style={{
              width: `${Math.max(4, 100 - mse * 60)}%`,
              background: mse < 0.08 ? "#34d399" : "#f472b6",
            }}
          />
        </div>

        <GameButton variant="ghost" onClick={autoFit}>
          Show the optimizer's answer
        </GameButton>

        {solved && (
          <div className="rounded-xl border border-emerald-400/30 bg-emerald-500/10 p-3 text-sm text-emerald-200">
            🎯 Loss below 0.08 — you found the best-fit line by hand!
          </div>
        )}

        <Insight>
          A single neuron <i>is</i> this line: <code className="text-amber-300">y = m·x + b</code>,
          where <b>m</b> is the weight and <b>b</b> the bias. Training just searches for the m and b
          that make the red error bars as short as possible (lowest MSE). Get the loss under 0.08 to
          claim your XP.
        </Insight>
      </div>
    </div>
  );
}
