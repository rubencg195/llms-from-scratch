import { useState } from "react";
import { Insight, Panel, GameButton, Stat } from "@/components/ui/primitives";
import type { ModuleProps } from "../types";

const W = 460;
const H = 240;

export default function STE({ onDiscover }: ModuleProps) {
  const [ste, setSte] = useState(true);

  const toggle = () => {
    setSte((s) => !s);
    onDiscover();
  };

  // staircase (rounding) vs identity gradient
  const sx = (x: number) => 30 + ((x + 1) / 2) * (W - 50);
  const sy = (y: number) => H - 20 - ((y + 1) / 2) * (H - 40);

  const stair: string[] = [];
  for (let i = 0; i <= 100; i++) {
    const x = -1 + (i / 100) * 2;
    const y = Math.round(x * 3) / 3;
    stair.push(`${sx(x)},${sy(y)}`);
  }

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_300px]">
      <Panel title="Forward = staircase · Backward = ?" hint="the rounding function">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
          <line x1={20} y1={sy(0)} x2={W - 10} y2={sy(0)} stroke="#26304f" />
          <line x1={sx(0)} y1={10} x2={sx(0)} y2={H - 10} stroke="#26304f" />

          {/* forward rounding (staircase) */}
          <polyline points={stair.join(" ")} fill="none" stroke="#fbbf24" strokeWidth={2.5} />

          {/* gradient line */}
          {ste ? (
            <line x1={sx(-1)} y1={sy(-1)} x2={sx(1)} y2={sy(1)} stroke="#34d399" strokeWidth={2.5} strokeDasharray="6 4" />
          ) : (
            <line x1={sx(-1)} y1={sy(0)} x2={sx(1)} y2={sy(0)} stroke="#f87171" strokeWidth={2.5} strokeDasharray="6 4" />
          )}
        </svg>
        <div className="mt-2 flex gap-4 text-xs">
          <span className="text-amber-300">— forward (round)</span>
          {ste ? (
            <span className="text-emerald-300">-- backward (STE: pretend slope = 1)</span>
          ) : (
            <span className="text-red-300">-- backward (true slope = 0 everywhere)</span>
          )}
        </div>
      </Panel>

      <div className="space-y-4">
        <GameButton onClick={toggle} variant={ste ? "success" : "ghost"}>
          {ste ? "STE: ON" : "STE: OFF"}
        </GameButton>

        <Stat
          label="Gradient that flows back"
          value={ste ? "≈ 1 (learns!)" : "0 (stuck)"}
          accent={ste ? "#34d399" : "#f87171"}
        />

        <div
          className={`rounded-xl border p-3 text-sm ${
            ste
              ? "border-emerald-400/30 bg-emerald-500/10 text-emerald-200"
              : "border-red-400/30 bg-red-500/10 text-red-200"
          }`}
        >
          {ste
            ? "Gradients pass through → the network can train with quantized weights."
            : "The staircase is flat between steps, so its true gradient is 0. Training dies."}
        </div>

        <Insight>
          Rounding is a staircase: flat between steps, so its real derivative is <b>zero</b> almost
          everywhere — backprop gets nothing and the model can't learn. The{" "}
          <b>Straight-Through Estimator</b> cheats: round in the forward pass, but in the backward
          pass <i>pretend</i> the function was the identity line (slope 1). Toggle STE to watch
          gradients live or die — that single trick is what makes quantization-aware training
          possible.
        </Insight>
      </div>
    </div>
  );
}
