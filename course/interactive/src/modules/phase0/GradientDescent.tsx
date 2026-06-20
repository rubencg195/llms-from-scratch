import { useEffect, useRef, useState } from "react";
import { Insight, Panel, Slider, Stat, GameButton } from "@/components/ui/primitives";
import type { ModuleProps } from "../types";

const W = 460;
const H = 280;

// Loss landscape: a wobbly bowl with a clear global min near x = 0.62.
const loss = (x: number) => 0.5 * (x - 0.62) ** 2 + 0.06 * Math.sin(9 * x) + 0.05;
const grad = (x: number) => (x - 0.62) + 0.06 * 9 * Math.cos(9 * x);

const toPx = (x: number) => 40 + x * (W - 80);
const toPy = (l: number) => H - 30 - l * (H - 70) * 1.4;

export default function GradientDescent({ onDiscover }: ModuleProps) {
  const [lr, setLr] = useState(0.08);
  const [x, setX] = useState(0.06);
  const [running, setRunning] = useState(false);
  const [steps, setSteps] = useState(0);
  const [converged, setConverged] = useState(false);
  const raf = useRef(0);

  useEffect(() => {
    if (!running) return;
    let cur = x;
    let n = steps;
    const tick = () => {
      const g = grad(cur);
      cur = Math.max(-0.05, Math.min(1.05, cur - lr * g));
      n += 1;
      setX(cur);
      setSteps(n);
      if (Math.abs(g) < 0.01 && Math.abs(cur - 0.62) < 0.04) {
        setRunning(false);
        setConverged(true);
        onDiscover();
        return;
      }
      if (n > 400) {
        setRunning(false);
        return;
      }
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running]);

  const reset = (nx = Math.random() * 0.15) => {
    setRunning(false);
    setX(nx);
    setSteps(0);
    setConverged(false);
  };

  const curve = Array.from({ length: 120 }, (_, i) => {
    const xx = i / 119;
    return `${toPx(xx)},${toPy(loss(xx))}`;
  }).join(" ");

  const ballX = toPx(x);
  const ballY = toPy(loss(x)) - 10;
  const tooBig = lr > 0.19;

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_320px]">
      <Panel title="Loss landscape" hint="the ball = current weight value">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
          <polyline points={curve} fill="none" stroke="#22d3ee" strokeWidth={2.5} />
          <line x1={toPx(0.62)} y1={20} x2={toPx(0.62)} y2={H - 30} stroke="#34d39955" strokeDasharray="4 4" />
          <text x={toPx(0.62)} y={16} fontSize={10} fill="#34d399" textAnchor="middle">
            global min
          </text>
          {/* gradient arrow */}
          <line
            x1={ballX}
            y1={ballY}
            x2={ballX - Math.sign(grad(x)) * 26}
            y2={ballY}
            stroke="#fbbf24"
            strokeWidth={2}
            markerEnd="url(#ar)"
          />
          <defs>
            <marker id="ar" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 Z" fill="#fbbf24" />
            </marker>
          </defs>
          <circle cx={ballX} cy={ballY} r={11} fill="#f472b6" stroke="#fff" strokeWidth={2} />
        </svg>
      </Panel>

      <div className="space-y-4">
        <Slider
          label="Learning rate"
          value={lr}
          min={0.005}
          max={0.3}
          step={0.005}
          onChange={setLr}
          display={lr.toFixed(3)}
          accent={tooBig ? "#f87171" : "#fbbf24"}
        />
        <div className="grid grid-cols-2 gap-3">
          <Stat label="Weight x" value={x.toFixed(3)} accent="#f472b6" />
          <Stat label="Loss" value={loss(x).toFixed(3)} accent="#22d3ee" />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <Stat label="Steps" value={steps} />
          <Stat label="Gradient" value={grad(x).toFixed(2)} accent="#fbbf24" />
        </div>

        <div className="flex gap-2">
          <GameButton onClick={() => setRunning((r) => !r)} variant={running ? "ghost" : "primary"}>
            {running ? "Pause" : "Roll ▶"}
          </GameButton>
          <GameButton variant="ghost" onClick={() => reset()}>
            Reset
          </GameButton>
        </div>

        {tooBig && (
          <div className="rounded-xl border border-red-400/30 bg-red-500/10 p-3 text-sm text-red-200">
            ⚠️ Learning rate too high — the ball will overshoot and bounce around instead of settling.
          </div>
        )}
        {converged && (
          <div className="rounded-xl border border-emerald-400/30 bg-emerald-500/10 p-3 text-sm text-emerald-200">
            🎯 Converged in {steps} steps! That's training in miniature.
          </div>
        )}

        <Insight>
          Training = repeatedly nudging each weight <i>downhill</i> on the loss curve. The{" "}
          <span className="text-amber-300">gradient</span> (yellow arrow) points uphill, so we step
          the opposite way, scaled by the learning rate. Too small = crawling; too big = chaos.
          Land the ball in the green valley to earn your XP.
        </Insight>
      </div>
    </div>
  );
}
