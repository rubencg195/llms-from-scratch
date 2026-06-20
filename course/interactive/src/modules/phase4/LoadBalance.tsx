import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Insight, Panel, Slider, Stat } from "@/components/ui/primitives";
import { seeded } from "@/lib/math";
import type { ModuleProps } from "../types";

const EXPERTS = ["E1", "E2", "E3", "E4"];
const COLORS = ["#f472b6", "#22d3ee", "#34d399", "#fbbf24"];

export default function LoadBalance({ onDiscover }: ModuleProps) {
  const [balance, setBalance] = useState(0); // 0 = no balancing, 1 = perfectly balanced
  const [solved, setSolved] = useState(false);

  // Without balancing, the router collapses onto one favorite expert.
  const loads = useMemo(() => {
    const rnd = seeded(5);
    const skewed = [0.62, 0.22, 0.11, 0.05];
    const uniform = [0.25, 0.25, 0.25, 0.25];
    return EXPERTS.map((_, i) => {
      const base = skewed[i] * (1 - balance) + uniform[i] * balance;
      return Math.max(0.02, base + (rnd() - 0.5) * 0.02);
    });
  }, [balance]);

  const total = loads.reduce((a, b) => a + b, 0);
  const norm = loads.map((l) => l / total);
  const maxLoad = Math.max(...norm);
  const cv = stddev(norm) / (1 / EXPERTS.length); // dispersion vs ideal

  if (maxLoad < 0.32 && !solved) {
    setSolved(true);
    onDiscover();
  }

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_300px]">
      <Panel title="Tokens handled per expert" hint="aim for balance">
        <div className="flex h-56 items-end justify-around gap-4">
          {norm.map((l, i) => (
            <div key={i} className="flex flex-1 flex-col items-center">
              <motion.div
                className="w-full rounded-t-lg"
                animate={{ height: `${l * 200}px`, background: COLORS[i] }}
                style={{ minHeight: 6 }}
              />
              <div className="mt-2 font-mono text-xs text-white/70">{(l * 100).toFixed(0)}%</div>
              <div className="text-[11px] text-white/40">{EXPERTS[i]}</div>
            </div>
          ))}
        </div>
        <div className="mt-2 border-t border-dashed border-white/15 pt-1 text-center text-[11px] text-white/40">
          ideal = 25% each
        </div>
      </Panel>

      <div className="space-y-4">
        <Slider label="Load-balance loss weight" value={balance} min={0} max={1} step={0.02} onChange={setBalance} display={balance.toFixed(2)} accent="#f472b6" />
        <div className="grid grid-cols-2 gap-3">
          <Stat label="Busiest expert" value={`${(maxLoad * 100).toFixed(0)}%`} accent={maxLoad > 0.4 ? "#f87171" : "#34d399"} />
          <Stat label="Imbalance" value={cv.toFixed(2)} accent="#fbbf24" />
        </div>

        {solved && (
          <div className="rounded-xl border border-emerald-400/30 bg-emerald-500/10 p-3 text-sm text-emerald-200">
            ✅ Well balanced — every expert is pulling its weight.
          </div>
        )}

        <Insight>
          Left alone, the router gets lazy and dumps almost everything on its favorite expert — the
          others never train and the extra capacity is wasted. An <b>auxiliary load-balancing
          loss</b> nudges the router to spread tokens evenly. Crank the slider up until no expert
          exceeds ~32% to bank your XP.
        </Insight>
      </div>
    </div>
  );
}

function stddev(xs: number[]) {
  const m = xs.reduce((a, b) => a + b, 0) / xs.length;
  return Math.sqrt(xs.reduce((s, x) => s + (x - m) ** 2, 0) / xs.length);
}
