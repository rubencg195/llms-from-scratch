import { useState } from "react";
import { motion } from "framer-motion";
import { Insight, Panel, GameButton, Stat } from "@/components/ui/primitives";
import type { ModuleProps } from "../types";

interface Fact {
  text: string;
  surprise: number; // 0..1
}

const FACTS: Fact[] = [
  { text: "The sky is blue.", surprise: 0.08 },
  { text: "Water is wet.", surprise: 0.05 },
  { text: "My passport number is X92-447-118.", surprise: 0.95 },
  { text: "The meeting moved to 3pm on Mars time.", surprise: 0.88 },
  { text: "Grass is green.", surprise: 0.06 },
  { text: "The vault code is 7741.", surprise: 0.97 },
];

export default function Surprise({ onDiscover }: ModuleProps) {
  const [i, setI] = useState(0);
  const [stored, setStored] = useState<Fact[]>([]);
  const [threshold] = useState(0.5);

  const cur = FACTS[i % FACTS.length];

  const feed = () => {
    if (cur.surprise > threshold) {
      setStored((s) => [...s, cur].slice(-6));
    }
    setI((x) => x + 1);
    if (i + 1 >= 3) onDiscover();
  };

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_300px]">
      <Panel title="Feed facts to the memory" hint="only surprising ones stick">
        <div className="rounded-2xl bg-black/30 p-5">
          <div className="text-xs text-white/40">incoming fact</div>
          <motion.div key={i} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="mt-1 text-lg text-white">
            "{cur.text}"
          </motion.div>

          <div className="mt-4">
            <div className="mb-1 flex justify-between text-xs">
              <span className="text-white/50">surprise</span>
              <span style={{ color: cur.surprise > threshold ? "#c084fc" : "#64748b" }}>
                {(cur.surprise * 100).toFixed(0)}%
              </span>
            </div>
            <div className="relative h-3 overflow-hidden rounded-full bg-white/10">
              <div className="absolute inset-y-0" style={{ left: `${threshold * 100}%`, width: 2, background: "#fff8" }} />
              <motion.div
                className="h-full rounded-full"
                animate={{ width: `${cur.surprise * 100}%` }}
                style={{ background: cur.surprise > threshold ? "#a855f7" : "#475569" }}
              />
            </div>
            <div className="mt-1 text-[10px] text-white/30">threshold = {threshold * 100}%</div>
          </div>

          <GameButton className="mt-4" onClick={feed}>
            Feed to memory →
          </GameButton>
          <p className="mt-2 text-xs text-white/45">
            {cur.surprise > threshold ? "→ surprising: will be written to memory" : "→ predictable: skipped"}
          </p>
        </div>
      </Panel>

      <div className="space-y-4">
        <Stat label="Facts stored" value={stored.length} accent="#c084fc" />
        <Panel title="Memory contents">
          {stored.length === 0 ? (
            <p className="text-sm text-white/40">Nothing stored yet. Feed a surprising fact.</p>
          ) : (
            <ul className="space-y-1.5">
              {stored.map((f, k) => (
                <motion.li key={k} initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} className="rounded-lg bg-purple-500/10 px-3 py-1.5 text-xs text-purple-100">
                  {f.text}
                </motion.li>
              ))}
            </ul>
          )}
        </Panel>

        <Insight>
          Titans doesn't memorize everything — that would fill up instantly. It measures{" "}
          <b>surprise</b>: how badly the memory <i>predicted</i> this input. Predictable facts ("the
          sky is blue") teach nothing and are skipped; surprising facts (your passport number) get
          written. It's gradient descent at <i>test time</i>, gated by surprise. Feed a few facts to
          bank your XP.
        </Insight>
      </div>
    </div>
  );
}
