import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Insight, Panel, Slider, Pill } from "@/components/ui/primitives";
import { seeded, softmax } from "@/lib/math";
import type { ModuleProps } from "../types";

const TOKENS = ["math", "code", "poem", "fact", "joke", "story", "sum", "loop"];
const EXPERTS = ["Math 🧮", "Code 💻", "Prose ✍️", "Facts 📚"];
const COLORS = ["#f472b6", "#22d3ee", "#34d399", "#fbbf24"];

export default function Router({ onDiscover }: ModuleProps) {
  const [topk, setTopk] = useState(2);
  const [picked, setPicked] = useState(0);

  // each token has fixed router logits over the 4 experts
  const logits = useMemo(() => {
    const rnd = seeded(13);
    return TOKENS.map(() => EXPERTS.map(() => rnd() * 3));
  }, []);

  const tok = picked;
  const probs = softmax(logits[tok]);
  const order = probs.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p);
  const chosen = new Set(order.slice(0, topk).map((o) => o.i));

  const pick = (i: number) => {
    setPicked(i);
    onDiscover();
  };

  return (
    <div className="space-y-5">
      <Panel title="Pick a token — see where the router sends it" hint={`top-${topk} experts win`}>
        <div className="mb-5 flex flex-wrap gap-2">
          {TOKENS.map((t, i) => (
            <button
              key={t}
              onClick={() => pick(i)}
              className={`rounded-lg px-3 py-1.5 text-sm font-mono ${
                picked === i ? "bg-pink-500/30 text-pink-200 ring-1 ring-pink-400" : "bg-white/5 text-white/60"
              }`}
            >
              {t}
            </button>
          ))}
        </div>

        <div className="grid grid-cols-[80px_1fr] items-center gap-x-4 gap-y-3">
          <div className="text-center">
            <div className="mx-auto grid h-12 w-12 place-items-center rounded-xl bg-pink-500/20 font-mono text-sm text-pink-200">
              {TOKENS[tok]}
            </div>
            <div className="mt-1 text-[10px] text-white/40">token</div>
          </div>

          <div className="space-y-2">
            {EXPERTS.map((e, i) => {
              const on = chosen.has(i);
              return (
                <div key={e} className="flex items-center gap-3">
                  <div className="relative h-8 flex-1 overflow-hidden rounded-lg bg-white/5">
                    <motion.div
                      className="h-full rounded-lg"
                      animate={{ width: `${probs[i] * 100}%`, opacity: on ? 1 : 0.35 }}
                      style={{ background: COLORS[i] }}
                    />
                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-sm font-semibold text-white">
                      {e}
                    </span>
                    <span className="absolute right-2 top-1/2 -translate-y-1/2 font-mono text-xs text-white/80">
                      {(probs[i] * 100).toFixed(0)}%
                    </span>
                  </div>
                  {on ? <Pill color={COLORS[i]}>active</Pill> : <span className="w-14 text-center text-xs text-white/25">idle</span>}
                </div>
              );
            })}
          </div>
        </div>
      </Panel>

      <Slider label="Top-k experts per token" value={topk} min={1} max={4} step={1} onChange={setTopk} accent="#f472b6" />

      <Insight>
        A Mixture-of-Experts layer holds several specialist FFNs but only runs <b>k</b> of them per
        token. A tiny <b>router</b> scores the experts (softmax) and the top-k win. This means you
        can have a huge total parameter count while only paying compute for k experts — more
        knowledge, same FLOPs. Click a token to watch routing happen and bank your XP.
      </Insight>
    </div>
  );
}
