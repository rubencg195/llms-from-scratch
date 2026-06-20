import { useRef, useState } from "react";
import { motion } from "framer-motion";
import { useMemoizedAttention } from "./useAttention";
import { Insight, Panel, Slider } from "@/components/ui/primitives";
import { heat } from "@/lib/math";
import type { ModuleProps } from "../types";

const SENTENCE = ["The", "cat", "sat", "on", "the", "mat", "because", "it", "was", "tired"];

export default function Attention({ onDiscover }: ModuleProps) {
  const [query, setQuery] = useState(7); // "it"
  const [temp, setTemp] = useState(1);
  const touched = useRef(0);

  const { weights } = useMemoizedAttention(SENTENCE, temp);
  const row = weights[query];

  const pick = (i: number) => {
    setQuery(i);
    touched.current += 1;
    if (touched.current >= 3) onDiscover();
  };

  return (
    <div className="space-y-5">
      <Panel title="Pick a word — see what it attends to" hint="brighter = stronger attention">
        <div className="flex flex-wrap gap-2">
          {SENTENCE.map((w, i) => (
            <button
              key={i}
              onClick={() => pick(i)}
              className={`rounded-lg px-3 py-2 text-sm font-semibold transition ${
                query === i ? "ring-2 ring-violet-400" : ""
              }`}
              style={{
                background: query === i ? "#7c3aed" : "rgba(255,255,255,0.06)",
                color: query === i ? "#fff" : "rgba(255,255,255,0.7)",
              }}
            >
              {w}
            </button>
          ))}
        </div>

        <div className="mt-5 space-y-2">
          {SENTENCE.map((w, i) => (
            <div key={i} className="flex items-center gap-3">
              <span className="w-20 text-right font-mono text-xs text-white/55">{w}</span>
              <div className="relative h-7 flex-1 overflow-hidden rounded-md bg-white/5">
                <motion.div
                  className="h-full rounded-md"
                  animate={{ width: `${row[i] * 100}%` }}
                  transition={{ type: "spring", stiffness: 160, damping: 20 }}
                  style={{ background: heat(row[i]) }}
                />
                <span className="absolute right-2 top-1/2 -translate-y-1/2 font-mono text-[11px] text-white/70">
                  {(row[i] * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </Panel>

      <Panel title="Full attention matrix">
        <div className="overflow-x-auto">
          <div className="inline-grid gap-0.5" style={{ gridTemplateColumns: `repeat(${SENTENCE.length}, 1fr)` }}>
            {weights.map((r, qi) =>
              r.map((v, ki) => (
                <div
                  key={`${qi}-${ki}`}
                  title={`${SENTENCE[qi]} → ${SENTENCE[ki]}: ${(v * 100).toFixed(0)}%`}
                  onClick={() => pick(qi)}
                  className="h-7 w-7 cursor-pointer rounded-sm"
                  style={{
                    background: heat(v),
                    outline: qi === query ? "2px solid #a78bfa" : "none",
                  }}
                />
              )),
            )}
          </div>
        </div>
        <p className="mt-2 text-xs text-white/40">Each row is one word's attention over all words (causal mask applied — words can't see the future).</p>
      </Panel>

      <Slider label="Temperature (softmax sharpness)" value={temp} min={0.3} max={3} step={0.1} onChange={setTemp} display={temp.toFixed(1)} accent="#22d3ee" />

      <Insight>
        Each word builds a <b>query</b> and compares it (via dot product — remember Phase 0!) to
        every other word's <b>key</b>. Softmax turns those scores into percentages that sum to
        100%. Notice how <i>"it"</i> often attends back to <i>"cat"</i> — that's the model resolving
        what "it" refers to. Low temperature = laser focus on one word; high = spread out. Click a
        few words to bank your XP.
      </Insight>
    </div>
  );
}
