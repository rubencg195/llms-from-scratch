import { useMemo, useState } from "react";
import { Insight, Panel, Pill, Stat, GameButton } from "@/components/ui/primitives";
import type { ModuleProps } from "../types";

/** Toy bigram counts — Karpathy makemore Part 1 style. */
const COUNTS: Record<string, Record<string, number>> = {
  the: { cat: 40, dog: 25, little: 18, girl: 12, boy: 10 },
  cat: { sat: 35, ran: 20, was: 15, and: 8 },
  sat: { on: 50, down: 12, by: 8 },
  on: { the: 45, a: 30, her: 10 },
  once: { upon: 60 },
  upon: { a: 55 },
  a: { time: 40, little: 35, big: 15, cat: 12 },
  little: { girl: 30, boy: 28, cat: 15, dog: 10 },
  girl: { was: 25, liked: 20, found: 12 },
  was: { playing: 18, happy: 15, sad: 8, very: 10 },
};

const STARTS = ["the", "once", "a", "little"];

function probs(word: string): [string, number][] {
  const row = COUNTS[word];
  if (!row) return [];
  const total = Object.values(row).reduce((a, b) => a + b, 0);
  return Object.entries(row)
    .map(([w, c]) => [w, c / total] as [string, number])
    .sort((a, b) => b[1] - a[1]);
}

export default function BigramLM({ onDiscover }: ModuleProps) {
  const [current, setCurrent] = useState("the");
  const [history, setHistory] = useState<string[]>(["the"]);
  const [sampled, setSampled] = useState(false);

  const distribution = useMemo(() => probs(current), [current]);

  const pickNext = (word: string) => {
    setCurrent(word);
    setHistory((h) => [...h, word]);
    setSampled(true);
    if (history.length >= 4) onDiscover();
  };

  const sampleRandom = () => {
    const row = probs(current);
    if (!row.length) return;
    const r = Math.random();
    let cum = 0;
    for (const [w, p] of row) {
      cum += p;
      if (r <= cum) {
        pickNext(w);
        return;
      }
    }
    pickNext(row[0][0]);
  };

  const reset = (start: string) => {
    setCurrent(start);
    setHistory([start]);
    setSampled(false);
  };

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_280px]">
      <Panel title="Bigram table" hint={`P(next | "${current}")`}>
        <div className="mb-3 flex flex-wrap gap-1">
          {history.map((w, i) => (
            <Pill key={i} color="#a78bfa">
              {w}
            </Pill>
          ))}
        </div>
        <div className="space-y-2">
          {distribution.map(([word, p]) => (
            <button
              key={word}
              type="button"
              onClick={() => pickNext(word)}
              className="flex w-full items-center gap-3 rounded-lg bg-white/5 px-3 py-2 text-left transition hover:bg-white/10"
            >
              <span className="w-20 font-mono text-sm text-white">{word}</span>
              <div className="h-3 flex-1 overflow-hidden rounded-full bg-white/10">
                <div
                  className="h-full rounded-full bg-violet-400 transition-all"
                  style={{ width: `${p * 100}%` }}
                />
              </div>
              <span className="w-12 text-right text-xs text-white/60">{(p * 100).toFixed(0)}%</span>
            </button>
          ))}
          {distribution.length === 0 && (
            <div className="text-sm text-white/50">No counts for "{current}" — try another start word.</div>
          )}
        </div>
      </Panel>

      <div className="space-y-4">
        <Stat label="Context word" value={current} accent="#a78bfa" />
        <Stat label="Tokens generated" value={history.length} />
        <div className="flex flex-wrap gap-2">
          {STARTS.map((s) => (
            <GameButton key={s} variant="ghost" onClick={() => reset(s)}>
              Start: {s}
            </GameButton>
          ))}
        </div>
        <GameButton onClick={sampleRandom} variant="primary">
          Sample next ▶
        </GameButton>
        {sampled && (
          <div className="rounded-xl border border-violet-400/30 bg-violet-500/10 p-3 text-sm text-violet-200">
            Same loop as Transformer generation — only the probability source changes.
          </div>
        )}
        <Insight>
          A language model predicts <span className="text-violet-300">P(next token | context)</span>.
          Bigrams use only the previous word. The Transformer uses attention over the full prefix —
          but the sampling loop is identical.
        </Insight>
      </div>
    </div>
  );
}
