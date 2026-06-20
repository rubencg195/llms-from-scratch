import { useState } from "react";
import { Insight, Panel, GameButton } from "@/components/ui/primitives";
import type { ModuleProps } from "../types";

interface Entry {
  key: string;
  value: string;
}

const SEED: Entry[] = [
  { key: "favorite color", value: "teal" },
  { key: "dog's name", value: "Pixel" },
];

export default function NeuralMemory({ onDiscover }: ModuleProps) {
  const [mem, setMem] = useState<Entry[]>(SEED);
  const [k, setK] = useState("");
  const [v, setV] = useState("");
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState<string | null>(null);

  const write = () => {
    if (!k.trim() || !v.trim()) return;
    setMem((m) => [...m.filter((e) => e.key !== k.trim()), { key: k.trim(), value: v.trim() }]);
    setK("");
    setV("");
  };

  const read = () => {
    const q = query.toLowerCase().trim();
    // fuzzy-ish nearest key by shared words
    let best: Entry | null = null;
    let bestScore = 0;
    for (const e of mem) {
      const score = overlap(q, e.key.toLowerCase());
      if (score > bestScore) {
        best = e;
        bestScore = score;
      }
    }
    setAnswer(best && bestScore > 0 ? best.value : "🤷 nothing close in memory");
    onDiscover();
  };

  return (
    <div className="grid gap-5 lg:grid-cols-2">
      <Panel title="Write a fact at test time" hint="no retraining, no context window">
        <div className="space-y-2">
          <input value={k} onChange={(e) => setK(e.target.value)} placeholder="key (e.g. 'birthday')" className="w-full rounded-lg bg-black/40 px-3 py-2 text-sm text-white outline-none ring-1 ring-white/10 focus:ring-purple-400/50" />
          <input value={v} onChange={(e) => setV(e.target.value)} placeholder="value (e.g. 'March 5')" className="w-full rounded-lg bg-black/40 px-3 py-2 text-sm text-white outline-none ring-1 ring-white/10 focus:ring-purple-400/50" />
          <GameButton onClick={write}>Write to memory ↓</GameButton>
        </div>

        <div className="mt-4">
          <div className="mb-1 text-xs text-white/40">memory matrix (key → value)</div>
          <ul className="space-y-1.5">
            {mem.map((e, i) => (
              <li key={i} className="flex items-center justify-between rounded-lg bg-purple-500/10 px-3 py-1.5 text-sm">
                <span className="text-white/60">{e.key}</span>
                <span className="font-mono text-purple-200">{e.value}</span>
              </li>
            ))}
          </ul>
        </div>
      </Panel>

      <div className="space-y-4">
        <Panel title="Query it later">
          <div className="flex gap-2">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && read()}
              placeholder="ask about a stored key…"
              className="flex-1 rounded-lg bg-black/40 px-3 py-2 text-sm text-white outline-none ring-1 ring-white/10 focus:ring-purple-400/50"
            />
            <GameButton onClick={read}>Recall</GameButton>
          </div>
          {answer !== null && (
            <div className="mt-3 rounded-xl bg-purple-500/15 p-3 text-sm text-purple-100">
              memory returns: <b>{answer}</b>
            </div>
          )}
        </Panel>

        <Insight>
          A normal transformer can only "remember" what fits in its context window — scroll past it
          and it's gone. Titans adds a small <b>neural memory</b> that it <i>writes into while
          running</i> (test-time training). Facts live in a fixed-size weight matrix, not the
          prompt, so they persist across an arbitrarily long conversation without growing the KV
          cache. Write and recall a fact to bank your XP.
        </Insight>
      </div>
    </div>
  );
}

function overlap(a: string, b: string): number {
  const wa = new Set(a.split(/\s+/).filter(Boolean));
  const wb = b.split(/\s+/).filter(Boolean);
  return wb.reduce((s, w) => s + (wa.has(w) ? 1 : 0), 0);
}
