import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Insight, Panel, Pill, Stat } from "@/components/ui/primitives";
import type { ModuleProps } from "../types";

type Mode = "char" | "word" | "bpe";

// Toy BPE: split words into a few common subword pieces so students see
// "sub-word" chunks without needing the real merge table.
const SUFFIXES = ["ing", "ed", "ly", "tion", "s", "er", "est", "ness"];
const PREFIXES = ["un", "re", "in", "pre"];

function bpeSplit(word: string): string[] {
  let w = word;
  const pieces: string[] = [];
  for (const p of PREFIXES) {
    if (w.toLowerCase().startsWith(p) && w.length > p.length + 2) {
      pieces.push(w.slice(0, p.length));
      w = w.slice(p.length);
      break;
    }
  }
  let suffix = "";
  for (const s of SUFFIXES) {
    if (w.toLowerCase().endsWith(s) && w.length > s.length + 2) {
      suffix = w.slice(w.length - s.length);
      w = w.slice(0, w.length - s.length);
      break;
    }
  }
  pieces.push(w);
  if (suffix) pieces.push(suffix);
  return pieces;
}

function tokenize(text: string, mode: Mode): string[] {
  if (mode === "char") return [...text].map((c) => (c === " " ? "␣" : c));
  const words = text.match(/\S+|\s+/g) ?? [];
  if (mode === "word") return words.filter((w) => w.trim().length);
  // bpe
  const out: string[] = [];
  for (const chunk of words) {
    if (!chunk.trim()) continue;
    const lead = chunk.match(/^[A-Za-z]+/);
    if (lead && lead[0].length > 3) {
      const sub = bpeSplit(lead[0]);
      out.push(...sub.map((s, i) => (i === 0 ? s : "·" + s)));
      const rest = chunk.slice(lead[0].length);
      if (rest) out.push(rest);
    } else {
      out.push(chunk);
    }
  }
  return out;
}

function idFor(tok: string): number {
  let h = 0;
  for (let i = 0; i < tok.length; i++) h = (h * 31 + tok.charCodeAt(i)) >>> 0;
  return h % 50257; // GPT-2-ish vocab size
}

const COLORS = ["#a78bfa", "#22d3ee", "#34d399", "#fbbf24", "#f472b6", "#60a5fa", "#fb923c"];

export default function Tokenizer({ onDiscover }: ModuleProps) {
  const [text, setText] = useState("Unbelievably, the running cats jumped quickly!");
  const [mode, setMode] = useState<Mode>("bpe");
  const [triedAll, setTriedAll] = useState<Set<Mode>>(new Set(["bpe"]));

  const tokens = useMemo(() => tokenize(text, mode), [text, mode]);

  const setM = (m: Mode) => {
    setMode(m);
    setTriedAll((s) => {
      const n = new Set(s).add(m);
      if (n.size >= 3) onDiscover();
      return n;
    });
  };

  const vocabHint =
    mode === "char" ? "~100" : mode === "word" ? "100,000+" : "8k–50k";

  return (
    <div className="space-y-5">
      <Panel title="Type anything" hint="watch it become tokens">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={2}
          className="w-full resize-none rounded-xl bg-black/40 p-3 font-mono text-sm text-white outline-none ring-1 ring-white/10 focus:ring-violet-400/50"
        />
        <div className="mt-3 flex flex-wrap gap-2">
          {(["char", "word", "bpe"] as Mode[]).map((m) => (
            <button
              key={m}
              onClick={() => setM(m)}
              className={`rounded-lg px-3 py-1.5 text-sm font-semibold transition ${
                mode === m
                  ? "bg-violet-500 text-white"
                  : "bg-white/5 text-white/60 hover:bg-white/10"
              }`}
            >
              {m === "char" ? "Character" : m === "word" ? "Word" : "Subword (BPE)"}
            </button>
          ))}
        </div>
      </Panel>

      <Panel title="Tokens" hint={`${tokens.length} tokens`}>
        <div className="flex flex-wrap gap-1.5">
          {tokens.map((t, i) => {
            const color = COLORS[idFor(t) % COLORS.length];
            return (
              <motion.div
                key={i}
                initial={{ scale: 0.6, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: Math.min(i * 0.01, 0.3) }}
                className="flex flex-col items-center"
              >
                <span
                  className="rounded-md px-2 py-1 font-mono text-sm"
                  style={{ background: `${color}22`, color, border: `1px solid ${color}55` }}
                >
                  {t}
                </span>
                <span className="mt-0.5 font-mono text-[10px] text-white/35">{idFor(t)}</span>
              </motion.div>
            );
          })}
        </div>
      </Panel>

      <div className="grid grid-cols-3 gap-3">
        <Stat label="Tokens" value={tokens.length} accent="#a78bfa" />
        <Stat label="Characters" value={text.length} accent="#22d3ee" />
        <Stat label="Vocab size" value={vocabHint} accent="#34d399" />
      </div>

      <div className="flex flex-wrap gap-2">
        <Pill color="#a78bfa">tried {triedAll.size}/3 strategies</Pill>
      </div>

      <Insight>
        Models never see letters — they see <b>integer IDs</b>. Character-level makes huge sequences;
        word-level needs a giant vocabulary and chokes on new words. <b>Subword (BPE)</b> is the
        sweet spot: it reuses pieces like <code className="text-violet-300">·ing</code> and{" "}
        <code className="text-violet-300">un·</code> so it can spell <i>any</i> word from a small
        vocab. Try all three strategies to see the trade-off and bank your XP.
      </Insight>
    </div>
  );
}
