import { useState } from "react";
import { Insight, Panel, GameButton, Pill } from "@/components/ui/primitives";
import type { ModuleProps } from "../types";

type Role = "user" | "assistant" | "tool";
interface Turn {
  role: Role;
  text: string;
}

const SPECIAL: Record<Role, string> = {
  user: "<|user|>",
  assistant: "<|assistant|>",
  tool: "<|tool|>",
};
const ROLE_COLOR: Record<Role, string> = {
  user: "#22d3ee",
  assistant: "#34d399",
  tool: "#fbbf24",
};

export default function ChatTemplate({ onDiscover }: ModuleProps) {
  const [turns, setTurns] = useState<Turn[]>([
    { role: "user", text: "What is 7 times 8?" },
    { role: "assistant", text: "7 times 8 is 56." },
  ]);

  const add = (role: Role) => {
    setTurns((t) => [...t, { role, text: role === "user" ? "..." : "..." }]);
    onDiscover();
  };
  const update = (i: number, text: string) =>
    setTurns((t) => t.map((x, j) => (j === i ? { ...x, text } : x)));
  const remove = (i: number) => setTurns((t) => t.filter((_, j) => j !== i));

  return (
    <div className="grid gap-5 lg:grid-cols-2">
      <Panel title="Build a conversation" hint="add turns">
        <div className="space-y-3">
          {turns.map((t, i) => (
            <div key={i} className="rounded-xl bg-white/5 p-3">
              <div className="mb-2 flex items-center justify-between">
                <Pill color={ROLE_COLOR[t.role]}>{t.role}</Pill>
                <button onClick={() => remove(i)} className="text-xs text-white/40 hover:text-red-300">
                  remove
                </button>
              </div>
              <input
                value={t.text}
                onChange={(e) => update(i, e.target.value)}
                className="w-full rounded-lg bg-black/40 px-3 py-2 text-sm text-white outline-none ring-1 ring-white/10 focus:ring-emerald-400/50"
              />
            </div>
          ))}
        </div>
        <div className="mt-4 flex flex-wrap gap-2">
          {(["user", "assistant", "tool"] as Role[]).map((r) => (
            <GameButton key={r} variant="ghost" onClick={() => add(r)}>
              + {r}
            </GameButton>
          ))}
        </div>
      </Panel>

      <div className="space-y-4">
        <Panel title="What the model actually reads">
          <pre className="overflow-x-auto whitespace-pre-wrap rounded-xl bg-black/50 p-4 font-mono text-xs leading-relaxed">
            {turns.map((t, i) => (
              <span key={i}>
                <span style={{ color: ROLE_COLOR[t.role] }}>{SPECIAL[t.role]}</span>
                <span className="text-white/85">{t.text}</span>
                <span className="text-white/40">{"<|end|>"}</span>
                {"\n"}
              </span>
            ))}
            <span style={{ color: ROLE_COLOR.assistant }}>{"<|assistant|>"}</span>
            <span className="animate-pulse text-white/40">▋</span>
          </pre>
        </Panel>

        <Insight>
          A base model just continues text — it has no idea who is "speaking". A <b>chat template</b>{" "}
          wraps each turn in <i>special tokens</i> like{" "}
          <code className="text-emerald-300">{"<|user|>"}</code> and{" "}
          <code className="text-emerald-300">{"<|assistant|>"}</code> so the model learns turn-taking.
          The trailing <code className="text-emerald-300">{"<|assistant|>"}</code> is the cue that
          says "your turn to generate". Add a turn to bank your XP.
        </Insight>
      </div>
    </div>
  );
}
