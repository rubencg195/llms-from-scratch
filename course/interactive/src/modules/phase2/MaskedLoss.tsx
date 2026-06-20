import { useState } from "react";
import { Insight, Panel, Stat, GameButton } from "@/components/ui/primitives";
import type { ModuleProps } from "../types";

interface Tok {
  text: string;
  role: "user" | "assistant" | "special";
}

const TOKENS: Tok[] = [
  { text: "<|user|>", role: "special" },
  { text: "What", role: "user" },
  { text: "is", role: "user" },
  { text: "2+2?", role: "user" },
  { text: "<|assistant|>", role: "special" },
  { text: "The", role: "assistant" },
  { text: "answer", role: "assistant" },
  { text: "is", role: "assistant" },
  { text: "4.", role: "assistant" },
];

export default function MaskedLoss({ onDiscover }: ModuleProps) {
  // mask[i] = 1 means "train on this token"
  const [mask, setMask] = useState<number[]>(TOKENS.map(() => 1));

  const correct = TOKENS.map((t) => (t.role === "assistant" ? 1 : 0));
  const isCorrect = mask.every((m, i) => m === correct[i]);

  const toggle = (i: number) =>
    setMask((m) => m.map((v, j) => (j === i ? (v ? 0 : 1) : v)));

  const autoMask = () => {
    setMask(correct);
    onDiscover();
  };

  if (isCorrect) onDiscover();

  const trained = mask.filter(Boolean).length;

  return (
    <div className="space-y-5">
      <Panel title="Click tokens to toggle the loss mask" hint="green = model learns from it">
        <div className="flex flex-wrap gap-2">
          {TOKENS.map((t, i) => {
            const on = mask[i] === 1;
            return (
              <button
                key={i}
                onClick={() => toggle(i)}
                className="flex flex-col items-center rounded-lg px-3 py-2 font-mono text-sm transition"
                style={{
                  background: on ? "#34d39922" : "rgba(255,255,255,0.04)",
                  border: `1px solid ${on ? "#34d399" : "rgba(255,255,255,0.1)"}`,
                  color: on ? "#6ee7b7" : "rgba(255,255,255,0.4)",
                  opacity: t.role === "special" ? 0.85 : 1,
                }}
              >
                {t.text}
                <span className="mt-1 text-[10px]">{on ? "loss ✓" : "ignored"}</span>
              </button>
            );
          })}
        </div>
      </Panel>

      <div className="grid grid-cols-2 gap-3">
        <Stat label="Tokens trained on" value={`${trained}/${TOKENS.length}`} accent="#34d399" />
        <Stat label="Mask correct?" value={isCorrect ? "Yes 🎯" : "Not yet"} accent={isCorrect ? "#34d399" : "#f472b6"} />
      </div>

      <GameButton variant="ghost" onClick={autoMask}>
        Show the correct mask
      </GameButton>

      {isCorrect && (
        <div className="rounded-xl border border-emerald-400/30 bg-emerald-500/10 p-3 text-sm text-emerald-200">
          🎯 Only the assistant's reply is trained on — exactly right.
        </div>
      )}

      <Insight>
        If you train on the <i>whole</i> conversation, the model wastes capacity learning to predict
        the user's questions — which it can't control. <b>Masked loss</b> zeros out the gradient for
        user/prompt tokens so the model only learns to produce good <i>assistant</i> replies. Turn
        the mask green on just the assistant tokens to bank your XP.
      </Insight>
    </div>
  );
}
