import { useState } from "react";
import { motion } from "framer-motion";
import { Insight, Panel, GameButton, Pill } from "@/components/ui/primitives";
import type { ModuleProps } from "../types";

type Tok = { kind: "text"; v: string } | { kind: "img"; v: number };

const TEXT_BEFORE = ["Describe", "this"];
const IMG = [0, 1, 2, 3]; // 4 image patches
const TEXT_AFTER = ["in", "detail", ":"];

export default function ModalityMix({ onDiscover }: ModuleProps) {
  const [mixed, setMixed] = useState(false);

  const seq: Tok[] = mixed
    ? [
        ...TEXT_BEFORE.map((v) => ({ kind: "text", v }) as Tok),
        ...IMG.map((v) => ({ kind: "img", v }) as Tok),
        ...TEXT_AFTER.map((v) => ({ kind: "text", v }) as Tok),
      ]
    : [...TEXT_BEFORE, ...TEXT_AFTER].map((v) => ({ kind: "text", v }) as Tok);

  const patchColor = (i: number) => `hsl(${190 + i * 30} 70% 45%)`;

  return (
    <div className="space-y-5">
      <Panel title="One sequence, two modalities" hint="interleave image + text">
        <div className="flex min-h-[90px] flex-wrap items-center gap-2 rounded-xl bg-black/30 p-4">
          {seq.map((t, i) => (
            <motion.div
              key={i}
              layout
              initial={{ scale: 0.5, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="flex flex-col items-center"
            >
              {t.kind === "text" ? (
                <span className="rounded-md bg-teal-500/15 px-2.5 py-1.5 font-mono text-sm text-teal-200 ring-1 ring-teal-400/40">
                  {t.v}
                </span>
              ) : (
                <span
                  className="grid h-9 w-9 place-items-center rounded-md text-[10px] font-bold text-black"
                  style={{ background: patchColor(t.v) }}
                  title={`image patch ${t.v}`}
                >
                  🖼{t.v}
                </span>
              )}
              <span className="mt-1 text-[9px] text-white/30">{i}</span>
            </motion.div>
          ))}
        </div>

        <div className="mt-3 flex gap-2">
          <Pill color="#2dd4bf">text token</Pill>
          <Pill color="#22d3ee">image patch token</Pill>
        </div>
      </Panel>

      <GameButton
        variant={mixed ? "ghost" : "primary"}
        onClick={() => {
          setMixed((m) => !m);
          onDiscover();
        }}
      >
        {mixed ? "Remove the image" : "Drop the image into the prompt 🖼️"}
      </GameButton>

      <Insight>
        Because image patches are projected into the <i>same</i> vector space as text tokens, you
        can <b>interleave</b> them anywhere in the sequence — "Describe &lt;image&gt; in detail".
        Self-attention then lets every text token look at every patch and vice-versa, with no
        separate vision tower. One stream, one transformer. Toggle the image in and out to bank your
        XP.
      </Insight>
    </div>
  );
}
