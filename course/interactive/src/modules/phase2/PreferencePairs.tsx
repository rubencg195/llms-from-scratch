import { useState } from "react";
import { Insight, Panel, Stat, GameButton } from "@/components/ui/primitives";
import type { ModuleProps } from "../types";

const EXAMPLE = {
  prompt: "Explain gravity to a 5-year-old.",
  chosen:
    "Gravity is like an invisible hug from the Earth. It pulls everything toward the ground so we don't float away.",
  rejected:
    "Gravity is a fundamental interaction described by general relativity and approximated by Newton's law F = G m1 m2 / r²...",
};

export default function PreferencePairs({ onDiscover }: ModuleProps) {
  const [pick, setPick] = useState<"chosen" | "rejected" | null>(null);
  const [confirmed, setConfirmed] = useState(false);

  const choose = (which: "chosen" | "rejected") => {
    setPick(which);
    if (which === "chosen") {
      setConfirmed(true);
      onDiscover();
    }
  };

  return (
    <div className="space-y-4">
      <Panel title="Prompt">
        <p className="text-sm text-white/80">{EXAMPLE.prompt}</p>
      </Panel>

      <div className="grid gap-4 md:grid-cols-2">
        <button
          type="button"
          onClick={() => choose("chosen")}
          className={`rounded-xl border p-4 text-left transition ${
            pick === "chosen"
              ? "border-emerald-400 bg-emerald-500/15"
              : "border-white/10 bg-white/5 hover:border-emerald-400/50"
          }`}
        >
          <div className="mb-2 text-xs font-semibold uppercase text-emerald-300">Chosen ✓</div>
          <p className="text-sm text-white/85">{EXAMPLE.chosen}</p>
        </button>

        <button
          type="button"
          onClick={() => choose("rejected")}
          className={`rounded-xl border p-4 text-left transition ${
            pick === "rejected"
              ? "border-red-400 bg-red-500/15"
              : "border-white/10 bg-white/5 hover:border-red-400/50"
          }`}
        >
          <div className="mb-2 text-xs font-semibold uppercase text-red-300">Rejected ✗</div>
          <p className="text-sm text-white/85">{EXAMPLE.rejected}</p>
        </button>
      </div>

      {pick === "rejected" && (
        <div className="rounded-xl border border-red-400/30 bg-red-500/10 p-3 text-sm text-red-200">
          Correct for a 5-year-old, but technically accurate. DPO would down-weight this style.
        </div>
      )}

      {confirmed && (
        <>
          <div className="grid grid-cols-2 gap-3">
            <Stat label="SFT teaches" value="Format & skills" accent="#34d399" />
            <Stat label="DPO teaches" value="Which answer wins" accent="#22d3ee" />
          </div>
          <GameButton variant="ghost" onClick={() => { setPick(null); setConfirmed(false); }}>
            Try again
          </GameButton>
        </>
      )}

      <Insight>
        After SFT (Phase 2 labs 2.1–2.3), <span className="text-emerald-300">DPO</span> trains on
        preference pairs: increase probability of the chosen answer vs a rejected one, relative to a
        frozen SFT copy. No reward model needed — Karpathy's post-training pipeline in one step.
      </Insight>
    </div>
  );
}
