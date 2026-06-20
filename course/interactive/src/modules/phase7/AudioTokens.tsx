import { useMemo, useState } from "react";
import { Insight, Panel, Slider, Stat } from "@/components/ui/primitives";
import type { ModuleProps } from "../types";

const W = 460;
const H = 200;
const SAMPLES = 240;

export default function AudioTokens({ onDiscover }: ModuleProps) {
  const [rate, setRate] = useState(12); // tokens per "second"
  const [low, setLow] = useState(false);

  const wave = useMemo(
    () =>
      Array.from({ length: SAMPLES }, (_, i) => {
        const t = i / SAMPLES;
        return (
          0.5 * Math.sin(2 * Math.PI * 6 * t) +
          0.3 * Math.sin(2 * Math.PI * 13 * t) +
          0.15 * Math.sin(2 * Math.PI * 21 * t)
        );
      }),
    [],
  );

  const tokens = rate;
  const win = Math.floor(SAMPLES / tokens);

  // each token = quantized mean amplitude of its window
  const codes = Array.from({ length: tokens }, (_, k) => {
    const slice = wave.slice(k * win, (k + 1) * win);
    const mean = slice.reduce((a, b) => a + b, 0) / (slice.length || 1);
    return Math.round(((mean + 1) / 2) * 7); // 8 codes (3 bits)
  });

  if (rate <= 8 && !low) {
    setLow(true);
    onDiscover();
  }

  const sx = (i: number) => 20 + (i / (SAMPLES - 1)) * (W - 40);
  const sy = (v: number) => H / 2 - v * (H / 2 - 20);

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_300px]">
      <Panel title="Waveform → discrete codec tokens" hint={`${tokens} tokens/sec`}>
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
          {Array.from({ length: tokens + 1 }, (_, k) => (
            <line key={k} x1={sx(k * win)} y1={10} x2={sx(k * win)} y2={H - 10} stroke="#fb923c22" />
          ))}
          <line x1={20} y1={sy(0)} x2={W - 20} y2={sy(0)} stroke="#26304f" />
          <polyline points={wave.map((v, i) => `${sx(i)},${sy(v)}`).join(" ")} fill="none" stroke="#fb923c" strokeWidth={2} />
        </svg>

        <div className="mt-3 flex flex-wrap gap-1">
          {codes.map((c, i) => (
            <span key={i} className="rounded bg-orange-500/15 px-2 py-1 font-mono text-xs text-orange-200">
              {c}
            </span>
          ))}
        </div>
      </Panel>

      <div className="space-y-4">
        <Slider label="Token rate (Hz)" value={rate} min={4} max={50} step={2} onChange={setRate} accent="#fb923c" />
        <div className="grid grid-cols-2 gap-3">
          <Stat label="Tokens / sec" value={tokens} accent="#fb923c" />
          <Stat label="Codebook" value="8 (3-bit)" accent="#22d3ee" />
        </div>

        <Insight>
          Raw audio is ~24,000 numbers per second — way too many to feed an LLM. A neural{" "}
          <b>codec</b> (VQ-VAE / EnCodec) chops it into short windows and replaces each with the{" "}
          <b>nearest codebook entry</b>, giving just 12–50 discrete tokens per second — the exact
          same trick as BPE for text. Fewer tokens/sec = more compression but less detail. Drop to
          ≤8 Hz to bank your XP.
        </Insight>
      </div>
    </div>
  );
}
