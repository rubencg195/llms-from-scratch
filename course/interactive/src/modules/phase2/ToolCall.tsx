import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Insight, Panel, GameButton } from "@/components/ui/primitives";
import type { ModuleProps } from "../types";

type Step = 0 | 1 | 2 | 3 | 4;
const CITY_TEMPS: Record<string, number> = { Paris: 14, Tokyo: 21, Cairo: 33, Oslo: 3 };

export default function ToolCall({ onDiscover }: ModuleProps) {
  const [city, setCity] = useState("Tokyo");
  const [step, setStep] = useState<Step>(0);

  const temp = CITY_TEMPS[city];
  const next = () =>
    setStep((s) => {
      const n = Math.min(4, s + 1) as Step;
      if (n === 4) onDiscover();
      return n;
    });
  const reset = () => setStep(0);

  const lines = [
    { who: "user", c: "#22d3ee", t: `What's the weather in ${city}?` },
    {
      who: "assistant",
      c: "#34d399",
      t: `<functioncall> {"name":"get_weather","arguments":{"city":"${city}"}}`,
    },
    { who: "tool", c: "#fbbf24", t: `{"temp_c": ${temp}, "city": "${city}"}` },
    { who: "assistant", c: "#34d399", t: `It's currently ${temp}°C in ${city}.` },
  ];

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_320px]">
      <Panel title="Tool-calling loop" hint="step through it">
        <div className="mb-4 flex items-center gap-2">
          <span className="text-sm text-white/55">City:</span>
          {Object.keys(CITY_TEMPS).map((c) => (
            <button
              key={c}
              onClick={() => {
                setCity(c);
                setStep(0);
              }}
              className={`rounded-lg px-2.5 py-1 text-sm ${
                city === c ? "bg-emerald-500/30 text-emerald-200" : "bg-white/5 text-white/60"
              }`}
            >
              {c}
            </button>
          ))}
        </div>

        <div className="space-y-2">
          <AnimatePresence>
            {lines.slice(0, step).map((l, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: l.who === "user" ? -20 : 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="rounded-xl bg-white/5 p-3"
              >
                <div className="mb-1 text-xs font-semibold" style={{ color: l.c }}>
                  {l.who}
                  {l.who === "tool" && " (your code runs)"}
                </div>
                <code className="break-words font-mono text-xs text-white/85">{l.t}</code>
              </motion.div>
            ))}
          </AnimatePresence>
          {step === 0 && <p className="text-sm text-white/45">Press "Next step" to watch the model use a tool.</p>}
        </div>

        <div className="mt-4 flex gap-2">
          <GameButton onClick={next} disabled={step === 4}>
            Next step →
          </GameButton>
          <GameButton variant="ghost" onClick={reset}>
            Reset
          </GameButton>
        </div>
      </Panel>

      <div className="space-y-4">
        <Panel title="The 4 stages">
          <ol className="space-y-2 text-sm">
            {["User asks", "Model emits JSON call", "Your code returns data", "Model answers in words"].map(
              (s, i) => (
                <li
                  key={i}
                  className={`flex items-center gap-2 ${step > i ? "text-white" : "text-white/35"}`}
                >
                  <span
                    className={`grid h-6 w-6 place-items-center rounded-full text-xs ${
                      step > i ? "bg-emerald-500 text-black" : "bg-white/10"
                    }`}
                  >
                    {i + 1}
                  </span>
                  {s}
                </li>
              ),
            )}
          </ol>
        </Panel>

        <Insight>
          The model can't actually check the weather — it has no internet. Instead it learns to emit
          structured <b>JSON</b> describing the call it <i>wants</i>. Your program runs the real
          function and feeds the result back as a <code className="text-amber-300">tool</code> turn,
          and the model turns that data into a natural answer. Reach the final step to bank your XP.
        </Insight>
      </div>
    </div>
  );
}
