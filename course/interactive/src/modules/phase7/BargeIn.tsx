import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Insight, Panel, GameButton } from "@/components/ui/primitives";
import type { ModuleProps } from "../types";

const REPLY = "Sure! The three main causes of the French Revolution were...".split(" ");

export default function BargeIn({ onDiscover }: ModuleProps) {
  const [spoken, setSpoken] = useState(0);
  const [speaking, setSpeaking] = useState(false);
  const [userTalking, setUserTalking] = useState(false);
  const [bargedIn, setBargedIn] = useState(false);
  const timer = useRef<number>(0);

  useEffect(() => {
    if (!speaking) return;
    timer.current = window.setInterval(() => {
      setSpoken((s) => {
        if (s >= REPLY.length) {
          setSpeaking(false);
          return s;
        }
        return s + 1;
      });
    }, 360);
    return () => clearInterval(timer.current);
  }, [speaking]);

  const start = () => {
    setSpoken(0);
    setBargedIn(false);
    setUserTalking(false);
    setSpeaking(true);
  };

  const bargeIn = () => {
    setSpeaking(false);
    setUserTalking(true);
    setBargedIn(true);
    onDiscover();
    setTimeout(() => setUserTalking(false), 1500);
  };

  return (
    <div className="space-y-5">
      <Panel title="Full-duplex conversation" hint="interrupt anytime">
        <div className="space-y-3">
          <div className="rounded-xl bg-emerald-500/10 p-4 ring-1 ring-emerald-400/30">
            <div className="mb-2 flex items-center gap-2 text-xs font-semibold text-emerald-300">
              🤖 Model {speaking && <span className="animate-pulse">● speaking</span>}
            </div>
            <p className="min-h-[48px] text-sm text-white/85">
              {REPLY.slice(0, spoken).map((w, i) => (
                <motion.span key={i} initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                  {w}{" "}
                </motion.span>
              ))}
              {bargedIn && <span className="text-red-300">— [cut off]</span>}
            </p>
          </div>

          <div
            className={`rounded-xl p-4 ring-1 transition ${
              userTalking ? "bg-cyan-500/15 ring-cyan-400/50" : "bg-white/5 ring-white/10"
            }`}
          >
            <div className="flex items-center gap-2 text-xs font-semibold text-cyan-300">
              🧑 You {userTalking && <span className="animate-pulse">● speaking</span>}
            </div>
            {userTalking && <p className="mt-2 text-sm text-white/80">"Wait — just give me the first one."</p>}
          </div>
        </div>

        <div className="mt-4 flex gap-2">
          <GameButton onClick={start} disabled={speaking}>
            ▶ Ask the model
          </GameButton>
          <GameButton variant="ghost" onClick={bargeIn} disabled={!speaking}>
            ✋ Barge in
          </GameButton>
        </div>
      </Panel>

      <Insight>
        Old voice assistants are <b>half-duplex</b>: they talk, then listen, like a walkie-talkie. A
        full-duplex model runs <b>two token streams at once</b> — generating speech <i>while</i>{" "}
        monitoring your mic. The instant it detects you talking (voice activity), it can stop and
        yield the floor. Start the model, then hit "Barge in" mid-sentence to feel real-time
        turn-taking and bank your XP.
      </Insight>
    </div>
  );
}
