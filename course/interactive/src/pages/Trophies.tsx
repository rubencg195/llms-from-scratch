import { motion } from "framer-motion";
import PageTransition from "@/components/PageTransition";
import {
  ACHIEVEMENTS,
  useProgress,
  levelForXp,
  TOTAL_MAX_XP,
  TOTAL_JOURNEY_ITEMS,
} from "@/store/progress";
import { TOTAL_MODULES } from "@/data/curriculum";
import { GameButton } from "@/components/ui/primitives";

export default function Trophies() {
  const { xp, unlocked, streak, reset, journeyStats } = useProgress();
  const { level } = levelForXp(xp);
  const journey = journeyStats();

  return (
    <PageTransition>
      <h1 className="text-2xl font-extrabold text-white sm:text-3xl">Your Trophy Room</h1>
      <p className="mt-1 text-white/55">
        Progress is saved automatically in <code className="text-indigo-300">localStorage</code> on this device.
      </p>

      <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <BigStat icon="⭐" label="Level" value={level} />
        <BigStat icon="⚡" label="XP" value={`${xp}/${TOTAL_MAX_XP}`} />
        <BigStat icon="🗺️" label="Journey" value={`${journey.itemsDone}/${TOTAL_JOURNEY_ITEMS}`} />
        <BigStat icon="🔥" label="Streak" value={`${streak}d`} />
      </div>

      <div className="surface-muted relative z-10 mt-4 grid grid-cols-3 gap-3 rounded-2xl p-4 text-center text-sm">
        <div>
          <div className="text-lg font-bold text-white">{journey.lecDone}</div>
          <div className="text-[11px] text-white/45">Lectures read</div>
        </div>
        <div>
          <div className="text-lg font-bold text-white">{journey.labDone}</div>
          <div className="text-[11px] text-white/45">Labs opened</div>
        </div>
        <div>
          <div className="text-lg font-bold text-white">{journey.playDone}/{TOTAL_MODULES}</div>
          <div className="text-[11px] text-white/45">Playgrounds done</div>
        </div>
      </div>

      <h2 className="mb-4 mt-10 text-lg font-bold text-white">
        Achievements ({unlocked.length}/{ACHIEVEMENTS.length})
      </h2>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        {ACHIEVEMENTS.map((a, i) => {
          const got = unlocked.includes(a.id);
          return (
            <motion.div
              key={a.id}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: i * 0.04 }}
              className={`surface relative z-10 rounded-2xl p-4 text-center transition ${
                got ? "border-amber-400/40 bg-amber-950" : "opacity-60"
              }`}
            >
              <div className={`text-4xl ${got ? "" : "grayscale"}`}>{got ? a.icon : "🔒"}</div>
              <div className="mt-2 text-sm font-semibold text-white">{a.title}</div>
              <div className="mt-0.5 text-xs text-white/50">{a.desc}</div>
            </motion.div>
          );
        })}
      </div>

      <div className="mt-10">
        <GameButton
          variant="ghost"
          onClick={() => {
            if (confirm("Reset all progress on this device? This cannot be undone.")) reset();
          }}
        >
          Reset progress
        </GameButton>
      </div>
    </PageTransition>
  );
}

function BigStat({ icon, label, value }: { icon: string; label: string; value: React.ReactNode }) {
  return (
    <div className="glass rounded-2xl p-4 text-center">
      <div className="text-3xl">{icon}</div>
      <div className="mt-1 font-mono text-base font-bold break-words text-white sm:text-xl">{value}</div>
      <div className="text-[11px] uppercase tracking-wide text-white/40">{label}</div>
    </div>
  );
}
