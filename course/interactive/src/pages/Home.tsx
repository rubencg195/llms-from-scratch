import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import PageTransition from "@/components/PageTransition";
import { PHASES, TOTAL_MODULES } from "@/data/curriculum";
import { LECTURES, LABS, labsForPhase, lectureForPhase } from "@/content";
import { useProgress, levelForXp, TOTAL_MAX_XP } from "@/store/progress";

export default function Home() {
  const { xp, journeyStats } = useProgress();
  const { level } = levelForXp(xp);
  const journey = journeyStats();

  return (
    <PageTransition>
      {/* Hero */}
      <section className="surface relative z-10 mb-10 overflow-hidden rounded-3xl p-8 md:p-12">
        <div className="relative">
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="max-w-2xl text-4xl font-extrabold leading-tight text-white md:text-5xl"
          >
            Build an LLM from scratch —{" "}
            <span className="text-glow text-fuchsia-300">
              by playing with it
            </span>
          </motion.h1>
          <p className="mt-4 max-w-xl text-white/60">
            The whole course in your browser: <b>{LECTURES.length} lectures</b>,{" "}
            <b>{LABS.length} labs</b>, and <b>{TOTAL_MODULES} interactive playgrounds</b>. Read the
            slides, poke at the concepts, then run the real labs in JupyterLab.
          </p>

          <div className="mt-7 flex flex-wrap items-center gap-3">
            <Link
              to={`/phase/${PHASES[0].slug}`}
              className="relative z-10 rounded-xl bg-indigo-500 px-5 py-3 text-sm font-bold text-white shadow-lg shadow-indigo-500/30 transition hover:brightness-110 active:scale-95"
            >
              {journey.itemsDone === 0 ? "Start the journey →" : "Continue →"}
            </Link>
            <div className="surface-muted flex items-center gap-4 rounded-xl px-5 py-3 text-sm">
              <Meter label="Level" value={level} />
              <div className="h-8 w-px bg-white/10" />
              <Meter label="XP" value={`${xp}/${TOTAL_MAX_XP}`} />
              <div className="h-8 w-px bg-white/10" />
              <Meter label="Journey" value={`${journey.pct}%`} />
            </div>
          </div>
        </div>
      </section>

      {/* Journey map */}
      <h2 className="mb-1 text-lg font-bold text-white">The Journey</h2>
      <p className="mb-6 text-sm text-white/50">
        Every phase has its full lecture, every lab section, and hands-on playgrounds.
      </p>

      <div className="relative grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
        {PHASES.map((phase, i) => (
          <PhaseCard key={phase.id} phaseIndex={i} />
        ))}
      </div>
    </PageTransition>
  );
}

function Meter({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="text-center">
      <div className="text-[11px] uppercase tracking-wide text-white/40">{label}</div>
      <div className="font-mono text-base font-bold text-white">{value}</div>
    </div>
  );
}

function PhaseCard({ phaseIndex }: { phaseIndex: number }) {
  const phase = PHASES[phaseIndex];
  const { phaseJourney } = useProgress();
  const j = phaseJourney(phase.id);
  const complete = j.done === j.total;

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: phaseIndex * 0.05 }}
      whileHover={{ y: -4 }}
    >
      <Link
        to={`/phase/${phase.slug}`}
        className="surface group relative z-10 block overflow-hidden rounded-2xl p-5"
      >
        <div
          className="absolute -right-8 -top-8 h-28 w-28 rounded-full opacity-0 blur-2xl transition group-hover:scale-125 group-hover:opacity-100"
          style={{ background: `${phase.color.ring}22` }}
        />
        <div className="relative flex items-start justify-between">
          <div
            className="grid h-14 w-14 place-items-center rounded-2xl text-3xl shadow-lg"
            style={{ background: `${phase.color.from}26`, border: `1px solid ${phase.color.ring}55` }}
          >
            {phase.icon}
          </div>
          <span className="surface-muted rounded-full px-2.5 py-1 text-xs font-semibold text-white/70">
            Phase {phase.id}
          </span>
        </div>

        <h3 className="relative mt-4 text-lg font-bold text-white">{phase.title}</h3>
        <p className="relative mt-1 text-sm text-white/55">{phase.tagline}</p>

        <div className="relative mt-3 flex flex-wrap gap-1.5 text-[11px] text-white/55">
          {lectureForPhase(phase.id) && (
            <span className="surface-muted rounded-md px-2 py-0.5">📽️ lecture</span>
          )}
          <span className="surface-muted rounded-md px-2 py-0.5">🧪 {labsForPhase(phase.id).length} labs</span>
          <span className="surface-muted rounded-md px-2 py-0.5">🎮 {phase.modules.length} play</span>
        </div>

        <div className="relative mt-4">
          <div className="mb-1.5 flex items-center justify-between text-xs text-white/50">
            <span>
              {j.done}/{j.total} items
            </span>
            <span>{complete ? "✓ Complete" : `${j.pct}%`}</span>
          </div>
          <div className="surface-muted h-2 overflow-hidden rounded-full">
            <div
              className="h-full rounded-full transition-all"
              style={{
                width: `${j.pct}%`,
                background: phase.color.from,
              }}
            />
          </div>
          <div className="mt-1.5 text-[10px] text-white/35">
            {j.lectureDone ? "📽️ ✓" : "📽️"} · {j.labsDone}/{j.labsTotal} labs · {j.playDone}/{j.playTotal} play
          </div>
        </div>

        <div className="relative mt-4 flex flex-wrap gap-1.5">
          {phase.modules.slice(0, 4).map((m) => (
            <span
              key={m.id}
              className="surface-muted rounded-md px-2 py-0.5 text-[11px] text-white/55"
            >
              {m.title}
            </span>
          ))}
        </div>
      </Link>
    </motion.div>
  );
}
