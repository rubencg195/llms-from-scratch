import { Link, useParams, Navigate } from "react-router-dom";
import { motion } from "framer-motion";
import PageTransition from "@/components/PageTransition";
import { getPhase, PHASES } from "@/data/curriculum";
import { useProgress } from "@/store/progress";
import { lectureForPhase, labsForPhase } from "@/content";

export default function Phase() {
  const { slug } = useParams();
  const phase = slug ? getPhase(slug) : undefined;
  const { isComplete, isRead, phaseJourney } = useProgress();

  if (!phase) return <Navigate to="/" replace />;

  const lecture = lectureForPhase(phase.id);
  const labs = labsForPhase(phase.id);
  const j = phaseJourney(phase.id);
  const next = PHASES.find((p) => p.id === phase.id + 1);
  const prev = PHASES.find((p) => p.id === phase.id - 1);

  return (
    <PageTransition>
      <div className="mb-2 flex items-center gap-2 text-sm text-white/50">
        <Link to="/" className="hover:text-white">
          Journey
        </Link>
        <span>/</span>
        <span className="text-white/80">Phase {phase.id}</span>
      </div>

      <div className="mb-8 flex items-center gap-4">
        <div
          className="grid h-16 w-16 place-items-center rounded-2xl text-4xl"
          style={{ background: `${phase.color.from}26`, border: `1px solid ${phase.color.ring}55` }}
        >
          {phase.icon}
        </div>
        <div>
          <h1 className="text-3xl font-extrabold text-white">{phase.title}</h1>
          <p className="text-white/55">{phase.tagline}</p>
        </div>
      </div>

      <div className="surface relative z-10 mb-8 rounded-2xl p-4">
        <div className="mb-2 flex items-center justify-between text-sm">
          <span className="text-white/60">Phase progress</span>
          <span className="font-semibold text-white">
            {j.done}/{j.total} · {j.pct}%
          </span>
        </div>
        <div className="surface-muted h-2 overflow-hidden rounded-full">
          <div
            className="h-full rounded-full transition-all"
            style={{ width: `${j.pct}%`, background: phase.color.from }}
          />
        </div>
        <div className="mt-2 flex flex-wrap gap-3 text-xs text-white/45">
          <span>{j.lectureDone ? "📽️ Lecture ✓" : "📽️ Lecture"}</span>
          <span>🧪 {j.labsDone}/{j.labsTotal} labs</span>
          <span>🎮 {j.playDone}/{j.playTotal} playgrounds</span>
        </div>
      </div>

      {/* Suggested flow hint */}
      <div className="mb-6 flex flex-wrap items-center gap-2 text-sm text-white/50">
        <span>Suggested flow:</span>
        <span className="surface-muted rounded-full px-2.5 py-0.5">1 · Watch lecture</span>
        <span>→</span>
        <span className="surface-muted rounded-full px-2.5 py-0.5">2 · Play the concepts</span>
        <span>→</span>
        <span className="surface-muted rounded-full px-2.5 py-0.5">3 · Do the labs</span>
      </div>

      {/* Lecture */}
      {lecture && (
        <Section title="Lecture" accent={phase.color.ring}>
          <Link
            to={`/phase/${phase.slug}/lecture`}
            className="surface group relative z-10 flex items-center justify-between rounded-2xl p-5 transition hover:brightness-105"
          >
            <div className="flex items-center gap-4">
              <span className="surface-muted grid h-12 w-12 place-items-center rounded-xl text-2xl">📽️</span>
              <div>
                <div className="font-semibold text-white">{lecture.title}</div>
                <div className="text-sm text-white/50">{lecture.slides.length} slides · concepts, math & intuition</div>
              </div>
            </div>
            <div className="flex items-center gap-3">
              {isRead(`lec:${phase.id}`) && (
                <span className="rounded-full bg-emerald-500/20 px-2 py-0.5 text-xs font-semibold text-emerald-300">
                  ✓ Read
                </span>
              )}
              <span className="text-sm font-semibold text-white/70 transition group-hover:translate-x-0.5">
                Open →
              </span>
            </div>
          </Link>
        </Section>
      )}

      {/* Playgrounds */}
      <Section title="Interactive playgrounds" accent={phase.color.ring}>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {phase.modules.map((m, i) => {
            const done = isComplete(phase.id, m.id);
            return (
              <motion.div
                key={m.id}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
                whileHover={{ y: -3 }}
              >
                <Link
                  to={`/phase/${phase.slug}/play/${m.id}`}
                  className="surface group relative z-10 block h-full rounded-2xl p-5 transition hover:brightness-105"
                >
                  <div className="flex items-start justify-between">
                    <h3 className="text-lg font-bold text-white">
                      {m.threeD && <span title="3D">🪐 </span>}
                      {m.title}
                    </h3>
                    {done ? (
                      <span className="rounded-full bg-emerald-500/20 px-2 py-0.5 text-xs font-semibold text-emerald-300">
                        ✓ Done
                      </span>
                    ) : (
                      <span
                        className="rounded-full px-2 py-0.5 text-xs font-semibold"
                        style={{ background: `${phase.color.from}22`, color: phase.color.ring }}
                      >
                        +{m.xp} XP
                      </span>
                    )}
                  </div>
                  <p className="mt-1.5 text-sm text-white/55">{m.blurb}</p>
                  <div className="mt-4 flex items-center justify-between">
                    <span className="text-[11px] text-white/35">{m.mirrors}</span>
                    <span className="text-sm font-semibold text-white/70 transition group-hover:translate-x-0.5">
                      Play →
                    </span>
                  </div>
                </Link>
              </motion.div>
            );
          })}
        </div>
      </Section>

      {/* Labs */}
      <Section title={`Labs (${labs.length})`} accent={phase.color.ring}>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {labs.map((lab, i) => (
            <motion.div
              key={lab.slug}
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.04 }}
            >
              <Link
                to={`/phase/${phase.slug}/lab/${lab.slug}`}
                className="surface group relative z-10 flex items-center justify-between rounded-xl p-4 transition hover:brightness-105"
              >
                <div className="flex items-center gap-3">
                  <span className="font-mono text-xs text-white/40">{lab.section}</span>
                  <span className="text-sm font-medium text-white/85">{lab.title.replace(/^Section\s+[\d.]+:\s*/, "")}</span>
                </div>
                <div className="flex items-center gap-2">
                  {isRead(`lab:${lab.slug}`) && <span className="text-xs text-emerald-300">✓</span>}
                  <span className="text-sm text-white/50 transition group-hover:translate-x-0.5">Read →</span>
                </div>
              </Link>
            </motion.div>
          ))}
        </div>
      </Section>

      <div className="mt-10 flex items-center justify-between">
        {prev ? (
          <Link to={`/phase/${prev.slug}`} className="surface-muted relative z-10 rounded-xl px-4 py-2.5 text-sm text-white/70 hover:brightness-110">
            ← Phase {prev.id}: {prev.title}
          </Link>
        ) : (
          <span />
        )}
        {next ? (
          <Link to={`/phase/${next.slug}`} className="surface-muted relative z-10 rounded-xl px-4 py-2.5 text-sm text-white/70 hover:brightness-110">
            Phase {next.id}: {next.title} →
          </Link>
        ) : (
          <Link to="/trophies" className="relative z-10 rounded-xl bg-orange-500 px-4 py-2.5 text-sm font-bold text-black">
            See your trophies 🏆
          </Link>
        )}
      </div>
    </PageTransition>
  );
}

function Section({ title, accent, children }: { title: string; accent: string; children: React.ReactNode }) {
  return (
    <section className="mb-8">
      <h2 className="mb-3 flex items-center gap-2 text-sm font-bold uppercase tracking-wide text-white/70">
        <span className="h-2 w-2 rounded-full" style={{ background: accent }} />
        {title}
      </h2>
      {children}
    </section>
  );
}
