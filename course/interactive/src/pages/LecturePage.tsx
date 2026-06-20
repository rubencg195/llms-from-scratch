import { useEffect, useState } from "react";
import { Link, useParams, Navigate } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import PageTransition from "@/components/PageTransition";
import Markdown from "@/components/Markdown";
import { getPhase } from "@/data/curriculum";
import { lectureForPhase } from "@/content";
import { useProgress, LECTURE_XP } from "@/store/progress";
import { GameButton } from "@/components/ui/primitives";

export default function LecturePage() {
  const { slug } = useParams();
  const phase = slug ? getPhase(slug) : undefined;
  const lecture = phase ? lectureForPhase(phase.id) : undefined;
  const { markRead, isRead } = useProgress();

  const [i, setI] = useState(0);
  const [showNotes, setShowNotes] = useState(false);

  const total = lecture?.slides.length ?? 0;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") setI((x) => Math.min(total - 1, x + 1));
      if (e.key === "ArrowLeft") setI((x) => Math.max(0, x - 1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [total]);

  useEffect(() => {
    if (phase && total && i === total - 1) {
      markRead(`lec:${phase.id}`, LECTURE_XP, `Finished the ${phase.title} lecture`);
    }
  }, [i, total, phase, markRead]);

  if (!phase || !lecture) return <Navigate to="/" replace />;

  const slide = lecture.slides[i];
  const pct = Math.round(((i + 1) / total) * 100);

  return (
    <PageTransition>
      <div className="mb-3 flex items-center gap-2 text-sm text-white/50">
        <Link to="/" className="hover:text-white">Journey</Link>
        <span>/</span>
        <Link to={`/phase/${phase.slug}`} className="hover:text-white">{phase.title}</Link>
        <span>/</span>
        <span className="text-white/80">Lecture</span>
      </div>

      <div className="mb-4 flex items-center justify-between gap-4">
        <h1 className="text-2xl font-extrabold text-white">📽️ {lecture.title}</h1>
        <span className="rounded-full bg-emerald-500/20 px-3 py-1 text-sm text-white/60">
          Slide {i + 1} / {total}
          {isRead(`lec:${phase.id}`) && " · ✓ Read"}
        </span>
      </div>

      <div className="surface-muted mb-4 h-1.5 overflow-hidden rounded-full">
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${pct}%`, background: phase.color.from }}
        />
      </div>

      <div className="surface relative z-10 min-h-[360px] rounded-3xl p-6 md:p-8">
        <AnimatePresence mode="wait">
          <motion.div
            key={i}
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -30 }}
            transition={{ duration: 0.25 }}
          >
            <Markdown>{slide.body}</Markdown>

            {slide.notes.length > 0 && (
              <div className="mt-6">
                <button
                  onClick={() => setShowNotes((s) => !s)}
                  className="text-xs font-semibold text-indigo-300 hover:text-indigo-200"
                >
                  {showNotes ? "▾ Hide speaker notes" : "▸ Show speaker notes"}
                </button>
                {showNotes && (
                  <div className="surface-muted mt-2 space-y-2 rounded-xl p-4 text-sm text-white/60">
                    {slide.notes.map((n, k) => (
                      <p key={k}>{n}</p>
                    ))}
                  </div>
                )}
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>

      <div className="mt-5 flex items-center justify-between">
        <GameButton variant="ghost" onClick={() => setI((x) => Math.max(0, x - 1))} disabled={i === 0}>
          ← Previous
        </GameButton>

        <div className="hidden items-center gap-1.5 sm:flex">
          {lecture.slides.map((_, k) => (
            <button
              key={k}
              onClick={() => setI(k)}
              className="h-2 rounded-full transition-all"
              style={{
                width: k === i ? 22 : 8,
                background: k === i ? phase.color.ring : "rgba(255,255,255,0.2)",
              }}
            />
          ))}
        </div>

        {i < total - 1 ? (
          <GameButton onClick={() => setI((x) => Math.min(total - 1, x + 1))}>Next →</GameButton>
        ) : (
          <GameButton variant="success" onClick={() => window.scrollTo({ top: 0 })}>
            <Link to={`/phase/${phase.slug}`}>Back to phase ✓</Link>
          </GameButton>
        )}
      </div>

      <p className="mt-3 text-center text-xs text-white/30">Tip: use ← → arrow keys to navigate slides</p>
    </PageTransition>
  );
}
