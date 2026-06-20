import { useEffect } from "react";
import { Link, useParams, Navigate } from "react-router-dom";
import PageTransition from "@/components/PageTransition";
import Markdown from "@/components/Markdown";
import { getPhase, PHASES } from "@/data/curriculum";
import { labBySlug, labsForPhase } from "@/content";
import { useProgress, LAB_XP } from "@/store/progress";
import { GameButton } from "@/components/ui/primitives";

export default function LabPage() {
  const { slug, labSlug } = useParams();
  const phase = slug ? getPhase(slug) : undefined;
  const lab = labSlug ? labBySlug(labSlug) : undefined;
  const { markRead, isRead } = useProgress();

  useEffect(() => {
    if (lab) markRead(`lab:${lab.slug}`, LAB_XP, `Opened lab ${lab.section}`);
  }, [lab, markRead]);

  if (!phase || !lab) return <Navigate to="/" replace />;

  const siblings = labsForPhase(phase.id);
  const idx = siblings.findIndex((l) => l.slug === lab.slug);
  const next = siblings[idx + 1];
  const nextPhase = PHASES.find((p) => p.id === phase.id + 1);

  return (
    <PageTransition>
      <div className="mb-3 flex items-center gap-2 text-sm text-white/50">
        <Link to="/" className="hover:text-white">Journey</Link>
        <span>/</span>
        <Link to={`/phase/${phase.slug}`} className="hover:text-white">{phase.title}</Link>
        <span>/</span>
        <span className="text-white/80">Lab {lab.section}</span>
      </div>

      <div className="mb-4 flex items-center gap-2">
        <span className="surface-muted rounded-full px-3 py-1 text-sm text-white/60">
          Lab {lab.section} · {idx + 1} of {siblings.length}
          {isRead(`lab:${lab.slug}`) && " · ✓ Read"}
        </span>
        <span className="rounded-full bg-emerald-500/15 px-3 py-1 text-xs text-emerald-300">
          🧪 runnable in JupyterLab
        </span>
      </div>

      <article className="surface relative z-10 rounded-3xl p-6 md:p-8">
        <Markdown>{lab.body}</Markdown>
      </article>

      <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
        <GameButton variant="ghost">
          <Link to={`/phase/${phase.slug}`}>← Back to {phase.title}</Link>
        </GameButton>
        {next ? (
          <GameButton>
            <Link to={`/phase/${phase.slug}/lab/${next.slug}`}>Next lab: {next.title} →</Link>
          </GameButton>
        ) : nextPhase ? (
          <GameButton>
            <Link to={`/phase/${nextPhase.slug}`}>On to Phase {nextPhase.id} →</Link>
          </GameButton>
        ) : (
          <GameButton variant="success">
            <Link to="/trophies">See your trophies 🏆</Link>
          </GameButton>
        )}
      </div>
    </PageTransition>
  );
}
