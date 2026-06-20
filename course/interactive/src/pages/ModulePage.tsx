import { Suspense, useMemo } from "react";
import { Link, useParams, Navigate, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import PageTransition from "@/components/PageTransition";
import { getPhase } from "@/data/curriculum";
import { useProgress } from "@/store/progress";
import { getModule } from "@/modules/registry";
import { GameButton } from "@/components/ui/primitives";
import { confettiBurst, chime } from "@/lib/celebrate";

export default function ModulePage() {
  const { slug, moduleId } = useParams();
  const navigate = useNavigate();
  const phase = slug ? getPhase(slug) : undefined;
  const mod = phase?.modules.find((m) => m.id === moduleId);

  const { isComplete, completeModule, visit3D, soundOn } = useProgress();

  const Comp = useMemo(
    () => (phase && moduleId ? getModule(phase.id, moduleId) : null),
    [phase, moduleId],
  );

  if (!phase || !mod || !Comp) return <Navigate to="/" replace />;

  const done = isComplete(phase.id, mod.id);
  const idx = phase.modules.findIndex((m) => m.id === mod.id);
  const nextMod = phase.modules[idx + 1];

  const handleDiscover = () => {
    if (mod.threeD) visit3D();
    if (!isComplete(phase.id, mod.id)) {
      completeModule(phase.id, mod.id, mod.xp);
      confettiBurst();
      chime(soundOn);
    }
  };

  return (
    <PageTransition>
      <div className="mb-3 flex items-center gap-2 text-sm text-white/50">
        <Link to="/" className="hover:text-white">
          Journey
        </Link>
        <span>/</span>
        <Link to={`/phase/${phase.slug}`} className="hover:text-white">
          {phase.title}
        </Link>
        <span>/</span>
        <span className="text-white/80">{mod.title}</span>
      </div>

      <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="flex items-center gap-2 text-3xl font-extrabold text-white">
            {mod.threeD && <span title="3D scene">🪐</span>}
            {mod.title}
          </h1>
          <p className="mt-1 max-w-2xl text-white/60">{mod.blurb}</p>
          <div className="mt-2 text-xs text-white/35">Mirrors {mod.mirrors}</div>
        </div>
        <div>
          {done ? (
            <span className="rounded-full bg-emerald-500/20 px-3 py-1.5 text-sm font-semibold text-emerald-300">
              ✓ Completed · +{mod.xp} XP
            </span>
          ) : (
            <span
              className="rounded-full px-3 py-1.5 text-sm font-semibold"
              style={{ background: `${phase.color.from}22`, color: phase.color.ring }}
            >
              Reward: +{mod.xp} XP
            </span>
          )}
        </div>
      </div>

      <motion.div
        key={mod.id}
        initial={{ opacity: 0, scale: 0.98 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
        className="surface relative z-10 rounded-3xl p-4 md:p-6"
      >
        <Suspense
          fallback={
            <div className="grid place-items-center py-24 text-white/40">
              <div className="animate-pulse">Loading playground…</div>
            </div>
          }
        >
          <Comp onDiscover={handleDiscover} completed={done} />
        </Suspense>
      </motion.div>

      <div className="mt-6 flex flex-wrap items-center justify-between gap-3">
        <GameButton variant="ghost" onClick={() => navigate(`/phase/${phase.slug}`)}>
          ← Back to {phase.title}
        </GameButton>

        <div className="flex items-center gap-3">
          {!done && (
            <GameButton variant="success" onClick={handleDiscover}>
              Mark complete · +{mod.xp} XP
            </GameButton>
          )}
          {nextMod ? (
            <GameButton onClick={() => navigate(`/phase/${phase.slug}/play/${nextMod.id}`)}>
              Next: {nextMod.title} →
            </GameButton>
          ) : (
            <GameButton onClick={() => navigate(`/phase/${phase.slug}`)}>Finish phase →</GameButton>
          )}
        </div>
      </div>
    </PageTransition>
  );
}
