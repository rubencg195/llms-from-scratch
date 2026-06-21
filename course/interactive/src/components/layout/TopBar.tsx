import { Link, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import { useProgress, levelForXp, TOTAL_JOURNEY_ITEMS } from "@/store/progress";
import { TOTAL_MODULES } from "@/data/curriculum";

export default function TopBar() {
  const { xp, streak, journeyStats, soundOn, toggleSound } = useProgress();
  const { level, into, span } = levelForXp(xp);
  const pct = Math.round((into / span) * 100);
  const journey = journeyStats();
  const loc = useLocation();

  return (
    <header className="sticky top-0 z-50 border-b border-purple-500/20 bg-black shadow-[0_12px_48px_rgba(88,28,135,0.22)]">
      <div className="mx-auto flex max-w-7xl items-center gap-2 px-3 py-2.5 sm:gap-4 sm:px-4 sm:py-3">
        <Link to="/" className="group flex min-w-0 items-center gap-2 sm:gap-2.5">
          <span className="grid h-8 w-8 shrink-0 place-items-center rounded-xl bg-indigo-500 text-base shadow-lg shadow-indigo-500/40 sm:h-9 sm:w-9 sm:text-lg">
            ⚡
          </span>
          <div className="min-w-0 leading-tight">
            <div className="text-sm font-bold tracking-wide text-white">LLMs From Scratch</div>
            <div className="hidden text-[11px] text-white/45 sm:block">by Ruben Chevez</div>
          </div>
        </Link>

        <nav className="ml-1 hidden items-center gap-1 md:flex">
          <NavLink to="/" active={loc.pathname === "/"}>
            Journey
          </NavLink>
          <NavLink to="/trophies" active={loc.pathname.startsWith("/trophies")}>
            Trophies
          </NavLink>
        </nav>

        <div className="ml-auto flex shrink-0 items-center gap-2 sm:gap-3">
          <div className="hidden items-center gap-1.5 rounded-full bg-orange-500/15 px-3 py-1.5 text-sm font-semibold text-orange-300 sm:flex">
            <span>🔥</span>
            <span>{streak}d</span>
          </div>

          <div className="hidden text-right sm:block">
            <div className="text-[11px] text-white/45">
              {journey.itemsDone}/{TOTAL_JOURNEY_ITEMS} journey
            </div>
            <div className="text-[10px] text-white/30">
              {journey.playDone}/{TOTAL_MODULES} play · {journey.lecDone} lec · {journey.labDone} labs
            </div>
          </div>

          <div className="flex items-center gap-1.5 rounded-full surface-muted px-2 py-1.5 sm:gap-2 sm:px-3">
            <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full bg-orange-500 text-[11px] font-bold text-black">
              {level}
            </span>
            <div className="w-16 sm:w-28">
              <div className="h-2 overflow-hidden rounded-full bg-white/10">
                <motion.div
                  className="h-full rounded-full bg-orange-500"
                  animate={{ width: `${pct}%` }}
                  transition={{ type: "spring", stiffness: 120, damping: 18 }}
                />
              </div>
              <div className="mt-0.5 text-[10px] text-white/40">{xp} XP</div>
            </div>
          </div>

          <button
            onClick={toggleSound}
            title={soundOn ? "Sound on" : "Sound off"}
            aria-label={soundOn ? "Sound on" : "Sound off"}
            className="surface-muted grid h-9 w-9 shrink-0 place-items-center rounded-full text-base hover:brightness-110"
          >
            {soundOn ? "🔊" : "🔇"}
          </button>
        </div>
      </div>
    </header>
  );
}

function NavLink({ to, active, children }: { to: string; active: boolean; children: string }) {
  return (
    <Link
      to={to}
      className={`rounded-lg px-3 py-1.5 text-sm font-medium transition ${
        active ? "surface-muted text-white" : "text-white/55 hover:text-white"
      }`}
    >
      {children}
    </Link>
  );
}
