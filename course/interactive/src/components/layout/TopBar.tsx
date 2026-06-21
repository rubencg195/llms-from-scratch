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
      <div className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-3">
        <Link to="/" className="group flex items-center gap-2.5">
          <span className="grid h-9 w-9 place-items-center rounded-xl bg-indigo-500 text-lg shadow-lg shadow-indigo-500/40">
            ⚡
          </span>
          <div className="leading-tight">
            <div className="text-sm font-bold tracking-wide text-white">LLMs From Scratch</div>
            <div className="text-[11px] text-white/45">by Ruben Chevez</div>
          </div>
        </Link>

        <nav className="ml-2 hidden items-center gap-1 md:flex">
          <NavLink to="/" active={loc.pathname === "/"}>
            Journey
          </NavLink>
          <NavLink to="/trophies" active={loc.pathname.startsWith("/trophies")}>
            Trophies
          </NavLink>
        </nav>

        <div className="ml-auto flex items-center gap-3">
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

          <div className="flex items-center gap-2 rounded-full surface-muted px-3 py-1.5">
            <span className="grid h-6 w-6 place-items-center rounded-full bg-orange-500 text-[11px] font-bold text-black">
              {level}
            </span>
            <div className="w-28">
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
            className="surface-muted grid h-9 w-9 place-items-center rounded-full text-base hover:brightness-110"
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
