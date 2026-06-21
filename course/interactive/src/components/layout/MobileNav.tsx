import { Link, useLocation } from "react-router-dom";

export default function MobileNav() {
  const loc = useLocation();
  const journeyActive =
    loc.pathname === "/" || loc.pathname.startsWith("/phase");
  const trophiesActive = loc.pathname.startsWith("/trophies");

  return (
    <nav
      aria-label="Mobile navigation"
      className="fixed inset-x-0 bottom-0 z-50 border-t border-purple-500/20 bg-black/95 backdrop-blur-md md:hidden"
      style={{ paddingBottom: "max(0.5rem, env(safe-area-inset-bottom))" }}
    >
      <div className="mx-auto grid max-w-lg grid-cols-2">
        <Tab to="/" active={journeyActive} icon="🗺️" label="Journey" />
        <Tab to="/trophies" active={trophiesActive} icon="🏆" label="Trophies" />
      </div>
    </nav>
  );
}

function Tab({
  to,
  active,
  icon,
  label,
}: {
  to: string;
  active: boolean;
  icon: string;
  label: string;
}) {
  return (
    <Link
      to={to}
      className={`flex min-h-14 flex-col items-center justify-center gap-0.5 px-4 py-2 text-xs font-semibold transition ${
        active ? "text-indigo-300" : "text-white/50 active:text-white"
      }`}
    >
      <span className="text-lg leading-none" aria-hidden>
        {icon}
      </span>
      {label}
    </Link>
  );
}
