import { GameButton } from "@/components/ui/primitives";
import type { ModuleProps } from "./types";

export default function ComingSoon({ onDiscover }: ModuleProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-16 text-center">
      <div className="text-6xl animate-[float_6s_ease-in-out_infinite]">🛠️</div>
      <h3 className="text-xl font-semibold text-white/90">This playground is under construction</h3>
      <p className="max-w-md text-white/60">
        The interactive activity for this concept is being built. In the meantime, the matching
        lecture and lab cover everything you need.
      </p>
      <GameButton variant="ghost" onClick={onDiscover}>
        Mark as explored
      </GameButton>
    </div>
  );
}
