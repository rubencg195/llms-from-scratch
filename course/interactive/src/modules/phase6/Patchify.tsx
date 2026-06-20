import { useState } from "react";
import { motion } from "framer-motion";
import { Insight, Panel, Slider, Stat } from "@/components/ui/primitives";
import { range } from "@/lib/math";
import type { ModuleProps } from "../types";

export default function Patchify({ onDiscover }: ModuleProps) {
  const [grid, setGrid] = useState(4);
  const [flattened, setFlattened] = useState(false);

  const colorAt = (r: number, c: number) => {
    // a smooth gradient "image" so patches look like a real picture
    const h = 200 + (r / grid) * 120;
    const l = 30 + (c / grid) * 40;
    return `hsl(${h} 70% ${l}%)`;
  };

  const patches = range(grid * grid);

  const doFlatten = () => {
    setFlattened(true);
    onDiscover();
  };

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_320px]">
      <Panel title="An image becomes a grid of patches" hint={`${grid}×${grid} patches`}>
        <div
          className="mx-auto grid aspect-square w-full max-w-[320px] gap-1 rounded-xl bg-black/30 p-1"
          style={{ gridTemplateColumns: `repeat(${grid}, 1fr)` }}
        >
          {patches.map((i) => {
            const r = Math.floor(i / grid);
            const c = i % grid;
            return (
              <motion.div
                key={i}
                layout
                className="rounded-md"
                style={{ background: colorAt(r, c) }}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              />
            );
          })}
        </div>

        {flattened && (
          <div className="mt-4">
            <div className="mb-1 text-xs text-white/40">flattened into a token sequence →</div>
            <div className="flex flex-wrap gap-1">
              {patches.map((i) => {
                const r = Math.floor(i / grid);
                const c = i % grid;
                return (
                  <motion.div
                    key={i}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: i * 0.015 }}
                    className="h-6 w-6 rounded"
                    style={{ background: colorAt(r, c) }}
                    title={`patch ${i}`}
                  />
                );
              })}
            </div>
          </div>
        )}
      </Panel>

      <div className="space-y-4">
        <Slider label="Patches per side" value={grid} min={2} max={8} step={1} onChange={setGrid} accent="#2dd4bf" />
        <div className="grid grid-cols-2 gap-3">
          <Stat label="Patches" value={grid * grid} accent="#2dd4bf" />
          <Stat label="= tokens" value={grid * grid} accent="#34d399" />
        </div>

        <button
          onClick={doFlatten}
          className="w-full rounded-xl bg-teal-500 px-4 py-2.5 text-sm font-semibold text-white transition hover:brightness-110 active:scale-95"
        >
          Flatten into tokens ↓
        </button>

        <Insight>
          Encoder-free multimodal models skip the heavy vision encoder. They just <b>cut the image
          into square patches</b>, flatten each patch into a vector, and feed them into the same
          transformer as text tokens. To the model, a picture is simply{" "}
          <b>{grid * grid} more tokens</b> in the stream. Flatten the grid to bank your XP.
        </Insight>
      </div>
    </div>
  );
}
