import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Insight, Panel, Pill } from "@/components/ui/primitives";
import { range, seeded } from "@/lib/math";
import type { ModuleProps } from "../types";

const RANKS = [
  { rank: 0, name: "Scalar", example: "42", desc: "a single number (e.g. a loss value)" },
  { rank: 1, name: "Vector", example: "[3, 1, 4]", desc: "a list (e.g. one word embedding)" },
  { rank: 2, name: "Matrix", example: "[[1,2],[3,4]]", desc: "a table (e.g. an attention map)" },
  { rank: 3, name: "Cube", example: "[[[…]]]", desc: "a stack of tables (e.g. an RGB image)" },
  { rank: 4, name: "Batch", example: "[[[[…]]]]", desc: "many cubes (e.g. a batch of images)" },
];

export default function TensorExplorer({ onDiscover }: ModuleProps) {
  const [rank, setRank] = useState(0);
  const [seen, setSeen] = useState<Set<number>>(new Set([0]));

  useEffect(() => {
    if (seen.size >= 5) onDiscover();
  }, [seen, onDiscover]);

  const setR = (r: number) => {
    setRank(r);
    setSeen((s) => new Set(s).add(r));
  };

  const meta = RANKS[rank];

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_360px]">
      <Panel title="Tensor shapes" hint="click a rank to morph it">
        <div className="mb-5 flex flex-wrap gap-2">
          {RANKS.map((r) => (
            <button
              key={r.rank}
              onClick={() => setR(r.rank)}
              className={`rounded-xl px-3 py-2 text-sm font-semibold transition ${
                rank === r.rank
                  ? "bg-cyan-500 text-white"
                  : "bg-white/5 text-white/60 hover:bg-white/10"
              }`}
            >
              {r.rank}D · {r.name}
            </button>
          ))}
        </div>

        <div className="grid min-h-[280px] place-items-center rounded-2xl bg-black/30 p-6">
          <TensorViz rank={rank} />
        </div>

        <div className="mt-4 flex items-center gap-3">
          <Pill color="#38bdf8">shape = {shapeFor(rank)}</Pill>
          <Pill color="#a78bfa">{numbersFor(rank)} numbers</Pill>
        </div>
      </Panel>

      <div className="space-y-4">
        <Panel title={`${meta.rank}D — ${meta.name}`}>
          <code className="block rounded-lg bg-black/40 px-3 py-2 font-mono text-sm text-cyan-300">
            {meta.example}
          </code>
          <p className="mt-3 text-sm text-white/70">{meta.desc}</p>
          <p className="mt-3 text-sm text-white/55">
            In PyTorch this is a <code className="text-cyan-300">torch.tensor</code> with{" "}
            <b>{meta.rank}</b> {meta.rank === 1 ? "axis" : "axes"}. Every model input, weight, and
            activation is just one of these.
          </p>
        </Panel>

        <Insight>
          A tensor is <b>only</b> a box of numbers with a shape. "Deep learning" is moving these
          boxes through multiplications and additions on the GPU. Open all five ranks to bank your
          XP — you now speak the language every later phase is built on.
        </Insight>

        <div className="text-center text-xs text-white/40">
          Explored {seen.size}/5 ranks {seen.size >= 5 ? "✓" : ""}
        </div>
      </div>
    </div>
  );
}

function shapeFor(rank: number): string {
  return ["()", "(3,)", "(2, 2)", "(3, 2, 2)", "(2, 3, 2, 2)"][rank];
}
function numbersFor(rank: number): number {
  return [1, 3, 4, 12, 24][rank];
}

function TensorViz({ rank }: { rank: number }) {
  const rnd = seeded(rank * 7 + 1);
  const cell = (key: string, v: number) => (
    <motion.div
      key={key}
      layout
      initial={{ scale: 0.4, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: "spring", stiffness: 220, damping: 18 }}
      className="grid h-9 w-9 place-items-center rounded-md font-mono text-xs"
      style={{ background: `hsl(${200 + v * 80} 80% ${30 + v * 30}%)`, color: "#021" }}
    >
      {(v * 9).toFixed(0)}
    </motion.div>
  );

  if (rank === 0) return cell("s", rnd());
  if (rank === 1)
    return <div className="flex gap-1.5">{range(3).map((i) => cell(`v${i}`, rnd()))}</div>;
  if (rank === 2)
    return (
      <div className="flex flex-col gap-1.5">
        {range(2).map((r) => (
          <div key={r} className="flex gap-1.5">
            {range(2).map((c) => cell(`m${r}-${c}`, rnd()))}
          </div>
        ))}
      </div>
    );
  if (rank === 3)
    return (
      <div className="flex gap-3">
        {range(3).map((d) => (
          <div key={d} className="flex flex-col gap-1.5" style={{ transform: `translateY(${d * 4}px)` }}>
            {range(2).map((r) => (
              <div key={r} className="flex gap-1.5">
                {range(2).map((c) => cell(`c${d}-${r}-${c}`, rnd()))}
              </div>
            ))}
          </div>
        ))}
      </div>
    );
  return (
    <div className="flex gap-5">
      {range(2).map((b) => (
        <div key={b} className="flex gap-2 rounded-xl border border-white/10 p-2">
          {range(3).map((d) => (
            <div key={d} className="flex flex-col gap-1" style={{ transform: `translateY(${d * 3}px)` }}>
              {range(2).map((r) => (
                <div key={r} className="flex gap-1">
                  {range(2).map((c) => cell(`b${b}-${d}-${r}-${c}`, rnd()))}
                </div>
              ))}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
