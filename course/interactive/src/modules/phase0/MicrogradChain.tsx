import { useState } from "react";
import { Insight, Panel, Stat, GameButton } from "@/components/ui/primitives";
import type { ModuleProps } from "../types";

/** Tiny graph: f = (a * b + c)^2 — same as Lab 0.25 worked example. */
const NODES = [
  { id: "a", label: "a = 2", op: "input", grad: -24 },
  { id: "b", label: "b = −3", op: "input", grad: 16 },
  { id: "c", label: "c = 10", op: "input", grad: 8 },
  { id: "ab", label: "a × b = −6", op: "×", grad: 8 },
  { id: "e", label: "d + c = 4", op: "+", grad: 8 },
  { id: "f", label: "e² = 16", op: "²", grad: 1 },
];

const EDGES: [string, string][] = [
  ["a", "ab"],
  ["b", "ab"],
  ["ab", "e"],
  ["c", "e"],
  ["e", "f"],
];

export default function MicrogradChain({ onDiscover }: ModuleProps) {
  const [selected, setSelected] = useState<string | null>(null);
  const [revealed, setRevealed] = useState(false);

  const node = NODES.find((n) => n.id === selected);

  const revealAll = () => {
    setRevealed(true);
    onDiscover();
  };

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_300px]">
      <Panel title="Computation graph" hint="click a node — backward() walks right to left">
        <svg viewBox="0 0 420 200" className="w-full">
          {EDGES.map(([from, to]) => {
            const fi = NODES.findIndex((n) => n.id === from);
            const ti = NODES.findIndex((n) => n.id === to);
            const x1 = 50 + fi * 70;
            const x2 = 50 + ti * 70;
            const y1 = fi < 3 ? 50 + fi * 40 : 130;
            const y2 = ti < 3 ? 50 + ti * 40 : 130;
            return (
              <line
                key={`${from}-${to}`}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke="#ffffff33"
                strokeWidth={2}
                markerEnd="url(#arrow)"
              />
            );
          })}
          <defs>
            <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 Z" fill="#ffffff55" />
            </marker>
          </defs>
          {NODES.map((n, i) => {
            const x = 50 + i * 70;
            const y = i < 3 ? 50 + i * 40 : 130;
            const active = selected === n.id;
            return (
              <g key={n.id} onClick={() => setSelected(n.id)} className="cursor-pointer">
                <rect
                  x={x - 32}
                  y={y - 16}
                  width={64}
                  height={32}
                  rx={8}
                  fill={active ? "#22d3ee44" : "#ffffff11"}
                  stroke={active ? "#22d3ee" : "#ffffff33"}
                  strokeWidth={2}
                />
                <text x={x} y={y + 4} textAnchor="middle" fontSize={9} fill="#fff">
                  {n.label.split("=")[0].trim()}
                </text>
              </g>
            );
          })}
          {revealed && (
            <text x={210} y={185} textAnchor="middle" fontSize={11} fill="#34d399">
              backward() complete — every input has a gradient
            </text>
          )}
        </svg>
      </Panel>

      <div className="space-y-4">
        {node ? (
          <>
            <Stat label="Node" value={node.label} accent="#22d3ee" />
            <Stat label="Operation" value={node.op} />
            {revealed && (
              <Stat label="∂f/∂(this node)" value={String(node.grad)} accent="#fbbf24" />
            )}
          </>
        ) : (
          <div className="text-sm text-white/50">Click any node in the graph.</div>
        )}

        <GameButton onClick={revealAll} variant={revealed ? "ghost" : "primary"}>
          {revealed ? "Gradients shown ✓" : "Run backward() ▶"}
        </GameButton>

        <Insight>
          Each operation stores a local derivative. <span className="text-cyan-300">backward()</span>{" "}
          multiplies them via the chain rule — from output (f, grad=1) back to inputs. PyTorch's{" "}
          <code className="text-white/80">loss.backward()</code> is this at scale. Karpathy's
          micrograd builds the same engine in ~100 lines.
        </Insight>
      </div>
    </div>
  );
}
