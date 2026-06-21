import { useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Html, Line } from "@react-three/drei";
import { Insight, Panel, Pill } from "@/components/ui/primitives";
import type { ModuleProps } from "../types";

// Hand-placed 3D "embeddings" grouped into semantic clusters so similar words
// sit near each other — the whole point of an embedding space.
interface Word {
  w: string;
  pos: [number, number, number];
  group: string;
}

const GROUPS: Record<string, string> = {
  royalty: "#fbbf24",
  animals: "#34d399",
  numbers: "#22d3ee",
  emotions: "#f472b6",
};

const WORDS: Word[] = [
  { w: "king", pos: [2.2, 1.6, 0.4], group: "royalty" },
  { w: "queen", pos: [2.6, 1.9, 0.1], group: "royalty" },
  { w: "prince", pos: [2.0, 1.2, 0.7], group: "royalty" },
  { w: "throne", pos: [2.8, 1.4, 0.5], group: "royalty" },
  { w: "cat", pos: [-2.2, 0.6, 1.4], group: "animals" },
  { w: "dog", pos: [-2.5, 0.9, 1.1], group: "animals" },
  { w: "puppy", pos: [-2.0, 1.2, 1.5], group: "animals" },
  { w: "kitten", pos: [-1.8, 0.4, 1.7], group: "animals" },
  { w: "one", pos: [0.4, -2.0, -1.6], group: "numbers" },
  { w: "two", pos: [0.7, -2.3, -1.3], group: "numbers" },
  { w: "three", pos: [0.2, -1.7, -1.9], group: "numbers" },
  { w: "ten", pos: [1.0, -2.1, -1.1], group: "numbers" },
  { w: "happy", pos: [-0.6, 2.2, -1.8], group: "emotions" },
  { w: "joy", pos: [-0.9, 2.5, -1.5], group: "emotions" },
  { w: "sad", pos: [-1.3, 1.8, -2.0], group: "emotions" },
  { w: "angry", pos: [-0.3, 1.6, -2.2], group: "emotions" },
];

function dist(a: [number, number, number], b: [number, number, number]) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

export default function Embeddings3D({ onDiscover }: ModuleProps) {
  const [selected, setSelected] = useState<string | null>("king");

  const sel = WORDS.find((w) => w.w === selected) ?? null;
  const neighbors = sel
    ? WORDS.filter((w) => w.w !== sel.w)
        .map((w) => ({ ...w, d: dist(w.pos, sel.pos) }))
        .sort((a, b) => a.d - b.d)
        .slice(0, 3)
    : [];

  const pick = (w: string) => {
    setSelected(w);
    onDiscover();
  };

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_300px]">
      <Panel title="Embedding space (drag to orbit)" hint="click any word">
        <div className="h-[280px] overflow-hidden rounded-2xl bg-black/40 sm:h-[360px] lg:h-[420px]">
          <Canvas camera={{ position: [0, 0, 8], fov: 55 }}>
            <ambientLight intensity={0.8} />
            <pointLight position={[10, 10, 10]} />
            <OrbitControls enablePan={false} minDistance={4} maxDistance={14} />
            <axesHelper args={[3]} />
            {sel &&
              neighbors.map((n) => (
                <Line
                  key={"l" + n.w}
                  points={[sel.pos, n.pos]}
                  color="#ffffff"
                  lineWidth={1}
                  dashed
                  transparent
                  opacity={0.4}
                />
              ))}
            {WORDS.map((word) => (
              <group key={word.w} position={word.pos}>
                <mesh onClick={() => pick(word.w)}>
                  <sphereGeometry args={[selected === word.w ? 0.28 : 0.18, 24, 24]} />
                  <meshStandardMaterial
                    color={GROUPS[word.group]}
                    emissive={GROUPS[word.group]}
                    emissiveIntensity={selected === word.w ? 0.9 : 0.3}
                  />
                </mesh>
                <Html distanceFactor={10} center>
                  <div
                    onClick={() => pick(word.w)}
                    className="cursor-pointer select-none whitespace-nowrap text-[11px] font-semibold"
                    style={{ color: GROUPS[word.group], textShadow: "0 0 6px #000" }}
                  >
                    {word.w}
                  </div>
                </Html>
              </group>
            ))}
          </Canvas>
        </div>
      </Panel>

      <div className="space-y-4">
        <Panel title="Nearest neighbors">
          {sel ? (
            <>
              <div className="mb-3 flex items-center gap-2">
                <span
                  className="h-3 w-3 rounded-full"
                  style={{ background: GROUPS[sel.group] }}
                />
                <span className="font-mono text-lg text-white">{sel.w}</span>
              </div>
              <ul className="space-y-2">
                {neighbors.map((n) => (
                  <li key={n.w} className="flex items-center justify-between text-sm">
                    <span className="text-white/75">{n.w}</span>
                    <span className="font-mono text-white/45">dist {n.d.toFixed(2)}</span>
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <p className="text-sm text-white/50">Click a sphere to see its closest words.</p>
          )}
        </Panel>

        <div className="flex flex-wrap gap-2">
          {Object.entries(GROUPS).map(([g, c]) => (
            <Pill key={g} color={c}>
              {g}
            </Pill>
          ))}
        </div>

        <Insight>
          Each word is a point in a high-dimensional space (we show 3D; real models use hundreds of
          dimensions). Training pulls words with similar meaning <b>close together</b> — that's why
          <i> king/queen</i> cluster apart from <i>cat/dog</i>. Distance ≈ meaning. The famous
          "king − man + woman ≈ queen" trick works because these are just vectors you can add and
          subtract.
        </Insight>
      </div>
    </div>
  );
}
