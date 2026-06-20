import { useMemo } from "react";
import { softmax } from "@/lib/math";

/**
 * Toy but believable attention: hand-tuned affinities between words plus a
 * causal mask, turned into a row-stochastic weight matrix via softmax. Good
 * enough to show the *shape* of attention (e.g. "it" -> "cat") without a model.
 */
const AFFINITY: Record<string, string[]> = {
  it: ["cat"],
  tired: ["cat", "it"],
  sat: ["cat", "mat"],
  mat: ["sat", "on"],
  because: ["sat", "tired"],
};

export function useMemoizedAttention(words: string[], temp: number) {
  const weights = useMemo(() => {
    return words.map((qw, qi) => {
      const scores = words.map((kw, ki) => {
        if (ki > qi) return -Infinity; // causal mask
        let s = 0.2;
        if (ki === qi) s += 0.6; // self
        if (AFFINITY[qw.toLowerCase()]?.includes(kw.toLowerCase())) s += 1.6;
        if (kw.toLowerCase() === "the") s -= 0.3; // stop-word-ish
        return s / temp;
      });
      return softmax(scores.map((s) => (s === -Infinity ? -1e9 : s)));
    });
  }, [words, temp]);

  return { weights };
}
