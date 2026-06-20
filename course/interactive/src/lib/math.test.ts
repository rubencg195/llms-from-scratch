import { describe, expect, it } from "vitest";
import { clamp, cosine, dot, fmtBytes, heat, lerp, norm, range, seeded, softmax } from "./math";

describe("math utilities", () => {
  it("clamp keeps values in range", () => {
    expect(clamp(5, 0, 10)).toBe(5);
    expect(clamp(-1, 0, 10)).toBe(0);
    expect(clamp(99, 0, 10)).toBe(10);
  });

  it("lerp interpolates", () => {
    expect(lerp(0, 10, 0.5)).toBe(5);
  });

  it("range builds index arrays", () => {
    expect(range(4)).toEqual([0, 1, 2, 3]);
  });

  it("softmax sums to 1", () => {
    const s = softmax([1, 2, 3]);
    expect(s.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
    expect(s[2]).toBeGreaterThan(s[0]);
  });

  it("dot and norm work for vectors", () => {
    expect(dot([1, 2], [3, 4])).toBe(11);
    expect(norm([3, 4])).toBe(5);
  });

  it("cosine similarity is 1 for identical vectors", () => {
    expect(cosine([1, 2, 3], [1, 2, 3])).toBeCloseTo(1, 5);
  });

  it("seeded RNG is deterministic", () => {
    const a = seeded(42);
    const b = seeded(42);
    expect(a()).toBe(b());
    expect(a()).toBe(b());
  });

  it("heat returns hsl color strings", () => {
    expect(heat(0)).toMatch(/^hsl\(/);
    expect(heat(1)).toMatch(/^hsl\(/);
  });

  it("fmtBytes formats storage sizes", () => {
    expect(fmtBytes(512)).toBe("512 B");
    expect(fmtBytes(2048)).toBe("2.0 KB");
    expect(fmtBytes(1024 ** 2)).toBe("1.0 MB");
    expect(fmtBytes(10 * 1024 ** 3)).toBe("10.00 GB");
  });
});
