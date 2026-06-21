import { describe, expect, it } from "vitest";
import {
  ALL_MODULES,
  PHASES,
  TOTAL_MODULES,
  TOTAL_XP,
  getPhase,
  moduleKey,
} from "./curriculum";

describe("curriculum", () => {
  it("defines 9 phases (0–8)", () => {
    expect(PHASES).toHaveLength(9);
    expect(PHASES.map((p) => p.id)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it("has unique slugs and module ids per phase", () => {
    const slugs = PHASES.map((p) => p.slug);
    expect(new Set(slugs).size).toBe(slugs.length);

    for (const phase of PHASES) {
      const ids = phase.modules.map((m) => m.id);
      expect(new Set(ids).size).toBe(ids.length);
      for (const mod of phase.modules) {
        expect(mod.xp).toBeGreaterThan(0);
        expect(mod.title.length).toBeGreaterThan(0);
        expect(mod.mirrors.length).toBeGreaterThan(0);
      }
    }
  });

  it("exports all interactive modules with positive total XP", () => {
    expect(TOTAL_MODULES).toBeGreaterThanOrEqual(23);
    expect(ALL_MODULES).toHaveLength(TOTAL_MODULES);
    expect(TOTAL_XP).toBeGreaterThan(1000);
  });

  it("getPhase resolves by slug", () => {
    expect(getPhase("dense-core")?.id).toBe(1);
    expect(getPhase("missing-phase")).toBeUndefined();
  });

  it("moduleKey encodes phase and module id", () => {
    expect(moduleKey(1, "tokenizer")).toBe("1:tokenizer");
  });
});
