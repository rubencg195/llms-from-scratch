import { describe, expect, it } from "vitest";
import { ALL_MODULES } from "@/data/curriculum";
import { REGISTRY, getModule } from "./registry";

describe("module registry", () => {
  it("registers every curriculum module (no ComingSoon fallbacks)", () => {
    expect(Object.keys(REGISTRY)).toHaveLength(ALL_MODULES.length);
    for (const mod of ALL_MODULES) {
      const key = `${mod.phaseId}:${mod.id}`;
      expect(REGISTRY[key], `missing registry entry for ${key}`).toBeDefined();
    }
    expect(Object.keys(REGISTRY)).toHaveLength(ALL_MODULES.length);
  });

  it("getModule returns registry entry or fallback", () => {
    expect(getModule(1, "tokenizer")).toBe(REGISTRY["1:tokenizer"]);
    expect(getModule(99, "nope")).toBeDefined();
  });
});
