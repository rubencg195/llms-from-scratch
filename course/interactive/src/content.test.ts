import { describe, expect, it } from "vitest";
import {
  LABS,
  LECTURES,
  labBySlug,
  labsForPhase,
  lectureForPhase,
} from "./content";

describe("course content loader", () => {
  it("loads all 9 lecture decks", () => {
    expect(LECTURES).toHaveLength(9);
    for (const lec of LECTURES) {
      expect(lec.title.length).toBeGreaterThan(0);
      expect(lec.slides.length).toBeGreaterThan(0);
      expect(lec.slides[0].body.length).toBeGreaterThan(0);
    }
  });

  it("loads all 39 lab sections", () => {
    expect(LABS).toHaveLength(39);
    for (const lab of LABS) {
      expect(lab.slug).toMatch(/^section-\d+-\d+-/);
      expect(lab.section).toMatch(/^\d+-\d+$/);
      expect(lab.title.length).toBeGreaterThan(0);
      expect(lab.body.length).toBeGreaterThan(100);
    }
  });

  it("maps lectures and labs to phases 0–8", () => {
    for (let phaseId = 0; phaseId <= 8; phaseId++) {
      expect(lectureForPhase(phaseId)).toBeDefined();
      expect(labsForPhase(phaseId).length).toBeGreaterThan(0);
    }
  });

  it("labBySlug finds labs and returns undefined for unknown slugs", () => {
    const first = LABS[0];
    expect(labBySlug(first.slug)?.title).toBe(first.title);
    expect(labBySlug("section-99-99-nope")).toBeUndefined();
  });

  it("parses speaker notes out of lecture slides", () => {
    const withNotes = LECTURES.flatMap((l) => l.slides).some((s) => s.notes.length > 0);
    expect(withNotes).toBe(true);
  });
});
