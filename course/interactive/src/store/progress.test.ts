import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PHASES, TOTAL_XP } from "@/data/curriculum";
import { LABS, LECTURES } from "@/content";
import { resetProgressStore } from "@/test/progress";
import {
  ACHIEVEMENTS,
  LAB_XP,
  LECTURE_XP,
  STORAGE_KEY,
  TOTAL_JOURNEY_ITEMS,
  TOTAL_LABS,
  TOTAL_LECTURES,
  TOTAL_MAX_XP,
  TOTAL_MODULES,
  levelForXp,
  useProgress,
} from "./progress";

describe("progress store", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-06-20T12:00:00Z"));
    resetProgressStore();
  });

  afterEach(() => {
    vi.useRealTimers();
    resetProgressStore();
  });

  it("starts with empty progress", () => {
    const s = useProgress.getState();
    expect(s.xp).toBe(0);
    expect(s.streak).toBe(0);
    expect(s.completed).toEqual({});
    expect(s.read).toEqual({});
    expect(s.unlocked).toEqual([]);
  });

  it("awards XP once per playground completion", () => {
    const { completeModule, isComplete } = useProgress.getState();
    completeModule(0, "tensor-explorer", 40);
    expect(useProgress.getState().xp).toBe(40);
    expect(isComplete(0, "tensor-explorer")).toBe(true);
    expect(useProgress.getState().unlocked).toContain("first-step");

    completeModule(0, "tensor-explorer", 40);
    expect(useProgress.getState().xp).toBe(40);
  });

  it("awards XP once per lecture/lab read and updates streak", () => {
    const { markRead, isRead } = useProgress.getState();
    markRead("lec:0", LECTURE_XP);
    markRead("lab:section-0-1-tensors", LAB_XP);

    expect(useProgress.getState().xp).toBe(LECTURE_XP + LAB_XP);
    expect(isRead("lec:0")).toBe(true);
    expect(isRead("lab:section-0-1-tensors")).toBe(true);
    expect(useProgress.getState().streak).toBe(1);
    expect(useProgress.getState().unlocked).toContain("first-read");

    markRead("lec:0", LECTURE_XP);
    expect(useProgress.getState().xp).toBe(LECTURE_XP + LAB_XP);
  });

  it("increments streak on consecutive active days", () => {
    const { completeModule } = useProgress.getState();
    completeModule(0, "tensor-explorer", 40);
    expect(useProgress.getState().streak).toBe(1);

    vi.setSystemTime(new Date("2026-06-21T12:00:00Z"));
    completeModule(0, "dot-product", 50);
    expect(useProgress.getState().streak).toBe(2);

    vi.setSystemTime(new Date("2026-06-23T12:00:00Z"));
    completeModule(0, "gradient-descent", 60);
    expect(useProgress.getState().streak).toBe(1);
  });

  it("unlocks streak-3 achievement after three consecutive days", () => {
    const { completeModule } = useProgress.getState();
    const days = ["2026-06-20", "2026-06-21", "2026-06-22"];
    const mods = ["tensor-explorer", "dot-product", "gradient-descent"] as const;
    days.forEach((day, i) => {
      vi.setSystemTime(new Date(`${day}T12:00:00Z`));
      completeModule(0, mods[i], 40);
    });
    expect(useProgress.getState().streak).toBe(3);
    expect(useProgress.getState().unlocked).toContain("streak-3");
  });

  it("tracks phase playground progress", () => {
    const phase0 = PHASES[0];
    for (const mod of phase0.modules) {
      useProgress.getState().completeModule(0, mod.id, mod.xp);
    }
    const { done, total } = useProgress.getState().phaseProgress(0);
    expect(done).toBe(total);
    expect(useProgress.getState().unlocked).toContain("phase-0");
  });

  it("tracks full phase journey (lecture + labs + playgrounds)", () => {
    useProgress.getState().markRead("lec:0", LECTURE_XP);
    for (const lab of LABS.filter((l) => l.phaseId === 0)) {
      useProgress.getState().markRead(`lab:${lab.slug}`, LAB_XP);
    }
    for (const mod of PHASES[0].modules) {
      useProgress.getState().completeModule(0, mod.id, mod.xp);
    }

    const j = useProgress.getState().phaseJourney(0);
    expect(j.lectureDone).toBe(true);
    expect(j.labsDone).toBe(j.labsTotal);
    expect(j.playDone).toBe(j.playTotal);
    expect(j.pct).toBe(100);
  });

  it("computes global journey stats", () => {
    useProgress.getState().markRead("lec:0", LECTURE_XP);
    useProgress.getState().completeModule(0, "tensor-explorer", 40);

    const j = useProgress.getState().journeyStats();
    expect(j.lecDone).toBe(1);
    expect(j.playDone).toBe(1);
    expect(j.itemsTotal).toBe(TOTAL_JOURNEY_ITEMS);
    expect(j.itemsDone).toBe(2);
    expect(j.pct).toBe(Math.round((2 / TOTAL_JOURNEY_ITEMS) * 100));
  });

  it("levelForXp follows escalating thresholds", () => {
    expect(levelForXp(0).level).toBe(1);
    expect(levelForXp(99).level).toBe(1);
    expect(levelForXp(100).level).toBe(2);
    expect(levelForXp(500).level).toBeGreaterThanOrEqual(3);
  });

  it("visit3D unlocks explorer achievement once", () => {
    useProgress.getState().visit3D();
    expect(useProgress.getState().unlocked).toContain("explorer");
    expect(useProgress.getState().toast?.title).toContain("Dimension Hopper");

    useProgress.getState().visit3D();
    expect(useProgress.getState().unlocked.filter((id) => id === "explorer")).toHaveLength(1);
  });

  it("toggleSound and clearToast work", () => {
    expect(useProgress.getState().soundOn).toBe(true);
    useProgress.getState().toggleSound();
    expect(useProgress.getState().soundOn).toBe(false);

    useProgress.getState().completeModule(0, "tensor-explorer", 40);
    expect(useProgress.getState().toast).not.toBeNull();
    useProgress.getState().clearToast();
    expect(useProgress.getState().toast).toBeNull();
  });

  it("reset clears progress but keeps sound preference", () => {
    useProgress.getState().completeModule(0, "tensor-explorer", 40);
    useProgress.getState().toggleSound();
    useProgress.getState().reset();

    const s = useProgress.getState();
    expect(s.xp).toBe(0);
    expect(s.completed).toEqual({});
    expect(s.unlocked).toEqual([]);
    expect(s.soundOn).toBe(false);
  });

  it("persists progress to localStorage", async () => {
    useProgress.getState().completeModule(0, "tensor-explorer", 40);
    useProgress.getState().markRead("lec:0", LECTURE_XP);

    await useProgress.persist.rehydrate();
    const raw = localStorage.getItem(STORAGE_KEY);
    expect(raw).toBeTruthy();
    const parsed = JSON.parse(raw!);
    expect(parsed.state.xp).toBe(40 + LECTURE_XP);
    expect(parsed.state.completed["0:tensor-explorer"]).toBeTypeOf("number");
    expect(parsed.state.read["lec:0"]).toBeTypeOf("number");
  });

  it("rehydrates persisted state on load", async () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        state: {
          version: 1,
          xp: 100,
          completed: { "0:tensor-explorer": 1 },
          read: { "lec:0": 2 },
          unlocked: ["first-step"],
          streak: 2,
          lastActive: "2026-06-20",
          soundOn: false,
        },
        version: 1,
      }),
    );

    await useProgress.persist.rehydrate();
    const s = useProgress.getState();
    expect(s.xp).toBe(100);
    expect(s.completed["0:tensor-explorer"]).toBe(1);
    expect(s.read["lec:0"]).toBe(2);
    expect(s.unlocked).toContain("first-step");
    expect(s.streak).toBe(2);
    expect(s.soundOn).toBe(false);
  });

  it("TOTAL_MAX_XP includes lectures, labs, and playgrounds", () => {
    expect(TOTAL_MAX_XP).toBe(
      TOTAL_XP + TOTAL_LECTURES * LECTURE_XP + TOTAL_LABS * LAB_XP,
    );
  });

  it("defines a unique achievement id for each entry", () => {
    const ids = ACHIEVEMENTS.map((a) => a.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});
