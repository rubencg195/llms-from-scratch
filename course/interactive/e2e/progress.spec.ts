import { test, expect } from "@playwright/test";
import {
  clearProgress,
  go,
  openPhase,
  finishLecture,
  markPlaygroundComplete,
  readProgress,
  xpFromPage,
  journeyCountFromTopBar,
  LECTURE_XP,
  LAB_XP,
  PHASES,
} from "./helpers";

test.beforeEach(async ({ page }) => {
  await clearProgress(page);
});

test.describe("XP & journey tracking", () => {
  test("completing a playground awards XP and updates journey", async ({ page }) => {
    const phase = PHASES[0];
    const mod = phase.modules[0];

    await go(page, `/phase/${phase.slug}/play/${mod.id}`);
    await expect(page.getByRole("heading", { name: mod.title })).toBeVisible();

    const xpBefore = await xpFromPage(page);
    await markPlaygroundComplete(page);
    await expect(page.getByText(/Completed · \+/)).toBeVisible();

    const xpAfter = await xpFromPage(page);
    expect(xpAfter).toBe(xpBefore + mod.xp);

    const progress = await readProgress(page);
    expect(progress?.state.completed[`${phase.id}:${mod.id}`]).toBeTruthy();
    expect(progress?.state.xp).toBe(mod.xp);
    expect(progress?.state.unlocked).toContain("first-step");
  });

  test("marking complete twice does not double XP", async ({ page }) => {
    const phase = PHASES[0];
    const mod = phase.modules[0];

    await go(page, `/phase/${phase.slug}/play/${mod.id}`);
    await markPlaygroundComplete(page);
    const xpOnce = await xpFromPage(page);

    await page.reload();
    await expect(page.getByText(/Completed · \+/)).toBeVisible();
    expect(await xpFromPage(page)).toBe(xpOnce);

    const progress = await readProgress(page);
    expect(progress?.state.xp).toBe(mod.xp);
  });

  test("finishing lecture awards XP and marks read", async ({ page }) => {
    const phase = PHASES[0];
    await go(page, `/phase/${phase.slug}/lecture`);
    await finishLecture(page);

    await expect(page.getByRole("button", { name: /Back to phase/i })).toBeVisible();
    await expect(page.getByText(/✓ Read/)).toBeVisible();

    const progress = await readProgress(page);
    expect(progress?.state.read[`lec:${phase.id}`]).toBeTruthy();
    expect(progress?.state.xp).toBeGreaterThanOrEqual(LECTURE_XP);
    expect(progress?.state.unlocked).toContain("first-read");
  });

  test("opening a lab awards XP and updates journey counter", async ({ page }) => {
    const phase = PHASES[0];
    await openPhase(page, phase.slug);

    const journeyBefore = await journeyCountFromTopBar(page);
    await page.getByRole("link", { name: /Read →/ }).first().click();
    await expect(page.getByText(/✓ Read/)).toBeVisible();

    const progress = await readProgress(page);
    expect(progress?.state.xp).toBeGreaterThanOrEqual(LAB_XP);
    expect(Object.keys(progress?.state.read ?? {}).some((k) => k.startsWith("lab:"))).toBe(true);

    await openPhase(page, phase.slug);
    const journeyAfter = await journeyCountFromTopBar(page);
    expect(journeyAfter).not.toBe(journeyBefore);
  });

  test("phase progress bar updates after lecture + playground", async ({ page }) => {
    const phase = PHASES[0];
    await openPhase(page, phase.slug);
    await expect(page.getByText(/Phase progress/)).toBeVisible();
    await expect(page.getByText(/0\/\d+ · 0%/)).toBeVisible();

    await go(page, `/phase/${phase.slug}/lecture`);
    await finishLecture(page);

    await openPhase(page, phase.slug);
    await expect(page.getByText(/Lecture ✓/)).toBeVisible();

    const mod = phase.modules[0];
    await go(page, `/phase/${phase.slug}/play/${mod.id}`);
    await markPlaygroundComplete(page);

    await openPhase(page, phase.slug);
    await expect(page.getByText(/Lecture ✓/)).toBeVisible();
    await expect(page.getByText(/1\/\d+ playgrounds/)).toBeVisible();
    await expect(page.getByText(/✓ Done/).first()).toBeVisible();
  });

  test("home phase card reflects completed playground", async ({ page }) => {
    const phase = PHASES[0];
    const mod = phase.modules[0];

    await go(page, `/phase/${phase.slug}/play/${mod.id}`);
    await markPlaygroundComplete(page);

    await go(page, "/");
    const card = page.getByRole("link", { name: new RegExp(phase.title) });
    await expect(card.getByText(/\d+\/\d+ items/)).toBeVisible();
    await expect(card.getByText(/1\/4 play/)).toBeVisible();
  });
});
