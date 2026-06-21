import { test, expect } from "@playwright/test";
import {
  clearProgress,
  go,
  openPhase,
  finishLecture,
  backToPhaseFromLecture,
  markPlaygroundComplete,
  readProgress,
  clickNav,
  PHASES,
} from "./helpers";

test.beforeEach(async ({ page }) => {
  await clearProgress(page);
});

test.describe("End-to-end journey flow", () => {
  test("full phase 0 path: home → phase → lecture → play → lab → trophies nav", async ({ page }) => {
    const phase = PHASES[0];

    await go(page, "/");
    await page.getByRole("link", { name: /Start the journey/i }).click();
    await expect(page.getByRole("heading", { name: phase.title })).toBeVisible();

    await page.getByRole("link", { name: /Open →/ }).click();
    await finishLecture(page);
    await backToPhaseFromLecture(page);
    await expect(page.getByText("✓ Read")).toBeVisible();

    const mod = phase.modules[0];
    await page.getByRole("link", { name: new RegExp(mod.title) }).click();
    await markPlaygroundComplete(page);
    await page.getByRole("button", { name: new RegExp(`Back to ${phase.title}`) }).click();
    await expect(page.getByText("✓ Done")).toBeVisible();

    await page.getByRole("link", { name: /Read →/ }).first().click();
    await expect(page.getByText(/✓ Read/)).toBeVisible();
    await page.getByRole("link", { name: new RegExp(`Back to ${phase.title}`) }).click();

    await clickNav(page, "Trophies");
    await expect(page.getByRole("heading", { name: /Trophy Room/i })).toBeVisible();

    const progress = await readProgress(page);
    expect(progress?.state.unlocked).toContain("first-step");
    expect(progress?.state.unlocked).toContain("first-read");
    expect(progress?.state.xp).toBeGreaterThan(0);
  });

  test("Continue button appears after progress started", async ({ page }) => {
    const phase = PHASES[0];
    await go(page, `/phase/${phase.slug}/play/${phase.modules[0].id}`);
    await markPlaygroundComplete(page);

    await go(page, "/");
    await expect(page.getByRole("link", { name: /Continue →/ })).toBeVisible();
  });

  test("transition home → phase → next phase via footer", async ({ page }) => {
    const p0 = PHASES[0];
    const p1 = PHASES[1];

    await openPhase(page, p0.slug);
    await page.getByRole("link", { name: new RegExp(`Phase ${p1.id}: ${p1.title}`) }).click();
    await expect(page.getByRole("heading", { name: p1.title })).toBeVisible();

    await clickNav(page, "Journey");
    await expect(page.getByRole("heading", { name: "The Journey" })).toBeVisible();
  });

  test("completing all phase 0 playgrounds unlocks Math Survivor", async ({ page }) => {
    const phase = PHASES[0];

    for (const mod of phase.modules) {
      await go(page, `/phase/${phase.slug}/play/${mod.id}`);
      await markPlaygroundComplete(page);
    }

    await go(page, "/trophies");
    const achievement = page.locator(".surface").filter({ hasText: "Math Survivor" });
    await expect(achievement.getByText("🧮")).toBeVisible();
  });
});
