import { test, expect } from "@playwright/test";
import { clearProgress, openPhase, PHASES, escapeRegex } from "./helpers";

test.beforeEach(async ({ page }) => {
  await clearProgress(page);
});

test.describe("Phase page", () => {
  test("shows lecture, playgrounds, and labs sections", async ({ page }) => {
    const phase = PHASES[0];
    await openPhase(page, phase.slug);

    await expect(page.getByRole("heading", { name: "Lecture" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Interactive playgrounds" })).toBeVisible();
    await expect(page.getByRole("heading", { name: /Labs \(\d+\)/ })).toBeVisible();
    await expect(page.getByText("Suggested flow:")).toBeVisible();
  });

  test("lecture card navigates to slide reader", async ({ page }) => {
    const phase = PHASES[0];
    await openPhase(page, phase.slug);
    await page.getByRole("link", { name: /Open →/ }).click();
    await expect(page).toHaveURL(/#\/phase\/bridging-the-gap\/lecture$/);
    await expect(page.getByText(/Slide 1 \//)).toBeVisible();
  });

  test("playground cards show XP badges and navigate", async ({ page }) => {
    const phase = PHASES[0];
    const mod = phase.modules[0];
    await openPhase(page, phase.slug);

    await page.getByRole("link", { name: new RegExp(mod.title) }).click();
    await expect(page).toHaveURL(new RegExp(`#/phase/${phase.slug}/play/${mod.id}$`));
    await expect(page.getByRole("heading", { name: mod.title })).toBeVisible();
    await expect(page.getByText(`Reward: +${mod.xp} XP`)).toBeVisible();
  });

  test("prev/next phase navigation", async ({ page }) => {
    const p0 = PHASES[0];
    const p1 = PHASES[1];
    await openPhase(page, p0.slug);

    await page.getByRole("link", { name: new RegExp(`Phase ${p1.id}: ${p1.title}`) }).click();
    await expect(page.getByRole("heading", { name: p1.title })).toBeVisible();

    await page.getByRole("link", { name: new RegExp(`Phase ${p0.id}: ${p0.title}`) }).click();
    await expect(page.getByRole("heading", { name: p0.title })).toBeVisible();
  });

  test("breadcrumb Journey link returns home", async ({ page }) => {
    await openPhase(page, PHASES[0].slug);
    await page.getByRole("link", { name: "Journey" }).first().click();
    await expect(page).toHaveURL(/#\/$/);
  });

  for (const phase of PHASES) {
    test(`phase ${phase.id} page loads all ${phase.modules.length} playgrounds`, async ({ page }) => {
      await openPhase(page, phase.slug);
      for (const mod of phase.modules) {
        await expect(
          page.getByRole("link", { name: new RegExp(escapeRegex(mod.title)) }),
        ).toBeVisible();
      }
    });
  }
});
