import { test, expect } from "@playwright/test";
import { clearProgress, openPhase, readProgress, LAB_XP, PHASES } from "./helpers";

test.beforeEach(async ({ page }) => {
  await clearProgress(page);
});

test.describe("Lab reader", () => {
  test("opening first lab marks read and awards XP", async ({ page }) => {
    const phase = PHASES[0];
    await openPhase(page, phase.slug);
    await page.getByRole("link", { name: /Read →/ }).first().click();

    await expect(page.getByText(/runnable in JupyterLab/i)).toBeVisible();
    await expect(page.getByText(/✓ Read/)).toBeVisible();

    const progress = await readProgress(page);
    expect(progress?.state.xp).toBeGreaterThanOrEqual(LAB_XP);
    expect(Object.keys(progress?.state.read ?? {}).length).toBeGreaterThan(0);
  });

  test("back to phase link works", async ({ page }) => {
    const phase = PHASES[0];
    await openPhase(page, phase.slug);
    await page.getByRole("link", { name: /Read →/ }).first().click();
    await page.getByRole("link", { name: new RegExp(`Back to ${phase.title}`) }).click();
    await expect(page.getByRole("heading", { name: phase.title })).toBeVisible();
  });

  test("next lab link navigates between labs in same phase", async ({ page }) => {
    const phase = PHASES[0];
    await openPhase(page, phase.slug);
    const labLinks = page.getByRole("link", { name: /Read →/ });
    const count = await labLinks.count();
    test.skip(count < 2, "Phase needs at least 2 labs");

    await labLinks.first().click();
    const nextLab = page.getByRole("link", { name: /Next lab:/i });
    if (await nextLab.isVisible()) {
      await nextLab.click();
      await expect(page.getByText(/Lab \d/)).toBeVisible();
      await expect(page.getByText(/✓ Read/)).toBeVisible();
    }
  });

  test("phase page shows checkmark after lab opened", async ({ page }) => {
    const phase = PHASES[0];
    await openPhase(page, phase.slug);
    await page.getByRole("link", { name: /Read →/ }).first().click();
    await expect(page.getByText(/✓ Read/)).toBeVisible();
    await openPhase(page, phase.slug);
    await expect(page.getByText(/1\/\d+ labs/)).toBeVisible();
  });

  test("lab content renders markdown", async ({ page }) => {
    const phase = PHASES[0];
    await openPhase(page, phase.slug);
    await page.getByRole("link", { name: /Read →/ }).first().click();
    await expect(page.locator("article")).toBeVisible();
    await expect(page.locator("article").locator("h1, h2, h3, p").first()).toBeVisible();
  });
});
