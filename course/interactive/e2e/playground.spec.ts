import { test, expect } from "@playwright/test";
import {
  clearProgress,
  go,
  openPhase,
  markPlaygroundComplete,
  readProgress,
  clickBreadcrumbJourney,
  PHASES,
  escapeRegex,
} from "./helpers";

test.beforeEach(async ({ page }) => {
  await clearProgress(page);
});

test.describe("Playgrounds", () => {
  test("Mark complete awards XP and shows completed badge", async ({ page }) => {
    const phase = PHASES[0];
    const mod = phase.modules[0];

    await go(page, `/phase/${phase.slug}/play/${mod.id}`);
    await expect(page.getByRole("button", { name: /Mark complete/i })).toBeVisible();
    await markPlaygroundComplete(page);

    await expect(page.getByText(/Completed · \+/)).toBeVisible();
    await expect(page.getByRole("button", { name: /Mark complete/i })).toHaveCount(0);

    const progress = await readProgress(page);
    expect(progress?.state.completed[`${phase.id}:${mod.id}`]).toBeTruthy();
  });

  test("Next playground button navigates within phase", async ({ page }) => {
    const phase = PHASES[0];
    const mod0 = phase.modules[0];
    const mod1 = phase.modules[1];

    await go(page, `/phase/${phase.slug}/play/${mod0.id}`);
    await page.getByRole("button", { name: new RegExp(`Next: ${mod1.title}`) }).click();
    await expect(page).toHaveURL(new RegExp(`#/phase/${phase.slug}/play/${mod1.id}$`));
    await expect(page.getByRole("heading", { name: mod1.title })).toBeVisible();
  });

  test("Back to phase button returns to phase page", async ({ page }) => {
    const phase = PHASES[0];
    const mod = phase.modules[0];

    await go(page, `/phase/${phase.slug}/play/${mod.id}`);
    await page.getByRole("button", { name: new RegExp(`Back to ${phase.title}`) }).click();
    await expect(page.getByRole("heading", { name: phase.title })).toBeVisible();
  });

  test("Finish phase button on last playground", async ({ page }) => {
    const phase = PHASES[0];
    const last = phase.modules[phase.modules.length - 1];

    await go(page, `/phase/${phase.slug}/play/${last.id}`);
    await page.getByRole("button", { name: /Finish phase/i }).click();
    await expect(page.getByRole("heading", { name: phase.title })).toBeVisible();
  });

  test("phase page shows Done badge after completion", async ({ page }) => {
    const phase = PHASES[0];
    const mod = phase.modules[0];

    await go(page, `/phase/${phase.slug}/play/${mod.id}`);
    await markPlaygroundComplete(page);
    await openPhase(page, phase.slug);
    const card = page.getByRole("link", { name: new RegExp(escapeRegex(mod.title)) });
    await expect(card.getByText("✓ Done")).toBeVisible();
  });

  test("breadcrumb links work from playground", async ({ page }) => {
    const phase = PHASES[2];
    const mod = phase.modules[0];

    await go(page, `/phase/${phase.slug}/play/${mod.id}`);
    await clickBreadcrumbJourney(page);
    await expect(page).toHaveURL(/#\/$/);

    await go(page, `/phase/${phase.slug}/play/${mod.id}`);
    await page.getByRole("link", { name: phase.title }).click();
    await expect(page.getByRole("heading", { name: phase.title })).toBeVisible();
  });

  for (const phase of PHASES) {
    for (const mod of phase.modules) {
      test(`loads: Phase ${phase.id} — ${mod.title}`, async ({ page }) => {
        await go(page, `/phase/${phase.slug}/play/${mod.id}`);
        await expect(page.getByRole("heading", { level: 1, name: mod.title })).toBeVisible({
          timeout: 30_000,
        });
        await expect(
          page.getByRole("button", { name: /Mark complete|Finish phase|Next:/i }).first(),
        ).toBeVisible();
      });
    }
  }
});
