import { test, expect } from "@playwright/test";
import {
  clearProgress,
  go,
  markPlaygroundComplete,
  readProgress,
  acceptNextDialog,
  finishLecture,
  LECTURE_XP,
  PHASES,
} from "./helpers";

test.beforeEach(async ({ page }) => {
  await clearProgress(page);
});

test.describe("Trophy room", () => {
  test("shows stats and locked achievements initially", async ({ page }) => {
    await go(page, "/trophies");
    await expect(page.getByRole("heading", { name: /Trophy Room/i })).toBeVisible();
    await expect(page.getByText(/Achievements \(0\//)).toBeVisible();
    await expect(page.getByText("First Contact")).toBeVisible();
    await expect(page.getByText("🔒").first()).toBeVisible();
  });

  test("unlocking first playground reveals First Contact achievement", async ({ page }) => {
    const phase = PHASES[0];
    const mod = phase.modules[0];

    await go(page, `/phase/${phase.slug}/play/${mod.id}`);
    await markPlaygroundComplete(page);

    await go(page, "/trophies");
    await expect(page.getByText(/Achievements \(1\//)).toBeVisible({ timeout: 15_000 });
    const firstContact = page.locator(".surface").filter({ hasText: "First Contact" });
    await expect(firstContact.getByText("✨")).toBeVisible();
    await expect(firstContact.getByText("🔒")).toHaveCount(0);
  });

  test("stats reflect XP and journey after activity", async ({ page }) => {
    const phase = PHASES[0];
    await go(page, `/phase/${phase.slug}/lecture`);
    await finishLecture(page);
    await expect.poll(async () => (await readProgress(page))?.state.xp ?? 0).toBeGreaterThanOrEqual(
      LECTURE_XP,
    );

    await go(page, "/trophies");
    await expect(page.locator(".surface").filter({ hasText: "Page Turner" }).getByText("📖")).toBeVisible();
    await expect(page.getByText(/Achievements \(1\//)).toBeVisible();
  });

  test("reset progress clears localStorage after confirm", async ({ page }) => {
    const phase = PHASES[0];
    await go(page, `/phase/${phase.slug}/play/${phase.modules[0].id}`);
    await markPlaygroundComplete(page);

    let progress = await readProgress(page);
    expect(progress?.state.xp).toBeGreaterThan(0);

    await go(page, "/trophies");
    acceptNextDialog(page);
    await page.getByRole("button", { name: "Reset progress" }).click();

    progress = await readProgress(page);
    expect(progress?.state.xp ?? 0).toBe(0);

    await page.reload();
    await expect(page.getByText(/Achievements \(0\//)).toBeVisible();
  });

  test("dismiss reset dialog keeps progress", async ({ page }) => {
    const phase = PHASES[0];
    await go(page, `/phase/${phase.slug}/play/${phase.modules[0].id}`);
    await markPlaygroundComplete(page);
    const xpBefore = (await readProgress(page))?.state.xp ?? 0;

    await go(page, "/trophies");
    page.once("dialog", (d) => d.dismiss());
    await page.getByRole("button", { name: "Reset progress" }).click();

    const xpAfter = (await readProgress(page))?.state.xp ?? 0;
    expect(xpAfter).toBe(xpBefore);
  });
});
