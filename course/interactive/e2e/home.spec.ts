import { test, expect } from "@playwright/test";
import { clearProgress, go, expectOnHome, PHASES } from "./helpers";

test.beforeEach(async ({ page }) => {
  await clearProgress(page);
});

test.describe("Home page", () => {
  test("shows hero, journey map, and all phase cards", async ({ page }) => {
    await go(page, "/");
    await expectOnHome(page);
    await expect(page.getByRole("heading", { name: "The Journey" })).toBeVisible();

    for (const phase of PHASES) {
      await expect(page.getByRole("link", { name: new RegExp(phase.title) })).toBeVisible();
      await expect(page.getByText(`Phase ${phase.id}`).first()).toBeVisible();
    }
  });

  test("Start the journey button opens phase 0", async ({ page }) => {
    await go(page, "/");
    await page.getByRole("link", { name: /Start the journey/i }).click();
    await expect(page).toHaveURL(/#\/phase\/bridging-the-gap$/);
    await expect(page.getByRole("heading", { name: "Bridging the Gap" })).toBeVisible();
  });

  test("hero meters show zero progress initially", async ({ page }) => {
    await go(page, "/");
    await expect(page.getByText("Level").locator("..").getByText("1")).toBeVisible();
    await expect(page.getByText("XP", { exact: true }).locator("..").getByText(/^0\//)).toBeVisible();
    await expect(page.getByText("Journey", { exact: true }).locator("..").getByText("0%")).toBeVisible();
  });

  test("phase cards show item counts and progress bars", async ({ page }) => {
    await go(page, "/");
    const first = PHASES[0];
    const card = page.getByRole("link", { name: new RegExp(first.title) });
    await expect(card.getByText(/0\/\d+ items/)).toBeVisible();
    await expect(card.getByText("📽️ lecture")).toBeVisible();
    await expect(card.getByText("🧪 4 labs")).toBeVisible();
    await expect(card.getByText("🎮 4 play")).toBeVisible();
  });
});
