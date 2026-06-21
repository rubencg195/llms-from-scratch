import { test, expect } from "@playwright/test";
import { clearProgress, go, expectOnHome, clickNav, PHASES } from "./helpers";

test.beforeEach(async ({ page }) => {
  await clearProgress(page);
});

test.describe("Top bar & routing", () => {
  test("logo returns home from any page", async ({ page }) => {
    await go(page, "/trophies");
    await expect(page.getByRole("heading", { name: /Trophy Room/i })).toBeVisible();
    await page.getByRole("link", { name: /LLMs From Scratch/i }).click();
    await expectOnHome(page);
  });

  test("navbar Journey and Trophies links work", async ({ page }) => {
    await go(page, "/");
    await clickNav(page, "Trophies");
    await expect(page).toHaveURL(/#\/trophies$/);
    await expect(page.getByRole("heading", { name: /Trophy Room/i })).toBeVisible();

    await clickNav(page, "Journey");
    await expect(page).toHaveURL(/#\/$/);
    await expectOnHome(page);
  });

  test("invalid phase slug redirects to home", async ({ page }) => {
    await go(page, "/phase/not-a-real-phase");
    await expectOnHome(page);
  });

  test("invalid playground redirects to home", async ({ page }) => {
    await go(page, "/phase/bridging-the-gap/play/fake-module");
    await expectOnHome(page);
  });

  test("invalid lab redirects to home", async ({ page }) => {
    await go(page, "/phase/bridging-the-gap/lab/fake-lab");
    await expectOnHome(page);
  });

  test("unknown hash route redirects to home", async ({ page }) => {
    await go(page, "/does-not-exist");
    await expectOnHome(page);
  });

  test("sound toggle persists in localStorage", async ({ page }) => {
    await go(page, "/");
    const soundBtn = page.locator("header").getByRole("button").last();
    await expect(soundBtn).toHaveAttribute("title", "Sound on");

    await soundBtn.click();
    await expect(soundBtn).toHaveAttribute("title", "Sound off");

    const stored = await page.evaluate(() => {
      const raw = localStorage.getItem("llms-interactive-progress");
      return raw ? JSON.parse(raw).state.soundOn : null;
    });
    expect(stored).toBe(false);

    await page.reload();
    await expect(soundBtn).toHaveAttribute("title", "Sound off");
  });
});

test.describe("Phase card navigation", () => {
  for (const phase of PHASES) {
    test(`phase card opens ${phase.title}`, async ({ page }) => {
      await go(page, "/");
      await page.getByRole("link", { name: new RegExp(phase.title) }).click();
      await expect(page).toHaveURL(new RegExp(`#/phase/${phase.slug}$`));
      await expect(page.getByRole("heading", { level: 1, name: phase.title })).toBeVisible();
    });
  }
});
