import { test, expect } from "@playwright/test";
import {
  clearProgress,
  go,
  expectOnHome,
  openPhase,
  finishLecture,
  markPlaygroundComplete,
  PHASES,
} from "./helpers";

const MOBILE = { width: 390, height: 844 };

test.use({ viewport: MOBILE });

test.beforeEach(async ({ page }) => {
  await clearProgress(page);
});

async function noHorizontalOverflow(page: import("@playwright/test").Page) {
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
  );
  expect(overflow).toBe(false);
}

async function clickMobileNav(page: import("@playwright/test").Page, label: "Journey" | "Trophies") {
  await page.getByRole("navigation", { name: "Mobile navigation" }).getByRole("link", { name: label }).click();
}

test.describe("Mobile layout", () => {
  test("home fits viewport and shows bottom navigation", async ({ page }) => {
    await go(page, "/");
    await expectOnHome(page);
    await expect(page.getByRole("navigation", { name: "Mobile navigation" })).toBeVisible();
    await expect(page.getByRole("link", { name: /Start the journey|Continue/ })).toBeVisible();
    await noHorizontalOverflow(page);
  });

  test("mobile nav reaches trophies and back to journey", async ({ page }) => {
    await go(page, "/");
    await clickMobileNav(page, "Trophies");
    await expect(page.getByRole("heading", { name: /Trophy Room/i })).toBeVisible();
    await noHorizontalOverflow(page);

    await clickMobileNav(page, "Journey");
    await expectOnHome(page);
  });

  test("phase page and playground actions fit on small screens", async ({ page }) => {
    const phase = PHASES[0];
    const mod = phase.modules[0];

    await openPhase(page, phase.slug);
    await noHorizontalOverflow(page);

    await page.getByRole("link", { name: new RegExp(mod.title) }).click();
    await expect(page.getByRole("heading", { level: 1, name: mod.title })).toBeVisible();
    await expect(page.getByRole("button", { name: /Mark complete/i })).toBeVisible();
    await markPlaygroundComplete(page);
    await noHorizontalOverflow(page);
  });

  test("lecture reader stacks controls on mobile", async ({ page }) => {
    const phase = PHASES[0];
    await go(page, `/phase/${phase.slug}/lecture`);
    await expect(page.getByRole("button", { name: "Next →" })).toBeVisible();
    await finishLecture(page);
    await expect(page.getByRole("link", { name: /Back to phase/i })).toBeVisible();
    await noHorizontalOverflow(page);
  });

  test("lab page renders without horizontal scroll", async ({ page }) => {
    const phase = PHASES[0];
    await openPhase(page, phase.slug);
    await page.getByRole("link", { name: /Read →/ }).first().click();
    await expect(page.locator("article")).toBeVisible();
    await noHorizontalOverflow(page);
  });
});
