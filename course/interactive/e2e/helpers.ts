import { expect, type Page } from "@playwright/test";
import { PHASES } from "../src/data/curriculum";

export { PHASES };

export const STORAGE_KEY = "llms-interactive-progress";
export const LECTURE_XP = 20;
export const LAB_XP = 10;

export interface PersistedProgress {
  state: {
    xp: number;
    completed: Record<string, number>;
    read: Record<string, number>;
    unlocked: string[];
    streak: number;
    lastActive: string | null;
    soundOn: boolean;
  };
}

/** Navigate using HashRouter paths (`/#/…`). */
export async function go(page: Page, hashPath: string) {
  const path = hashPath.startsWith("/") ? hashPath : `/${hashPath}`;
  await page.goto(`/#${path}`);
  await page.waitForLoadState("domcontentloaded");
}

export async function clearProgress(page: Page) {
  await go(page, "/");
  await page.evaluate((key) => localStorage.removeItem(key), STORAGE_KEY);
  await page.reload();
  await page.waitForLoadState("domcontentloaded");
}

export async function readProgress(page: Page): Promise<PersistedProgress | null> {
  return page.evaluate((key) => {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    try {
      return JSON.parse(raw) as PersistedProgress;
    } catch {
      return null;
    }
  }, STORAGE_KEY);
}

export async function xpFromPage(page: Page): Promise<number> {
  const text = await page.locator("header").getByText(/^\d+ XP$/).textContent();
  const m = text?.match(/^(\d+)/);
  return m ? parseInt(m[1], 10) : 0;
}

export async function waitForXp(page: Page, min: number) {
  await expect.poll(async () => xpFromPage(page)).toBeGreaterThanOrEqual(min);
}

/** Advance lecture slides until the last slide (Back to phase control visible). */
export async function finishLecture(page: Page) {
  await page.locator("main").click();
  const next = page.getByRole("button", { name: "Next →" });
  for (let i = 0; i < 100 && (await next.isVisible()); i++) {
    await next.click();
  }
  await expect(page.getByRole("link", { name: /Back to phase/i })).toBeVisible();
}

export async function backToPhaseFromLecture(page: Page) {
  await page.getByRole("link", { name: /Back to phase/i }).click();
}

export async function markPlaygroundComplete(page: Page) {
  const btn = page.getByRole("button", { name: /Mark complete/i });
  await expect(btn).toBeVisible({ timeout: 30_000 });
  await btn.click();
  await expect(page.getByText(/Completed · \+/)).toBeVisible();
  await expect.poll(async () => (await readProgress(page))?.state.xp ?? 0).toBeGreaterThan(0);
}

export async function expectOnHome(page: Page) {
  await expect(page.getByRole("heading", { name: /Build an LLM from scratch/i })).toBeVisible();
}

export async function openPhase(page: Page, slug: string) {
  await go(page, `/phase/${slug}`);
  const phase = PHASES.find((p) => p.slug === slug)!;
  await expect(page.getByRole("heading", { level: 1, name: phase.title })).toBeVisible();
}

/** Parse "Slide N / M" from the lecture header badge. */
export async function lectureSlideTotal(page: Page): Promise<number> {
  const badge = page.locator("text=/Slide \\d+ \\/ \\d+/").first();
  const text = await badge.textContent();
  const m = text?.match(/\/ (\d+)/);
  return m ? parseInt(m[1], 10) : 0;
}

export async function journeyCountFromTopBar(page: Page): Promise<string> {
  return (await page.locator("text=/\\d+\\/\\d+ journey/").first().textContent()) ?? "";
}

export async function clickNav(page: Page, label: "Journey" | "Trophies") {
  await page.getByRole("navigation").getByRole("link", { name: label, exact: true }).click();
}

/** Breadcrumb "Journey" link in page content (not the top nav). */
export async function clickBreadcrumbJourney(page: Page) {
  await page.locator("main").getByRole("link", { name: "Journey", exact: true }).click();
}

export async function acceptNextDialog(page: Page) {
  page.once("dialog", (d) => d.accept());
}

export function escapeRegex(text: string) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
