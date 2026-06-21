import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, screen } from "@testing-library/react";
import { TOTAL_MODULES } from "@/data/curriculum";
import Home from "./Home";
import { renderWithMemoryRouter } from "@/test/render";
import { resetProgressStore } from "@/test/progress";
import { useProgress } from "@/store/progress";

describe("Home journey map", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-06-20T12:00:00Z"));
    resetProgressStore();
  });

  it("shows hero stats and all phase cards", () => {
    renderWithMemoryRouter(<Home />);
    expect(screen.getByText(/9 lectures/i)).toBeInTheDocument();
    expect(screen.getByText(/43 labs/i)).toBeInTheDocument();
    expect(screen.getByText(new RegExp(`${TOTAL_MODULES} interactive playgrounds`, "i"))).toBeInTheDocument();
    expect(screen.getAllByText(/Phase \d/).length).toBeGreaterThanOrEqual(9);
  });

  it("updates XP after progress", () => {
    useProgress.getState().completeModule(0, "tensor-explorer", 40);
    expect(useProgress.getState().xp).toBe(40);
    renderWithMemoryRouter(<Home />);
    expect(screen.getAllByText("Level").length).toBeGreaterThan(0);
  });

  it("links to the first phase", () => {
    renderWithMemoryRouter(<Home />);
    const start = screen.getAllByRole("link", { name: /Start the journey/i })[0];
    expect(start.getAttribute("href")).toContain("bridging-the-gap");
  });
});

describe("Trophies page", () => {
  beforeEach(() => resetProgressStore());

  it("lists achievements and supports reset", async () => {
    const { default: Trophies } = await import("./Trophies");
    renderWithMemoryRouter(<Trophies />);
    expect(screen.getByText(/Achievements/i)).toBeInTheDocument();
    expect(screen.getByText(/First Contact/i)).toBeInTheDocument();
    expect(screen.getByText(/Completionist/i)).toBeInTheDocument();

    vi.spyOn(window, "confirm").mockReturnValue(true);
    fireEvent.click(screen.getByRole("button", { name: /Reset progress/i }));
    expect(useProgress.getState().xp).toBe(0);
  });
});
