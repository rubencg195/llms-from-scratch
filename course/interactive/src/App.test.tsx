import { describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import App from "./App";
import { renderWithMemoryRouter } from "@/test/render";
import { resetProgressStore } from "@/test/progress";

describe("App routing", () => {
  it("renders the journey home page", () => {
    resetProgressStore();
    renderWithMemoryRouter(<App />, { initialEntries: ["/"] });
    expect(screen.getByRole("heading", { name: /Build an LLM from scratch/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "The Journey", level: 2 })).toBeInTheDocument();
  });

  it("renders a phase detail page", () => {
    resetProgressStore();
    renderWithMemoryRouter(<App />, { initialEntries: ["/phase/bridging-the-gap"] });
    expect(screen.getByRole("heading", { level: 1, name: /Bridging the Gap/i })).toBeInTheDocument();
    expect(screen.getAllByText("Interactive playgrounds").length).toBeGreaterThan(0);
    expect(screen.getByText(/Labs \(\d+\)/)).toBeInTheDocument();
  });

  it("renders the trophies page", () => {
    resetProgressStore();
    renderWithMemoryRouter(<App />, { initialEntries: ["/trophies"] });
    expect(screen.getByRole("heading", { name: /Your Trophy Room/i })).toBeInTheDocument();
    expect(screen.getByText(/localStorage/i)).toBeInTheDocument();
  });

  it("renders a lab reader page", () => {
    resetProgressStore();
    renderWithMemoryRouter(<App />, {
      initialEntries: ["/phase/bridging-the-gap/lab/section-0-1-tensors"],
    });
    expect(screen.getByText(/runnable in JupyterLab/i)).toBeInTheDocument();
    expect(screen.getByRole("article")).toBeInTheDocument();
  });

  it("shows top bar navigation and stats", () => {
    resetProgressStore();
    renderWithMemoryRouter(<App />, { initialEntries: ["/"] });
    expect(screen.getAllByText("LLMs From Scratch").length).toBeGreaterThan(0);
    expect(screen.getAllByRole("link", { name: "Journey" }).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByRole("link", { name: "Trophies" }).length).toBeGreaterThanOrEqual(1);
  });
});
