import { type ReactElement, type ReactNode } from "react";
import { render, type RenderOptions } from "@testing-library/react";
import { HashRouter, MemoryRouter, type MemoryRouterProps } from "react-router-dom";

export function renderWithRouter(ui: ReactElement, route = "/") {
  window.location.hash = route.startsWith("#") ? route : `#${route}`;
  return render(<HashRouter>{ui}</HashRouter>);
}

export function renderWithMemoryRouter(
  ui: ReactNode,
  routerProps: MemoryRouterProps = { initialEntries: ["/"] },
  options?: Omit<RenderOptions, "wrapper">,
) {
  return render(<MemoryRouter {...routerProps}>{ui}</MemoryRouter>, options);
}
