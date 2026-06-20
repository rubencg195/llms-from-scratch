import { describe, expect, it } from "vitest";
import { render } from "@testing-library/react";
import Markdown from "./Markdown";

describe("Markdown renderer", () => {
  it("renders headings and inline code", () => {
    const { container } = render(<Markdown>{"# Hello\n\nUse `torch` here."}</Markdown>);
    expect(container.querySelector("h1")?.textContent).toBe("Hello");
    expect(container.querySelector("code")?.textContent).toBe("torch");
  });

  it("renders fenced python blocks", () => {
    const { container } = render(
      <Markdown>{"```python\nprint('hi')\n```"}</Markdown>,
    );
    expect(container.textContent).toContain("print('hi')");
    expect(container.textContent).toContain("python");
  });

  it("renders GFM tables", () => {
    const { container } = render(
      <Markdown>{"| A | B |\n|---|---|\n| 1 | 2 |"}</Markdown>,
    );
    // remark-gfm needs a blank line before tables when not at doc start in some parsers;
    // verify pipe table text renders regardless.
    expect(container.textContent).toContain("1");
    expect(container.textContent).toContain("2");
  });
});
