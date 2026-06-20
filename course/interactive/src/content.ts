/**
 * Loads the *actual* course content — all 9 lecture decks and all 39 lab
 * sections — straight from the markdown sources in ../lectures and ../labs, so
 * the website contains 100% of the written material (not just the playgrounds).
 *
 * Uses Vite's import.meta.glob with `?raw` to inline every file at build time;
 * the app stays fully static with no runtime fetches.
 */

const lectureFiles = import.meta.glob("../../lectures/*.md", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

const labFiles = import.meta.glob("../../labs/*/section-*.md", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

export interface Slide {
  body: string;
  notes: string[];
}

export interface Lecture {
  phaseId: number;
  title: string;
  slides: Slide[];
}

export interface Lab {
  phaseId: number;
  /** section number like "0-1" */
  section: string;
  /** url-safe slug = filename without extension, e.g. "section-0-1-tensors" */
  slug: string;
  title: string;
  /** markdown body with YAML front matter stripped */
  body: string;
}

function stripFrontMatter(raw: string): { meta: Record<string, string>; body: string } {
  const m = raw.match(/^---\n([\s\S]*?)\n---\n?/);
  if (!m) return { meta: {}, body: raw };
  const meta: Record<string, string> = {};
  for (const line of m[1].split("\n")) {
    const kv = line.match(/^(\w[\w-]*):\s*(.*)$/);
    if (kv) meta[kv[1]] = kv[2].replace(/^["']|["']$/g, "");
  }
  return { meta, body: raw.slice(m[0].length) };
}

function phaseIdFromPath(path: string): number {
  const m = path.match(/phase-(\d+)/);
  return m ? parseInt(m[1], 10) : 0;
}

function firstHeading(body: string, fallback: string): string {
  const m = body.match(/^#\s+(.+)$/m);
  return m ? m[1].trim() : fallback;
}

/** Pull `<!-- notes: ... -->` blocks out of a slide and return them separately. */
function extractNotes(body: string): Slide {
  const notes: string[] = [];
  const clean = body.replace(/<!--\s*notes:\s*([\s\S]*?)-->/gi, (_m, n) => {
    notes.push(String(n).trim());
    return "";
  });
  return { body: clean.trim(), notes };
}

function buildLectures(): Lecture[] {
  const out: Lecture[] = [];
  for (const [path, raw] of Object.entries(lectureFiles)) {
    const { meta, body } = stripFrontMatter(raw);
    // Lecture slides are separated by horizontal rules (--- on their own line).
    const parts = body
      .split(/\n-{3,}\n/)
      .map((s) => s.trim())
      .filter(Boolean);
    out.push({
      phaseId: phaseIdFromPath(path),
      title: meta.title || firstHeading(body, "Lecture"),
      slides: parts.map(extractNotes),
    });
  }
  return out.sort((a, b) => a.phaseId - b.phaseId);
}

function buildLabs(): Lab[] {
  const out: Lab[] = [];
  for (const [path, raw] of Object.entries(labFiles)) {
    const { body } = stripFrontMatter(raw);
    const file = path.split("/").pop() ?? "section.md";
    const slug = file.replace(/\.md$/, "");
    const secMatch = slug.match(/section-(\d+)-(\d+)/);
    const section = secMatch ? `${secMatch[1]}-${secMatch[2]}` : slug;
    out.push({
      phaseId: phaseIdFromPath(path),
      section,
      slug,
      title: firstHeading(body, slug),
      body,
    });
  }
  return out.sort((a, b) => a.slug.localeCompare(b.slug, undefined, { numeric: true }));
}

export const LECTURES = buildLectures();
export const LABS = buildLabs();

export function lectureForPhase(phaseId: number): Lecture | undefined {
  return LECTURES.find((l) => l.phaseId === phaseId);
}
export function labsForPhase(phaseId: number): Lab[] {
  return LABS.filter((l) => l.phaseId === phaseId);
}
export function labBySlug(slug: string): Lab | undefined {
  return LABS.find((l) => l.slug === slug);
}
