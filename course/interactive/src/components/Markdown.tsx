import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";

/**
 * Shared markdown renderer for lecture slides and lab sections. Handles GitHub
 * tables, LaTeX math ($...$), and renders fenced code (the labs' Python cells)
 * in styled blocks so students can read the real code in the browser.
 */
export default function Markdown({ children }: { children: string }) {
  return (
    <div className="prose-invert max-w-none text-[15px] leading-relaxed text-white/85">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          h1: (p) => <h1 className="mb-3 mt-2 text-2xl font-extrabold text-white" {...p} />,
          h2: (p) => <h2 className="mb-2 mt-6 text-xl font-bold text-white" {...p} />,
          h3: (p) => <h3 className="mb-2 mt-5 text-lg font-semibold text-white/95" {...p} />,
          p: (p) => <p className="my-3 text-white/80" {...p} />,
          ul: (p) => <ul className="my-3 list-disc space-y-1 pl-6 text-white/80" {...p} />,
          ol: (p) => <ol className="my-3 list-decimal space-y-1 pl-6 text-white/80" {...p} />,
          li: (p) => <li className="text-white/80" {...p} />,
          a: (p) => <a className="text-indigo-300 underline underline-offset-2" {...p} />,
          strong: (p) => <strong className="font-semibold text-white" {...p} />,
          blockquote: (p) => (
            <blockquote
              className="my-4 border-l-4 border-indigo-400/50 bg-indigo-500/10 px-4 py-2 text-white/75"
              {...p}
            />
          ),
          table: (p) => (
            <div className="my-4 overflow-x-auto">
              <table className="w-full border-collapse text-sm" {...p} />
            </div>
          ),
          th: (p) => (
            <th className="border border-white/15 bg-white/5 px-3 py-2 text-left font-semibold text-white" {...p} />
          ),
          td: (p) => <td className="border border-white/10 px-3 py-2 text-white/75" {...p} />,
          code: ({ className, children, ...rest }) => {
            const isBlock = /language-/.test(className ?? "");
            if (!isBlock) {
              return (
                <code
                  className="rounded bg-white/10 px-1.5 py-0.5 font-mono text-[13px] text-cyan-200"
                  {...rest}
                >
                  {children}
                </code>
              );
            }
            const lang = (className ?? "").replace("language-", "") || "text";
            return (
              <span className="my-4 block overflow-hidden rounded-xl border border-white/10 bg-black/50">
                <span className="flex items-center justify-between border-b border-white/10 bg-white/5 px-3 py-1.5 text-[11px] uppercase tracking-wide text-white/40">
                  {lang}
                </span>
                <code className="block overflow-x-auto p-4 font-mono text-[13px] leading-relaxed text-emerald-100" {...rest}>
                  {children}
                </code>
              </span>
            );
          },
          pre: (p) => <pre className="my-0" {...p} />,
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
