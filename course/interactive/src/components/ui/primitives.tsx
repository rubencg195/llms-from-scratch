import { type ReactNode } from "react";

export function Panel({
  children,
  className = "",
  title,
  hint,
}: {
  children: ReactNode;
  className?: string;
  title?: string;
  hint?: string;
}) {
  return (
    <div className={`glass rounded-2xl p-5 ${className}`}>
      {title && (
        <div className="mb-3 flex items-center justify-between gap-3">
          <h3 className="text-sm font-semibold tracking-wide text-white/80 uppercase">{title}</h3>
          {hint && <span className="text-xs text-white/40">{hint}</span>}
        </div>
      )}
      {children}
    </div>
  );
}

export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  display,
  accent = "#7d9cff",
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  display?: string;
  accent?: string;
}) {
  return (
    <label className="block select-none">
      <div className="mb-1.5 flex items-center justify-between text-sm">
        <span className="text-white/70">{label}</span>
        <span className="font-mono text-white/90" style={{ color: accent }}>
          {display ?? value}
        </span>
      </div>
      <input
        type="range"
        className="w-full"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </label>
  );
}

export function Pill({
  children,
  color = "#7d9cff",
  className = "",
}: {
  children: ReactNode;
  color?: string;
  className?: string;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-semibold ${className}`}
      style={{ background: `${color}22`, color, border: `1px solid ${color}55` }}
    >
      {children}
    </span>
  );
}

export function Stat({ label, value, accent }: { label: string; value: ReactNode; accent?: string }) {
  return (
    <div className="surface-muted rounded-xl px-3 py-2">
      <div className="text-[11px] uppercase tracking-wide text-white/45">{label}</div>
      <div className="font-mono text-lg" style={{ color: accent ?? "#e7ecff" }}>
        {value}
      </div>
    </div>
  );
}

export function Insight({ children, title = "Key insight" }: { children: ReactNode; title?: string }) {
  return (
    <div className="rounded-2xl border border-indigo-400/30 bg-indigo-500/10 p-4">
      <div className="mb-1 flex items-center gap-2 text-sm font-semibold text-indigo-200">
        <span>💡</span>
        {title}
      </div>
      <div className="text-sm leading-relaxed text-white/75">{children}</div>
    </div>
  );
}

export function GameButton({
  children,
  onClick,
  variant = "primary",
  disabled,
  className = "",
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "ghost" | "success";
  disabled?: boolean;
  className?: string;
}) {
  const styles: Record<string, string> = {
    primary:
      "relative z-10 bg-indigo-500 text-white shadow-lg shadow-indigo-500/30 hover:brightness-110",
    success:
      "relative z-10 bg-emerald-500 text-white shadow-lg shadow-emerald-500/30 hover:brightness-110",
    ghost: "relative z-10 surface-muted text-white/80 hover:brightness-110",
  };
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`min-h-11 rounded-xl px-4 py-2.5 text-sm font-semibold transition active:scale-95 disabled:cursor-not-allowed disabled:opacity-40 ${styles[variant]} ${className}`}
    >
      {children}
    </button>
  );
}
