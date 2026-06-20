/**
 * Tiny dependency-free celebration helpers: a confetti burst on a canvas and a
 * pleasant chime via the Web Audio API. Both no-op gracefully if unavailable.
 */

export function confettiBurst(): void {
  if (typeof document === "undefined") return;
  const canvas = document.createElement("canvas");
  canvas.style.cssText =
    "position:fixed;inset:0;pointer-events:none;z-index:60;width:100vw;height:100vh";
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  document.body.appendChild(canvas);
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    canvas.remove();
    return;
  }

  const colors = ["#7d9cff", "#a78bfa", "#34d399", "#fbbf24", "#f472b6", "#22d3ee"];
  const N = 140;
  const parts = Array.from({ length: N }, () => ({
    x: window.innerWidth / 2,
    y: window.innerHeight / 3,
    vx: (Math.random() - 0.5) * 14,
    vy: Math.random() * -12 - 4,
    r: Math.random() * 6 + 3,
    c: colors[(Math.random() * colors.length) | 0],
    rot: Math.random() * Math.PI,
    vr: (Math.random() - 0.5) * 0.3,
    life: 1,
  }));

  let raf = 0;
  const start = performance.now();
  const tick = (now: number) => {
    const dt = Math.min((now - start) / 1000, 4);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let alive = false;
    for (const p of parts) {
      p.vy += 0.35;
      p.x += p.vx;
      p.y += p.vy;
      p.rot += p.vr;
      p.life = Math.max(0, 1 - dt / 2.4);
      if (p.life > 0) alive = true;
      ctx.save();
      ctx.globalAlpha = p.life;
      ctx.translate(p.x, p.y);
      ctx.rotate(p.rot);
      ctx.fillStyle = p.c;
      ctx.fillRect(-p.r / 2, -p.r / 2, p.r, p.r * 1.6);
      ctx.restore();
    }
    if (alive) {
      raf = requestAnimationFrame(tick);
    } else {
      cancelAnimationFrame(raf);
      canvas.remove();
    }
  };
  raf = requestAnimationFrame(tick);
}

let audioCtx: AudioContext | null = null;
export function chime(enabled: boolean): void {
  if (!enabled || typeof window === "undefined") return;
  try {
    audioCtx =
      audioCtx ?? new (window.AudioContext || (window as any).webkitAudioContext)();
    const notes = [523.25, 659.25, 783.99]; // C5 E5 G5
    notes.forEach((freq, i) => {
      const osc = audioCtx!.createOscillator();
      const gain = audioCtx!.createGain();
      osc.type = "triangle";
      osc.frequency.value = freq;
      const t0 = audioCtx!.currentTime + i * 0.08;
      gain.gain.setValueAtTime(0.0001, t0);
      gain.gain.exponentialRampToValueAtTime(0.18, t0 + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.4);
      osc.connect(gain).connect(audioCtx!.destination);
      osc.start(t0);
      osc.stop(t0 + 0.45);
    });
  } catch {
    /* ignore audio errors */
  }
}
