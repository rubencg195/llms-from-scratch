import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import { ALL_MODULES, moduleKey, PHASES, TOTAL_MODULES, TOTAL_XP } from "@/data/curriculum";
import { LABS, LECTURES, labsForPhase, lectureForPhase } from "@/content";

export const LECTURE_XP = 20;
export const LAB_XP = 10;
export const STORAGE_KEY = "llms-interactive-progress";
export const STORAGE_VERSION = 1;

export const TOTAL_LECTURES = LECTURES.length;
export const TOTAL_LABS = LABS.length;
/** Max XP if every lecture, lab, and playground is completed once. */
export const TOTAL_MAX_XP =
  TOTAL_XP + TOTAL_LECTURES * LECTURE_XP + TOTAL_LABS * LAB_XP;
export const TOTAL_JOURNEY_ITEMS = TOTAL_MODULES + TOTAL_LECTURES + TOTAL_LABS;

export interface Achievement {
  id: string;
  title: string;
  desc: string;
  icon: string;
}

export const ACHIEVEMENTS: Achievement[] = [
  { id: "first-step", title: "First Contact", desc: "Complete your first playground", icon: "✨" },
  { id: "first-read", title: "Page Turner", desc: "Read a lecture or open a lab", icon: "📖" },
  { id: "scholar", title: "Scholar", desc: "Finish every lecture deck", icon: "📽️" },
  { id: "lab-rat", title: "Lab Rat", desc: "Open every lab section", icon: "🧪" },
  { id: "phase-0", title: "Math Survivor", desc: "Finish every Phase 0 playground", icon: "🧮" },
  { id: "phase-1", title: "Transformer Tamer", desc: "Finish every Phase 1 playground", icon: "🔤" },
  { id: "streak-3", title: "On a Roll", desc: "Reach a 3-day streak", icon: "🔥" },
  { id: "halfway", title: "Halfway There", desc: "Complete half of all playgrounds", icon: "🌗" },
  { id: "explorer", title: "Dimension Hopper", desc: "Open a 3D module", icon: "🪐" },
  { id: "level-5", title: "Rising Star", desc: "Reach level 5", icon: "⭐" },
  { id: "grand-slam", title: "Grand Architect", desc: "Complete every playground", icon: "🏆" },
  { id: "completionist", title: "Completionist", desc: "Finish the entire journey (lectures, labs & play)", icon: "👑" },
];

/** Level curve: each level needs a bit more XP than the last. */
export function levelForXp(xp: number): { level: number; into: number; span: number } {
  let level = 1;
  let needed = 100;
  let acc = 0;
  while (xp >= acc + needed) {
    acc += needed;
    level += 1;
    needed = Math.round(needed * 1.35);
  }
  return { level, into: xp - acc, span: needed };
}

function todayStamp(): string {
  return new Date().toISOString().slice(0, 10);
}

function touchStreak(lastActive: string | null, streak: number): { streak: number; lastActive: string } {
  const today = todayStamp();
  if (lastActive === today) {
    return { streak: streak || 1, lastActive: today };
  }
  const yesterday = new Date(Date.now() - 864e5).toISOString().slice(0, 10);
  const nextStreak = lastActive === yesterday ? streak + 1 : 1;
  return { streak: nextStreak, lastActive: today };
}

export interface Toast {
  title: string;
  icon: string;
  sub?: string;
}

export interface JourneyStats {
  playDone: number;
  playTotal: number;
  lecDone: number;
  lecTotal: number;
  labDone: number;
  labTotal: number;
  itemsDone: number;
  itemsTotal: number;
  pct: number;
}

export interface PhaseJourneyStats {
  lectureDone: boolean;
  labsDone: number;
  labsTotal: number;
  playDone: number;
  playTotal: number;
  done: number;
  total: number;
  pct: number;
}

interface ProgressState {
  version: number;
  xp: number;
  completed: Record<string, number>;
  /** lecture/lab content marked as read; separate from module completion. */
  read: Record<string, number>;
  unlocked: string[];
  streak: number;
  lastActive: string | null;
  soundOn: boolean;
  toast: Toast | null;

  completeModule: (phaseId: number, moduleId: string, xp: number) => void;
  markRead: (key: string, xp: number, label?: string) => void;
  visit3D: () => void;
  toggleSound: () => void;
  clearToast: () => void;
  reset: () => void;

  isComplete: (phaseId: number, moduleId: string) => boolean;
  isRead: (key: string) => boolean;
  phaseProgress: (phaseId: number) => { done: number; total: number };
  phaseJourney: (phaseId: number) => PhaseJourneyStats;
  journeyStats: () => JourneyStats;

  /** internal — recompute achievements and surface a toast */
  _evaluate: (note?: string) => void;
}

export const useProgress = create<ProgressState>()(
  persist(
    (set, get) => ({
      version: STORAGE_VERSION,
      xp: 0,
      completed: {},
      read: {},
      unlocked: [],
      streak: 0,
      lastActive: null,
      soundOn: true,
      toast: null,

      isComplete: (phaseId, moduleId) =>
        Boolean(get().completed[moduleKey(phaseId, moduleId)]),

      isRead: (key) => Boolean(get().read[key]),

      phaseProgress: (phaseId) => {
        const phase = PHASES.find((p) => p.id === phaseId);
        if (!phase) return { done: 0, total: 0 };
        const done = phase.modules.filter((m) =>
          get().completed[moduleKey(phaseId, m.id)],
        ).length;
        return { done, total: phase.modules.length };
      },

      phaseJourney: (phaseId) => {
        const s = get();
        const phase = PHASES.find((p) => p.id === phaseId);
        const labs = labsForPhase(phaseId);
        const hasLecture = Boolean(lectureForPhase(phaseId));
        const lectureDone = hasLecture ? Boolean(s.read[`lec:${phaseId}`]) : true;
        const labsDone = labs.filter((l) => s.read[`lab:${l.slug}`]).length;
        const playDone = phase
          ? phase.modules.filter((m) => s.completed[moduleKey(phaseId, m.id)]).length
          : 0;
        const playTotal = phase?.modules.length ?? 0;
        const total = (hasLecture ? 1 : 0) + labs.length + playTotal;
        const done =
          (lectureDone ? 1 : 0) + labsDone + playDone;
        return {
          lectureDone,
          labsDone,
          labsTotal: labs.length,
          playDone,
          playTotal,
          done,
          total,
          pct: total ? Math.round((done / total) * 100) : 0,
        };
      },

      journeyStats: () => {
        const s = get();
        const playDone = Object.keys(s.completed).length;
        const lecDone = LECTURES.filter((l) => s.read[`lec:${l.phaseId}`]).length;
        const labDone = LABS.filter((l) => s.read[`lab:${l.slug}`]).length;
        const itemsDone = playDone + lecDone + labDone;
        return {
          playDone,
          playTotal: TOTAL_MODULES,
          lecDone,
          lecTotal: TOTAL_LECTURES,
          labDone,
          labTotal: TOTAL_LABS,
          itemsDone,
          itemsTotal: TOTAL_JOURNEY_ITEMS,
          pct: Math.round((itemsDone / TOTAL_JOURNEY_ITEMS) * 100),
        };
      },

      completeModule: (phaseId, moduleId, xp) => {
        const key = moduleKey(phaseId, moduleId);
        const state = get();
        if (state.completed[key]) return;

        const { streak, lastActive } = touchStreak(state.lastActive, state.streak);

        set({
          completed: { ...state.completed, [key]: Date.now() },
          xp: state.xp + xp,
          streak,
          lastActive,
        });
        get()._evaluate(`Playground complete · +${xp} XP`);
      },

      markRead: (key, xp, label) => {
        const state = get();
        if (state.read[key]) return;

        const { streak, lastActive } = touchStreak(state.lastActive, state.streak);

        set({
          read: { ...state.read, [key]: Date.now() },
          xp: state.xp + xp,
          streak,
          lastActive,
        });
        get()._evaluate(label ? `${label} · +${xp} XP` : `+${xp} XP`);
      },

      visit3D: () => {
        const state = get();
        if (state.unlocked.includes("explorer")) return;
        const meta = ACHIEVEMENTS.find((a) => a.id === "explorer");
        set({
          unlocked: [...state.unlocked, "explorer"],
          toast: meta
            ? { icon: meta.icon, title: `Achievement: ${meta.title}`, sub: meta.desc }
            : { icon: "🪐", title: "Opened a 3D module" },
        });
      },

      toggleSound: () => set((s) => ({ soundOn: !s.soundOn })),
      clearToast: () => set({ toast: null }),
      reset: () =>
        set({
          version: STORAGE_VERSION,
          xp: 0,
          completed: {},
          read: {},
          unlocked: [],
          streak: 0,
          lastActive: null,
          toast: null,
        }),

      _evaluate: (note?: string) => {
        const s = get();
        const newly: string[] = [];
        const add = (id: string) => {
          if (!s.unlocked.includes(id) && !newly.includes(id)) newly.push(id);
        };

        const playDone = Object.keys(s.completed).length;
        const readCount = Object.keys(s.read).length;

        if (playDone >= 1) add("first-step");
        if (readCount >= 1) add("first-read");
        if (playDone >= Math.ceil(TOTAL_MODULES / 2)) add("halfway");
        if (playDone >= TOTAL_MODULES) add("grand-slam");
        if (s.streak >= 3) add("streak-3");
        if (levelForXp(s.xp).level >= 5) add("level-5");

        if (
          ALL_MODULES.some(
            (m) => m.threeD && s.completed[moduleKey(m.phaseId, m.id)],
          )
        ) {
          add("explorer");
        }

        if (LECTURES.every((l) => s.read[`lec:${l.phaseId}`])) add("scholar");
        if (LABS.every((l) => s.read[`lab:${l.slug}`])) add("lab-rat");

        for (const phase of PHASES) {
          const allPlay = phase.modules.every((m) => s.completed[moduleKey(phase.id, m.id)]);
          if (allPlay && phase.id <= 1) add(`phase-${phase.id}`);
        }

        const journey = get().journeyStats();
        if (journey.itemsDone >= journey.itemsTotal) add("completionist");

        if (newly.length > 0) {
          const meta = ACHIEVEMENTS.find((a) => a.id === newly[newly.length - 1]);
          set({
            unlocked: [...s.unlocked, ...newly],
            toast: meta
              ? { icon: meta.icon, title: `Achievement: ${meta.title}`, sub: meta.desc }
              : note
                ? { icon: "✅", title: note }
                : null,
          });
        } else if (note) {
          set({ toast: { icon: "⚡", title: note } });
        }
      },
    }),
    {
      name: STORAGE_KEY,
      version: STORAGE_VERSION,
      storage: createJSONStorage(() => localStorage),
      partialize: (s) => ({
        version: s.version,
        xp: s.xp,
        completed: s.completed,
        read: s.read,
        unlocked: s.unlocked,
        streak: s.streak,
        lastActive: s.lastActive,
        soundOn: s.soundOn,
      }),
      migrate: (persisted, _version) => {
        const state = persisted as Partial<ProgressState>;
        return {
          version: STORAGE_VERSION,
          xp: state.xp ?? 0,
          completed: state.completed ?? {},
          read: state.read ?? {},
          unlocked: state.unlocked ?? [],
          streak: state.streak ?? 0,
          lastActive: state.lastActive ?? null,
          soundOn: state.soundOn ?? true,
        };
      },
    },
  ),
);