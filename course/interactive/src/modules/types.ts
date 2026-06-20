/**
 * Contract every interactive module implements.
 *
 * A module is a focused, playable activity. When the student performs the
 * "aha" interaction (the thing that proves they got the concept), it calls
 * `onDiscover()` once. The host page turns that into XP + celebration. Modules
 * never touch the store directly, which keeps them easy to test and reuse.
 */
export interface ModuleProps {
  /** Call when the student hits the key insight. Safe to call repeatedly. */
  onDiscover: () => void;
  /** True once this module has already been completed (persisted). */
  completed: boolean;
}
