import { useProgress, STORAGE_KEY } from "@/store/progress";

/** Reset Zustand progress state and clear persisted localStorage between tests. */
export function resetProgressStore() {
  localStorage.removeItem(STORAGE_KEY);
  useProgress.getState().reset();
  useProgress.setState({ toast: null, soundOn: true });
}
