import { useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useProgress } from "@/store/progress";

export default function ToastHost() {
  const { toast, clearToast } = useProgress();

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(clearToast, 3200);
    return () => clearTimeout(t);
  }, [toast, clearToast]);

  return (
    <div className="pointer-events-none fixed inset-x-0 top-20 z-50 flex justify-center px-4">
      <AnimatePresence>
        {toast && (
          <motion.div
            key={toast.title + (toast.sub ?? "")}
            initial={{ y: -24, opacity: 0, scale: 0.9 }}
            animate={{ y: 0, opacity: 1, scale: 1 }}
            exit={{ y: -16, opacity: 0, scale: 0.95 }}
            transition={{ type: "spring", stiffness: 260, damping: 20 }}
            className="glass pointer-events-auto flex max-w-[min(100%,24rem)] items-center gap-3 rounded-2xl px-5 py-3 shadow-2xl"
          >
            <span className="text-2xl">{toast.icon}</span>
            <div>
              <div className="text-sm font-semibold text-white">{toast.title}</div>
              {toast.sub && <div className="text-xs text-white/55">{toast.sub}</div>}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
