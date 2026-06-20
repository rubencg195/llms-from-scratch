import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { HashRouter } from "react-router-dom";
import App from "./App";
import { useProgress } from "./store/progress";
import "./index.css";

useProgress.persist.onFinishHydration(() => {
  useProgress.getState()._evaluate();
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <HashRouter>
      <App />
    </HashRouter>
  </StrictMode>,
);
