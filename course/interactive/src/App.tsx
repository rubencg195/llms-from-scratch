import { Routes, Route, useLocation } from "react-router-dom";
import { AnimatePresence } from "framer-motion";
import TopBar from "@/components/layout/TopBar";
import ToastHost from "@/components/ToastHost";
import Home from "@/pages/Home";
import Phase from "@/pages/Phase";
import ModulePage from "@/pages/ModulePage";
import LecturePage from "@/pages/LecturePage";
import LabPage from "@/pages/LabPage";
import Trophies from "@/pages/Trophies";

export default function App() {
  const location = useLocation();
  return (
    <>
      <div aria-hidden className="ambient-bg">
        <div className="ambient-bg__top" />
        <div className="ambient-bg__bottom" />
      </div>
      <div className="app-shell min-h-full">
        <TopBar />
        <ToastHost />
        <AnimatePresence mode="wait">
          <Routes location={location} key={location.pathname}>
            <Route path="/" element={<Home />} />
            <Route path="/phase/:slug" element={<Phase />} />
            <Route path="/phase/:slug/lecture" element={<LecturePage />} />
            <Route path="/phase/:slug/lab/:labSlug" element={<LabPage />} />
            <Route path="/phase/:slug/play/:moduleId" element={<ModulePage />} />
            <Route path="/trophies" element={<Trophies />} />
            <Route path="*" element={<Home />} />
          </Routes>
        </AnimatePresence>
      </div>
    </>
  );
}
