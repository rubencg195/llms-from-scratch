import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { fileURLToPath, URL } from "node:url";

// Deployed at https://rubenchevez.com/llms-from-scratch/ — override with VITE_BASE=./ for local preview/tests.
export default defineConfig({
  base: process.env.VITE_BASE ?? "/llms-from-scratch/",
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    fs: {
      // The lecture/lab markdown lives in ../lectures and ../labs (the course/
      // dir), one level above this app, so allow the dev server to read it.
      allow: [fileURLToPath(new URL("..", import.meta.url))],
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    css: true,
    restoreMocks: true,
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
    exclude: ["e2e/**", "node_modules/**"],
  },
});
