import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development, /api is proxied to the Python server (python -m labeltriage.api).
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8000" } },
  build: { target: "es2020", sourcemap: false, chunkSizeWarningLimit: 200 },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.js"],
    globals: true,
    include: ["src/**/*.test.{js,jsx}"],
  },
});
