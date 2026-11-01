import { defineConfig } from "@playwright/test";
import { existsSync } from "node:fs";

// Drives the real, installed Google Chrome (no browser download). The webServer builds nothing:
// run `npm run build` first. It serves the built app and the API on one port, like production.
const chrome = [
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "/usr/bin/google-chrome",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
].find((p) => existsSync(p));

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  expect: { timeout: 15_000 },
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: "http://127.0.0.1:8123",
    viewport: { width: 1360, height: 900 },
    launchOptions: chrome ? { executablePath: chrome } : { channel: "chrome" },
  },
  webServer: {
    command: process.env.LT_PYTHON
      ? `${process.env.LT_PYTHON} -m labeltriage.api --port 8123`
      : "python -m labeltriage.api --port 8123",
    cwd: "..",
    url: "http://127.0.0.1:8123/api/health",
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
