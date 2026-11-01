import { test } from "@playwright/test";
import { mkdirSync } from "node:fs";

// Regenerates the README screenshots from the real app and real results: `npm run shots`.
const OUT = "../docs/screenshots";
mkdirSync(OUT, { recursive: true });

async function settle(page) {
  await page.waitForLoadState("networkidle");
  await page.evaluate(() => document.fonts.ready);
  // Let the item picture finish decoding, if there is one.
  await page.waitForFunction(() => [...document.images].every((i) => i.complete));
}

test("screenshots", async ({ browser }) => {
  const light = await browser.newContext({ viewport: { width: 1360, height: 900 }, colorScheme: "light" });
  const page = await light.newPage();
  const shot = async (name, opts = {}) => {
    await settle(page);
    await page.screenshot({ path: `${OUT}/${name}.png`, ...opts });
  };

  await page.goto("/#/overview");
  await page.getByText("120 runs").waitFor();
  await shot("overview");

  await page.goto("/#/results");
  await page.getByText("Does triage help the classifier?").waitFor();
  await shot("results-images", { fullPage: true });
  await page.getByRole("button", { name: "Text (ChaosNLI)" }).click();
  await page.getByText("Gates this run misses").waitFor();
  await shot("results-text", { fullPage: true });

  await page.goto("/#/explorer");
  await page.getByLabel("Selected item").getByRole("img").waitFor();
  await shot("explorer-image", { fullPage: true });

  await page.getByRole("button", { name: "Text (ChaosNLI)" }).click();
  await page.getByLabel("Error type").selectOption("lazy");
  await page.getByLabel("Selected item").getByText("Premise.").waitFor();
  await shot("explorer-text", { fullPage: true });
  await light.close();

  const dark = await browser.newContext({ viewport: { width: 1360, height: 900 }, colorScheme: "dark" });
  const d = await dark.newPage();
  await d.goto("/#/overview");
  await d.getByText("120 runs").waitFor();
  await d.waitForLoadState("networkidle");
  await d.screenshot({ path: `${OUT}/overview-dark.png` });
  await dark.close();
});
