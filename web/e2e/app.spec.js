import { expect, test } from "@playwright/test";

// End to end, in real Chrome, against the real server and the real mini results.
// Run `npm run build` first, then `npm run e2e` (see playwright.config.js).

test.beforeEach(async ({ page }) => {
  page.problems = [];
  page.on("console", (m) => m.type() === "error" && page.problems.push(`console: ${m.text()}`));
  page.on("pageerror", (e) => page.problems.push(`pageerror: ${e.message}`));
  page.on("requestfailed", (r) => page.problems.push(`request failed: ${r.url()}`));
  page.on("response", (r) => r.status() >= 400 && page.problems.push(`${r.status()} ${r.url()}`));
});

test.afterEach(async ({ page }) => {
  expect(page.problems, "browser console, network and page errors").toEqual([]);
});

test("overview shows the headline findings from the real runs", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveTitle("LabelTriage");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("mistake, or a real difference");
  await expect(page.getByText("120 runs")).toBeVisible();
  await expect(page.getByText("VariErr", { exact: false }).first()).toBeVisible();
  await expect(page.getByText("Read these numbers as direction")).toBeVisible();
  await expect(page.locator(".stat .value")).toHaveCount(4);
});

test("navigation works with the URL hash, reload and back", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("link", { name: "Results", exact: true }).click();
  await expect(page).toHaveURL(/#\/results$/);
  await expect(page.getByRole("link", { name: "Results", exact: true })).toHaveAttribute("aria-current", "page");
  await page.reload();
  await expect(page.getByText("Does triage help the classifier?")).toBeVisible();
  await page.getByRole("link", { name: "About", exact: true }).click();
  await expect(page.getByText("The three routes")).toBeVisible();
  await page.goBack();
  await expect(page.getByText("Does triage help the classifier?")).toBeVisible();
});

test("results: both datasets, charts, tables and the visible misses", async ({ page }) => {
  await page.goto("/#/results");
  await expect(page.getByRole("img", { name: /Validation accuracy by error type.*Images/ })).toBeVisible();
  await expect(page.locator("svg.chart rect")).toHaveCount(16); // 4 error types x 4 methods
  await expect(page.getByRole("row", { name: /Lazy.*ahead/ })).toBeVisible();

  await page.getByRole("button", { name: "Text (ChaosNLI)" }).click();
  await expect(page.getByRole("img", { name: /Validation accuracy by error type.*Text/ })).toBeVisible();
  await expect(page.getByText("Gates this run misses")).toBeVisible();
  await expect(page.getByText(/G1 alone reaches 0.588.*text, lazy/)).toBeVisible();
  await expect(page.getByText(/level with vote counting, not better/)).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "G3 plausibility" })).toBeVisible();
});

test("explorer: scatter, table, filters, search, paging and the item detail", async ({ page }) => {
  await page.goto("/#/explorer");
  const rows = page.getByRole("table", { name: "Items" }).locator("tbody tr");
  await expect(rows).toHaveCount(10);
  await expect(page.locator(".scatter-wrap circle[data-i]")).toHaveCount(1000);
  await expect(page.getByText(/1,000 items, page 1 of 100/)).toBeVisible();

  // The most suspicious item opens by default, with its picture and its five votes.
  const detail = page.getByLabel("Selected item");
  await expect(detail.getByRole("heading", { level: 2 })).toContainText("Item");
  await expect(detail.getByRole("table", { name: "Votes on this item" }).locator("tbody tr")).toHaveCount(5);
  const img = detail.getByRole("img");
  await expect(img).toBeVisible();
  await expect.poll(() => img.evaluate((el) => el.complete && el.naturalWidth > 0)).toBe(true);

  // Clicking a dot opens that item.
  const before = await detail.getByRole("heading", { level: 2 }).innerText();
  await page.locator(".scatter-wrap circle[data-i]").nth(300).dispatchEvent("click");
  await expect(detail.getByRole("heading", { level: 2 })).not.toHaveText(before);

  // Clicking a table row opens that item too.
  const third = await rows.nth(2).getByRole("button").innerText();
  await rows.nth(2).click();
  await expect(detail.getByRole("heading", { level: 2 })).toHaveText(`Item ${third}`);

  // Paging and page size.
  await page.getByRole("button", { name: "Next" }).click();
  await expect(page.getByText(/page 2 of 100/)).toBeVisible();
  await page.getByLabel("Per page").selectOption("50");
  await expect(rows).toHaveCount(50);
  await expect(page.getByText(/page 1 of 20/)).toBeVisible();

  // Route filter: switching a route off shrinks the list and the plot together.
  await page.getByRole("button", { name: "clear" }).click();
  await expect(page.getByRole("button", { name: "clear" })).toHaveAttribute("aria-pressed", "false");
  await expect.poll(() => page.locator(".scatter-wrap circle[data-i]").count()).toBeLessThan(1000);
  await expect(rows.first().locator(".chip")).not.toHaveText("clear");
  await page.getByRole("button", { name: "clear" }).click();

  // Injected-error filter.
  await page.getByLabel("Only items with an injected error").check();
  await expect.poll(() => page.locator(".scatter-wrap circle[data-i]").count()).toBeLessThan(1000);
  await page.getByLabel("Only items with an injected error").uncheck();

  // Debounced search narrows to one item.
  const target = await rows.first().getByRole("button").innerText();
  await page.getByLabel("Find an item").fill(target);
  await expect(page.getByText(/\b1 items, page 1 of 1/)).toBeVisible();
  await expect(rows).toHaveCount(1);
  await page.getByLabel("Find an item").fill("no-such-item-zzz");
  await expect(page.getByText("No items match these filters.").first()).toBeVisible();
  await page.getByLabel("Find an item").fill("");
});

test("explorer: switching to text shows the premise and hypothesis, other error types load", async ({ page }) => {
  await page.goto("/#/explorer");
  await page.getByRole("button", { name: "Text (ChaosNLI)" }).click();
  const detail = page.getByLabel("Selected item");
  await expect(detail.getByText("Premise.")).toBeVisible();
  await expect(detail.getByText("Hypothesis.")).toBeVisible();
  await expect(detail.getByRole("img")).toHaveCount(0);
  await expect(page.locator(".scatter-wrap circle[data-i]")).toHaveCount(800);

  for (const type of ["swap", "lazy", "plausible_wrong", "spammer"]) {
    await page.getByLabel("Error type").selectOption(type);
    await expect(page.locator(".scatter-wrap circle[data-i]")).toHaveCount(800);
    await expect(detail.getByRole("table", { name: "Votes on this item" })).toBeVisible();
  }
  await page.getByLabel("Seed").selectOption("2");
  await expect(detail.getByRole("heading", { level: 2 })).toContainText("Item");
});

test("the server compresses, revalidates and refuses path tricks", async ({ request }) => {
  const runs = await (await request.get("/api/runs")).json();
  expect(runs).toHaveLength(24);
  const run = runs[0].id;

  const points = await request.get(`/api/points?run=${encodeURIComponent(run)}`, {
    headers: { "Accept-Encoding": "gzip" },
  });
  expect(points.status()).toBe(200);
  expect(points.headers()["content-encoding"]).toBe("gzip");
  const tag = points.headers()["etag"];
  const again = await request.get(`/api/points?run=${encodeURIComponent(run)}`, {
    headers: { "If-None-Match": tag },
  });
  expect(again.status()).toBe(304);

  for (const bad of ["../..", "..%2F..%2F..", "C:/Windows", "/etc/passwd"]) {
    const res = await request.get(`/api/points?run=${bad}`);
    expect(res.status(), bad).toBe(404);
  }
  expect((await request.get(`/api/items?run=${encodeURIComponent(run)}&page_size=9999`)).status()).toBe(422);

  const asset = await request.get("/");
  const html = await asset.text();
  const js = html.match(/assets\/index-[^"]+\.js/)[0];
  const cached = await request.get(`/${js}`);
  expect(cached.headers()["cache-control"]).toContain("immutable");
});

test("pages load fast and the main bundle stays small", async ({ page }) => {
  const sizes = [];
  page.on("response", async (r) => {
    if (r.url().endsWith(".js")) sizes.push((await r.body()).length);
  });
  const t0 = Date.now();
  await page.goto("/#/explorer");
  await expect(page.getByLabel("Selected item")).toBeVisible();
  expect(Date.now() - t0, "explorer ready (ms)").toBeLessThan(5000);
  expect(sizes.reduce((a, b) => a + b, 0), "JS bytes on first Explorer view").toBeLessThan(250_000);
});
