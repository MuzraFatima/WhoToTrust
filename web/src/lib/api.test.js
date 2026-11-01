import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { clearApiCache, getJSON, imageUrl } from "./api.js";

const ok = (body) => Promise.resolve({ ok: true, json: () => Promise.resolve(body) });

beforeEach(() => {
  clearApiCache();
  vi.stubGlobal("fetch", vi.fn());
});
afterEach(() => vi.unstubAllGlobals());

describe("getJSON", () => {
  it("builds the query string, repeats array values and drops empty ones", async () => {
    fetch.mockReturnValue(ok({}));
    await getJSON("/api/items", { run: "a/b", route: ["clear", "unreliable"], q: "", only_errors: false, page: 2 });
    expect(fetch.mock.calls[0][0]).toBe("/api/items?run=a%2Fb&route=clear&route=unreliable&page=2");
  });

  it("answers a repeat request from memory and shares a request in flight", async () => {
    fetch.mockReturnValue(ok({ n: 1 }));
    const [a, b] = await Promise.all([getJSON("/api/runs"), getJSON("/api/runs")]);
    await getJSON("/api/runs");
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(a).toBe(b);
  });

  it("throws the server's message and does not cache the failure", async () => {
    fetch.mockReturnValueOnce(
      Promise.resolve({ ok: false, status: 404, statusText: "Not Found", json: () => Promise.resolve({ error: "no scored run" }) }),
    );
    await expect(getJSON("/api/points", { run: "x" })).rejects.toThrow("no scored run");
    fetch.mockReturnValueOnce(ok({ fine: true }));
    await expect(getJSON("/api/points", { run: "x" })).resolves.toEqual({ fine: true });
    expect(fetch).toHaveBeenCalledTimes(2);
  });
});

describe("imageUrl", () => {
  it("encodes the run and item", () => {
    expect(imageUrl("dev/r", "a#1")).toBe("/api/image?run=dev%2Fr&item=a%231");
  });
});
