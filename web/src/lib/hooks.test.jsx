import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useAsync, useDebounced } from "./hooks.js";

describe("useDebounced", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("follows the value only after it stops changing", () => {
    const { result, rerender } = renderHook(({ v }) => useDebounced(v, 300), { initialProps: { v: "a" } });
    rerender({ v: "ab" });
    rerender({ v: "abc" });
    act(() => vi.advanceTimersByTime(299));
    expect(result.current).toBe("a");
    act(() => vi.advanceTimersByTime(2));
    expect(result.current).toBe("abc");
  });
});

describe("useAsync", () => {
  it("reports loading, then data", async () => {
    const { result } = renderHook(() => useAsync(() => Promise.resolve(42), []));
    expect(result.current.loading).toBe(true);
    await act(async () => {});
    expect(result.current).toMatchObject({ data: 42, loading: false, error: null });
  });

  it("reports an error", async () => {
    const { result } = renderHook(() => useAsync(() => Promise.reject(new Error("boom")), []));
    await act(async () => {});
    expect(result.current.error.message).toBe("boom");
  });

  it("is already loading in the render where the inputs change, and keeps the old data", async () => {
    const { result, rerender } = renderHook(({ k }) => useAsync(() => Promise.resolve(k * 10), [k]), {
      initialProps: { k: 1 },
    });
    await act(async () => {});
    expect(result.current).toMatchObject({ data: 10, loading: false });
    rerender({ k: 2 });
    expect(result.current).toMatchObject({ data: 10, loading: true });
    await act(async () => {});
    expect(result.current).toMatchObject({ data: 20, loading: false });
  });

  it("ignores an answer that arrives after the inputs changed", async () => {
    let resolveSlow;
    const slow = new Promise((r) => (resolveSlow = r));
    const { result, rerender } = renderHook(({ k }) => useAsync(() => (k === 1 ? slow : Promise.resolve("new")), [k]), {
      initialProps: { k: 1 },
    });
    rerender({ k: 2 });
    await act(async () => {});
    await act(async () => resolveSlow("stale"));
    expect(result.current.data).toBe("new");
  });
});
