import { describe, expect, it } from "vitest";
import { num, pct, signed, voteSummary } from "./format.js";

describe("voteSummary", () => {
  const classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"];

  it("names the classes and counts repeats, most common first", () => {
    expect(voteSummary("2 8 8 8 9", classes)).toBe("ship ×3, bird, truck");
  });

  it("breaks ties by class number and falls back to the number without names", () => {
    expect(voteSummary("1 0 1 0", classes)).toBe("plane ×2, car ×2");
    expect(voteSummary("3 3 1")).toBe("3 ×2, 1");
  });

  it("copes with no votes", () => {
    expect(voteSummary("", classes)).toBe("");
  });
});

describe("number formatting", () => {
  it("shows a dash for missing values and a real minus sign for negatives", () => {
    expect(num(null)).toBe("–");
    expect(pct(undefined)).toBe("–");
    expect(signed(-0.0123)).toBe("−0.012");
    expect(signed(0.5)).toBe("+0.500");
    expect(pct(0.1234)).toBe("12.3%");
  });
});
