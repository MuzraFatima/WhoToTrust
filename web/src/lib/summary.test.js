import { describe, expect, it } from "vitest";
import { findMisses, gains, summarise } from "./summary.js";

const row = (dataset, error_type, method, val_acc, val_acc_std = 0.01) => ({
  dataset,
  error_type,
  method,
  val_acc,
  val_acc_std,
});

const compare = [
  row("cifar10h", "lazy", "majority_vote", 0.91),
  row("cifar10h", "lazy", "soft_labels", 0.82),
  row("cifar10h", "lazy", "labeltriage", 0.95),
  row("cifar10h", "swap", "majority_vote", 0.94),
  row("cifar10h", "swap", "soft_labels", 0.88),
  row("cifar10h", "swap", "labeltriage", 0.94),
  row("cifar10h", "spammer", "majority_vote", 0.95),
  row("cifar10h", "spammer", "soft_labels", 0.9),
  row("cifar10h", "spammer", "labeltriage", 0.9),
];

describe("gains", () => {
  const byType = Object.fromEntries(gains(compare, "cifar10h").map((g) => [g.type, g]));

  it("compares LabelTriage with the better of the two simple rules", () => {
    expect(byType.lazy.against).toBe("majority_vote");
    expect(byType.lazy.gain).toBeCloseTo(0.04);
  });

  it("calls a gap inside one seed deviation level, and outside it ahead or behind", () => {
    expect(byType.lazy.verdict).toBe("ahead");
    expect(byType.swap.verdict).toBe("level");
    expect(byType.spammer.verdict).toBe("behind");
  });

  it("ignores other datasets and error types with no LabelTriage run", () => {
    expect(gains(compare, "chaosnli")).toEqual([]);
    expect(gains([row("cifar10h", "lazy", "majority_vote", 0.9)], "cifar10h")).toEqual([]);
  });
});

describe("summarise", () => {
  it("counts wins and losses and reads the transfer row", () => {
    const s = summarise({
      n_runs: 9,
      train_seconds_total: 120,
      sections: { compare, transfer: [{ auroc: 0.8 }] },
    });
    expect(s.runs).toBe(9);
    expect(s.minutes).toBe(2);
    expect(s.images.ahead).toBe(1);
    expect(s.images.behind).toBe(1);
    expect(s.images.mean).toBeCloseTo((0.04 + 0 - 0.05) / 3);
    expect(s.text.mean).toBeNull();
    expect(s.transfer.auroc).toBe(0.8);
  });

  it("survives an empty result set", () => {
    expect(summarise({ n_runs: 0, sections: {} }).images.gains).toEqual([]);
    expect(summarise(null).runs).toBe(0);
  });
});

describe("findMisses", () => {
  it("lists weak feature groups and failed gates, and nothing when all pass", () => {
    const misses = findMisses({
      features: [{ dataset: "chaosnli", error_type: "lazy", g1: 0.58, g2: 0.8 }],
      triage: [
        { dataset: "chaosnli", error_type: "lazy", auroc: 0.83, auroc_pass: true, calibration_error: 0.07, calibration_pass: false },
        { dataset: "cifar10h", error_type: "swap", auroc: 0.97, auroc_pass: true, calibration_error: 0.02, calibration_pass: true },
      ],
    });
    expect(misses).toHaveLength(2);
    expect(misses[0]).toMatch(/G1 alone reaches 0.580.*text, lazy/);
    expect(misses[1]).toMatch(/calibration error 0.070/);
    expect(findMisses({})).toEqual([]);
  });
});
