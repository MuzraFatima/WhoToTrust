import { ERROR_TYPES, METHODS } from "./format.js";

/** compare rows -> { errorType: { method: row } } for one dataset. */
export function compareByType(compare, dataset) {
  const out = {};
  for (const row of compare ?? []) {
    if (row.dataset !== dataset) continue;
    (out[row.error_type] ??= {})[row.method] = row;
  }
  return out;
}

/**
 * LabelTriage's validation accuracy minus the better of majority vote and soft labels, for each
 * error type. "Level" means the gap is inside one seed standard deviation of either side.
 */
export function gains(compare, dataset) {
  const byType = compareByType(compare, dataset);
  return ERROR_TYPES.filter((t) => byType[t]?.labeltriage).map((type) => {
    const m = byType[type];
    const rivals = ["majority_vote", "soft_labels"].filter((k) => m[k]);
    const best = rivals.reduce((a, b) => (m[a].val_acc >= m[b].val_acc ? a : b));
    const gain = m.labeltriage.val_acc - m[best].val_acc;
    const spread = Math.max(m.labeltriage.val_acc_std ?? 0, m[best].val_acc_std ?? 0);
    return {
      type,
      gain,
      against: best,
      verdict: gain > spread ? "ahead" : gain < -spread ? "behind" : "level",
    };
  });
}

const range = (xs) => (xs.length ? [Math.min(...xs), Math.max(...xs)] : null);

export function summarise(results) {
  const s = results?.sections ?? {};
  const out = { runs: results?.n_runs ?? 0, minutes: (results?.train_seconds_total ?? 0) / 60 };
  for (const [key, dataset] of [
    ["images", "cifar10h"],
    ["text", "chaosnli"],
  ]) {
    const g = gains(s.compare, dataset);
    out[key] = {
      gains: g,
      mean: g.length ? g.reduce((a, b) => a + b.gain, 0) / g.length : null,
      ahead: g.filter((x) => x.verdict === "ahead").length,
      behind: g.filter((x) => x.verdict === "behind").length,
      detection: range(
        (s.triage ?? []).filter((r) => r.dataset === dataset).map((r) => r.auroc),
      ),
    };
  }
  out.transfer = s.transfer?.[0] ?? null;
  out.methods = METHODS;
  return out;
}

const GROUP_KEYS = ["g1", "g2", "g3", "g4", "g5"];

/** Every gate the results miss, as plain sentences. Empty when nothing is missed. */
export function findMisses(sections) {
  const where = (r) => `${r.dataset === "chaosnli" ? "text" : "images"}, ${r.error_type}`;
  const out = [];
  for (const r of sections?.features ?? []) {
    for (const g of GROUP_KEYS) {
      if (r[g] != null && r[g] < 0.6) {
        out.push(`${g.toUpperCase()} alone reaches ${r[g].toFixed(3)} (under 0.60) for ${where(r)}`);
      }
    }
  }
  for (const r of sections?.triage ?? []) {
    if (!r.auroc_pass) out.push(`error detection AUROC ${r.auroc.toFixed(3)} (under 0.80) for ${where(r)}`);
    if (!r.calibration_pass) {
      out.push(`calibration error ${r.calibration_error.toFixed(3)} is over its bar for ${where(r)}`);
    }
  }
  return out;
}
