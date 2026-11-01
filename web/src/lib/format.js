export const pct = (x, digits = 1) => (x == null ? "–" : `${(x * 100).toFixed(digits)}%`);
export const num = (x, digits = 3) => (x == null ? "–" : x.toFixed(digits));
export const signed = (x, digits = 3) =>
  x == null ? "–" : `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(digits)}`;

export const DATASETS = { cifar10h: "Images (CIFAR-10H)", chaosnli: "Text (ChaosNLI)" };
export const ERROR_TYPES = ["spammer", "swap", "lazy", "plausible_wrong"];
export const ERROR_LABELS = {
  spammer: "Spammer",
  swap: "Swap",
  lazy: "Lazy",
  plausible_wrong: "Plausible wrong",
};
export const METHODS = ["majority_vote", "soft_labels", "labeltriage", "oracle"];
export const METHOD_LABELS = {
  majority_vote: "Majority vote",
  soft_labels: "Soft labels",
  labeltriage: "LabelTriage",
  oracle: "Oracle",
};
export const METHOD_COLOURS = {
  majority_vote: "var(--m-majority)",
  soft_labels: "var(--m-soft)",
  labeltriage: "var(--m-triage)",
  oracle: "var(--m-oracle)",
};
export const ROUTE_NOTES = {
  clear: "Votes agree. Train on the hard label.",
  ambiguous: "Real disagreement. Keep it as a soft label.",
  unreliable: "A vote looks like an error. Down-weight the item.",
};
export const label = (map, key) => map[key] ?? key;

/** "2 8 8 8 9" with class names -> "ship ×3, bird, truck" (most common first). */
export function voteSummary(labels, classes = []) {
  const counts = new Map();
  for (const token of String(labels).split(" ").filter(Boolean)) {
    counts.set(token, (counts.get(token) ?? 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || Number(a[0]) - Number(b[0]))
    .map(([token, n]) => `${classes[Number(token)] ?? token}${n > 1 ? ` ×${n}` : ""}`)
    .join(", ");
}
