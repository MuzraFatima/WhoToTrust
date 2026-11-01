"""Verdicts on H1–H4 from finished Phase 6 / 7 grids.

    python scripts/analyze.py outputs/grid --out docs/results-final.md

Reads every finished run under the grid root, applies the pre-registered tests and writes a
Markdown report. Runs whose config is not stage=final are ignored, so pilot or dev runs never
leak into the verdicts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from labeltriage.data.splits import splits_dir
from labeltriage.eval import hypotheses as H
from labeltriage.eval.summary import collect


def _fmt(x) -> str:
    if isinstance(x, tuple):
        return "[" + ", ".join(_fmt(v) for v in x) + "]"
    if isinstance(x, float | np.floating):
        return f"{x:.4f}"
    return str(x)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--out", default="docs/results-final.md")
    args = parser.parse_args(argv)

    hyp = OmegaConf.to_container(OmegaConf.load(splits_dir().parent / "prereg" / "hypotheses.yaml"))
    runs = collect(args.root)
    if runs.empty:
        print("no finished runs")
        return 1
    runs = runs[runs["stage"] == "final"]
    main_runs = runs[runs["experiment"] == "main"]
    harm = runs[runs["experiment"] == "harm"]
    transfer_rows = runs[runs["experiment"] == "transfer"]
    transfer = transfer_rows.iloc[0].to_dict() if len(transfer_rows) else None

    results = {
        "H1": H.h1(main_runs, transfer, hyp),
        "H2": H.h2(main_runs, hyp),
        "H3": H.h3(harm, hyp),
        "H4": H.h4(main_runs, hyp),
    }
    rejected = H.holm_family(results)

    lines = [
        "# Final results",
        "",
        "| Hypothesis | Verdict | Holm-significant | Key numbers |",
        "|---|---|---|---|",
    ]
    for name, res in results.items():
        keys = {k: v for k, v in res.items() if k not in ("verdict", "table")}
        nums = "; ".join(f"{k} = {_fmt(v)}" for k, v in keys.items())
        sig = {True: "yes", False: "no"}.get(rejected.get(name), "—")
        lines.append(f"| {name} | **{res['verdict']}** | {sig} | {nums} |")
    if "table" in results["H2"] and len(results["H2"]["table"]):
        lines += [
            "",
            "## H2 per setting (clean-test accuracy gain over the better global rule)",
            "",
        ]
        lines.append(results["H2"]["table"].to_markdown(index=False, floatfmt=".4f"))
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
