"""Run the whole project locally on the mini profile: a laptop CPU, no GPU, no Kaggle.

    python scripts/make_mini.py --seed-cache   # once: freeze the subsets, reuse cached embeddings
    python scripts/run_mini.py                 # every step, then outputs_mini/report.md
    python scripts/run_mini.py --steps mini_5_compare   # one step

Resumable: finished runs are skipped, so after an interruption run the same command again.
Every run is dev stage (val and clean_dev only). These are mini results: a subset of the data
and a frozen-embedding classifier (amendment A15). They show the pipeline working end to end and
the direction of the effects; they are not the pre-registered results, and scripts/analyze.py
refuses them.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from labeltriage.data.splits import splits_dir
from labeltriage.eval.summary import aggregate, collect
from labeltriage.experiments import embed
from labeltriage.grid import run_grid

SWEEPS = splits_dir().parent / "sweep"
STEPS = (
    "mini_1_reproduce",
    "mini_2_injection",
    "mini_3_features_gate",
    "mini_4_triage_gate",
    "mini_5_compare",
    "mini_6_transfer_dev",
)

FEATURE_COLS = [
    "data",
    *(f"auroc/{g}" for g in ("g1", "g2", "g3", "g4", "g5", "all")),
    "auroc/minority_share_only",
    "auroc/leakage_shuffled",
]
TRIAGE_COLS = [
    "data",
    "triage/val_auroc",
    "triage/val_minority_share_only",
    "triage/val_calibration_error",
    "gate/auroc_pass",
    "gate/calibration_pass",
]
COMPARE_COLS = [
    "data",
    "method",
    "n_seeds",
    "val/acc_mean",
    "val/acc_std",
    "val/kl_mean",
    "clean_dev/acc_mean",
]
TRANSFER_COLS = [
    "transfer/dev_auroc",
    "transfer/dev_minority_share_only",
    "transfer/dev_gain_over_minority_lo",
    "transfer/dev_gain_over_minority_hi",
]


def embed_spec(cache_dir: Path) -> dict | None:
    """Frozen-model outputs not yet in ``cache_dir`` (make_mini --seed-cache copies most)."""
    missing = [
        b
        for b in ("cifar10h", "chaosnli", "varierr")
        if not all(p.exists() for p in embed.cache_paths(cache_dir, b).values())
    ]
    if not missing:
        return None
    return {
        "name": "mini_0_embed",
        "fixed": ["profile=mini", "experiment=embed", "method=majority_vote"],
        "axes": {"data": missing, "seed": [0]},
    }


def _table(df: pd.DataFrame, cols: list[str]) -> str:
    cols = [c for c in cols if c in df]
    return df[cols].to_markdown(index=False, floatfmt=".3f") if len(df) else "_no runs yet_"


def report(root: Path) -> str:
    runs = collect(root)
    if runs.empty:
        return "# Mini results\n\nNo finished runs.\n"
    by = {name: runs[runs["experiment"] == name] for name in runs["experiment"].unique()}
    lines = [
        "# Mini results (local CPU, dev stage)",
        "",
        "Mini profile (A15): a stratified subset of each dataset and an MLP head on frozen",
        "embeddings. Val and clean_dev only. Not the pre-registered results.",
        "",
        "## Reproduce: majority vote vs soft labels",
        "",
        _table(
            by.get("reproduce", pd.DataFrame()),
            ["data", "method", "val/acc", "val/kl", "clean_dev/acc", "train_seconds"],
        ),
        "",
        "## Injection model (held-out accuracy against the majority label)",
        "",
        _table(by.get("injection", pd.DataFrame()), ["data", "heldout_acc_vs_majority"]),
        "",
        "## Error detection per feature group (AUROC)",
        "",
        _table(
            by.get("features", pd.DataFrame()),
            FEATURE_COLS,
        ),
        "",
        "## LabelTriage error detection (calibrated, val votes)",
        "",
        _table(
            by.get("triage", pd.DataFrame()),
            TRIAGE_COLS,
        ),
    ]
    compare = by.get("pilot", pd.DataFrame())
    if len(compare):
        agg = aggregate(compare)
        lines += [
            "",
            "## Training on the triaged labels (mean over seeds)",
            "",
            _table(
                agg,
                COMPARE_COLS,
            ),
        ]
    lines += [
        "",
        "## Transfer to VariErr's real errors (dev half)",
        "",
        _table(
            by.get("transfer", pd.DataFrame()),
            TRANSFER_COLS,
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs_mini/grid")
    parser.add_argument("--cache", default="cache_mini")
    parser.add_argument("--report", default="outputs_mini/report.md")
    parser.add_argument("--steps", nargs="*", default=list(STEPS), choices=STEPS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    specs = [embed_spec(Path(args.cache))]
    specs += [OmegaConf.to_container(OmegaConf.load(SWEEPS / f"{s}.yaml")) for s in args.steps]
    for spec in filter(None, specs):
        start = time.perf_counter()
        counts = run_grid(OmegaConf.create(spec), args.out)
        minutes = (time.perf_counter() - start) / 60
        logging.info("%s: %s in %.1f min", spec["name"], counts, minutes)
        if counts["failed"]:
            logging.error("%s had failed runs; see the logs above. Stopping.", spec["name"])
            return 1

    text = report(Path(args.out))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(text, encoding="utf8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
