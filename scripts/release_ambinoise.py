"""Build the AmbiNoise release: every (base, K, rate, error type, seed) draw as Parquet.

    python scripts/release_ambinoise.py --out release/ambinoise --seeds 0 1 2

Writes one Parquet file per draw plus the dataset card as README.md. Plausible-wrong draws need
the injection rankings from sweep/injection.yaml and are skipped (with a note) when absent.
Uploading is a separate, deliberate step: see scripts/upload_ambinoise.py.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from labeltriage.ambinoise.errors import ERROR_TYPES
from labeltriage.ambinoise.injector import NoiseSpec, inject
from labeltriage.data import prepare

BASES = {"cifar10h": ("10h", 10), "chaosnli": ("nli", 3)}
# Hugging Face dataset-card header. `license: other` is deliberate: the owner sets the real
# licence after checking CIFAR-10H's and ChaosNLI's terms (see the card's Licensing section).
HF_FRONT_MATTER = """---
pretty_name: AmbiNoise
license: other
task_categories:
  - image-classification
  - text-classification
tags:
  - label-noise
  - annotator-disagreement
  - human-label-variation
  - crowdsourcing
configs:
  - config_name: cifar10h
    data_files: "10h/*.parquet"
  - config_name: chaosnli
    data_files: "nli/*.parquet"
---

"""

PUBLIC_COLUMNS = [
    "item_id",
    "annotator_id",
    "label",
    "split",
    "original_label",
    "is_error",
    "error_type",
    "annotator_propensity",
]


def build(
    out: Path,
    data_dir: str,
    cache_dir: str,
    seeds: list[int],
    ks=(3, 5, 10),
    rates=(0.1, 0.2, 0.4),
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"draws": [], "skipped": []}
    for base, (short, num_classes) in BASES.items():
        votes, _, _ = prepare.load_prepared(base, data_dir)
        dev = votes[votes["split"].isin(["train", "val"])]
        ranking_path = Path(cache_dir) / "injection" / f"{base}.parquet"
        ranking = None
        if ranking_path.exists():
            ranking = pd.read_parquet(ranking_path)
            ranking.columns = ranking.columns.astype(int)
        for k, rate, etype, seed in itertools.product(ks, rates, ERROR_TYPES, seeds):
            name = f"ambinoise_{short}_k{k}_r{rate}_{etype}_seed{seed}"
            if etype == "plausible_wrong" and ranking is None:
                manifest["skipped"].append(f"{name} (no injection ranking)")
                continue
            draw = inject(
                dev, num_classes, NoiseSpec(k=k, rate=rate, error_type=etype, seed=seed), ranking
            )
            path = out / short / f"{name}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            draw[PUBLIC_COLUMNS].to_parquet(path, index=False)
            manifest["draws"].append(
                {"name": name, "votes": len(draw), "error_share": float(draw["is_error"].mean())}
            )
    card = Path(__file__).resolve().parents[1] / "docs" / "ambinoise-card.md"
    (out / "README.md").write_text(
        HF_FRONT_MATTER + card.read_text(encoding="utf8"), encoding="utf8"
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="release/ambinoise")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args(argv)
    m = build(Path(args.out), args.data_dir, args.cache_dir, args.seeds)
    print(f"wrote {len(m['draws'])} draws, skipped {len(m['skipped'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
