"""Run a sweep, skipping finished runs.

python scripts/run_grid.py src/labeltriage/conf/sweep/smoke.yaml --out outputs/grid
python scripts/run_grid.py SPEC --out DIR -- stage=final data.num_items=500
"""

from __future__ import annotations

import argparse
import logging
import sys

from omegaconf import OmegaConf

from labeltriage.grid import run_grid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", help="path to a sweep YAML")
    parser.add_argument("--out", default="outputs/grid", help="root directory for run outputs")
    parser.add_argument("extra", nargs="*", help="overrides applied to every run")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    counts = run_grid(OmegaConf.load(args.spec), args.out, args.extra)
    logging.info("grid done: %s", counts)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
