"""Run a sweep, skipping finished runs.

python scripts/run_grid.py src/labeltriage/conf/sweep/smoke.yaml --out outputs/grid
python scripts/run_grid.py SPEC --out DIR --shard 1/2 --max-hours 8
python scripts/run_grid.py SPEC --out DIR -- stage=final data.num_items=500

Exit code 0 means everything in the (shard of the) sweep is finished; 2 means some runs were
deferred by the time budget and the same command should be run again in a new session.
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
    parser.add_argument("--shard", help="run only shard i of n, written as i/n")
    parser.add_argument("--max-hours", type=float, help="start no new run after this many hours")
    parser.add_argument("extra", nargs="*", help="overrides applied to every run")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None
    counts = run_grid(OmegaConf.load(args.spec), args.out, args.extra, shard, args.max_hours)
    logging.info("grid done: %s", counts)
    if counts["failed"]:
        return 1
    return 2 if counts["deferred"] else 0


if __name__ == "__main__":
    sys.exit(main())
