"""Summarise a grid directory into per-run and per-method CSV tables.

python scripts/summarize.py outputs/grid --out outputs/summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from labeltriage.eval.summary import aggregate, collect


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--out", default="summary.csv")
    args = parser.parse_args(argv)

    runs = collect(args.root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out.with_name(out.stem + "_runs.csv"), index=False)
    if runs.empty:
        print("no finished runs found")
        return
    table = aggregate(runs)
    table.to_csv(out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
