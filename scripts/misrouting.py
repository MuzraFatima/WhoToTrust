"""Error analysis of misrouted items across every LabelTriage run under a folder.

python scripts/misrouting.py outputs/grid --classes 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from labeltriage.explorer import misrouting


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--classes", type=int, default=10)
    args = parser.parse_args(argv)

    rows = []
    for path in sorted(Path(args.root).rglob("scored_votes.parquet")):
        scored = pd.read_parquet(path)
        if "is_error" not in scored:
            continue  # real noise: no oracle to compare with
        rep = misrouting(scored, args.classes)
        rows.append(
            {
                "run": str(path.parent.relative_to(args.root)),
                "misrouted_share": rep["misrouted_share"],
                **{f"misrouted_{k}": v for k, v in rep["misrouted_by_error_type"].items()},
            }
        )
    if not rows:
        print("no scored LabelTriage runs with error labels found")
        return 1
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
