"""Run bookkeeping that makes grids resumable across Kaggle / Colab sessions.

A run directory holds ``config.yaml``, ``result.json`` and, once finished, a ``DONE`` marker.
Grid launchers skip any run whose directory already has the marker.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DONE = "DONE"
RESULT = "result.json"


class StageError(RuntimeError):
    """Raised when a dev-stage run tries to read a test split."""


def require_stage(stage: str, split: str) -> None:
    """Refuse test-split access unless the run is in the ``final`` stage."""
    if split == "test" and stage != "final":
        raise StageError(
            "test splits are locked until Phase 6; rerun with stage=final once pre-registered"
        )


def is_done(run_dir: str | Path) -> bool:
    return (Path(run_dir) / DONE).exists()


def write_result(run_dir: str | Path, result: dict[str, Any]) -> Path:
    """Write ``result.json`` and mark the run finished. The marker is written last."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / RESULT
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True, default=float))
    tmp.replace(path)
    (run_dir / DONE).touch()
    return path


def read_result(run_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(run_dir) / RESULT).read_text())
