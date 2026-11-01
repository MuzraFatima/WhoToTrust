"""Where a run came from: code commit, uncommitted changes, and which hypotheses were in force.

Recorded in every result so ``scripts/analyze.py`` can refuse runs that were not produced under
the frozen pre-registration.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from importlib import resources
from pathlib import Path

HYPOTHESES = Path("src/labeltriage/conf/prereg/hypotheses.yaml")


def _git(*args: str, cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def repo_root() -> Path | None:
    here = Path(__file__).resolve().parent
    root = _git("rev-parse", "--show-toplevel", cwd=here)
    return Path(root) if root else None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hypotheses_sha256() -> str:
    path = resources.files("labeltriage") / "conf" / "prereg" / "hypotheses.yaml"
    return sha256_bytes(path.read_bytes())


def current() -> dict[str, object]:
    """Provenance of the code running right now (``LT_GIT_COMMIT`` overrides the commit)."""
    root = repo_root()
    commit = os.environ.get("LT_GIT_COMMIT") or (
        _git("rev-parse", "HEAD", cwd=root) if root else None
    )
    dirty = None
    if root is not None:
        status = _git("status", "--porcelain", "--untracked-files=no", cwd=root)
        dirty = bool(status) if status is not None else None
    return {"git_commit": commit, "git_dirty": dirty, "hypotheses_sha256": hypotheses_sha256()}
