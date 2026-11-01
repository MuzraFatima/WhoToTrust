"""Refuse to analyse runs that were not produced under the frozen pre-registration.

A final-stage run is accepted only if:

1. the ``prereg-v1`` tag exists;
2. the run recorded the same hypotheses file as the one at the tag;
3. the run's code commit is the tagged commit or a descendant of it;
4. the run's working copy had no uncommitted changes;
5. the run used the full data, not a subset such as the mini profile (A15).

The working copy's own hypotheses file must also still match the tag.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from labeltriage.utils import provenance

TAG = "prereg-v1"


@dataclass
class GuardReport:
    ok: bool
    problems: list[str] = field(default_factory=list)
    accepted: int = 0
    rejected: list[str] = field(default_factory=list)


class Git:
    """Thin wrapper so tests can stand in a fake repository."""

    def __init__(self, root: Path | None):
        self.root = root

    def tag_commit(self, tag: str) -> str | None:
        return provenance._git("rev-parse", f"{tag}^{{commit}}", cwd=self.root)

    def file_at(self, rev: str, path: Path) -> bytes | None:
        """Exact bytes of ``path`` at ``rev``, so its hash matches the file on disk."""
        try:
            out = subprocess.run(
                ["git", "show", f"{rev}:{path.as_posix()}"],
                cwd=self.root,
                capture_output=True,
                check=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout

    def is_ancestor(self, ancestor: str, commit: str) -> bool:
        return (
            provenance._git("merge-base", "--is-ancestor", ancestor, commit, cwd=self.root)
            is not None
        )


def frozen_hypotheses_sha(git: Git) -> str | None:
    data = git.file_at(TAG, provenance.HYPOTHESES)
    return None if data is None else provenance.sha256_bytes(data)


def check(runs: pd.DataFrame, git: Git | None = None) -> GuardReport:
    """``runs`` holds final-stage rows with ``run``, ``prov.git_commit``, ``prov.git_dirty`` and
    ``prov.hypotheses_sha256`` columns (see :func:`labeltriage.eval.summary.collect`)."""
    git = git or Git(provenance.repo_root())
    rep = GuardReport(ok=True)
    tag_commit = git.tag_commit(TAG)
    if tag_commit is None:
        rep.ok = False
        rep.problems.append(f"the {TAG} tag does not exist: the pre-registration is not frozen")
        return rep
    frozen = frozen_hypotheses_sha(git)
    if frozen != provenance.hypotheses_sha256():
        rep.ok = False
        rep.problems.append("hypotheses.yaml in this checkout differs from the tagged version")

    for _, r in runs.iterrows():
        why = []
        if r.get("prov.hypotheses_sha256") != frozen:
            why.append("hypotheses differ from the tag")
        commit = r.get("prov.git_commit")
        if not isinstance(commit, str) or not commit:
            why.append("no code commit recorded")
        elif not git.is_ancestor(tag_commit, commit):
            why.append(f"commit {commit[:10]} is not at or after {TAG}")
        if r.get("prov.git_dirty") is not False:
            why.append("uncommitted changes (or unknown)")
        subset = r.get("subset")
        if isinstance(subset, str) and subset:
            why.append(f"{subset}-profile run on a data subset")
        if why:
            rep.rejected.append(f"{r.get('run', '?')}: {'; '.join(why)}")
        else:
            rep.accepted += 1
    if rep.rejected:
        rep.ok = False
        rep.problems.append(f"{len(rep.rejected)} final run(s) not produced under {TAG}")
    return rep
