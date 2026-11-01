import shutil
import subprocess

import pandas as pd
import pytest

from labeltriage.eval import guard
from labeltriage.utils import provenance

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


def _git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A tiny repository: one commit before the tag, the tagged commit, one commit after."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    _git(tmp_path, "config", "core.autocrlf", "false")
    hyp = tmp_path / provenance.HYPOTHESES
    hyp.parent.mkdir(parents=True)
    (tmp_path / "x.txt").write_text("0")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "before")
    before = _git(tmp_path, "rev-parse", "HEAD")
    package_file = provenance.resources.files("labeltriage") / "conf" / "prereg" / "hypotheses.yaml"
    hyp.write_bytes(package_file.read_bytes())
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "freeze")
    _git(tmp_path, "tag", guard.TAG)
    (tmp_path / "x.txt").write_text("1")
    _git(tmp_path, "commit", "-qam", "after")
    after = _git(tmp_path, "rev-parse", "HEAD")
    return guard.Git(tmp_path), before, after


def _run(name, commit, dirty=False, sha=None):
    return {
        "run": name,
        "prov.git_commit": commit,
        "prov.git_dirty": dirty,
        "prov.hypotheses_sha256": sha or provenance.hypotheses_sha256(),
    }


def test_runs_after_the_tag_with_frozen_hypotheses_pass(repo):
    git, _, after = repo
    rep = guard.check(pd.DataFrame([_run("good", after)]), git)
    assert rep.ok and rep.accepted == 1


def test_bad_runs_are_rejected_with_reasons(repo):
    git, before, after = repo
    runs = pd.DataFrame(
        [
            _run("early", before),
            _run("dirty", after, dirty=True),
            _run("other-hypotheses", after, sha="0" * 64),
            _run("no-commit", None),
        ]
    )
    rep = guard.check(runs, git)
    assert not rep.ok and rep.accepted == 0
    text = "\n".join(rep.rejected)
    assert "not at or after" in text
    assert "uncommitted" in text
    assert "hypotheses differ" in text
    assert "no code commit" in text


def test_missing_tag_fails(tmp_path):
    _git(tmp_path, "init", "-q")
    rep = guard.check(pd.DataFrame(), guard.Git(tmp_path))
    assert not rep.ok and "does not exist" in rep.problems[0]


def test_every_run_records_provenance(tmp_path):
    from hydra import compose, initialize_config_module

    from labeltriage import run

    with initialize_config_module(config_module="labeltriage.conf", version_base="1.3"):
        cfg = compose(config_name="config")
    result = run.execute(cfg, tmp_path)
    prov = result["provenance"]
    assert prov["hypotheses_sha256"] == provenance.hypotheses_sha256()
    assert set(prov) == {"git_commit", "git_dirty", "hypotheses_sha256"}
