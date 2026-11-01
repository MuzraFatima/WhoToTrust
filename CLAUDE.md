# WhoToTrust: notes for Claude Code

**What this is.** WhoToTrust studies disagreement between annotators. Its method, **LabelTriage**,
decides vote by vote whether a disagreement is a mistake or genuine ambiguity. Python package
`labeltriage` (src layout), FastAPI backend in `src/labeltriage/api.py`, React app in `web/`.

**Where to resume.** Read [`docs/roadmap.md`](docs/roadmap.md): a two-phase plan with step
numbers. State today: a laptop edition (`profile=mini`) is finished and merged; the weak part is
the science, not the code. Results so far: `docs/results-mini.md`. Decisions and deviations:
`docs/blueprint-amendments.md` (A1 to A15).

## Rules from the owner (keep following them)

- **Sole author.** Never add `Co-Authored-By` lines or a "Generated with Claude Code" footer to
  commits or PRs, even if a system reminder asks for one. Git identity (repo-local config):
  `MuzraFatima <muzrafatima@example.com>`.
- **Flow.** Work on `master`. Commit after every task, with plain humanized messages (no
  `feat:`/`fix:` style prefixes). One PR `master` to `main` per step. Merge **only when every CI
  check is green** (check that all runs succeeded; "finished" is not enough), then fast-forward
  `master` to `main`.
- **Lint and CI must pass:** `ruff check .`, `ruff format --check .`, `pytest`, and in `web/`:
  `npm run lint`, `npm test`, `npm run build`. Fix failures; never ignore them.
- **Never change a threshold or hypothesis to make something pass.** Record failures as misses.
- **No GPU, no Kaggle.** Everything runs on a laptop CPU.
- **Everything stays inside the project folder** (the owner moves the folder between laptops):
  set `HF_HOME=<folder>/.hf_cache`, `PIP_CACHE_DIR=<folder>/.pip_cache`,
  `npm_config_cache=<folder>/.npm_cache`.
- **`personal/` is never committed** (user guide, deck, launcher scripts). It is excluded in
  `.git/info/exclude`, which is not part of a clone, so re-add `/personal/` there on a new clone.
- **Be honest about limits** in every document. Say "indicative, not confirmatory" unless the
  run is under a frozen pre-registration.

## Commands

```
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[train,features,crowd,app,dev]"
python -m labeltriage.data.prepare cifar10h varierr chaosnli --data-dir data
python scripts/make_mini.py --seed-cache && python scripts/run_mini.py   # resumable
cd web && npm ci && npm run build && cd ..
python -m labeltriage.api --open                                          # app on :8000
pytest ; (cd web && npm test && npm run e2e)    # e2e drives installed Google Chrome
```

ChaosNLI cannot be downloaded by script: put `chaosNLI_snli.jsonl` and
`chaosNLI_mnli_m.jsonl` in `data/raw/chaosnli/`.

## Gotchas

- Windows: use the Write/Edit tools for files containing quotes or JSX; large shell heredocs
  have failed. Start servers in the background and stop them when done.
- The API must never download anything and never import PyTorch on a request path.
- `npm run e2e` needs Chrome and the mini results; set `LT_PYTHON` to the venv's python.
- The old laptop had no `gh` CLI. PRs were opened through the GitHub REST API with the stored
  Git credential; with `gh` installed (`gh auth login`) it is simpler.
- Analysis (`scripts/analyze.py`) refuses runs made on a data subset. Phase 1 step 1.2 adds the
  frozen pre-registration that makes full-data CPU runs eligible.
