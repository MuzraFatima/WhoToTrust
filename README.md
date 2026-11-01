# WhoToTrust — LabelTriage

**When Labels Disagree: learning to separate annotation error from genuine ambiguity, and training on each differently.**

When several annotators label the same item and disagree, existing pipelines either delete the
minority votes (noisy-label learning) or keep all of them as a soft label (human label variation).
LabelTriage decides *per annotation* whether a disagreement is a mistake or a genuine ambiguity,
then routes each item to a hard, soft or down-weighted training target.

The project has three parts:

- **AmbiNoise** — a benchmark built from real many-annotator vote distributions (CIFAR-10H, ChaosNLI)
  with controlled, labelled errors injected on top.
- **LabelTriage** — a calibrated per-annotation triage model plus a three-route target builder.
- **Harm analysis** — counterfactual retraining that measures which kinds of disagreement hurt
  accuracy and which hurt calibration.

The full research design lives in [`docs/`](docs/).

## Repository layout

```
src/labeltriage/
  conf/         Hydra configs (data, method, experiment, sweep)
  data/         dataset loaders and the shared annotation store
  utils/        seeding, run bookkeeping, optional W&B tracking
  run.py        single entry point for one run
  grid.py       resumable sweep runner
scripts/        grid launcher and Kaggle / Colab bootstrap
tests/          pytest suite (runs on CPU, no downloads)
docs/           blueprint, amendments and build plan
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # core + lint/test tools
pip install -e ".[train,features,crowd]"   # add when you need models and baselines
pre-commit install

pytest                             # unit tests
python -m labeltriage.run seed=1   # one smoke run, written under outputs/
python scripts/run_grid.py src/labeltriage/conf/sweep/smoke.yaml --out outputs/grid
```

Environment variables `LT_DATA_DIR`, `LT_CACHE_DIR` and `LT_OUTPUT_DIR` move data, caches and run
outputs. On Kaggle or Colab, `scripts/kaggle_bootstrap.sh SPEC` clones the repo, installs it and
runs a sweep. Re-running the same command after a timeout picks up where it stopped.

Test splits are locked: any run that asks for one fails unless it sets `stage=final`, which is
reserved for Phase 6, after the pre-registration tag.

## Preparing data

```bash
python -m labeltriage.data.prepare cifar10h cifar10n varierr chaosnli --data-dir data
```

Everything is fetched and checksum-verified automatically except ChaosNLI, whose Dropbox link
blocks scripts. Download `chaosNLI_v1.0.zip` from the
[ChaosNLI repository](https://github.com/easonnie/ChaosNLI) in a browser and save it as
`data/raw/chaosnli/chaosNLI_v1.0.zip`. Until then, an MNLI-only mirror is used.

## Running on Kaggle

In a GPU notebook (T4), with internet switched on:

```bash
!curl -sSLO https://raw.githubusercontent.com/MuzraFatima/WhoToTrust/main/scripts/kaggle_bootstrap.sh
!CHAOSNLI_ZIP=/kaggle/input/chaosnli/chaosNLI_v1.0.zip bash kaggle_bootstrap.sh src/labeltriage/conf/sweep/reproduce.yaml
```

Then, for Phase 2, run `sweep/injection.yaml` (writes the plausible-wrong ranking) followed by
`sweep/oracle_gate.yaml` the same way.

The first command prepares the data, runs the Phase 1 reproduction (majority vote vs soft labels on every
dataset, one seed) and prints a summary with `train_seconds` per run for the compute check.

## GPU run order

Each line is one `kaggle_bootstrap.sh <sweep>` call, using sweeps in `src/labeltriage/conf/sweep/`:

1. `reproduce.yaml`: Phase 1 baselines and compute check
2. `injection.yaml`, then `embed.yaml`: frozen-model caches for plausible-wrong errors and features
3. `oracle_gate.yaml`, `features_gate.yaml`, `triage_gate.yaml`: Phase 2–4 gates
4. `pilot.yaml` and `tune_router.yaml`, then
   `python scripts/preregister.py outputs/pilot_runs.csv --tune outputs/tune_runs.csv`.
   Commit `docs/preregistration.md` and `git tag prereg-v1` **before** any `stage=final` run.
5. Phase 6: `main_10h.yaml`, `main_10h_headline_seeds.yaml`, `k_sweep.yaml`, `main_nli.yaml`,
   `cifar10n.yaml`, `transfer_final.yaml`, then `python scripts/analyze.py outputs/grid`.

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Repo, CI, configs, run plumbing | done |
| 1 | Data loaders, trainer, evaluation, baseline reproduction | code done; GPU numbers pending |
| 2 | AmbiNoise benchmark | code done; injection model and oracle gate pending GPU |
| 3 | Feature library | done on NLI; image side pending GPU |
| 4 | Triage model and router | code done; first VariErr transfer is a miss (see docs/results.md) |
| 5 | Baselines, pilot, pre-registration | code done; pilot and prereg-v1 tag pending GPU |
| 6 | Main experiments | grids and analysis ready; runs after prereg-v1 on GPU |
| 7 | Harm analysis, ablations, explorer | — |
| 8 | Paper, release, defence | — |
