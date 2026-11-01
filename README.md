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

## Reproduce a headline number

On any machine with Python 3.11+, from a fresh clone, on CPU (about 40 minutes):

```bash
bash scripts/reproduce_headline.sh
```

It prints the VariErr dev-half transfer AUROC of the LabelTriage transfer model (expected
**0.833** with the official ChaosNLI files, 0.756 with only the MNLI mirror) next to the
untrained minority-share reference (expected **0.857**). All results so
far, including the ones that went against the method, are in
[`docs/results.md`](docs/results.md).

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
blocks scripts. Download it from the [ChaosNLI repository](https://github.com/easonnie/ChaosNLI)
in a browser and put `chaosNLI_snli.jsonl` and `chaosNLI_mnli_m.jsonl` in `data/raw/chaosnli/`
(the unextracted `chaosNLI_v1.0.zip` there works too). Both files are checksum-verified.
`chaosNLI_alphanli.jsonl` is a different task and is not used. Without the official files, an
MNLI-only mirror is used, with a warning.

## Running locally on a laptop: the mini profile

No GPU and no Kaggle needed. The mini profile (amendment A15) runs every stage of the project
on an ordinary CPU, using only the data in this folder:

- **Smaller data.** A fixed, stratified subset of each dataset: 1,500 of the 10,000 CIFAR-10H
  images, 1,200 of the 2,613 ChaosNLI items, and all 500 VariErr items. Each item keeps its
  original train / val / test split. The subsets are committed in `conf/splits/mini/`, and
  `data/` is never changed.
- **A lighter classifier.** A small MLP head on frozen embeddings replaces fine-tuning
  ResNet-18 / DeBERTa: DINOv2-small for images, and for NLI a small encoder already trained
  on NLI, `cross-encoder/nli-MiniLM2-L6-H768`. Embeddings are computed once and cached, so a
  training run takes seconds.
- **Its own folders.** `cache_mini/` and `outputs_mini/`, so nothing mixes with full runs.
  `run_mini.py` also downloads pretrained models into the project's `.hf_cache/` (about
  0.8 GB), unless `HF_HOME` points somewhere else.

```bash
pip install -e ".[train,features,crowd]"
python -m labeltriage.data.prepare cifar10h varierr chaosnli --data-dir data   # if not done
python scripts/make_mini.py --seed-cache   # subsets + reuse embeddings already in cache/
python scripts/run_mini.py                 # all steps, then outputs_mini/report.md
```

`run_mini.py` is resumable: if it is interrupted, run it again and it continues. Any single
run also works with `profile=mini`, for example
`python -m labeltriage.run profile=mini experiment=reproduce data=chaosnli method=soft_labels`.

Mini runs are **dev stage only** (val and clean_dev). Their test items belong to the real test
split, which stays sealed until the `prereg-v1` tag. Mini numbers show that the pipeline works
and which way the effects go. They are not the pre-registered results, and
`scripts/analyze.py` refuses them. The latest local results are in
[`docs/results-mini.md`](docs/results-mini.md).

## Running on Kaggle: Tier 1 run order

Tier 1 (about 23 GPU-hours) is the approved minimum that supports H1–H4. Its scope and cut
order are in [`docs/prereg-v1.md`](docs/prereg-v1.md); Tier 2 waits.

**One-time setup.** Create a Kaggle notebook with a T4 GPU and internet on. Upload
`chaosNLI_snli.jsonl` and `chaosNLI_mnli_m.jsonl` as a Kaggle dataset named `chaosnli`. In each
session, run:

```bash
!curl -sSLO https://raw.githubusercontent.com/MuzraFatima/WhoToTrust/main/scripts/kaggle_bootstrap.sh
!CHAOSNLI_DIR=/kaggle/input/chaosnli RESUME_FROM=/kaggle/input/<previous-session-output>   bash kaggle_bootstrap.sh src/labeltriage/conf/sweep/<sweep>.yaml
```

Leave out `RESUME_FROM` in session 1. From session 2 on, attach the previous session's saved
output as an input and point `RESUME_FROM` at it, so finished runs and caches carry over.

**Every session ends the same way:** "Save Version" (so `outputs/` and `cache/` survive), then
**turn the session off** (Run → Stop session) so it stops using GPU quota. Each sweep is
resumable: if a session dies or prints `TIME BUDGET REACHED`, run the same command again in a
new session with `RESUME_FROM`. Finished runs are skipped. Each session is sized to stay under
8 hours (`MAX_HOURS=8`, inside Kaggle's 12-hour limit).

| Session | Sweeps, in order | Est. GPU-h | Then |
|---|---|---|---|
| **S1** | `t1_s1_reproduce`, then the GPU smoke test (`SMOKE=1`, no sweep argument) | 0.3 + ~0.6 | **Stop. Report the measured `train_seconds` per run** (in `outputs/summary_runs.csv`) **and the smoke test's PASS/FAIL list** before continuing. |
| S2 | `injection`, `t1_s2_embed`, `features_gate`, `triage_gate`, `t1_s2_transfer_dev` | ~1 + CPU | **Report `transfer/dev_auroc`** (the VariErr dev re-measure with plausible-wrong errors). |
| S3 | `t1_s3_oracle_gate`, `t1_s3_pilot`, `tune_router` | 5.0 | Run `python scripts/preregister.py outputs/summary_runs.csv --tune outputs/summary_runs.csv`, fill every "to fill" in `docs/prereg-v1.md`, commit. **The owner reads it and tags `prereg-v1`.** |
| S4 | `t1_s4_main_core` with `SHARD=1/2` | 4.8 | — |
| S5 | `t1_s4_main_core` with `SHARD=2/2` | 4.8 | — |
| S6 | `t1_s6_headline_seeds`, `t1_s6_strong_baselines`, `harm`, `transfer_final` | 6.9 | `python scripts/analyze.py outputs/grid`, `python scripts/misrouting.py outputs/grid` |

**The 1.5× rule.** The plan assumes about 240 s per CIFAR-10H run and about 240 s per ChaosNLI
run. If session 1 measures more than 1.5× that (360 s) for either, **stop and replan before
session 2**. Cuts follow the order in `docs/prereg-v1.md`: step 9 first, then the pilot's
second seed. The same rule applies to any later session that runs more than 1.5× over its
estimate.

The smoke test runs every GPU code path once, for one epoch (training, Co-teaching, the
injection model, the embedding models, LabelTriage, CROWDLAB, Confident Learning, harm targets
and the DINOv2 backbone). It writes only to `outputs_smoke/` and `cache_smoke/`, so nothing in it
counts as a real run. Any FAIL is fixed before session 2.

Nothing with `stage=final` (sessions S4–S6) may start before the `prereg-v1` tag exists.
`scripts/analyze.py` enforces this. Every run records its code commit, whether the code had
uncommitted changes, and a hash of the hypotheses file. Verdicts are only written if the tag
exists and every final run came from clean code at or after it, under the tagged hypotheses.
`--draft` gives a preview stamped DRAFT.

## Disagreement Explorer

```bash
pip install -e ".[demo]"
streamlit run app/explorer.py
```

It reads the `scored_votes.parquet` every LabelTriage run saves. Each item is plotted by vote
entropy against its most suspicious vote, coloured by route. Clicking one shows the image, the
votes, and plain-language reasons for each vote's score.

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Repo, CI, configs, run plumbing | done |
| 1 | Data loaders, trainer, evaluation, baseline reproduction | code done; GPU numbers pending |
| 2 | AmbiNoise benchmark | code done; injection model and oracle gate pending GPU |
| 3 | Feature library | done on NLI; image side pending GPU |
| 4 | Triage model and router | code done; first VariErr transfer is a miss (see docs/results.md) |
| 5 | Baselines, pilot, pre-registration | code done; Tier 1 plan in docs/prereg-v1.md, tag pending |
| 6 | Main experiments | grids and analysis ready; runs after prereg-v1 on GPU |
| 7 | Harm analysis, ablations, explorer | code done; runs after prereg-v1 on GPU |
| 8 | Paper, release, defence | release tooling, repro script and paper draft done; final numbers pending GPU |

## Credits

This project builds on open data and tools. Thank you to their authors:

- **CIFAR-10H**: Peterson, Battleday, Griffiths and Russakovsky, *Human Uncertainty Makes
  Classification More Robust*, ICCV 2019.
- **CIFAR-10N**: Wei, Zhu, Cheng, Liu, Niu and Liu, *Learning with Noisy Labels Revisited*,
  ICLR 2022.
- **ChaosNLI**: Nie, Zhou and Bansal, *What Can We Learn from Collective Human Opinions on
  Natural Language Inference Data?*, EMNLP 2020.
- **VariErr NLI**: Weber-Genzel, Peng, de Marneffe and Plank, ACL 2024. It is the closest prior
  work, and it is used here only as a transfer test.
- **CIFAR-10**, **CIFAR-10.1** (Recht et al.) and **CIFAR-10-C** (Hendrycks and Dietterich).
- **cleanlab** (Confident Learning, CROWDLAB) and the Dawid–Skene model (Dawid and Skene, 1979),
  reimplemented here in NumPy.
- **Hugging Face Transformers**, **PyTorch**, **LightGBM**, **scikit-learn** and **Hydra**.

Code is MIT-licensed (see `LICENSE`). Each dataset keeps its own licence.
