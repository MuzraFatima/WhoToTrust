# The full-scale study (optional, not part of this edition)

The blueprint's pre-registered study fine-tunes ResNet-18 and DeBERTa on a GPU and reads sealed
test splits. **That study was not run.** This repository is complete as the laptop edition (see
the README), and every result in it comes from that edition. Nothing below is needed to use,
test or demo the project.

This page keeps the instructions for anyone who later wants to run the full-scale version on a
Kaggle T4. The plan, its cut order and the pre-registration checklist are in
[`prereg-v1.md`](prereg-v1.md); the amendments that shaped it are in
[`blueprint-amendments.md`](blueprint-amendments.md), and the paper and defence drafts that wait
on its numbers are [`paper/draft.md`](paper/draft.md) and [`defence-outline.md`](defence-outline.md).

## Run order on Kaggle (Tier 1)

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
