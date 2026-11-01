# Pre-registration record: prereg-v1

This file is read by the project owner before the `prereg-v1` tag is created. **The tag is
applied by the owner, by hand, after reading it.** Everything below the "Pre-tag measurements"
heading must be filled in first. Until the tag exists, no `stage=final` run may start.

## What the tag freezes

- The hypotheses and thresholds in `src/labeltriage/conf/prereg/hypotheses.yaml`.
- Amendments A1–A14 in `docs/blueprint-amendments.md`.
- The method, data and experiment configs, and the Tier 1 sweep files (`conf/sweep/t1_*.yaml`,
  `harm.yaml`, `transfer_final.yaml`), at the tagged commit.
- The code at the tagged commit. `scripts/analyze.py` checks that every final run was produced
  under these frozen hypotheses.

## Tier 1 scope (about 23–24 GPU-hours)

| Step | Session | Sweep | Runs | Est. GPU-h | Serves |
|---|---|---|---|---|---|
| 1 | S1 | `t1_s1_reproduce`, then the GPU smoke test | 4 + 13 steps | 0.3 + ~0.6 | per-run cost check; every GPU code path once |
| 2 | S2 | `injection`, then `t1_s2_embed` | 2 + 3 | 0.9 | A1; features for H1 |
| 3 | S3 | `t1_s3_oracle_gate` | 24 | 1.6 | Phase 2 gate |
| 4 | S2 | `features_gate`, `triage_gate`, `t1_s2_transfer_dev` | 8 + 8 + 1 | ~0 (CPU) | Phase 3–4 gates; VariErr dev re-measure |
| 5 | S3 | `t1_s3_pilot` | 32 | 2.2 | detectability |
| 6 | S3 | `tune_router` | 18 | 1.2 | router ν, h₀ |
| — | — | **owner tags `prereg-v1`** | | | |
| 7 | S4–S5 | `t1_s4_main_core` (2 shards) | 144 | 9.6 | H1 (AmbiNoise), H2, H4 |
| 8 | S6 | `t1_s6_headline_seeds` | 32 | 2.1 | H2 (5 seeds in main setting) |
| 9 | S6 | `t1_s6_strong_baselines` | 24 | 1.6 | CROWDLAB, Confident Learning at headline settings |
| 10 | S6 | `harm` | 48 | 3.2 | H3 |
| 11 | S6 | `transfer_final` | 2 | ~0 (CPU) | H1 (VariErr test half) |

Methods in the main grid: majority vote, soft labels, oracle and LabelTriage, at all 12 settings
(4 error types × 3 rates, K = 5) with 3 seeds, plus seeds 3–4 in the main setting.

**Not in Tier 1** (Tier 2, run later if budget allows, never needed for H1–H4 verdicts):

- Dawid–Skene, Co-teaching and entropy-weight in the main grid.
- CROWDLAB and Confident Learning outside the headline settings.
- The K sweep, AmbiNoise-NLI generalisation and CIFAR-10N real noise.
- Ablations and the DINOv2 backbone.

The paper states plainly that its baseline table is partial.

## Cut order

If the measured per-run cost or the remaining quota means Tier 1 must shrink, cuts happen
**before the tag** and only in this order:

1. **Step 9** (`t1_s6_strong_baselines`, about 1.6 GPU-h).
2. **The pilot's second seed** (seed 1 in `t1_s3_pilot`, about 1.1 GPU-h). The detectability
   check then uses the main-grid seed spread instead, and that is recorded here.

Nothing else may be cut without a new amendment written before the tag. After the tag, no
sweep serving H1–H4 is dropped. A budget overrun after the tag delays runs; it does not cancel
them.

## Known misses carried into the tag

- **Phase 3 gate, lazy errors (A14):** G1 alone reaches 0.571 against the 0.60 bar. This is
  recorded as a miss, with no threshold changed.
- **H1's A9 clause on the VariErr dev half:** 0.833 against 0.857 for minority share. The gap's
  CI includes 0, so the clause is not met on the dev evidence available before the tag.

## Pre-tag measurements

### Per-run cost (session 1)

Estimates: about 240 s per CIFAR-10H run and about 240 s per ChaosNLI run on a T4. **If either
measured mean exceeds 1.5× its estimate (360 s), stop and replan before continuing.**

| Dataset | Measured mean `train_seconds` | Within 1.5×? |
|---|---|---|
| CIFAR-10H | _to fill_ | _to fill_ |
| ChaosNLI | _to fill_ | _to fill_ |

### VariErr dev-half re-measure with plausible-wrong errors (session 2, A14)

| Training mix | Dev AUROC | − minority share (95% CI) |
|---|---|---|
| spammer, swap, lazy (CPU, full ChaosNLI) | 0.833 | −0.024 [−0.059, +0.009] |
| spammer, swap, lazy, plausible-wrong (pre-declared) | _to fill_ | _to fill_ |

### Phase 2 oracle gate and Phase 3–4 gates on AmbiNoise-10H (sessions 2–3)

_to fill from `t1_s3_oracle_gate`, `features_gate` and `triage_gate`; misses are logged as misses._

### Detectability and router settings (filled by `scripts/preregister.py`)

<!-- detectability:begin -->
_not yet generated: run `python scripts/preregister.py outputs/pilot_runs.csv --tune outputs/tune_runs.csv`_
<!-- detectability:end -->

## Tagging (owner only)

After every "to fill" above is filled and reviewed:

```bash
git tag -a prereg-v1 -m "Pre-registration v1: Tier 1 scope, hypotheses frozen" <commit>
git push origin prereg-v1
```
