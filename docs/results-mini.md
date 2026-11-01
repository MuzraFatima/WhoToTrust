# Mini results: the whole project on a laptop CPU

Run on 2026-11-01 on a 2-core Intel i5-7200U with 8 GB RAM, no GPU, using only the data in this
folder: `python scripts/make_mini.py --seed-cache`, then `python scripts/run_mini.py`. Every step
finished, with no failed runs.

**What these numbers are.** The mini profile (amendment A15) uses a stratified subset of each
dataset and an MLP head on frozen embeddings. All runs are dev stage, so scores are on val and
clean_dev only, never on test items. These results show that the pipeline works end to end and
which way the effects go. They are **not** the pre-registered results: `scripts/analyze.py`
refuses mini runs.

## Time

From a clean slate (empty `cache_mini/`, `outputs_mini/` and `.hf_cache/`) with the two
commands above, on 2026-11-01:

| Step | Runs | Minutes |
|---|---|---|
| Image embeddings: CLIP and DINOv2 downloads plus encoding the 1,500 mini images (once) | 1 | 22.0 |
| Reproduce (includes encoding the train and val sets for the classifier the first time) | 4 | 5.9 |
| Injection model | 2 | 1.8 |
| Feature gates | 8 | 2.6 |
| LabelTriage gates | 8 | 0.7 |
| Comparison: 4 methods × 4 error types × 2 datasets × 3 seeds | 96 | 11.0 |
| VariErr transfer (dev half) | 1 | 0.6 |
| **Total** | **120** | **about 45** |

**Reproducible exactly.** The clean-slate run matches the earlier run on every metric of all 120
runs (largest difference 0.0). Once embeddings are cached, a single training run takes 2–3
seconds, and rerunning `run_mini.py` skips finished runs.

## What the mini run shows

- **Images (CIFAR-10H): LabelTriage matches the direction of the blueprint's claim.** At K = 5
  and 20% errors, training on LabelTriage's targets gives a higher val accuracy than majority
  vote and soft labels for every error type, and a lower KL to the human votes. It comes within
  1 point of the oracle that knows which votes are wrong. Example, lazy errors: 0.953 vs 0.913
  (majority) and 0.818 (soft); oracle 0.960. One exception on clean_dev: with spammer errors,
  majority vote is slightly ahead (0.948 vs 0.940).
- **NLI (ChaosNLI): a gain for lazy errors, level otherwise.** Against the better global rule,
  LabelTriage gains 9.1 points for lazy errors (0.629 vs 0.538) and 0.7 for spammer errors. It
  is 0.2 points behind for plausible-wrong and 1.1 behind for swap; with seed spreads of 1–2
  points, those three are level. Its KL to the human votes is the lowest of the three methods
  for every error type.
- **Error detection works.** Calibrated triage reaches val AUROC 0.96–0.99 on images and
  0.76–0.95 on NLI. The shuffled-label leakage check stays at about 0.5 throughout. On NLI the
  Phase 4 gate misses for lazy errors (calibration error 0.072) and for plausible-wrong errors
  (AUROC 0.758).
- **The known lazy-error weakness reappears.** On NLI, G1 (vote statistics) alone reaches 0.588
  for lazy errors, below the 0.60 bar. This matches the full-data miss (0.571, A14).
- **VariErr transfer is level with vote counting, as on the full data.** AUROC 0.826 against
  0.857 for minority share, gain CI [−0.073, +0.009], against 0.833 on the full data.

**The NLI encoder was changed once, and the first result is kept here.** The first run used a
general sentence encoder (all-MiniLM-L6-v2). Its NLI head reached only 0.45–0.49 val accuracy,
so the NLI training comparison was noise. It was replaced with `cross-encoder/nli-MiniLM2-L6-H768`,
which was trained on SNLI and MNLI, and every NLI step that depends on the classifier was
rerun. Val accuracy rose to 0.69–0.71. Because that encoder already knows NLI, mini NLI accuracy
mostly reflects its training, not this project's (A15). The image results are unchanged.

## Full tables

## Reproduce: majority vote vs soft labels

| data     | method        |   val/acc |   val/kl |   clean_dev/acc |   train_seconds |
|:---------|:--------------|----------:|---------:|----------------:|----------------:|
| chaosnli | majority_vote |     0.693 |    0.320 |         nan     |         161.496 |
| chaosnli | soft_labels   |     0.707 |    0.204 |         nan     |           2.443 |
| cifar10h | majority_vote |     0.960 |    0.284 |           0.960 |         314.446 |
| cifar10h | soft_labels   |     0.960 |    0.141 |           0.946 |           2.968 |

## Injection model (held-out accuracy against the majority label)

| data     |   heldout_acc_vs_majority |
|:---------|--------------------------:|
| chaosnli |                     0.720 |
| cifar10h |                     0.958 |

## Error detection per feature group (AUROC)

| data                                  |   auroc/g1 |   auroc/g2 |   auroc/g3 |   auroc/g4 |   auroc/g5 |   auroc/all |   auroc/minority_share_only |   auroc/leakage_shuffled |
|:--------------------------------------|-----------:|-----------:|-----------:|-----------:|-----------:|------------:|----------------------------:|-------------------------:|
| ambinoise_10h_k5_r0.2_lazy            |      0.936 |      0.864 |      0.990 |      0.998 |      0.982 |       0.999 |                       0.928 |                    0.503 |
| ambinoise_10h_k5_r0.2_plausible_wrong |      0.956 |      0.719 |      0.966 |      0.981 |      0.977 |       0.987 |                       0.950 |                    0.482 |
| ambinoise_10h_k5_r0.2_spammer         |      0.994 |      0.722 |      0.997 |      0.996 |      0.998 |       0.998 |                       0.991 |                    0.466 |
| ambinoise_10h_k5_r0.2_swap            |      0.958 |      0.725 |      0.959 |      0.989 |      0.984 |       0.988 |                       0.952 |                    0.488 |
| ambinoise_nli_k5_r0.2_lazy            |      0.588 |      0.763 |      0.712 |      0.778 |      0.740 |       0.867 |                       0.575 |                    0.498 |
| ambinoise_nli_k5_r0.2_plausible_wrong |      0.692 |      0.683 |      0.658 |      0.628 |      0.636 |       0.739 |                       0.689 |                    0.498 |
| ambinoise_nli_k5_r0.2_spammer         |      0.779 |      0.725 |      0.805 |      0.756 |      0.745 |       0.868 |                       0.769 |                    0.474 |
| ambinoise_nli_k5_r0.2_swap            |      0.736 |      0.820 |      0.901 |      0.807 |      0.724 |       0.940 |                       0.727 |                    0.474 |

## LabelTriage error detection (calibrated, val votes)

| data                                  |   triage/val_auroc |   triage/val_minority_share_only |   triage/val_calibration_error |   gate/auroc_pass |   gate/calibration_pass |
|:--------------------------------------|-------------------:|---------------------------------:|-------------------------------:|------------------:|------------------------:|
| ambinoise_10h_k5_r0.2_lazy            |              0.992 |                            0.910 |                          0.017 |             1.000 |                   1.000 |
| ambinoise_10h_k5_r0.2_plausible_wrong |              0.976 |                            0.951 |                          0.022 |             1.000 |                   1.000 |
| ambinoise_10h_k5_r0.2_spammer         |              0.964 |                            0.991 |                          0.023 |             1.000 |                   1.000 |
| ambinoise_10h_k5_r0.2_swap            |              0.972 |                            0.956 |                          0.025 |             1.000 |                   1.000 |
| ambinoise_nli_k5_r0.2_lazy            |              0.834 |                            0.607 |                          0.072 |             1.000 |                   0.000 |
| ambinoise_nli_k5_r0.2_plausible_wrong |              0.758 |                            0.682 |                          0.055 |             0.000 |                   1.000 |
| ambinoise_nli_k5_r0.2_spammer         |              0.858 |                            0.756 |                          0.038 |             1.000 |                   1.000 |
| ambinoise_nli_k5_r0.2_swap            |              0.953 |                            0.767 |                          0.035 |             1.000 |                   1.000 |

## Training on the triaged labels (mean over seeds)

| data                                  | method        |   n_seeds |   val/acc_mean |   val/acc_std |   val/kl_mean |   clean_dev/acc_mean |
|:--------------------------------------|:--------------|----------:|---------------:|--------------:|--------------:|---------------------:|
| ambinoise_10h_k5_r0.2_lazy            | labeltriage   |         3 |          0.953 |         0.012 |         0.225 |                0.946 |
| ambinoise_10h_k5_r0.2_lazy            | majority_vote |         3 |          0.913 |         0.008 |         0.368 |                0.906 |
| ambinoise_10h_k5_r0.2_lazy            | oracle        |         3 |          0.960 |         0.009 |         0.248 |                0.945 |
| ambinoise_10h_k5_r0.2_lazy            | soft_labels   |         3 |          0.818 |         0.016 |         0.480 |                0.817 |
| ambinoise_10h_k5_r0.2_plausible_wrong | labeltriage   |         3 |          0.950 |         0.005 |         0.222 |                0.945 |
| ambinoise_10h_k5_r0.2_plausible_wrong | majority_vote |         3 |          0.935 |         0.017 |         0.310 |                0.931 |
| ambinoise_10h_k5_r0.2_plausible_wrong | oracle        |         3 |          0.957 |         0.006 |         0.247 |                0.945 |
| ambinoise_10h_k5_r0.2_plausible_wrong | soft_labels   |         3 |          0.907 |         0.016 |         0.275 |                0.902 |
| ambinoise_10h_k5_r0.2_spammer         | labeltriage   |         3 |          0.955 |         0.013 |         0.221 |                0.940 |
| ambinoise_10h_k5_r0.2_spammer         | majority_vote |         3 |          0.943 |         0.006 |         0.281 |                0.948 |
| ambinoise_10h_k5_r0.2_spammer         | oracle        |         3 |          0.957 |         0.006 |         0.247 |                0.945 |
| ambinoise_10h_k5_r0.2_spammer         | soft_labels   |         3 |          0.903 |         0.013 |         0.406 |                0.901 |
| ambinoise_10h_k5_r0.2_swap            | labeltriage   |         3 |          0.955 |         0.005 |         0.216 |                0.943 |
| ambinoise_10h_k5_r0.2_swap            | majority_vote |         3 |          0.938 |         0.003 |         0.309 |                0.915 |
| ambinoise_10h_k5_r0.2_swap            | oracle        |         3 |          0.957 |         0.006 |         0.247 |                0.945 |
| ambinoise_10h_k5_r0.2_swap            | soft_labels   |         3 |          0.875 |         0.030 |         0.378 |                0.886 |
| ambinoise_nli_k5_r0.2_lazy            | labeltriage   |         3 |          0.629 |         0.019 |         0.258 |              nan     |
| ambinoise_nli_k5_r0.2_lazy            | majority_vote |         3 |          0.538 |         0.019 |         0.502 |              nan     |
| ambinoise_nli_k5_r0.2_lazy            | oracle        |         3 |          0.664 |         0.004 |         0.239 |              nan     |
| ambinoise_nli_k5_r0.2_lazy            | soft_labels   |         3 |          0.536 |         0.010 |         0.321 |              nan     |
| ambinoise_nli_k5_r0.2_plausible_wrong | labeltriage   |         3 |          0.678 |         0.014 |         0.221 |              nan     |
| ambinoise_nli_k5_r0.2_plausible_wrong | majority_vote |         3 |          0.669 |         0.015 |         0.247 |              nan     |
| ambinoise_nli_k5_r0.2_plausible_wrong | oracle        |         3 |          0.684 |         0.004 |         0.218 |              nan     |
| ambinoise_nli_k5_r0.2_plausible_wrong | soft_labels   |         3 |          0.680 |         0.018 |         0.225 |              nan     |
| ambinoise_nli_k5_r0.2_spammer         | labeltriage   |         3 |          0.689 |         0.008 |         0.212 |              nan     |
| ambinoise_nli_k5_r0.2_spammer         | majority_vote |         3 |          0.682 |         0.010 |         0.220 |              nan     |
| ambinoise_nli_k5_r0.2_spammer         | oracle        |         3 |          0.684 |         0.004 |         0.218 |              nan     |
| ambinoise_nli_k5_r0.2_spammer         | soft_labels   |         3 |          0.682 |         0.014 |         0.244 |              nan     |
| ambinoise_nli_k5_r0.2_swap            | labeltriage   |         3 |          0.640 |         0.007 |         0.236 |              nan     |
| ambinoise_nli_k5_r0.2_swap            | majority_vote |         3 |          0.640 |         0.018 |         0.313 |              nan     |
| ambinoise_nli_k5_r0.2_swap            | oracle        |         3 |          0.664 |         0.004 |         0.239 |              nan     |
| ambinoise_nli_k5_r0.2_swap            | soft_labels   |         3 |          0.651 |         0.008 |         0.302 |              nan     |

## Transfer to VariErr's real errors (dev half)

|   transfer/dev_auroc |   transfer/dev_minority_share_only |   transfer/dev_gain_over_minority_lo |   transfer/dev_gain_over_minority_hi |
|---------------------:|-----------------------------------:|-------------------------------------:|-------------------------------------:|
|                0.826 |                              0.857 |                               -0.073 |                                0.009 |
