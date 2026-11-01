# Results log

Phase gates and checks as they are run. GPU-dependent results are added after each Kaggle
session.

## Phase 1

| Check | Result |
|---|---|
| Real-data pipeline, CIFAR-10H, CPU, 1 epoch, width 16 | val acc 39.4%, clean-dev acc 37.6%, 71 s training: end to end works |
| MV vs soft labels, 60 epochs, T4 | *pending GPU* |
| Compute within 1.5× of budget | *pending GPU* |

## Phase 2

| Check | Result |
|---|---|
| Exact injected rates (4 types × 10/20/40%, both bases) | exact ✅ |
| Native errors | 2.5% of votes (CIFAR-10H), 1.3–1.7% (ChaosNLI) |
| Sampler KL, median (gate < 0.05, A8) | 0.0001 CIFAR-10H, 0.003 ChaosNLI ✅ |
| Oracle within 1.0 point of clean votes (A5) | *pending GPU* |

## Phase 3: AmbiNoise-NLI, K = 5, 20% error, seed 0, val AUROC

Run on CPU against the MNLI-only ChaosNLI mirror (1,099 items after removing VariErr; 545 val votes).

| Error type | G1 | G2 | G3 | G4 | G5 | All | Entropy only | Minority share only | Shuffled (A10) |
|---|---|---|---|---|---|---|---|---|---|
| spammer | 0.753 | 0.688 | 0.788 | 0.730 | 0.717 | **0.840** | 0.647 | 0.750 | 0.457 ± 0.042 ✅ |
| swap | 0.745 | 0.773 | 0.901 | 0.752 | 0.679 | **0.922** | 0.635 | 0.730 | 0.442 (1 draw) |
| lazy | 0.643 | 0.794 | 0.814 | 0.812 | 0.774 | **0.920** | 0.626 | 0.601 | 0.422 (1 draw) |

Every group clears the 0.60 bar for every error type ✅. The plausible-wrong type and the image
side wait on the GPU-built injection model and image embeddings.

## Phase 4: triage gate, AmbiNoise-NLI dev data (K = 5, 20%, seed 0)

Held-out half of val, about 200 votes.

| Error type | Val AUROC (≥ 0.80) | Calibration error | Bar (A11) | Pass |
|---|---|---|---|---|
| spammer | 0.866 ✅ | 0.067 | 0.073 | ✅ |
| swap | 0.928 ✅ | 0.056 | 0.055 | ❌ narrowly |
| lazy | 0.900 ✅ | 0.021 | 0.056 | ✅ |

The swap miss is within sampling noise at 200 votes, but it is recorded as a miss. The decisive
run uses full ChaosNLI and CIFAR-10H, where the bar is the original 0.05.

## Early VariErr signal (no training on VariErr)

| Untrained score | AUROC on official error labels (878 item–label pairs, 129 errors) |
|---|---|
| Item vote entropy | 0.596 |
| G3, −p(label) from roberta-large-mnli | 0.636 |
| Minority share | 0.872 |

This motivated amendment A9.
