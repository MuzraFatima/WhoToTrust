# Build plan

The engineering view of the blueprint's eight phases, with the v1.1 amendments
([blueprint-amendments.md](blueprint-amendments.md)) folded in. Each phase ends with a pull request
from `master` into `main`, merged only once CI is green and the phase gate is met.

| Phase | Weeks | Build | Gate |
|---|---|---|---|
| 0 | — | Package, lint, CI, Hydra configs, resumable runs, Kaggle bootstrap | CI green; smoke grid resumes |
| 1 | 1–2 | Loaders (CIFAR-10H, CIFAR-10N, CIFAR-10 / 10.1 / 10-C, ChaosNLI, VariErr) into the store; fixed splits; VariErr overlap filter (A3); VariErr error mapping (A2); weighted soft-CE trainer (A5); evaluation harness | MV vs soft on CIFAR-10H in published range; budget within 1.5× |
| 2 | 3–4 | AmbiNoise injector: K sampling, simulated annotators, 4 error types, 3 rates, native-error rule; cross-fitted injection model for plausible-wrong (A1); oracle triage; dataset card | Clean-vote median KL < 0.05; oracle within 1.0 point of clean votes (A5) |
| 3 | 5–6 | Embedding cache + kNN; feature groups G1–G5 behind one cross-fit interface; cheap cached G5 probe (A4); leakage test | Each group AUROC ≥ 0.60 on val; shuffled labels give ≈ 0.5 |
| 4 | 7–8 | LightGBM triage + isotonic calibration (MLP variant); three-route router; first VariErr transfer | Val AUROC ≥ 0.80; calibration error < 0.05 |
| 5 | 9–10 | Remaining baselines (Dawid–Skene, CROWDLAB, Confident Learning, entropy weight, DivideMix / Co-teaching); stats (bootstrap, Wilcoxon, Holm, MDE); one-seed pilot; tune ν and h₀; `prereg-v1` tag | Every threshold above the MDE |
| 6 | 11–12 | Full grids with `stage=final`; results aggregator | All tables regenerate from configs + tag |
| 7 | 13–14 | Harm counterfactuals; ablations; misrouting analysis; Disagreement Explorer | RQ1–RQ4 answered with CIs |
| 8 | 15–16 | Paper, AmbiNoise on Hugging Face, public README repro, defence deck | A headline number reproduces from the README alone |

## Conventions

- Every stage reads and writes the annotation store (`labeltriage.data.store`).
- Unit tests run on CPU with synthetic data and no downloads. Anything needing PyTorch is marked
  `@pytest.mark.torch` and skipped where torch is absent.
- GPU work runs on Kaggle / Colab via `scripts/kaggle_bootstrap.sh`. Results land in run
  directories with a `DONE` marker, so interrupted grids resume.
- Test splits are only readable with `stage=final`.
