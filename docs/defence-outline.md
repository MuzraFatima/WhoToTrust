# Defence outline

About 15 slides for a 20-minute talk plus questions. **[GPU: …]** marks numbers that come from
the Tier 1 runs. Each placeholder names the file the number is read from, so the deck can be
finished without re-deriving anything. Known results and misses are already filled in. Misses
stay on the slides.

---

### 1. Title

*When Labels Disagree: separating annotation error from genuine ambiguity.* Name, programme,
date.

### 2. The problem in one picture

Five annotators: dog, dog, dog, wolf, wolf. Noisy-label methods delete the wolf votes;
label-variation methods keep them. Neither asks *which kind* of disagreement this is.

### 3. Why one global rule fails

The best rule flips between datasets: soft labels win on CIFAR-10N, majority vote on CIFAR-10H
(Wei et al., 2023). Observation 1: q = (1 − ε)p\* + εu. Soft labels are hurt by the error rate,
majority vote by genuine ambiguity, and removing the errors first helps in both regimes.

### 4. Research questions and what was pre-registered

H1–H4 in one line each, with their thresholds. The `prereg-v1` tag froze hypotheses, configs
and code before any test split was read, and amendments A1–A14 record every deviation.

### 5. AmbiNoise

Real many-annotator votes (CIFAR-10H, ChaosNLI with 2,613 items) plus four labelled error
processes at exact rates. Measured: rates hit exactly, sampler KL 0.0001 / 0.003, native errors
1.3–2.5%. The oracle gate result is **[GPU: oracle vs clean-vote clean-dev accuracy, from
`t1_s3_oracle_gate` in `outputs/summary.csv`]**.

### 6. LabelTriage

Five signal groups, then calibrated p(error) per vote (out of fold), then three routes:
clear (hard target), ambiguous (soft target), unreliable (down-weighted). Diagram from blueprint
section 7.

### 7. Detecting errors on AmbiNoise (H1, part 1)

| | spammer | swap | lazy |
|---|---|---|---|
| All groups (NLI, dev) | 0.864 | 0.960 | 0.856 |
| Minority share | 0.767 | 0.740 | 0.549 |

On CIFAR-10H: **[GPU: `triage/val_auroc` by error type, from `t1_s4_main_core` LabelTriage
runs]**. **Miss shown here:** for lazy errors, G1 alone reaches 0.571 against the 0.60 gate
(A14).

### 8. Transfer to real errors (H1, part 2)

The VariErr story in three bars: first check 0.60, then 0.76 after two artefact features were
removed (dev half, A13), then 0.83 with full ChaosNLI, against 0.857 for vote counting. Sealed
test half: **[GPU: `transfer/test_auroc` and its gap CI, from `transfer_final` in
`docs/results-final.md`]**. Honest reading: level with vote counting, not better (A9).

### 9. Does triage help generalisation? (H2)

The headline. LabelTriage minus the better global rule, per setting, with CIs: **[GPU: the H2
table from `docs/results-final.md`]**. Verdict: **[GPU: holds / not supported / falsified]**.

### 10. Which disagreement hurts? (H3)

Errors removed vs ambiguity collapsed, accuracy and ECE deltas: **[GPU: H3 row of
`docs/results-final.md`, from `harm`]**.

### 11. Human-like uncertainty (H4)

KL to human votes for LabelTriage, soft labels and majority vote: **[GPU: H4 row of
`docs/results-final.md`]**.

### 12. Baselines in context

CROWDLAB and Confident Learning at the headline settings: **[GPU: from
`t1_s6_strong_baselines`]**. State plainly that the full nine-method table is Tier 2 and not in
this work.

### 13. What went wrong, and what it taught

- Synthetic errors created shortcuts (G3 plausibility, an annotation-design feature), and
  removing them helped transfer (A13).
- Vote counting is a strong baseline, so H1 is held to it (A9).
- The lazy-error gate miss (A14).
- The pre-split VariErr exposure and test-half votes in features, as stated in the paper's
  limitations.

### 14. Contributions that hold either way

AmbiNoise and its release tooling; the harm map **[GPU]**; the transfer result (synthetic noise
as a proxy for real errors); the pre-registered, resumable pipeline (182+ tests, one-command
reproduction).

### 15. Questions

Backup slides: the amendment list A1–A14, the sample-size-aware calibration bar (A11), the
one-sided leakage test (A10), and the compute log per session.

---

**Likely questions to prepare for:**

- Why not DivideMix? (A12.)
- Is VariErr's test half really unseen? (Disclosed: the pre-split aggregate, and test-half votes
  used in unsupervised features.)
- Why does LabelTriage not beat vote counting on real errors?
- What would Tier 2 change?
- How sensitive are the router settings ν and h₀? (From `tune_router`.)
