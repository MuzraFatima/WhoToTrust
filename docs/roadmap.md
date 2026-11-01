# Roadmap: from a good demo to a solid result

**Where things stand.** The software is in good shape (tests, CI, a web app, Lighthouse 99 to 100).
The science is the weak part: the results come from a subset, a small classifier on frozen
features, 3 seeds and 4 of the 9 planned methods, and the paper and defence drafts still have
`[GPU]` placeholders. An honest rating today is about **6.5 out of 10**. The goal of this
roadmap is **at least 8.5**.

Everything below runs on a laptop CPU. No GPU and no Kaggle. Each step is its own pull request
with CI green before merging, and no threshold or hypothesis is changed to make anything pass.
Failures are recorded as misses.

Ranges are rough: the old 2-core laptop did the whole mini pipeline in about 45 minutes.

---

## Phase 1: make the evidence solid (about 2 to 3 days, mostly unattended compute)

Goal: results on the **full data**, with **every method**, **enough seeds**, **real statistics**,
and a **legitimate verdict**.

| # | Step | Done when |
|---|---|---|
| 1.0 | **Restore the environment** on the new laptop (see `CLAUDE.md`). Run the tests and the web app once. | `pytest` and `npm test` pass; the app opens. |
| 1.1 | **A full-data CPU profile** (`profile=full_cpu`): all 10,000 CIFAR-10H images, all 2,613 ChaosNLI items, all 500 VariErr items, still with frozen encoders. Chunk the long steps so they resume after an interruption. | The pipeline completes on full data; `docs/results-full-cpu.md` exists. |
| 1.2 | **Freeze a pre-registration for that profile.** Same hypotheses and thresholds as `prereg-v1`, written down before the final runs, with the known misses listed. The owner reads it and creates the tag. | A tag exists; the analysis guard accepts runs made after it. |
| 1.3 | **All nine methods**: majority vote, soft labels, Dawid-Skene, entropy weighting, Confident Learning, CROWDLAB, Co-teaching, the oracle, LabelTriage. | Every method has results for every setting. |
| 1.4 | **10 seeds** per setting, with paired tests, confidence intervals, effect sizes and a multiple-comparison correction. | Verdicts for H1 to H4 come from `scripts/analyze.py` with the guard on. |
| 1.5 | **The sweeps that already have configs**: error rate (10, 20, 40%), K (3, 5, 10), router settings, feature-group ablations. | Tables and plots for each, in the results doc and the app. |
| 1.6 | **The harm analysis (H3)**, which was never run in the laptop edition. | H3 has a verdict and its own page in the app. |
| 1.7 | **A second real-error test on CIFAR-10N** (real annotator noise, already on disk), next to VariErr. | Transfer is reported on two real-noise sets. |
| 1.8 | **App update**: a Methods page with all nine methods, confidence intervals in the tables, a Harm page, a sweeps view. Update the Chrome end-to-end tests and the screenshots. | Lint, unit and e2e tests pass; Lighthouse stays at 95 or more. |

**Exit criteria for Phase 1**
- Full data, nine methods, 10 seeds, verdicts under a frozen pre-registration.
- Every miss is shown, not hidden.
- Re-running from an empty cache reproduces the numbers exactly.

---

## Phase 2: finish and ship (about 1 to 2 days)

Goal: the written work matches the evidence, the known weaknesses are addressed openly, and
the repo looks and behaves like a finished project.

| # | Step | Done when |
|---|---|---|
| 2.1 | **Address the known misses as labelled amendments**: text lazy errors (vote statistics alone reach 0.588 against a bar of 0.60) and the text calibration gates. Try pre-declared fixes (for example a feature built for lazy annotators). Report them as post-hoc, and **do not move any threshold**. | Each miss is either fixed with evidence or kept as a documented limitation. |
| 2.2 | **Control for the text encoder.** The NLI encoder was already trained on inference tasks. Rerun the text comparison with a general-purpose encoder as a control. | The paper states how much of the text result depends on the encoder. |
| 2.3 | **Rewrite the paper draft and defence outline** from the final numbers. No `[GPU]` placeholders remain. Rebuild the deck. | `grep -r "\[GPU" docs` finds nothing. |
| 2.4 | **Engineering polish**: run the Chrome end-to-end tests in CI, add a coverage floor and type checking, pin Python dependencies, add Dependabot. App: keyboard access to the scatter plot, CSV export, a run-comparison view. | CI covers Python, web and Chrome; coverage and types are enforced. |
| 2.5 | **Release**: a CHANGELOG, a `v1.0.0` tag and GitHub release, and a final README with fresh screenshots. | The release page exists and the README matches reality. |
| 2.6 | **Optional, needs the owner**: check the CIFAR-10H and ChaosNLI licences, set the licence in the dataset card, then publish AmbiNoise to Hugging Face (`scripts/upload_ambinoise.py`, dry run first). | Published, or consciously skipped. |

**Exit criteria for Phase 2**
- Paper, defence, deck and README all describe the same, final numbers.
- CI is green and covers everything that runs locally.
- A self-review against the rubric below scores at least 8.5.

---

## Rubric used for the rating

| Area | Today | Target |
|---|---|---|
| Software quality | 8.5 | 9 |
| Documentation and presentation | 8 | 9 |
| Scientific strength | 5 | 8 |
| Completeness | 6 | 9 |

## Not in scope

The blueprint's GPU study (fine-tuned ResNet-18 and DeBERTa, sealed test splits) stays optional;
see [`full-scale-study.md`](full-scale-study.md). The full-data CPU profile above is the
realistic substitute, and the documents must keep saying which one produced each number.
