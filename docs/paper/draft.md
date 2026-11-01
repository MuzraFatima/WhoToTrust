# When Labels Disagree: Separating Annotation Error from Genuine Ambiguity

*Workshop draft. Sections marked **[GPU]** are filled from `scripts/analyze.py` once the
pre-registered runs finish. Every number below comes from `docs/results.md`.*

## Abstract

When several annotators disagree on an item, noisy-label methods delete the minority votes and
human-label-variation methods keep all of them. We ask whether a model can tell, per vote, which
kind of disagreement it is looking at, and whether training on that distinction helps. We
contribute **AmbiNoise**, a benchmark that injects controlled, labelled errors on top of real
many-annotator vote distributions (CIFAR-10H, ChaosNLI), so that every vote is known to be an
error or genuine variation. We also contribute **LabelTriage**, a calibrated per-vote error scorer
that routes each item to a hard, soft or down-weighted target. In development runs on AmbiNoise-NLI,
LabelTriage detects injected errors well (AUROC 0.84–0.91). On VariErr's *real* errors, a model
trained on synthetic errors transfers poorly: it reaches 0.60 AUROC on a development half, against
0.86 for simply counting votes. Removing two features that encode synthetic artefacts lifts this
to 0.76, still short of vote counting. **[GPU: generalisation, calibration and harm results.]**

## 1. Introduction

The best global treatment of disagreement flips between datasets. Soft labels win on CIFAR-10N,
with three noisy votes per image, while majority vote wins on CIFAR-10H, with about fifty
low-noise votes (Wei et al., 2023). A simple decomposition explains why. If item *i*'s observed
votes mix the clean human distribution *p\** with an error distribution *u* at rate ε, then
q = (1 − ε)p\* + εu. A soft label is off by ε‖u − p\*‖, which grows with the error rate. A
majority label is off by ‖onehot − p\*‖, which grows with genuine ambiguity. Removing the error
votes first approaches *p\** in every regime. This holds only if errors can be told apart from
variation, which is the question this work tests.

## 2. AmbiNoise

AmbiNoise samples K real votes per item, without replacement, from CIFAR-10H (about 51 votes per
image) or ChaosNLI (100 per item). The 500 VariErr items are removed from ChaosNLI because all of
them appear in it. Each vote is credited to one of 200 simulated annotators with Beta-distributed
error propensities. Exactly a share ε ∈ {10%, 20%, 40%} of votes is then corrupted, chosen in
proportion to propensity, by one of four processes:

- spammer: a uniform other class;
- swap: a fixed confusable pair;
- lazy: the most common class;
- plausible-wrong: the top other class of a cross-fitted injection model that no feature uses.

Real votes for classes under 10% of an item's full pool are marked as native errors.

Properties measured on the real data:

- Rates are hit exactly for every type and base.
- Native errors are 2.5% (CIFAR-10H) and 1.3–1.7% (ChaosNLI) of votes.
- Averaged over 20 draws, the sampler reproduces each item's clean pool, with median KL 0.0001
  and 0.003.

## 3. LabelTriage

Each vote is described by five groups of cheap signals. Every group that fits a model is
cross-fitted over item folds.

- **G1, vote statistics:** share, rank, entropy, lone dissent.
- **G2, annotator reliability:** Dawid–Skene accuracy, agreement on high-consensus items.
- **G3, semantic plausibility:** CLIP zero-shot or MNLI-model probability of the label.
- **G4, neighbourhood consistency:** how often the 20 nearest items got the same label.
- **G5, training dynamics:** data-map confidence and variability, area under the margin, and an
  out-of-fold probability, from a probe on frozen embeddings.

A LightGBM scorer, calibrated with isotonic regression, gives p(error) per vote. Train votes are
scored out of fold. Weighting votes by 1 − p(error) gives a cleaned distribution *q* with
effective count *n*. An item is **unreliable** if *n* < ν (target *q*, weight *n*/K), **clear** if
H(*q*) ≤ h₀ (one-hot), and **ambiguous** otherwise (soft *q*). One weighted soft cross-entropy
trainer serves every method.

## 4. Experimental protocol

There are nine methods, sharing backbones (ResNet-18; DeBERTa-v3-base), epochs and seeds:

- majority vote, soft labels, Dawid–Skene, CROWDLAB, Confident Learning, Co-teaching (replacing
  DivideMix, A12), an entropy-weight rule;
- an oracle router that sees the true error labels;
- LabelTriage.

Hypotheses, thresholds and router settings are frozen in a pre-registration tag before any test
split is read. Thirteen documented amendments (`docs/blueprint-amendments.md`) record every
deviation from the original plan, together with the reason and the order in which it happened.

## 5. Results

### 5.1 Error detection on AmbiNoise (development data, CPU)

On AmbiNoise-NLI (K = 5, 20% error) every feature group alone exceeds 0.64 val AUROC.

| Error type | All groups | Entropy only | Minority share only |
|---|---|---|---|
| spammer | 0.825 | 0.647 | 0.750 |
| swap | 0.919 | 0.635 | 0.730 |
| lazy | 0.924 | 0.626 | 0.601 |

Shuffled-label detectors average 0.47–0.48, below chance, as a noise-fitted model can be. They
never exceed it, so there is no sign of leakage. The calibrated triage model reaches AUROC 0.84 /
0.91 / 0.89. Calibration meets the sample-size-aware bar for spammer and lazy but not for swap
(0.056 against 0.054).

### 5.2 Transfer to real errors (VariErr)

Trained only on synthetic AmbiNoise-NLI errors, LabelTriage reached **0.60** AUROC on VariErr's
development half, while untrained minority share reached **0.86**. A group ablation, restricted to
the development half, found two culprits:

- a reliability feature that measured annotation design (annotators label 7–23 items in
  AmbiNoise, about 400 in VariErr);
- the plausibility group, which synthetic errors had turned into a shortcut. Spammer, swap and
  lazy votes look implausible to an MNLI model; real errors do not.

Removing both, chosen among three pre-declared variants, gives **0.76** (gap to minority share
−0.10, 95% CI [−0.16, −0.04]). Vote-pattern features alone reach 0.89 on development data. That
variant was not pre-declared, so it is reported only as a secondary analysis.

**[GPU: sealed VariErr test-half results for the transfer model and the G1-only ablation.]**

### 5.3 Does triage help generalisation? (H2) **[GPU]**

### 5.4 Which disagreement hurts? (H3) **[GPU]**

### 5.5 Human-like uncertainty (H4) **[GPU]**

## 6. Discussion

The development results already support one of the blueprint's fallback contributions: synthetic
noise is an unreliable proxy for real annotation error. The obvious synthetic errors are separable
by model-based features that do not carry over to real data. Vote patterns transfer, and simple
vote counting is a strong baseline that learned detectors must be held to (A9). Whether the
plausible-wrong error type, built to resemble real errors, closes the gap is the main open question
for the pre-registered runs.

## 7. Limitations

- Development numbers use an MNLI-only ChaosNLI mirror (1,099 items).
- AmbiNoise annotators are simulated; ChaosNLI has no annotator identities.
- VariErr is NLI-only, with 4 annotators and 250 test items (CIs near ±0.07).
- The Phase 4 transfer check reported one aggregate number over all of VariErr before the
  development/test split existed (disclosed in A13).

## References

See `docs/` and the README credits. Full bibliography in the blueprint, section 19.
