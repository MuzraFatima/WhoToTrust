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
that routes each item to a hard, soft or down-weighted target. In development runs on AmbiNoise-NLI
(2,613 items), LabelTriage detects injected errors well (AUROC 0.84–0.96). On VariErr's *real*
errors, a model trained on synthetic errors first transferred poorly: 0.60 AUROC on a development
half, against 0.86 for simply counting votes. Removing two features that encode synthetic
artefacts, and training on the full data, lifts this to 0.83. That is level with vote counting
within noise, but not better. **[GPU: generalisation, calibration and harm results.]**

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

On AmbiNoise-NLI with full ChaosNLI (K = 5, 20% error, 2,613 items):

| Error type | All groups | Entropy only | Minority share only | Weakest single group |
|---|---|---|---|---|
| spammer | 0.864 | 0.646 | 0.767 | G5 0.749 |
| swap | 0.960 | 0.641 | 0.740 | G1 0.744 |
| lazy | 0.856 | 0.567 | 0.549 | **G1 0.571** |

The pre-registered Phase 3 gate asks every feature group, alone, to reach 0.60. **It is missed for
lazy errors**, where vote statistics alone reach 0.571 (A14). Lazy votes pick the most common
class and so often side with the majority. Shuffled-label detectors average 0.48–0.50 and never
exceed chance. The calibrated triage model passes the Phase 4 gate for every error type: AUROC
0.845 / 0.955 / 0.841 and calibration error 0.047 / 0.027 / 0.037 on 605 held-out votes.

### 5.2 Transfer to real errors (VariErr)

Trained only on synthetic AmbiNoise-NLI errors (MNLI-only data at the time), LabelTriage reached
**0.60** AUROC on VariErr's development half, while untrained minority share reached **0.86**. A group ablation, restricted to
the development half, found two culprits:

- a reliability feature that measured annotation design (annotators label 7–23 items in
  AmbiNoise, about 400 in VariErr);
- the plausibility group, which synthetic errors had turned into a shortcut. Spammer, swap and
  lazy votes look implausible to an MNLI model; real errors do not.

Removing both, chosen among three pre-declared variants, gives **0.76** (gap to minority share
−0.10, 95% CI [−0.16, −0.04]). Trained on the full ChaosNLI data, with nothing else changed, the
same model reaches **0.83** (gap −0.02, 95% CI [−0.06, +0.01]). That is level within noise but
not better, so H1's A9 clause is not met on development evidence. Vote-pattern features alone
reached 0.89 on development data. That variant was not pre-declared, so it is reported only as a
secondary analysis.

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

- **Transfer to real errors is unresolved.** On VariErr's development half the transfer model
  reaches 0.833 AUROC against 0.857 for untrained minority share. It is not better than counting
  votes, and the pre-registered A9 clause is not met on that evidence (Section 5.2).
- **Pre-split exposure of VariErr.** The first transfer check (Phase 4) reported one aggregate
  AUROC (0.63) over all 500 VariErr items, before the 250/250 dev/test split existed (A13). No
  design decision used a test-half number, but the test half is not perfectly unseen.
- **Test-half votes feed unsupervised features.** Annotator reliability (G2), neighbourhood
  consistency (G4) and the training-dynamics probe (G5) are computed over the whole VariErr vote
  table, including the test half's votes, though never its error labels. The final run does the
  same, so the test half is treated no differently, but it is not held out from feature
  computation.
- **Phase 3 gate miss.** For lazy errors, vote statistics alone reach 0.571 against the
  pre-registered 0.60 (A14). The threshold was not changed.
- Earlier development numbers used an MNLI-only ChaosNLI mirror (1,099 items). Full-data
  numbers (2,613 items) replace them where re-run, and both are kept in `docs/results.md`.
- AmbiNoise annotators are simulated; ChaosNLI has no annotator identities.
- VariErr is NLI-only, with 4 annotators and 250 test items (CIs near ±0.07).
- The Phase 4 transfer check reported one aggregate number over all of VariErr before the
  development/test split existed (disclosed in A13).

## References

See `docs/` and the README credits. Full bibliography in the blueprint, section 19.
