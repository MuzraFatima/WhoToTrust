# Blueprint amendments (v1.1)

These amendments close gaps found while turning the v1.0 blueprint
(`When Labels Disagree — Master Research Blueprint.docx`) into a build plan. Where this file and
the blueprint disagree, this file wins. All amendments are fixed before the Phase 5
pre-registration tag, so none of them is chosen after seeing test results.

---

## A1. Plausible-wrong errors are injected with a model the triage features never see

**Gap.** The plausible-wrong error type uses "the model's second-ranked class" without naming the
model. If that model were CLIP, every injected error would look plausible to the G3 feature group
by construction, and G3's contribution would be circular.

**Amendment.**

- Images: the injection model is a **ResNet-18 cross-fitted on CIFAR-10H majority labels**
  (5 folds, so no item is ranked by a model that saw it). It is trained once, with seeds disjoint
  from every other run, and its ranked logits are cached and shipped with AmbiNoise.
- NLI: the injection model is a **DeBERTa-v3-small cross-fitted on ChaosNLI majority labels**.
  G3 uses a different public MNLI model (`roberta-large-mnli`).
- The injection model's identity and checksum are recorded in the AmbiNoise dataset card.
- Every triage result is reported per error type **with and without G3**, so any remaining overlap
  between "plausible to the injector" and "plausible to CLIP" is visible.

Cost: about 0.5 T4 GPU-hours, taken from the 15% buffer.

## A2. VariErr's explanation-level judgments are mapped to a precise error label

**Gap.** VariErr labels *explanations*, not votes. H1's transfer target ("AUROC ≥ 0.70 on VariErr's
real errors") was undefined at the annotation level.

**Amendment.** VariErr annotators may give several labels per item, each with one or more
explanations, and each explanation receives validity judgments from the other annotators.

- **Primary unit (used for H1): the item–label pair.** A pair `(item, label)` is an **error** if
  *none* of its explanations is judged valid by a majority of its validators. This matches
  VariErr's own definition of annotation error. Every other pair is **variation**.
- **Secondary unit: the annotator–label vote** `(item, annotator, label)`, labelled an error if
  that annotator's explanations for the label are all judged invalid. Reported, not tested.
- Features for VariErr are computed exactly as for AmbiNoise-NLI, with K = 4. G2 runs Dawid–Skene
  over VariErr's 4 annotators. G4 uses VariErr items only, excluding the item itself. G5's probe is
  cross-fitted on VariErr votes. None of these use the validity judgments, so the error labels never
  reach the features.
- Ties in validity judgments count as **valid** (conservative: fewer errors).

## A3. VariErr items are removed from AmbiNoise-NLI

**Gap (new).** VariErr's 500 items are drawn from MNLI matched-dev, which is also the source of
ChaosNLI's MNLI part. Without a filter, the transfer test could include items the triage model
was trained or tuned on.

**Amendment.** Phase 1 matches VariErr against ChaosNLI on `(premise, hypothesis)` after
whitespace and case normalisation, and records every overlapping ID. Those items are dropped
from all AmbiNoise-NLI splits before the 70/10/20 split is drawn. The overlap count goes in the
dataset card. A unit test asserts that the intersection is empty.

## A4. G5 training-dynamics probes are cheap and cached

**Gap.** A 5-fold full-backbone probe per AmbiNoise setting (4 types × 3 rates × 3 K × 2 domains)
does not fit the blueprint's 3 GPU-hour line for "feature extraction and probe models".

**Amendment.**

- The default G5 probe is an **MLP head on frozen embeddings** (DINOv2 ViT-S/14 for images,
  the sentence encoder for NLI). It trains on **annotation-level samples**: every vote is a training
  example, so data-map confidence, variability and AUM come out per `(item, label)` directly.
- Probes are trained once per `(dataset, K, error rate, error type, seed)`, 5-fold cross-fitted,
  and their per-epoch probabilities are cached to disk. On frozen embeddings this runs on CPU or in
  a couple of GPU-minutes per setting.
- A full ResNet-18 / DeBERTa probe is run **only for the 4 headline settings**, as a sensitivity
  check that the cheap probe does not change conclusions.

## A5. Phase 1 must deliver the trainer, not just baseline numbers

**Gap.** Phase 2's exit gate ("oracle triage recovers clean accuracy") needs a working training
loop, which the blueprint first describes in Phase 4–5.

**Amendment.** Phase 1's deliverable includes the **weighted soft-cross-entropy trainer**
(ResNet-18 and DeBERTa-v3-base) and the evaluation harness. Majority vote and soft labels are both
run through it, as one-hot and vote-share targets.

"Recovers clean accuracy" is made numeric: **oracle triage is within 1.0 clean-accuracy point of
training on the uncorrupted sampled votes**, averaged over 3 seeds, at K = 5 and 20% error for
every error type.

## A6. Smaller definitional fixes

- **"Main setting" for H2's falsification clause** is K = 5, 20% error, pooled over the four error
  types (the four headline settings, 5 seeds each).
- **"Best global rule"** in H2 and the regret metric is the better of majority vote and soft labels
  **chosen on the test split, per setting**. This is deliberately generous to the baselines.
- **CIFAR-10N models are also evaluated on the CIFAR-10 test set** (10,000 clean-labelled images
  unseen in CIFAR-10N training), in addition to CIFAR-10.1. This reduces the standard error from
  about 1 point to about 0.4.
- **DINOv2 and CLIP inputs** are CIFAR images upsampled to 224 × 224 with bicubic interpolation.
  The embeddings are computed once and cached.
- **Native-error sensitivity** (5% and 15% cut-offs) is computed for the headline settings only,
  3 seeds.

## Budget impact

A1 adds about 0.5 GPU-hours. A4 brings G5 well under its original line, and its four headline
full-probe runs fit in the remaining slack. Total stays at about 88 T4 GPU-hours.
