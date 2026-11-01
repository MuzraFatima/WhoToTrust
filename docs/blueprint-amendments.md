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

- **Primary unit (used for H1): the item–label pair.** VariErr's release ships an official
  `error_labels` field per item. A pair `(item, label)` is an **error** if its label is listed
  there, and **variation** otherwise. The release has 878 such pairs, 129 of them errors.
- **Consistency check.** Our own reconstruction, "no explanation for the label is judged valid by
  at least half of its validators", agrees with `error_labels` on 92% of pairs (82% if
  self-judgments are excluded). This is reported in the dataset card, but the official field is
  what H1 is tested against.
- **Secondary unit: the annotator–label vote** `(item, annotator, label)`, labelled an error if the
  item–label pair is an error. Reported, not tested.
- Features for VariErr are computed exactly as for AmbiNoise-NLI, with K = 4. G2 runs Dawid–Skene
  over VariErr's 4 annotators. G4 uses VariErr items only, excluding the item itself. G5's probe is
  cross-fitted on VariErr votes. None of these use the validity judgments, so the error labels never
  reach the features.
- Ties in validity judgments count as **valid** (conservative: fewer errors).

## A3. VariErr items are removed from AmbiNoise-NLI

**Gap (new).** VariErr's 500 items are drawn from MNLI matched-dev, which is also the source of
ChaosNLI's MNLI part. Without a filter, the transfer test could include items the triage model
was trained or tuned on.

**Measured.** The overlap is total: all 500 VariErr items appear in ChaosNLI-MNLI (1,599 items),
matched both by `uid` and by normalised `(premise, hypothesis)` text. VariErr even carries each
item's ChaosNLI vote counts.

**Amendment.** VariErr items are dropped from AmbiNoise-NLI before the 70/10/20 split is drawn.
That leaves 1,099 MNLI items plus the SNLI part (1,514 items), about 2,600 in total, so the
AmbiNoise-NLI test split holds about 520 items (CIs near ±1.7 points instead of ±1.5). The overlap
count goes in the dataset card, and a unit test asserts that the intersection is empty.

**Data access.** ChaosNLI's official zip sits on Dropbox, which blocks scripted downloads. It is
fetched once by hand and placed at `$LT_DATA_DIR/raw/chaosnli/chaosNLI_v1.0.zip`. The loader
verifies it by checksum. The `tasksource/chaos-mnli-ambiguity` Hugging Face mirror (MNLI part
only) is accepted as a fallback for development.

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

## A7. A clean development set, so gates never touch the test images

**Gap.** AmbiNoise-10H's clean-accuracy test set is the 50,000 CIFAR-10 training images. The
Phase 1 reproduction, the Phase 2 oracle gate and the Phase 5 pilot all need a clean accuracy
number, but the blueprint also says test splits are opened only once, in Phase 6.

**Amendment.** The 50,000 CIFAR-10 training images are split once, stratified by class, into
**5,000 clean-dev** images (readable at any stage) and **45,000 clean-test** images (Phase 6
only). The split is committed as `conf/splits/cifar10_train_clean.csv`. Clean-test accuracy on
45,000 images still has a standard error near 0.2 points, so H2's power is unchanged. CIFAR-10.1,
CIFAR-10-C, the CIFAR-10 test set (for CIFAR-10N models) and the 2,000 held-out CIFAR-10H images
are all test-only.

## A8. The AmbiNoise distribution check is defined on the sampler

**Gap.** The Phase 2 gate asks that "the clean votes' distribution stays close to the full-pool
distribution (median KL under 0.05)". Read per realisation, this cannot hold: five votes can never
reproduce a 50-vote distribution, and KL is infinite whenever a pool class is not drawn.

**Amendment.** The check is on the sampler, not on one draw. For each item, the clean (non-native,
uncorrupted) votes are averaged over 20 independent AmbiNoise draws at K = 5, and compared with the
item's full-pool distribution after native-error classes are removed. KL uses 1e-3 smoothing,
and the gate is a median below 0.05.

**Measured.** Median KL is 0.0001 on CIFAR-10H and 0.003 on ChaosNLI (90th percentiles 0.0001 and
0.016), so the sampler passes.

## A9. H1 is also judged against the minority-share rule

**Gap.** H1 asks LabelTriage to beat "annotator entropy alone" by 0.10 AUROC. Item vote entropy
is the same for every vote on an item, so it cannot tell the dissenting vote from the majority
votes. That makes it a weak reference. On VariErr's real errors it reaches only 0.60, while the
untrained rule "the smaller this vote's share of the item, the more likely it is an error"
reaches **0.87**.

**Amendment.** Every triage result reports both untrained references, `entropy_only` and
`minority_share_only`. H1 keeps its entropy clause as written, and adds: *LabelTriage's VariErr
AUROC must also exceed the minority-share rule's, with a bootstrap 95% CI above zero.* Without
this, H1 could pass while learning nothing beyond counting votes.

## A10. The leakage test is one-sided and averaged over shuffles

**Gap.** The risk register asks that "shuffled labels must give AUROC near 0.5". A single shuffle
on about 550 val votes has a standard error near 0.03, so one draw cannot settle the question.
On the first real AmbiNoise-NLI run, 20 shuffles gave a mean of 0.462 (SD 0.037): consistently
slightly *below* 0.5.

**Amendment.** The detector is refitted on 10 independent shuffles of the train labels. The test
**fails only if the mean shuffled AUROC exceeds 0.5 + max(0.02, 2 SE)**. Leakage, meaning answer
information reaching the features, can only raise AUROC above chance. A mean below 0.5 comes
from how a noise-fitted LightGBM behaves in sparse feature regions, where errors concentrate.
That is not a pipeline fault, and feature groups are structurally barred from the answer columns
anyway (`features.check_inputs`). The cross-fitting itself is checked separately: out-of-fold
train AUROC must sit close to held-out val AUROC.

## A11. The calibration gate respects what the sample can measure

**Gap.** Phase 4 asks for triage calibration error under 0.05. ECE is biased upward on small
samples. On the MNLI-only development data, the held-out val half has about 200 votes, and a
*perfectly* calibrated scorer with the same predictions shows a mean ECE of 0.043 and a 95th
percentile of 0.064. The bar cannot be checked there.

**Amendment.** The calibration bar is `max(0.05, q95)`, where `q95` is the 95th-percentile ECE of
a perfectly calibrated scorer with the same predictions and sample size (300 simulated label
draws). The bar is reported next to every result. With about 5,000 held votes (CIFAR-10H) `q95`
is about 0.015, so the original 0.05 applies unchanged. The bar only relaxes where it would
otherwise be unmeasurable.

## A12. Co-teaching is the robust-training baseline for both domains

**Gap.** The blueprint plans DivideMix for images and Co-teaching for NLI, with DivideMix's
official code. DivideMix needs its own two-stage semi-supervised pipeline (Gaussian-mixture
splits, MixMatch), which does not fit the shared weighted-soft-target trainer that every other
method uses. It is also the method the risk register already flags as "slow or unstable on
7,000 images".

**Amendment.** The risk register's fallback is taken up front: **Co-teaching runs in both
domains**, inside the shared training loop (same backbone, epochs, augmentation and seeds as
every other method). The forget rate ramps linearly to τ over the first 20% of epochs. τ is
estimated from the votes as the share disagreeing with their item's majority, capped at 0.5,
since the true noise rate is not available in practice. The baseline table names it
"Co-teaching". If the compute buffer allows after Phase 6, DivideMix may be added for images as
an extra row, but it is not part of any pre-registered comparison.

## A13. VariErr is split into a dev half and a sealed test half

**Gap.** The first transfer check (Phase 4) was a clear miss: LabelTriage scored 0.63 AUROC on
VariErr's real errors, while the untrained minority-share rule scored 0.87. With all 500 items
as the only real-error test, any attempt to improve transfer would be tuning on the test set,
and keeping the method frozen would waste the one real-error signal the project has.

**Amendment.** VariErr is split once, stratified by whether an item carries an official error,
into **250 dev items** (60 with errors) and **250 test items** (59 with errors). The split is
committed in `conf/splits/varierr.csv`. Design choices aimed at transfer may use only the dev
half. The test half is scored only with `stage=final` after the `prereg-v1` tag, and that score
is what H1's VariErr clauses are judged on. The cost is precision: with half the pairs, the
test AUROC CI widens from about ±0.05 to about ±0.07.

**Dev-half diagnosis and the one design change it led to.** Fitting the transfer model on one
feature group at a time, or with one removed, scored on the 250 dev items only:

| Variant | Dev AUROC |
|---|---|
| all groups (as in Phase 4) | 0.596 |
| G1 alone / G5 alone / G4 alone / G3 alone / G2 alone | 0.891 / 0.760 / 0.634 / 0.594 / 0.540 |
| untrained minority share | 0.857 |

Two causes. (1) `g2_n_votes` ranged 7–23 in AmbiNoise and 367–404 in VariErr: it described the
annotation design, not the vote. It was removed from G2 outright (feature cache bumped to v2).
(2) G3 acts as a shortcut, because spammer, swap and lazy errors are implausible to the MNLI model
and real errors are not. Three variants were declared before comparing them: (A) all groups minus
`g2_n_votes`, 0.603; (B) A without G3, **0.756**; (C) B without G2, 0.708. The rule was "highest
dev AUROC", so **B became the transfer model** (`experiment/transfer.yaml`). Iteration on the dev
half stops here. G1 alone scored highest of all but was not a declared variant. It is reported
on the test half as a pre-registered secondary ablation, not adopted.

Variant B still trails minority share on dev, so on this evidence A9's clause is expected to
fail. Plausible-wrong training errors (GPU) are the remaining planned lever.

**Disclosure.** The Phase 4 check reported one aggregate number over all 500 items (0.63)
before this split existed. No design decision uses a test-half number, and the paper reports
this sequence as it happened.

## Budget impact

A1 adds about 0.5 GPU-hours. A4 brings G5 well under its original line, and its four headline
full-probe runs fit in the remaining slack. Total stays at about 88 T4 GPU-hours.
