# AmbiNoise dataset card

AmbiNoise is a label-noise benchmark in which **every annotation is labelled as an error or as
genuine variation**, at error rates and error types the experimenter controls, while the
ambiguity stays real.

## How it is built

1. **Real ambiguity.** Start from datasets with many human votes per item:
   - **AmbiNoise-10H:** CIFAR-10H, about 51 votes per image on the 10,000 CIFAR-10 test images
     (514,200 votes, 2,571 annotators).
   - **AmbiNoise-NLI:** ChaosNLI, 100 votes per item. The 500 VariErr items are removed, because
     all of them also appear in ChaosNLI-MNLI (amendment A3).
2. **Sampling.** For each item, draw K real votes without replacement (K ∈ {3, 5, 10}; main
   setting K = 5).
3. **Simulated annotators.** Credit each vote to one of 200 simulated annotators. An annotator
   labels an item at most once, and each has an error propensity drawn from
   Beta(2·rate, 2·(1 − rate)).
4. **Injected errors.** Corrupt exactly `round(rate × N)` votes (rate ∈ {10%, 20%, 40%}). Votes are
   chosen in proportion to their annotator's propensity, and the error process is one of:

   | Type | Wrong label | Can corrupt |
   |---|---|---|
   | `spammer` | uniform over the other classes | any vote |
   | `swap` | the paired class (cat↔dog, car↔truck, deer↔horse, plane↔ship, bird↔frog; entailment↔contradiction) | votes in a pair |
   | `lazy` | the dataset's most common class | votes not already that class |
   | `plausible_wrong` | the injection model's top class other than the true vote | any vote |

5. **Native errors.** A real vote is also labelled an error when its class got under 10% of the
   item's full pool.

## Columns

| Column | Meaning |
|---|---|
| `item_id`, `annotator_id`, `label` | the vote as a method sees it |
| `split` | train / val (test items keep their full human distribution and are never injected) |
| `original_label` | the real human vote before corruption |
| `is_error` | injected or native error |
| `error_type` | process name, `native`, or `none` |
| `annotator_propensity` | the simulated annotator's propensity, for analysis only and never used as a feature |

## Injection model (amendment A1)

Plausible-wrong errors use a model the triage features never use: a ResNet-18 (images) or
DeBERTa-v3-small (NLI), 5-fold cross-fitted on majority labels with seeds starting at 10,000.
Run `sweep/injection.yaml` once. The ranking's SHA-256 is recorded in that run's `result.json`
and should be copied here when the benchmark is released.

| Base | Ranking file | SHA-256 |
|---|---|---|
| CIFAR-10H | `cache/injection/cifar10h.parquet` | *recorded after the Kaggle run* |
| ChaosNLI | `cache/injection/chaosnli.parquet` | *recorded after the Kaggle run* |

## Measured properties

- **Exact error rates.** Every error type hits 10%, 20% and 40% exactly on both bases.
- **Native errors** are about 2.5% of sampled votes on CIFAR-10H and 1.3–1.7% on ChaosNLI.
- **Sampler check (A8).** Averaged over 20 draws at K = 5, clean votes reproduce each item's pool
  with median KL 0.0001 (CIFAR-10H) and 0.003 (ChaosNLI), against a gate of 0.05.
- **Oracle gate (A5).** The oracle must come within 1.0 clean-dev accuracy point of the
  uncorrupted votes at K = 5 and 20%, for every error type. Run with `sweep/oracle_gate.yaml`;
  results pending a GPU session.

## Known limitations

- Synthetic errors may be easier to detect than real ones. That is why `plausible_wrong` exists
  and why VariErr's real errors are used as a transfer test.
- ChaosNLI has no annotator identities, so on NLI, annotator structure comes only from the
  simulated annotators.
- The swap and lazy processes can only corrupt eligible votes. At 40% they still reach the target
  on both bases, but they concentrate on fewer classes than spammer errors do.
