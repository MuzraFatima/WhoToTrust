# WhoToTrust — LabelTriage

**When Labels Disagree: learning to separate annotation error from genuine ambiguity, and training on each differently.**

When several annotators label the same item and disagree, existing pipelines either delete the
minority votes (noisy-label learning) or keep all of them as a soft label (human label variation).
LabelTriage decides *per annotation* whether a disagreement is a mistake or a genuine ambiguity,
then routes each item to a hard, soft or down-weighted training target.

The project has three parts:

- **AmbiNoise** — a benchmark built from real many-annotator vote distributions (CIFAR-10H, ChaosNLI)
  with controlled, labelled errors injected on top.
- **LabelTriage** — a calibrated per-annotation triage model plus a three-route target builder.
- **Harm analysis** — counterfactual retraining that measures which kinds of disagreement hurt
  accuracy and which hurt calibration.

The full research design lives in [`docs/`](docs/).
