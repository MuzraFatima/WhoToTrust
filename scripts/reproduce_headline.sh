#!/usr/bin/env bash
# Reproduce a headline number on a CPU, from a fresh clone, in about 40 minutes:
# the VariErr dev-half transfer AUROC of the LabelTriage transfer model (amendment A13).
#
# Expected with the official ChaosNLI files in data/raw/chaosnli/: transfer/dev_auroc 0.833,
# minority share 0.857. With only the MNLI mirror: 0.756.
set -euo pipefail

pip install -q -e ".[train,crowd]"
python -m labeltriage.data.prepare varierr chaosnli --data-dir data
python -m labeltriage.run experiment=embed data=varierr method=majority_vote
python -m labeltriage.run experiment=embed data=chaosnli method=majority_vote
python -m labeltriage.run experiment=transfer method=labeltriage data=ambinoise_nli \
  "experiment.train_error_types=[spammer,swap,lazy]" experiment.n_boot=1000 \
  hydra.run.dir=outputs/headline
python - <<'PY'
import json
m = json.load(open("outputs/headline/result.json"))["metrics"]
print(f"VariErr dev transfer AUROC: {m['transfer/dev_auroc']:.3f}  (expected 0.833; 0.756 on the MNLI-only mirror)")
print(f"minority-share reference:  {m['transfer/dev_minority_share_only']:.3f}  (expected 0.857)")
PY
