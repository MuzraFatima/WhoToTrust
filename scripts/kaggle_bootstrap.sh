#!/usr/bin/env bash
# Bootstrap a Kaggle or Colab session and run one sweep.
#
#   !bash kaggle_bootstrap.sh src/labeltriage/conf/sweep/smoke.yaml [extra overrides...]
#
# Outputs go to /kaggle/working (or /content on Colab) so they survive as notebook output.
# Re-running the same sweep after a timeout resumes from the first unfinished run.
set -euo pipefail

REPO_URL=${REPO_URL:-https://github.com/MuzraFatima/WhoToTrust.git}
BRANCH=${BRANCH:-main}
SPEC=${1:?usage: kaggle_bootstrap.sh SPEC [overrides...]}
shift

if [ -d /kaggle/working ]; then WORK=/kaggle/working; else WORK=/content; fi
cd "$WORK"
[ -d WhoToTrust ] || git clone --depth 1 --branch "$BRANCH" "$REPO_URL"
cd WhoToTrust
git pull --ff-only || true

pip install -q -e ".[train,features,crowd,tracking]"

export LT_DATA_DIR="$WORK/data"
export LT_CACHE_DIR="$WORK/cache"
export LT_OUTPUT_DIR="$WORK/outputs"

# ChaosNLI's official zip cannot be fetched by script. Upload it once as a Kaggle dataset (or to
# Drive on Colab) and point CHAOSNLI_ZIP at it; otherwise the MNLI-only mirror is used.
if [ -n "${CHAOSNLI_ZIP:-}" ] && [ -f "$CHAOSNLI_ZIP" ]; then
  mkdir -p "$LT_DATA_DIR/raw/chaosnli"
  cp -n "$CHAOSNLI_ZIP" "$LT_DATA_DIR/raw/chaosnli/chaosNLI_v1.0.zip"
fi

for name in cifar10h cifar10n varierr chaosnli; do
  [ -f "$LT_DATA_DIR/processed/$name/meta.json" ] || \
    python -m labeltriage.data.prepare "$name" --data-dir "$LT_DATA_DIR"
done

python scripts/run_grid.py "$SPEC" --out "$LT_OUTPUT_DIR/grid" "$@"
python scripts/summarize.py "$LT_OUTPUT_DIR/grid" --out "$LT_OUTPUT_DIR/summary.csv"
