#!/usr/bin/env bash
# Bootstrap a Kaggle (or Colab) GPU session and run one sweep, or one shard of it.
#
#   !CHAOSNLI_DIR=/kaggle/input/chaosnli RESUME_FROM=/kaggle/input/<previous-output> \
#    SHARD=1/2 MAX_HOURS=8 bash kaggle_bootstrap.sh src/labeltriage/conf/sweep/<sweep>.yaml
#
# Every sweep is resumable. Finished runs carry a DONE marker and are skipped, so running the
# same command again after a restart continues where it stopped. On Kaggle, /kaggle/working only
# survives if the notebook is saved as a version; attach that saved output as an input of the
# next session and point RESUME_FROM at it, and its outputs and caches are copied back first.
#
#   SHARD=i/n    run only shard i of n (balanced; the README says how many shards each sweep needs)
#   MAX_HOURS=h  start no new run after h hours (default 8, safely inside Kaggle's 12 h limit)
#   RESUME_FROM  a previous session's saved /kaggle/working (its outputs/ and cache/ are reused)
#   CHAOSNLI_DIR folder holding chaosNLI_snli.jsonl and chaosNLI_mnli_m.jsonl (or the zip)
set -euo pipefail

REPO_URL=${REPO_URL:-https://github.com/MuzraFatima/WhoToTrust.git}
BRANCH=${BRANCH:-main}
MAX_HOURS=${MAX_HOURS:-8}
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

# Carry finished runs and expensive caches (embeddings, injection rankings, feature tables)
# over from the previous session. Existing files are never overwritten.
if [ -n "${RESUME_FROM:-}" ]; then
  for d in outputs cache; do
    if [ -d "$RESUME_FROM/$d" ]; then
      mkdir -p "$WORK/$d"
      cp -rn "$RESUME_FROM/$d/." "$WORK/$d/"
    fi
  done
  echo "resumed from $RESUME_FROM: $(find "$LT_OUTPUT_DIR" -name DONE 2>/dev/null | wc -l) finished runs"
fi

# ChaosNLI cannot be fetched by script: upload the official files once as a Kaggle dataset.
# The MNLI-only mirror is refused here unless ALLOW_MIRROR=1, so no sweep silently runs on it.
mkdir -p "$LT_DATA_DIR/raw/chaosnli"
if [ -n "${CHAOSNLI_DIR:-}" ] && [ -d "$CHAOSNLI_DIR" ]; then
  for f in chaosNLI_snli.jsonl chaosNLI_mnli_m.jsonl chaosNLI_v1.0.zip; do
    if [ -f "$CHAOSNLI_DIR/$f" ]; then cp -n "$CHAOSNLI_DIR/$f" "$LT_DATA_DIR/raw/chaosnli/"; fi
  done
fi
raw="$LT_DATA_DIR/raw/chaosnli"
if ! { [ -f "$raw/chaosNLI_snli.jsonl" ] && [ -f "$raw/chaosNLI_mnli_m.jsonl" ]; } \
   && [ ! -f "$raw/chaosNLI_v1.0.zip" ] && [ "${ALLOW_MIRROR:-0}" != "1" ]; then
  echo "ERROR: official ChaosNLI files not found; set CHAOSNLI_DIR (or ALLOW_MIRROR=1)" >&2
  exit 1
fi

for name in cifar10h cifar10n varierr chaosnli; do
  [ -f "$LT_DATA_DIR/processed/$name/meta.json" ] || \
    python -m labeltriage.data.prepare "$name" --data-dir "$LT_DATA_DIR"
done

status=0
python scripts/run_grid.py "$SPEC" --out "$LT_OUTPUT_DIR/grid" ${SHARD:+--shard "$SHARD"} \
  --max-hours "$MAX_HOURS" "$@" || status=$?
python scripts/summarize.py "$LT_OUTPUT_DIR/grid" --out "$LT_OUTPUT_DIR/summary.csv" || true

echo
case $status in
  0) echo "SWEEP FINISHED: $SPEC ${SHARD:+(shard $SHARD)}" ;;
  2) echo "TIME BUDGET REACHED: save this version, then run the same command in a new session" \
          "with RESUME_FROM pointing at this output." ;;
  *) echo "SOME RUNS FAILED (exit $status): check the log above before continuing." ;;
esac
echo "Save the notebook version now (outputs/ and cache/ are needed by later sessions),"
echo "then TURN THE SESSION OFF (Kaggle: Run > Stop session) so it stops using GPU quota."
exit "$status"
