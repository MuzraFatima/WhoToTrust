#!/usr/bin/env bash
# GPU smoke test: every GPU code path once, for one epoch, in about 30-45 minutes on a T4.
#
# Run it through the bootstrap so data is prepared first:
#   !SMOKE=1 CHAOSNLI_DIR=/kaggle/input/chaosnli bash kaggle_bootstrap.sh
#
# It writes to separate folders (outputs_smoke/, cache_smoke/), so nothing it produces is ever
# counted as a real run or reused as a real cache. Each step prints PASS or FAIL with its time;
# a failing step does not stop the others. Exit code = number of failed steps.
set -uo pipefail

WORK=${WORK:-$(dirname "$LT_OUTPUT_DIR")}
export LT_OUTPUT_DIR="$WORK/outputs_smoke"
export LT_CACHE_DIR="$WORK/cache_smoke"
mkdir -p "$LT_OUTPUT_DIR" "$LT_CACHE_DIR"
FAILED=0
RESULTS=()

step() {
  local name=$1; shift
  local t0=$SECONDS
  if python -m labeltriage.run hydra.run.dir="$LT_OUTPUT_DIR/$name" "$@" \
       > "$LT_OUTPUT_DIR/$name.log" 2>&1; then
    RESULTS+=("PASS  $name  ($((SECONDS - t0)) s)")
  else
    RESULTS+=("FAIL  $name  ($((SECONDS - t0)) s)  -> see $LT_OUTPUT_DIR/$name.log")
    FAILED=$((FAILED + 1))
  fi
  echo "${RESULTS[-1]}"
}

ONE="data.train.epochs=1"
NOISE="data=ambinoise_10h data.noise.k=5 data.noise.rate=0.2"

# Image and text training paths, mixed precision on GPU.
step image_mv        experiment=reproduce data=cifar10h method=majority_vote $ONE
step text_soft       experiment=reproduce data=chaosnli method=soft_labels $ONE
# Co-teaching trains two networks at once.
step image_coteach   experiment=reproduce data=cifar10h method=coteaching $ONE
# Injection model (A1) on both bases, two folds only.
step injection_10h   experiment=injection data=cifar10h method=majority_vote $ONE experiment.injection.folds=2
step injection_nli   experiment=injection data=chaosnli method=majority_vote $ONE experiment.injection.folds=2
# Frozen-model caches: CLIP + DINOv2 on images, MNLI model + sentence encoder on text.
step embed_10h       experiment=embed data=cifar10h method=majority_vote
step embed_varierr   experiment=embed data=varierr method=majority_vote
# AmbiNoise runs: plausible-wrong needs the injection ranking written above.
step oracle_pw       experiment=ambinoise_gate $NOISE data.noise.error_type=plausible_wrong method=oracle $ONE
step labeltriage_pw  experiment=ambinoise_gate $NOISE data.noise.error_type=plausible_wrong method=labeltriage $ONE +experiment.probe_epochs=2
step crowdlab        experiment=ambinoise_gate $NOISE method=crowdlab $ONE
step confident_lrn   experiment=ambinoise_gate $NOISE method=confident_learning $ONE
step harm_both       experiment=ambinoise_gate $NOISE method=both $ONE
# Second backbone: real frozen DINOv2 at 224 px.
step dinov2_head     experiment=ambinoise_gate $NOISE method=lt_dinov2_head $ONE +experiment.probe_epochs=2

echo
echo "===== GPU smoke test: $((${#RESULTS[@]} - FAILED)) passed, $FAILED failed ====="
printf '%s\n' "${RESULTS[@]}"
exit "$FAILED"
