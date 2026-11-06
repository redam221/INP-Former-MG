#!/usr/bin/env bash
# Ablation ladder for INP-Former-MG (see README "Ablation design").
#
# Usage:  bash scripts/run_ablation.sh <dataset> <data-path>
#   e.g.  bash scripts/run_ablation.sh MVTec-AD ../mvtec_anomaly_detection
#
# Each row keeps everything else at the original INP-Former setting so that the
# contribution of a single axis is isolated.

set -euo pipefail

DATASET="${1:?dataset required (MVTec-AD / VisA / Real-IAD)}"
DATA_PATH="${2:?data path required}"

COMMON="--dataset $DATASET --data-path $DATA_PATH --phase train"

run() {
  local name="$1"; shift
  echo "=============================================================="
  echo "ABLATION: $name"
  echo "=============================================================="
  python train.py $COMMON --save-name "ablation_$name" "$@"
}

# 1. baseline (single granularity, hard coherence) == original INP-Former
run "1_baseline" \
  --layer-groups "0,1,2,3,4,5,6,7" --inp-nums 6 --coherence-mode hard

# 2. two granularities (shallow / deep)
run "2_two_gran" \
  --layer-groups "0,1,2,3;4,5,6,7" --inp-nums 6,6 --coherence-mode hard

# 3. three granularities (shallow / middle / deep)
run "3_three_gran" \
  --layer-groups "0,1,2;3,4,5;6,7" --inp-nums 4,4,4 --coherence-mode hard

# 4. three granularities + soft coherence assignment
run "4_soft_coherence" \
  --layer-groups "0,1,2;3,4,5;6,7" --inp-nums 4,4,4 --coherence-mode soft

# 5. + prototype diversity regularization
run "5_diversity" \
  --layer-groups "0,1,2;3,4,5;6,7" --inp-nums 4,4,4 \
  --coherence-mode soft --diversity-weight 0.05

# 6. + adaptive prototype gating (learned INP count)
run "6_adaptive" \
  --layer-groups "0,1,2;3,4,5;6,7" --inp-nums 4,4,4 \
  --coherence-mode soft --diversity-weight 0.05 \
  --adaptive-prototypes --gate-sparsity-weight 0.01

echo "Ablation ladder complete. Checkpoints under saved_results/ablation_*/"
