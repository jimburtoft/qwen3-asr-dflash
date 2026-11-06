#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Stage 1: N_PROD feature producers (cores 0..N_PROD-1) + 1 draft trainer (core TRAIN_CORE).
# Default = one trn2.3xlarge at LNC=2: producers on cores 0-2, trainer on core 3.
# Usage: sh train/launch_queue_run.sh <run_name> <total_steps> [extra trainer args...]
# Env: see train/_env.sh; SUBSET (producer --subset-frac, 1.0), BS (block size, 16), LAYERS (layer ids),
#      EVAL (eval item dir), N_PROD (3), TRAIN_CORE (N_PROD)
set -e
. "$(dirname "$0")/_env.sh"
RUN=$1; STEPS=$2; shift 2
SUBSET=${SUBSET:-1.0}; BS=${BS:-16}; LAYERS=${LAYERS:-}; N_PROD=${N_PROD:-3}; TRAIN_CORE=${TRAIN_CORE:-$N_PROD}
LARG=""; [ -n "$LAYERS" ] && LARG="--layer-ids $LAYERS"
Q=${QUEUE_ROOT:-/dev/shm}/q3df_q_$RUN
OUT=$Q3DF_WORK/runs/$RUN
EVAL=${EVAL:-$Q3DF_WORK/eval_batches/b4_per64}
mkdir -p "$Q" "$OUT"
for r in $(seq 0 $((N_PROD-1))); do
  ( pin $r; exec python train/feature_producer.py --device "$DEVICE" --labels "$Q3DF_WORK/labels/train/*.labels.parquet" \
      --audio-root "$Q3DF_CORPUS/shards" --queue "$Q" --rank $r --world $N_PROD --batch 4 --max-queue 8 --epochs ${EPOCHS:-100} \
      --subset-frac $SUBSET --block-size $BS $LARG > "$OUT/producer$r.out" 2>&1 ) &
done
( pin $TRAIN_CORE; exec python train/train_draft_queue.py --device "$DEVICE" --queue "$Q" --eval-dir "$EVAL" \
    --out "$OUT" --total-steps $STEPS $LARG "$@" > "$OUT/trainer.out" 2>&1 ) &
TP=$!
echo "launched $RUN (trainer pid $TP); logs in $OUT"
wait $TP
pkill -f "feature_producer.py.*$Q" || true
