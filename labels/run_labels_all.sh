#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Greedy labels for the short eval + train sets, one process per core / device (default 4 =
# one trn2.3xlarge at LNC=2). Usage: sh labels/run_labels_all.sh   (env: see train/_env.sh, NPROC)
set -e
. "$(dirname "$0")/../train/_env.sh"
NPROC=${NPROC:-4}
mkdir -p "$Q3DF_WORK/labels"
for r in $(seq 0 $((NPROC-1))); do
  ( pin $r
    for s in eval train; do
      python labels/generate_labels.py --device "$DEVICE" --rank $r --world $NPROC --batch ${BATCH:-16} --max-new 224 \
        --manifest "$Q3DF_CORPUS/shards/${s}_manifest.parquet" --root "$Q3DF_CORPUS/shards" \
        --out "$Q3DF_WORK/labels/$s" > "$Q3DF_WORK/labels/${s}_rank$r.out" 2>&1
    done ) &
  # let the first process populate the compile cache before the others start
  [ $r = 0 ] && sleep ${FIRST_WAIT:-600}
done
wait
