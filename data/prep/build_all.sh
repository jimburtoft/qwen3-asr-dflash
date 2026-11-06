#!/bin/bash
# Full corpus build (CPU instance). Eval sets first (the train text-overlap filter needs every eval
# transcript), then train. Resumable: re-run to continue. Logs: $Q3DF_CORPUS/logs/<name>.log
set -u
cd "$(dirname "$0")"
PY=${PY:-python}
W=${WORKERS:-8}
ROOT=${Q3DF_CORPUS:-$HOME/q3df_corpus}
mkdir -p "$ROOT/logs"
names() { $PY -c "import build_corpus as b; print(' '.join(n for n in b.DATASETS if $1))"; }
run() { for n in "$@"; do
  if grep -q '"total_clips"' "$ROOT/stats/$n.json" 2>/dev/null; then echo "skip $n"; continue; fi
  echo "== $n $(date -u +%H:%M:%S)"; $PY build_corpus.py all "$n" --workers "$W" > "$ROOT/logs/$n.log" 2>&1 \
    || echo "FAILED $n (see $ROOT/logs/$n.log)"; done; }
EVAL=$(names "b.DATASETS[n][3] in ('eval','pool')")
TRAIN=$(names "b.DATASETS[n][3]=='train'")
for n in $EVAL; do $PY build_corpus.py meta "$n" >> "$ROOT/logs/meta.log" 2>&1 || echo "META FAILED $n"; done
rm -f "$ROOT/meta/eval_pool_texts.json"; $PY build_corpus.py evalpool >> "$ROOT/logs/meta.log" 2>&1
run $(echo $EVAL | tr ' ' '\n' | grep -v pool)
$PY build_corpus.py meta ls_train >> "$ROOT/logs/meta.log" 2>&1; $PY build_corpus.py meta ltts_train >> "$ROOT/logs/meta.log" 2>&1
run $TRAIN
echo BUILD_DONE
