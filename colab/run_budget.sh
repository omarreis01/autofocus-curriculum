#!/bin/bash
# Usage: colab/run_budget.sh <short|full> <seed...>
# Trains and evaluates all 7 conditions for the given seeds; shuffled runs after teacher_pg.
B=$1; shift
D=/content/drive/MyDrive/auto-curriculum
cd /content/autofocus-curriculum
run() {  # condition seed
  local out=$D/results/$B/$1/$2 log=$D/logs/${B}_$1_$2.log
  python -m src.train --config configs/$B/$1_s$2.json --data-dir /content/data --out-dir $out --resume > $log 2>&1 &&
  python -m src.eval --ckpt $out/last.pt --data-dir /content/data --out $out/eval.json >> $log 2>&1
}
for s in "$@"; do for c in teacher_pg teacher_spg sharp random staircase reverse; do run $c $s & done; done
wait
for s in "$@"; do run shuffled $s & done
wait
echo "done $B $*"
