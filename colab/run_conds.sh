#!/bin/bash
# Usage: colab/run_conds.sh <short|full> <parallel> <seeds> <condition...>
# Trains, evaluates and probes every (condition, seed), at most <parallel> runs at a time.
# Skips runs that already have probe.json; others resume from their checkpoint.
B=$1; P=$2; SEEDS=$3; shift 3
D=/content/drive/MyDrive/auto-curriculum
cd /content/autofocus-curriculum
for c in "$@"; do for s in $SEEDS; do echo "$c $s"; done; done | xargs -P "$P" -L 1 bash -c '
  out='"$D"'/results/'"$B"'/$0/$1; log='"$D"'/logs/'"$B"'_$0_$1.log
  [ -f $out/probe.json ] && exit 0
  python -m src.train --config configs/'"$B"'/$0_s$1.json --data-dir /content/data --out-dir $out --resume > $log 2>&1 &&
  python -m src.eval --ckpt $out/last.pt --data-dir /content/data --out $out/eval.json >> $log 2>&1 &&
  python -m src.probe --ckpt $out/last.pt --data-dir /content/data --out $out/probe.json >> $log 2>&1'
echo "done $B $*"
