#!/bin/bash
# Waits for the extended runner, re-runs runs that died (e.g. CUDA OOM) with lower parallelism,
# fills in missing probes on old checkpoints, then writes short/full summaries.
D=/content/drive/MyDrive/auto-curriculum
cd /content/autofocus-curriculum
while ! grep -q "all done" $D/logs/ext_runner.log; do sleep 20; done
git pull -q
NEW="teacher_rpg teacher_pg_eta0.003 teacher_pg_eta0.03 teacher_pg_noiw teacher_alp random_image random_blur random_contrast sharp_gn random_gn teacher_pg_gn staircase_f025 staircase_f075"
bash colab/run_conds.sh short 6 "0 1 2" $NEW
ls -d $D/results/short/*/[0-9] $D/results/full/*/[0-9] | while read d; do [ -f $d/probe.json ] || echo $d; done |
  xargs -P 4 -I{} sh -c 'python -m src.probe --ckpt {}/last.pt --data-dir /content/data --out {}/probe.json > /dev/null 2>&1'
for b in short full; do e=30; [ $b = full ] && e=100; python -m src.collect --results $D/results/$b --epochs $e > $D/results/${b}_summary.json; done
echo "missing eval: $(ls -d $D/results/short/*/[0-9] | while read d; do [ -f $d/eval.json ] || echo $d; done | wc -l), missing probe: $(ls -d $D/results/*/*/[0-9] | while read d; do [ -f $d/probe.json ] || echo $d; done | wc -l)"
echo finished
