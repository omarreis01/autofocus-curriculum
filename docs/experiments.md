# Experiment log

Each trial: what we tried, why, the result, the evidence. Newest at the bottom. Plan: `docs/plans/genis-deneyler.md`.

| # | Date | Trial | Why | Result | Evidence |
|---|---|---|---|---|---|
| 1 | 11-03 | Sharp, 30 ep, seed 0 (F1 acceptance) | Is the pipeline correct? | Clean 93.2% (target ~93) | Drive `results/sharp/0/eval.json` |
| 2 | 11-03 | Speed: eager vs `torch.compile` | Which mode to use | 2.0 vs 1.7 s/epoch, compile costs 30 s → eager | notebook output |
| 3 | 11-03 | Pilot: 7 conditions, seed 0 | Do all conditions run? | Ran; mixed conditions get 13–60% clean → suspected BN | report v1 |
| 4 | 11-03 | BN recalibration (clean train, 100 batches) | Test the BN-mismatch hypothesis | Confirmed: random 60→89, teacher_spg 16→87 clean; sharp unchanged | `results/short/pilot_bn_recal.json` |
| 5 | 11-04 | F5: 7 conditions × 3 seeds × 30 ep | Main comparison | random best robustness (81.4/82.2); teacher goes to the hardest arm; order gain small (+0.3–0.6) | `results/short_summary.json`, report v2 |
| 6 | 11-08 | F6: 7 × 3 × 100 ep | Does it hold with the full budget? | Same ranking as F5. Clean/seen/unseen: sharp 94.8/73.5/75.0, staircase 93.1/77.0/78.8, reverse 86.0/80.9/71.9, random 92.2/85.3/84.4, shuffled 85.4/80.9/79.5, teacher_pg 86.2/81.5/79.7, teacher_spg 87.1/82.5/80.7 | `results/full_summary.json` |
| 7 | 11-08 | Extended: 13 new conditions × 3 seeds × 30 ep + probes on all checkpoints (plan A/B/C) | `docs/plans/genis-deneyler.md` | running | `logs/ext_runner.log` |
