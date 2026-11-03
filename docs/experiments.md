# Experiment log

Each trial: what we tried, why, the result, the evidence. Newest at the bottom. Plan: `docs/plans/genis-deneyler.md`.

| # | Date | Trial | Why | Result | Evidence |
|---|---|---|---|---|---|
| 1 | 11-03 | Sharp, 30 ep, seed 0 (F1 acceptance) | Is the pipeline correct? | Clean 93.2% (target ~93) | Drive `results/sharp/0/eval.json` |
| 2 | 11-03 | Speed: eager vs `torch.compile` | Which mode to use | 2.0 vs 1.7 s/epoch, compile costs 30 s → eager | notebook output |
| 3 | 11-03 | Pilot: 7 conditions, seed 0 | Do all conditions run? | Ran; mixed conditions get 13–60% clean → suspected BN | report v1 |
| 4 | 11-03 | BN recalibration (clean train, 100 batches) | Test the BN-mismatch hypothesis | Confirmed: random 60→89, teacher_spg 16→87 clean; sharp unchanged | `results/short/pilot_bn_recal.json` |
| 5 | 11-04 | F5: 7 conditions × 3 seeds × 30 ep | Main comparison | random best robustness (81.4/82.2); teacher goes to the hardest arm; order gain small (+0.3–0.6) | `results/short_summary.json`, report v2 |
| 6 | 11-04 | F6: 7 × 3 × 100 ep | Does it hold with the full budget? | running | `logs/f6_runner.log` |
