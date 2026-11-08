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
| 7 | 11-08 | A1 teacher_rpg (PG ÷ loss) | Is hard-first caused by the PG scale? | Yes: hardest-arm share 1–3% (PG: 71–74%); 90.0/81.1/82.4 ≈ random | `short_summary.json` |
| 8 | 11-08 | A2 η 0.003 / 0.03 | η sensitivity | Only speed changes; both still hard-first | same |
| 9 | 11-08 | A4 no importance weighting | Update detail | Slow drift (14% late); 88.8/81.6/81.7 | same |
| 10 | 11-08 | A3 ALP-GMM (quantized to 16 arms) | Different teacher | Near uniform, slightly harder; 89.0/82.1/82.0 ≈ random | same |
| 11 | 11-08 | B1 random arm per image | Does contrast become real? | Yes: contrast corruption 88.9 (per batch 69.3); C-seen 86.9 best; PGD 1/255 low (37.0) | same |
| 12 | 11-08 | B2 GroupNorm (sharp, random, teacher_pg) | BN dependence | GN sharp contrast 87.6 (BN 73.8); clean −3 pts; teacher still hard-first | same |
| 13 | 11-08 | B3 random_blur / random_contrast | Which axis? | Blur: 82.1/81.9; contrast only ≈ sharp (68.2/73.6) | same |
| 14 | 11-08 | B4 staircase 25% / 75% | Schedule length | Longer blur phase → more robust, less clean (75%: 88.5/78.9/80.3) | same |
| 15 | 11-08 | C1 shape bias (hybrids σ=2), low/high-pass | Shape vs texture | Follows blur exposure, not order: sharp 6, random 23, teacher_pg 54, shuffled 60 (100 ep: 67 vs 69) | `*/probe.json` |
| 16 | 11-08 | C2 FGSM, PGD-10 | Adversarial | Blur helps PGD 1/255 (sharp 34.8 → random 59.5); FGSM likely gradient-masked, not used | same |
| 17 | 11-08 | OOM at 12 parallel runs (GN, per-image) | — | 3 runs re-run at 6 parallel from checkpoint | `logs/finish.log` |
