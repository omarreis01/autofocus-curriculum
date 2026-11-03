# Extended experiments (2026-11-04)

The user asked: "let's not skip anything, test as much as possible." Every trial goes into `docs/experiments.md` (what, why, result, evidence). A negative result is also a result.

Base: F5 (7 conditions × 3 seeds × 30 epochs) and F6 (100 epochs). Report: https://claude.ai/artifact/4uPcbPyQs3RmhzjeWz8FAi

## Group A — Teacher (why does it go "hard first"?)
| ID | Experiment | Question | Code |
|---|---|---|---|
| A1 | `teacher_rpg`: PG / loss_before (relative progress) | Is "hard first" caused by the PG scale? | teacher.py |
| A2 | η sweep (0.003, 0.03) for teacher_pg | Is the result sensitive to η? | config only |
| A3 | ALP-GMM teacher (continuous σ, c space; Portelas 2019) | Does a different teacher choose the same thing? (F7 in the proposal) | teacher_alp.py |
| A4 | Teacher without importance weighting / with a short reward window (500) | Sensitivity of the update details | config |

## Group B — Training regime (BN and contrast)
| ID | Experiment | Question | Code |
|---|---|---|---|
| B1 | `random_image`: random arm per image (mixed batch) | Does per-batch vs per-image matter? Does contrast become real? | train.py |
| B2 | GroupNorm model: sharp, random, teacher_pg | Do the results depend on BN? | model.py |
| B3 | `random_blur` (contrast=1) and `random_contrast` (σ=0) | Which axis gives the robustness? | schedule.py |
| B4 | Staircase with frac 0.25 / 0.75 | Is the hand-made schedule length important? | config |

## Group C — Measurements (what the proposal promised)
| ID | Measurement | Question | Code |
|---|---|---|---|
| C1 | Shape bias, CIFAR version: hybrid images (low frequency of class A + high frequency of class B). The share of choices for the low-frequency class = shape-like bias. Plus accuracy on low-pass and high-pass filtered images. | Do blur-trained models use more low-frequency (shape) information? | probe.py |
| C2 | Small-ε adversarial (FGSM, PGD-10, ε=1/255, 2/255) | Robustness of another kind | probe.py |
| C3 | Per-severity curves for CIFAR-10-C (already in eval.json) | Where is the gap: low or high severity? | analysis |

## Order
1. F6 (running).
2. Code: A1+A3 (agent 1), C1+C2 (agent 2), B1+B2+B3 (agent 3), in parallel, different files.
3. GPU: short runs (30 ep × 3 seeds) for every new condition, then the probes on all F5/F6 checkpoints.
4. The best variants go to the 100-epoch run.
5. Report + `docs/experiments.md`.

## Acceptance
- Every new condition: tests pass, the CPU `--fake-data` run works, the resume test passes.
- Every result: 3 seeds, mean ± std, file evidence in the report.
