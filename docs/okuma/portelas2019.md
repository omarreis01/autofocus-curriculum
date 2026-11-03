# Portelas, Colas, Hofmann, Oudeyer (2019). Teacher algorithms for curriculum learning of Deep RL in continuously parameterized environments. CoRL 2019. arXiv:1910.07224 (v1, 16 Oct 2019)

Source read: the full arXiv PDF (20 pages: main text pp. 1-9, appendix A-E pp. 11-19). The arXiv abstract page confirmed venue and version.
Code: https://github.com/flowersteam/teachDeepRL (named in the abstract, p. 1). I did not read the code.
Labels: "(paper)" = stated in the paper, with location. "(mine)" = my inference or recommendation, NOT in the paper.

Page numbers below are the PDF page numbers (the printed page numbers match).

---

## 1. Main claim (Abstract p.1; Sec. 6 p.8)

- Learning-progress (LP) based teachers can guide a deep RL student through continuously parameterized environments. This still works when the space has irrelevant dimensions and large unlearnable regions (Sec. 6).
- The teacher needs no prior knowledge of the student. It only needs loose bounds on the parameter space. Its proposed method, ALP-GMM, consistently beat random sampling and sometimes beat a hand-made expert curriculum (Sec. 6).
- The teacher problem is cast as a "surrogate continuous bandit problem". The teacher samples environments to maximize the student's absolute learning progress (Abstract). The real goal is high final competence over the whole parameter space (Eq. 1, Sec. 3 p.3). The teacher is not rewarded for that goal directly. It is rewarded for ALP only.
- Contributions list (Sec. 1 p.2): the Continuous Teacher-Student framework (Sec. 3), two parameterized BipedalWalker environments (Sec. 4.3), ALP-GMM (Sec. 4.1), and the first study of ALP teachers for generalist DRL agents (Sec. 5).

Relevance to us (mine): the paper has the same structure as our design. The teacher sees only ALP. The outcome that matters, mastery over the whole space on a uniform test set, is measured at evaluation time only (Sec. 5, p.5).

---

## 2. How they tested it

### Setup
- Student: OpenAI's SAC implementation (Spinning Up) is the only student algorithm used (Sec. 4.3 p.5; App. B p.13). Three bodies (Fig. 1): default walker, short walker with 50% shorter legs, quadrupedal walker. Same SAC settings for all (see Sec. 6 below).
- Environment: BipedalWalker. The walker gets reward for moving forward and penalties for torque and head movement. Episodes last up to 2000 steps. A -100 penalty is given if the head touches an obstacle (Sec. 4.3 p.5).
- Each parameter vector p maps to a distribution of tracks. The student never sees the same exact track twice (Sec. 1 p.2). The teacher samples one task per parameter (m=1) and uses uniform weight w_p=1 (end of Sec. 3 p.3).

### Parameter spaces
| Env | Dim | Parameters | Bounds | Location |
|---|---|---|---|---|
| Stump Tracks | 2 | mean stump height mu_h, stump spacing Delta_s. Heights are drawn from N(mu_h, 0.1). | mu_h in [0,3], Delta_s in [0,6] | Sec. 4.3 p.5; App. E p.18 |
| Hexagon Tracks | 12 | 10 positive offsets applied to the vertex positions of a default hexagon obstacle, plus 2 colour parameters called distractors | each dim in [0,4] | Sec. 4.3 p.5; App. E p.18; Fig. 13 p.19 |

Hexagon Tracks has no subspace with trivial tracks, because offsets are strictly positive. It has non-linear difficulty gradients (Sec. 4.3).

### Teachers compared
- Random: uniform sample in P each episode (Sec. 4.2 p.4).
- RIAC: split the space into hyperboxes by absolute LP, sample regions in proportion to ALP. The authors added changes to limit over-splitting (Sec. 4.1 p.3; details App. B p.13).
- Covar-GMM: GMM fit on (parameter, time, episodic reward); sample a Gaussian by positive time-reward covariance. It ignores negative LP (App. B p.13).
- ALP-GMM: the proposed method (Sec. 4.1, Alg. 1).
- Oracle: hand-made sliding window over the space, moved to harder tasks when the mean reward over the last 50 tasks passes 230 (Sec. 4.2 p.5; App. B p.13; Alg. 2 p.14).
- GoalGAN: NOT a baseline. It appears only in the related work (Sec. 2 p.2), where the authors criticize it. It was not tested.
- Exp3.S / Graves et al. 2017 (their ref [16]): NOT tested. It is cited in Sec. 2 only as prior LP use for LSTM and neural Turing machine training. Bandit-based teacher baselines are not in the experiments.

### Evaluation (Sec. 5 p.5)
- A track distribution counts as "mastered" if the student's episodic reward > 230 (threshold taken from POET). The threshold is used for evaluation only, and inside Oracle.
- A fixed test set of 50 distributions, sampled uniformly from the full parameter space. One track per distribution. Learning is turned off during evaluation. The metric is % of mastered test distributions.
- 32 seeded runs per condition (25 for Random on Hexagon Tracks). Welch's t-test for significance. 20M steps for Stump Tracks, 80M steps for Hexagon Tracks.

### Main results
- Stump Tracks, Fig. 3 p.7 (Sec. 5.1 pp.6-7):
  - Short walkers: Oracle best (over 20% mastered). ALP-GMM 14.9%, Covar-GMM 10.6%, RIAC 8.6%. ALP-GMM beats RIAC (p<0.04). ALP-GMM vs Covar-GMM is not significant (p=0.16). All LP teachers beat Random (p<0.001).
  - Default walkers: Oracle is best for the first 10M steps, then declines. At the end ALP-GMM is best and significantly better than Oracle (p<0.04), RIAC (p<0.01) and Covar-GMM (p<0.01). All LP teachers beat Random (p<1e-4).
  - Quadrupedal walkers: Random, ALP-GMM, Covar-GMM and RIAC all reach nearly 100%. Oracle is significantly worse than all of them (p<1e-5). The authors say this space is simple enough for random sampling to work.
  - Explanation given for Oracle's decline: training only in a sliding window on the hardest subspace leads to forgetting of simpler distributions (p.7; App. C Fig. 9 p.16; Fig. 7 p.15).
- Unfeasible tasks, Table 1 and Fig. 10 (Sec. 5.2 p.7; App. C p.16): see item 7.
- Hexagon Tracks, Fig. 4 p.8 (Sec. 5.3): Oracle is ahead early. At 80M steps ALP-GMM reaches about 80% against 68% for Oracle (p<0.02). ALP-GMM beats Covar-GMM (p<0.01) and RIAC (p<0.005). All LP teachers beat Random (p<0.01). ALP-GMM also has less variance (App. D Fig. 11 p.17).
- Fig. 2 p.6 (not asked, but useful): shows ALP-GMM's Gaussians moving over Stump Tracks. Early on there is no progress. After about 1500 episodes the student progresses on low stump heights with spacing above 1.5, and ALP-GMM focuses there. After 15k episodes sampling has moved to more complex regions.

---

## 3. Figure 1 (Sec. 4.3 p.5)

Figure 1 is a picture of the student bodies and environments, not a result plot.
- (a) Stump Tracks walkers: a short walker (50% shorter legs, left), the default bipedal walker (middle), and a bigger quadrupedal walker (right).
- (b) The Hexagon Tracks walker: the quadrupedal walker facing a hexagon obstacle.
- Purpose: to test whether teachers are robust to students with different abilities (Sec. 4.3).

Related figures: Fig. 6 p.14 is the schematic of the ALP-GMM workflow (teacher, task generator, student, ALP computer, FIFO window W, history H). Fig. 2 p.6 is the sampling trajectory.

---

## 4. Conclusion (Sec. 6 p.8)

1. LP-based teachers can guide DRL agents in difficult continuously parameterized environments with irrelevant dimensions and large proportions of unlearnable tasks.
2. With no prior knowledge of the student and only loose boundaries on the task space, ALP-GMM consistently beat random heuristics and occasionally expert-designed curricula.
3. ALP-GMM is conceptually simple and, in the authors' words, has "very few crucial hyperparameters".
4. Possible uses named by the authors: autonomous goal exploration with incremental building of goal spaces (several ALP-GMM instances, one per discovered goal space, their ref [30]), and assisted education (their ref [17]), where current methods use expert knowledge and discrete task sets.

Section 5.3 (p.8) also states that ALP-GMM is the most suitable of the tested conditions for complex high-dimensional spaces with irrelevant dimensions, non-linear spaces and many initially unfeasible tasks. Section 6 contains no explicit limitations paragraph. See item 8.

---

## 5. ALP-GMM, step by step (Sec. 4.1 pp.3-4, Eq. 2, Algorithm 1 p.4; schematic Fig. 6 p.14)

Notation (Sec. 3 p.3): p is a parameter vector in the bounded space P. r_p is the episodic reward of the one task sampled from T(p). H is the history database of (p, r_p). W is the FIFO window of (p, ALP_p) pairs.

| Step | What happens | Where |
|---|---|---|
| 0. Bootstrap | Repeat N times: sample p uniformly at random in P, send one task to the student, observe r_p, compute ALP (step 1), store (p, r_p) in H, store (p, ALP_p) in W. | Alg. 1 lines 3-6 (comment "Bootstrap phase") |
| 1. ALP of a new parameter | Find the closest previously sampled parameter p_old to p_new by Euclidean distance, using a nearest-neighbour search implemented with a KD-tree. Then alp_new = abs(r_new - r_old), where r_old is the episodic reward stored with p_old. The absolute difference gives ALP. One neighbour is used. | Sec. 4.1 p.4; Eq. 2; Alg. 1 lines 5 and 14 |
| 2. Which history | The nearest-neighbour search uses H, the database of ALL previously sampled parameters and rewards. The authors say this prevents forgetting of long-term progress. W (recent pairs only) is used for fitting only. The new pair is stored after its ALP is computed, so it does not compare with itself. | Sec. 4.1 p.4; Alg. 1 lines 5-6, 14-15 |
| 3. GMM features | The GMM is fit on the window W. Each point is the parameter vector concatenated with its ALP value, so the Gaussians live in the joint (P x ALP) space. | Sec. 4.1 p.4; Fig. 6 p.14 |
| 4. Window and refit rate | W holds only the most recent pairs (the last 250). The GMM is refit periodically. Alg. 1 uses the same symbol N for the window size, the bootstrap length and the number of samples between two fits. With N=250 from the text, that gives: bootstrap 250 episodes, refit every 250 episodes. (The paper gives 250 explicitly for the window only. The rest follows from Alg. 1 using one symbol N. Treat as implied.) | Sec. 4.1 p.4; Alg. 1 "Require", lines 2, 3, 10 |
| 5. Number of Gaussians | At each refit, fit several GMMs with 2 to k_max Gaussians (k_max=10), using EM. Keep the one with the best Akaike Information Criterion (AIC). This adapts the number of Gaussians online. | Sec. 4.1 p.4; Alg. 1 lines 8-9 |
| 6. Sampling a new parameter | With probability p_rnd = 20%: sample p uniformly at random in P (exploration). Otherwise: pick a Gaussian with probability proportional to its mean ALP value, then sample p from that Gaussian. The text says the Gaussian is chosen with an EXP4 bandit scheme (their ref [24]), where each Gaussian is an arm and ALP is its utility. The pseudo-code just says "proportionally to its mean ALP value". | Sec. 4.1 p.4; Alg. 1 lines 11-12 |
| 7. Loop | Send the task to the student, observe r_p, compute ALP from H, store the pair in H and W. Repeat N times, then refit (back to step 5). The outer loop stops after K total teacher-student interactions (comment on line 7). | Alg. 1 lines 7-16 |

NOT CONFIRMED in the paper text:
- The GMM covariance type, the EM initialisation, and the software library.
- How "mean ALP of a Gaussian" is computed in detail (the ALP coordinate of the Gaussian's mean is the natural reading).
- How p is drawn from a joint (P x ALP) Gaussian. Fig. 6 has a box for sampling p inside the chosen Gaussian, which suggests using the P part. Whether samples are clipped to the bounds of P is not stated.
- Whether the EXP4 weight updates exist in the code beyond proportional sampling.
- Whether parameters are normalised before the Euclidean nearest-neighbour search. (The toy space in App. A is [0,1]^n. For Stump Tracks the bounds in App. E are [0,3] and [0,6].)
- How the KD-tree is updated as H grows.
- Any maximum distance or minimum time gap on the nearest neighbour.

---

## 6. Key hyperparameters and defaults

| Item | Value | Where |
|---|---|---|
| ALP-GMM window W (FIFO) | last 250 (parameter, ALP) pairs | Sec. 4.1 p.4 |
| Refit rate / bootstrap length | N, same symbol as window size; so 250 by Alg. 1 (implied) | Alg. 1 p.4 |
| Gaussians | 2 to k_max=10; chosen by AIC | Sec. 4.1; Alg. 1 |
| Random exploration p_rnd | 20% | Sec. 4.1 p.4 |
| Nearest neighbour | 1 neighbour, Euclidean, KD-tree, over all of H | Sec. 4.1 p.4 |
| Covar-GMM | same hyperparameters as ALP-GMM | App. B p.13 |
| RIAC | region max size max_s=200; n=50 candidate splits; child min size min_s=20; min region side min_d=1/6 of initial range; sampling split 20% random / 70% in an ALP-chosen region / 10% mutate (Gaussian noise N(p,0.1)) the worst-reward parameter of a region; if no valid split, drop the oldest quarter of the region's pairs | App. B p.13 |
| Oracle | step sigma_W = R/30, window size = R/6 (R is the per-dimension range), threshold r_thr=230, mean reward over last 50 tasks | App. B p.13; Alg. 2 p.14 |
| SAC student | 2 layers (400,300) ReLU for Q, V and policy; tanh policy output; entropy coefficient 0.005; second value 0.001 printed as "learning progress" (almost certainly the learning rate, a typo; mine); one gradient step every 10 env steps on 1000 samples; replay buffer 2M | App. B p.13 |
| Evaluation | threshold 230; 50 uniformly sampled test distributions; 32 seeds; 20M steps (Stump) / 80M (Hexagon) | Sec. 5 p.5; Fig. 3, Fig. 4 |

---

## 7. Robustness to irrelevant dimensions and infeasible regions (as reported)

Irrelevant dimensions:
- Hexagon Tracks, 12 dims with 2 colour distractors (Sec. 5.3 p.8, Fig. 4). ALP-GMM about 80% vs Oracle 68%, with the narrowest spread (Fig. 11 p.17). Note: only 2 of the 12 dimensions are irrelevant here. The non-linear difficulty is also present (Sec. 4.3).
- Toy space, App. A (p.11, Fig. 5 middle row p.12): a 2D space plus 10, 20 or 50 useless dimensions. Random is unaffected. RIAC is consistently worse than the GMM methods. ALP-GMM is best with 20 and 50 useless dims. With 10 useless dims, Covar-GMM reaches 100% median first (44k vs 52k episodes), after ALP-GMM led early (first 40k episodes). The toy student is a deterministic rule-based simulator, not a DRL agent (App. A).
- More meaningful dimensions (Fig. 5 top row): 2D to 6D. At 6D, ALP-GMM and Covar-GMM both reach 100% median after 1M episodes. RIAC reaches 80%. Random is below.

Infeasible regions:
- Stump Tracks with the stump height range widened from [0,3] to [0,4] and [0,5], test set fixed to [0,3] (Sec. 5.2 p.7, Table 1 p.7, Fig. 10 p.16). Average % mastered after 20M steps (mean +- std, 32 seeds; p-value = Welch test vs the same condition on [0,3]):

| Teacher | [0,3] | [0,4] | [0,5] |
|---|---|---|---|
| ALP-GMM | 39.6 +- 9.6 | 38.6 +- 11.6 (p=0.71) | 34.3 +- 15.8 (p=0.12) |
| Covar-GMM | 33.3 +- 7.1 | 27.6 +- 14.0 (p=0.05) | 24.1 +- 14.7 (p=0.01) |
| RIAC | 32.1 +- 12.2 | 23.2 +- 17.2 (p=0.02) | 20.5 +- 15.4 (p=0.002) |
| Random | 18.2 +- 11.5 | 17.4 +- 11.8 (p=0.78) | 12.6 +- 11.0 (p=0.05) |

  ALP-GMM is the only LP teacher that keeps most of its performance. The authors say Random's apparent robustness is due to stagnation at low performance. ALP-GMM stays best in both wider spaces (p<0.02).
- Toy space with more hypercubes, i.e. a smaller initially rewarding area (App. A Fig. 5 bottom row p.12): ALP-GMM does NOT win here. Covar-GMM is best at 50 and 100 cubes/dim. At 20 cubes/dim Covar-GMM and RIAC are best. At 50 and 100 cubes RIAC falls below ALP-GMM.
- Short walkers (large unlearnable share): 13/32 ALP-GMM runs, 15/32 Covar-GMM runs and 19/32 RIAC runs still ended at 0% mastered (Fig. 9 caption p.16).

---

## 8. Caveats and limitations

Stated or directly visible in the paper:
- Restricted setting: one task per parameter (m=1), uniform weights (w_p=1), so no prioritisation of subspaces (end of Sec. 3 p.3).
- ALP-GMM is not the best everywhere. Oracle is better on short walkers (Sec. 5.1) and in the early phase for default walkers and Hexagon Tracks. Covar-GMM is best on the toy spaces (App. A p.11; the authors note this is contrary to the DRL results). On quadrupedal walkers random sampling already suffices (Sec. 5.1).
- Not reliable per run: with short walkers 13/32 ALP-GMM runs end at 0% (Fig. 9). Fig. 11 shows low outliers for ALP-GMM on Hexagon Tracks.
- The Oracle result depends on hand-tuned hyperparameters taken from earlier runs (App. B p.13). The comparison is therefore "ALP-GMM vs a well-informed hand curriculum".
- Scope: one student algorithm (SAC), one simulator family (BipedalWalker), 3 body types. The authors point to other domains (goal exploration, education) only as future applications (Sec. 6).

My own observations, NOT in the paper:
- Welch tests on 32 seeds with many comparisons and no stated correction. Treat borderline p-values (0.04, 0.05) as weak.
- Eq. 2 compares the new reward to a reward from a different location (distance > 0) and possibly from a much earlier time. So |r_new - r_old| mixes temporal progress with the spatial difficulty gradient and with noise. The paper does not discuss this.
- ALP is non-negative even when the student does not learn. Pure noise gives positive ALP. The paper does not discuss noise.

---

## 9. Adapting ALP-GMM to our case (all mine; NOT in the paper)

Parameter space: P = (blur level, contrast level), 2D, bounded. Rescale both axes to [0,1] before the nearest-neighbour search. Euclidean distance is scale-sensitive and the paper does not say it rescales.

What replaces the RL episodic reward r_p. The paper's r_p is one episode reward from the current student at p (Sec. 3). Options:
1. Training mini-batch reward, free. Render a mini-batch at the sampled (blur, contrast). Use the batch accuracy or the negative loss, measured in the forward pass before the gradient step, as r_p. One teacher step is one mini-batch. Zero extra compute. Noisy: with batch 128 and accuracy near 50% the standard error of one batch accuracy is about 4 points. The difference of two such values is about 1.4 times noisier. The 250-point GMM fit averages part of this out.
2. Held-out probe at p. Evaluate a small held-out set rendered at p. Cleaner signal, but costs extra forward passes at every sample. Use it only periodically, for example refresh a small number of probe points, not every step.
3. Loss drop on the same batch (before vs after the update). This is the Graves et al. style "prediction gain". It is a different LP definition from Eq. 2 and it removes the need for the nearest neighbour. Not what this paper does. Mention as an alternative, not as ALP-GMM.

Cost per sample (estimate): one KD-tree query over H plus an O(1) store. GMM refit every N=250 steps on 250 points in 3 dims (2 parameters + ALP), 9 candidate fits (k=2..10). Both are milliseconds to tens of milliseconds. This is negligible next to a CNN training step. The real cost is H growing without bound (all past samples are kept, Sec. 4.1). For long runs, rebuild the KD-tree periodically or cap H. The paper does not say how it handles this (NOT CONFIRMED).

Why absolute LP matters when the student can forget (paper's argument, Sec. 4.1 p.3; supported by Sec. 5.1 p.7, App. D p.17, Fig. 8 caption p.15):
- The authors write that ALP "gives a richer signal than (positive) LP". It lets the teacher notice when the student is losing competence on a region it had mastered, so the teacher revisits it and avoids catastrophic forgetting.
- For us: a CNN trained mostly on blurry, low-contrast images may lose accuracy on sharp, high-contrast images. Signed LP there is negative. A signed teacher would never sample it again. ALP sees |change| and brings the teacher back.
- Evidence in the paper: Oracle forgets because it only slides forward (Fig. 7, Fig. 9). In Hexagon Tracks every Covar-GMM, RIAC and Random condition had runs that reached non-zero mastery and later ended at 0%. The authors read this as forgetting that ALP-GMM avoids (App. D). Covar-GMM, which uses only positive LP, is more variable (Fig. 11). Caution: that evidence is an interpretation by the authors, not an ablation of signed vs absolute LP.
- Risk (mine): ALP also rewards noise. In a classifier, binomial noise in batch accuracy is largest near 50% accuracy and smallest near 0% or 100%. So the teacher may be pulled to regions of intermediate difficulty by noise alone. Check this with a control: run the teacher with a student that does not learn (frozen weights) and see where it samples.

Another point for our research question (mine): ALP has no built-in preference for blurry or low-contrast inputs. If the student ends up favouring them early, that is a finding. It must come from where accuracy changes fastest, not from the design.

---

## 10. Discrete Exp3.S bandit over a (blur, contrast) grid vs continuous ALP-GMM (my judgment; the paper does not test Exp3.S)

Exp3.S over grid cells is the better first choice for our 2D space. It has few hyperparameters, it needs one scalar reward per batch, it handles non-stationary rewards by design, and its arm-pull histograms read directly as a curriculum for the thesis analysis. A coarse grid (for example 5x5) is enough for two bounded dimensions, and Graves et al. (2017) already tested it on supervised networks. ALP-GMM is the better first choice when the space is continuous, higher-dimensional (more than 2 or 3 axes, for example blur, contrast, noise, colour), or contains large unlearnable regions. In those cases a grid explodes or wastes arms, while the GMM adapts its resolution and focuses on the high-ALP regions (Sec. 5.2-5.3, Table 1, Fig. 4). It also keeps a full memory of past rewards, so it can detect forgetting in distant regions. Its costs are more moving parts (window, k_max, p_rnd, refit rate, nearest-neighbour scale) and a noisy, location-mixed ALP estimate (item 8). In the paper ALP-GMM was not best everywhere. A hand curriculum won on short walkers, and Covar-GMM won on the toy spaces. Recommended plan: keep Exp3.S as the default, and run ALP-GMM on the same 2D space as a robustness check, to show that the self-selected curriculum is not an artifact of the grid.

---

## Quotes used (3, each under 15 words)
- "surrogate continuous bandit problem" (Abstract p.1)
- "gives a richer signal than (positive) LP" (Sec. 4.1 p.3)
- "very few crucial hyperparameters" (Sec. 6 p.8)

## Reading notes
- WebFetch could not extract text from the PDF (it returned a binary-format error). The abstract page worked. The tool saved its own binary copy of the PDF under the Claude tool-results folder. I read that copy with the Read tool, page by page (pages 1-20, in 4 calls). I did not download anything myself.
- Everything in items 1-8 comes from reading the PDF pages. Item 9 and item 10 are my reasoning, as labelled.
