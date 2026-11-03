# Hacohen & Weinshall (2019) - reading note

Paper: "On The Power of Curriculum Learning in Training Deep Networks". ICML 2019 (PMLR 97). arXiv 1904.03626 (v3, 29 May 2019).

## How this note was checked

- The arXiv abstract page and the ar5iv HTML page were fetched with WebFetch. Both are summaries made by a small model. I did not trust them alone.
- WebFetch also saved the arXiv PDF to disk by itself (`~/.claude/projects/.../tool-results/webfetch-1791750525519-6bmbnn.pdf`). I did not download it. I read PDF pages 2-14 with the Read tool and checked every formula, figure caption and theory item against them.
- PDF page 1 (title, abstract, first half of the Introduction) was NOT read. The abstract comes from the arXiv abs page. The contribution list in Item 1 comes from ar5iv plus PDF page 2.
- Page references below are PDF pages (13 pages in total: main text pp. 1-8, appendix and references pp. 9-13).
- The paper gives almost no accuracy numbers in text. Where I give a number from a figure, I mark it "eyeballed". These are approximate.

---

## 1. Main claim

- Location: Abstract; Sec. 1 (pp. 1-2).
- Non-uniform sampling of mini-batches, from easy to hard, speeds up training and raises final test accuracy of CNNs on image recognition.
- Curriculum learning (CL) has two parts. A scoring function ranks examples by difficulty. A pacing function sets how fast harder examples enter the training pool.
- Theory (Sec. 4): CL changes the optimization landscape. Under an "ideal curriculum" and mild conditions, the global optimum stays the same.
- The authors also claim that earlier CL gains (Weinshall et al. 2018) may partly come from untuned learning rates (Sec. 1, p. 2; Sec. 3 "Hyper-parameter tuning", p. 4).

## 2. How they tested it

Location: Sec. 3 (pp. 4-7), Appendix A-C (pp. 9-12).

| Case | Data | Network | Figure |
|---|---|---|---|
| 1 | CIFAR-100 "small mammals" super-class: 5 classes, 3000 images (500 train + 100 test per class, App. A) | moderate hand-built CNN | Fig. 2, 5, 7 |
| 2 | full CIFAR-10 | same moderate CNN | Fig. 4b |
| 3 | full CIFAR-100 | same moderate CNN | Fig. 3 |
| 4 | CIFAR-10 | public VGG-based net (Liu & Deng 2015), no augmentation | Fig. 4c |
| 5 | CIFAR-100 | same VGG-based net, no augmentation | Fig. 4d |
| 6 | ImageNet (ILSVRC 2012), 7 cat classes, resized to 56x56, 1300 train + 50 test per class (App. C) | same moderate CNN | Fig. 4a |

- Moderate CNN (App. C, p. 11): 8 conv layers (32,32,64,64,128,128,256,256 filters), max-pool + dropout 0.25 every second layer, one 512-unit dense layer with dropout 0.5, batch size 100, SGD, cross-entropy.
- Baselines (Sec. 2.2, p. 3): vanilla (uniform sampling); anti-curriculum (f' = -f, hardest first); random curriculum (random scores).
- Main curriculum settings: Inception-based transfer scoring with fixed exponential pacing (cases 1-3, 6). Varied exponential pacing for the VGG cases 4-5.
- Repetitions: 50 (case 1), 25 (case 6), 5 (cases 2, 3), 3 (cases 4, 5) (Fig. 2-4 captions).
- Tuning (Sec. 3 "Hyper-parameter tuning", "Cross-validation", p. 4; App. B-C): grid search with cross-validation. Curriculum, anti and random get the same grid. Vanilla gets a finer and wider learning-rate grid because it has no pacing parameters.
- Results, from text:
  - Case 1 (Sec. 3.1, p. 4-5, Fig. 2): CL starts faster and ends higher. Random scoring is similar to vanilla. The authors read this as proof that the gain comes from the transfer score.
  - Cases 2-6: same qualitative result. Larger benefit on CIFAR-100 than CIFAR-10. Smaller but still significant benefit on VGG (p. 5).
  - CL benefit is larger when the task is harder, i.e. lower vanilla accuracy (Sec. 3.1, p. 5).
  - Other CIFAR-100 super-classes give similar results (App. A, Fig. 8).
- Results, eyeballed from figures (not in text):
  - Fig. 7 (case 1, final accuracy %): vanilla about 56, random about 55.5, anti about 55, curriculum / single step / varied / self-taught about 57, self-paced about 54.5.
  - Fig. 3 inset (CIFAR-100): vanilla about 0.38 top-1, curriculum about 0.40.
  - Fig. 4: curriculum gain about 0.5 to 1 point in each panel.
  - So the gains are small, about 1 to 2 points.

## 3. Figure 1

- Location: Sec. 2.3, p. 4.
- It plots the three pacing functions: fraction of data used (y, 0 to 1) against iteration (x, 0 to about 2500).
- Red = fixed exponential (equal-width steps). Green dashed = varied exponential (unequal step widths). Blue dotted = single step.
- Annotations mark the hyperparameters: starting percent, step length, and the update H(t+1) = H(t) * Increase, where H(t) is the data size at time t.
- The caption says the hyperparameter values are arbitrary and only for illustration.

## 4. Conclusion

- Location: there is NO separate Conclusion section. The paper ends with the "Discussion" paragraph at the end of Sec. 4 (p. 8), then Acknowledgements and the Appendix (p. 9). The closest summary of results is Sec. 3.5 (p. 7, Fig. 7).
- Sec. 3.5: all curriculum variants (three pacings, self-taught) beat the controls and reach similar accuracy. Self-paced scoring is the exception and hurts. Transfer-based CL is easier to obtain and more robust (App. C).
- Sec. 3.3 (p. 6): single-step pacing uses only a small share of easy data and still gets competitive results. Quote: "most of the power of CL lies at the beginning of training."
- Sec. 4 Discussion (p. 8): a prior that rises with current utility (as in self-paced learning) or falls with it (as in hard-example mining or boosting) can both help. What counts is positive correlation with the optimal utility. The authors say a curriculum tied directly to the optimal utility might beat both. This is speculation, hedged in the text.

## 5. Formal definitions

Location: Sec. 2.1 (p. 3), Sec. 2.2 and Algorithm 1 (p. 3).

- Data: X = {(x_i, y_i)}, i = 1..N, with x_i in R^d and y_i in [K]. Learner h_theta: R^d -> [K]. Training uses a sequence of M mini-batches B_1..B_M of constant size.
- Difficulty (Sec. 2.1): the minimal loss of an example with respect to the set of optimal hypotheses under consideration. So difficulty is relative to a TARGET hypothesis, not to the current model.
- Scoring function f: X -> R.
  - Input: one labelled example (x_i, y_i). Output: a real number.
  - Example i is harder than j if f(x_i, y_i) > f(x_j, y_j).
  - The paper states "Choosing f is the main challenge of CL" (Sec. 2.1, p. 3).
- Pacing function g_theta: [M] -> [N].
  - Input: the iteration (mini-batch) index i in 1..M. Output: an integer, the size of the training subset at that iteration.
  - It can depend on the learner (Sec. 2.1). All experiments use fixed staircase functions with grid-searched hyperparameters. No learner-dependent g is tested.
  - Constraint (Sec. 2.3, p. 3-4): monotone non-decreasing, staircase-shaped. The reason given is that easy examples can only lose likelihood over time.
- Curriculum (Sec. 2.1-2.2, Algorithm 1): sort X by f ascending. For i = 1..M, the subset X'_i is the first g(i) examples. Draw mini-batch B'_i uniformly from X'_i. Class balance must be kept in each subset.
- Control conditions (Sec. 2.2): anti-curriculum uses f' = -f. Random uses random scores.
- Order and pace are cleanly separate in the paper. The order comes only from f. The pace comes only from g. Examples already released stay in the pool (the subsets are nested).

## 6. Scoring methods

Location: Sec. 2.3 (p. 3) with footnotes 1-2; Sec. 3.2 (p. 5-6); App. A (p. 9); App. B (p. 10).

- Transfer scoring (from Weinshall et al. 2018):
  - Teacher: Inception, pre-trained on ImageNet. Each training image is resized to 299x299 and passed through it.
  - Features: the 2048-d activation of the penultimate layer (App. A).
  - Classifier: RBF-kernel SVM; its confidence comes from sklearn `predict_proba` (cross-validation based) (App. A).
  - Footnote 1 and App. A, Fig. 9: other teachers (VGG-16, ResNet), a linear SVM, and other confidence measures (e.g. margin) give similar results. The authors call the Inception and RBF-SVM choice reasonable but arbitrary.
  - The text says the classifier confidence is used as the score and sets the difficulty of each image. Whether it is the confidence in the TRUE label, and how the sign is flipped so that high f means hard: NOT CONFIRMED in the text.
- Self-taught (bootstrapping):
  - Step 1: train the same architecture with vanilla uniform sampling.
  - Step 2: use that trained network's confidence (its loss on each training point under its final hypothesis, Sec. 3.2) as f.
  - Step 3: retrain from scratch with the curriculum.
  - It needs no outside resources. It improves accuracy through the whole run (Fig. 7, 12).
  - Repeating it (footnote 2; App. B): errors accumulate. No gain, and some harm after many repeats.
- Contrast case, self-paced scoring (Sec. 3.2, Fig. 7 and Fig. 12): score = loss under the CURRENT network, updated during training. It lowers test accuracy throughout, most at the start (Fig. 12 eyeballed: starts near 0.25 against 0.4 for vanilla). The exact implementation of the self-paced variant is NOT CONFIRMED in the text I read.

## 7. Pacing functions

Location: Sec. 2.3 (p. 3-4, two unnumbered equations); Fig. 1 (p. 4); App. A "Varied exponential pacing" (p. 10); App. C (pp. 11-12).

Shared hyperparameters (Sec. 2.3): step_length = iterations per constant step; increase = exponential growth factor of the data size per step (written "inc" in the formula); starting_percent = fraction of data in the first step.

- Fixed exponential (Sec. 2.3, unnumbered equation):
  g(i) = min( starting_percent * inc^floor(i / step_length), 1 ) * N
  Three hyperparameters: starting_percent, increase, step_length.
- Varied exponential (Sec. 2.3 text; App. A formula, p. 10):
  g(i) = min( starting_percent * increase^z(i), 1 ) * N, with z(i) = sum over k of 1[i > step_length_k]
  Number of steps = ceil( -log_increase(starting_percent) ).
  Five hyperparameters in practice: only the first two step lengths vary and the rest are fixed (Sec. 3 "Hyper-parameter tuning"; App. A). Its purpose is to avoid re-tuning the learning rate. The gain is smaller than with fixed exponential (Fig. 10, case 1).
- Single step (Sec. 2.3, unnumbered equation):
  g(i) = starting_percent^(1[i < step_length]) * N
  This equals starting_percent * N for i < step_length, then N (the full data). Two hyperparameters: starting_percent, step_length.
  Result (Sec. 3.3, Fig. 5): comparable to exponential pacing in case 1.
- Count of new pacing hyperparameters: 3, 5 and 2 (Sec. 3, p. 4).
- Search ranges, NOT chosen values (App. C, pp. 11-12). The chosen values are NOT CONFIRMED in the text I read.
  - Case 1: step_length 20-400 (fixed and varied); increase 1.1-3; starting_percent 4%-15% (4% is the size of one mini-batch).
  - Cases 2, 3: step_length 100-2000; increase 1.1-3; starting_percent 0.4%-15%.
  - Cases 4, 5: step_length 50-2500; increase 1.1-2; starting_percent 2%-20%.
  - Case 6: step_length 50-400; increase 1.9; starting_percent 2%-15%.
- Important side effect (Sec. 3 "Hyper-parameter tuning", p. 4): pacing shrinks the data early, which acts like a higher effective learning rate. So the learning rate must be re-tuned per condition for fixed exponential and single-step pacing. Varied exponential removes most of this need.
- Learning-rate schedule has its own three hyperparameters: initial rate, exponential decay factor, step length (p. 4).

## 8. Theoretical result

Location: Sec. 4 (pp. 7-8), Eq. 1-7, Propositions 1-3, Corollary 1; proof of Prop. 2 in App. C (p. 12).

- Setup: ERM (Eq. 1) is rewritten as maximum likelihood. Utility U_theta(X) = exp(-L_theta(X)) (Eq. 2). The scoring and pacing functions induce a prior p_i over examples (Eq. 3). The prior is non-increasing in difficulty. In the algorithm, p_i = 1/M for points below the current difficulty threshold and 0 otherwise. The threshold grows with the pacing function.
- Prop. 1 (Eq. 4): prior-weighted utility = plain utility + covariance between U_theta(X) and p(X). This is an identity, not an assumption.
- Assumption (Eq. 5): the optimum theta~ of the plain utility also maximizes the covariance term.
- Prop. 2, under Eq. 5: (1) theta~ is the optimum of both the plain and the weighted utility, so the global optimum is unchanged; (2) the gap U_p(theta~) - U_p(theta) is at least U(theta~) - U(theta) for all theta. In the authors' words, the optimum becomes more pronounced. The proof is two lines (App. C).
- Ideal curriculum (p. 8): the prior induced by the optimal hypothesis, p_i = exp(-L_theta~(X_i)) / C. It gives Eq. 6-7 (a covariance and a Cauchy-Schwarz bound).
- Corollary 1: for the ideal curriculum, Prop. 2 holds if the variance of the utility is roughly constant over plausible parameter values. The text derives it for exactly constant variance.
- Prop. 3: for the ideal curriculum, the gap inequality holds for all theta whose covariance with U_theta~ is at most Var[U_theta~]. This includes uncorrelated and negatively correlated parameter vectors.
- Other assumptions: N -> infinity so that empirical estimates equal expectations (footnote 10).
- What is NOT proved: that gradients are steeper. The text only says gradients "may be overall steeper" (after Prop. 1). The Introduction states "steeper" more strongly (p. 2). No convergence-rate or gradient result exists.
- What is not addressed: the ideal curriculum needs the unknown optimum theta~. The paper gives no way to build it, except to argue that the transfer and self-taught scores approximate it (Sec. 3.3, 3.4).
- Scope: the theory reweights or subsamples a FIXED set of examples. It does not cover changing the examples themselves.
- Empirical support (Sec. 3.4, Fig. 6, case 1, first mini-batch = 10% of data = 250 examples): mean gradients from the three transfer scores (VGG-16, ResNet, Inception) are close to each other. They are far from the random-subset and full-data gradients. Total variance of transfer-subset gradients is much smaller than for random subsets. The authors read this as CL steering the weights toward different local minima. This is an interpretation.

## 9. Caveats and negative results

- Random scoring behaves like vanilla (Fig. 2, 7). The gain needs an informative score (Sec. 3.1).
- Anti-curriculum: the text does not state its result anywhere I read (Sec. 2.2, 3.1, 3.5). It appears only as a control in Fig. 2, 5, 7, 11. Eyeballed, it ends slightly below vanilla and also slightly below random (Fig. 7 about 55 against 56; Fig. 11 clearer). Treat "anti is worse than vanilla" as read from a figure, with no stated significance test.
- Self-paced scoring (loss under the current model) hurts accuracy throughout, mainly early (Sec. 3.2, Fig. 7, Fig. 12).
- Repeated self-taught bootstrapping gives no gain and some harm (App. B).
- Smaller benefit on easier tasks (CIFAR-10 vs CIFAR-100) and on the VGG network (Sec. 3.1, p. 5; App. A).
- Gains are small (about 1 to 2 points, eyeballed) and some conditions have only 3 to 5 repetitions, so error bars are large (App. A, Fig. 4 caption).
- Learning rate confound: earlier CL gains may be an artifact of untuned learning rates (Sec. 1; Sec. 3). With cyclic learning rates (Smith 2017) both vanilla and CL improve, and CL still helps, more on the harder dataset (App. B, Fig. 13, case 3).
- Most of the benefit comes at the start of training (Sec. 3.3). The authors guess that the score is more reliable for easy examples than for hard ones (Sec. 3.3). This is hedged speculation.
- Robustness in the paper means stability of the result under a different grid-search criterion (AUC instead of final accuracy; App. C, Fig. 11). It does NOT mean robustness to image corruptions, blur, contrast change or adversarial input. Shape bias is not tested. Everything is clean test accuracy on image classification with CNNs. I found no corruption or shape-bias experiment on pages 2-14.
- Theory only holds for the ideal or near-ideal curriculum and the assumptions in Item 8.

## 10. Mapping to our project

All of this section is MY INFERENCE, not a claim from the paper.

| Paper | Our setup |
|---|---|
| Scoring function f: example -> difficulty number, fixed and pre-computed (teacher net or earlier vanilla run) | Nothing computed in advance. At most a hypothesised fixed ordering over conditions (blur sigma and contrast level), e.g. "high blur and low contrast = infant-like = early". That ordering belongs to the transformation, not to each image. |
| Pacing function g: iteration -> how much of the sorted data is open | The bandit's arm choice over time (blur and contrast arms visited and how long it stays). |
| Order = sort by f | Order = the sequence of (blur, contrast) arms the bandit visits. |
| Pace = shape of g | Pace = dwell time per arm, and when the bandit reaches sharp, full-contrast images. |

How it lines up with "which order" versus "how long":

- In the paper the split is clean and was tested on its own. Order is changed by reversing f (anti) or randomising it (random), with pacing fixed (Fig. 2, 7). Pace is changed with f fixed (Fig. 5, 10).
- In our setup one decision, the arm choice, sets order and pace together. There is no separate f and g. We can only split them after the run, from the logged arm sequence.
- Order can be scored as the rank correlation between the visited arms and a blur-descending reference order.
- Pace can be scored as the dwell time per arm and the iteration where the policy first reaches the clean setting.
- Two matching controls follow from the paper. (a) A reversed-order or random-arm control, like anti and random. (b) A fixed schedule whose dwell times match the bandit's, like the pacing-only comparison. This gives a clean test of whether order matters or only pace.

Differences that matter (each one limits how far the paper's results transfer):

1. No pre-computed score. The paper defines difficulty against a target hypothesis (Sec. 2.1). Our bandit uses learning progress (the change in loss) under the current network. That is a learner-dependent signal, closer to Graves et al. 2017 (cited in the paper's Previous work, p. 2). It is not the same as the paper's "self-paced" score, which is the current loss, not its change. The paper's only evidence on current-model difficulty is negative (Sec. 3.2, Fig. 12). That is a warning, not a verdict for learning progress.
2. Online and non-monotone. The paper's g is a monotone staircase and the data pool only grows. The definition allows g to depend on the learner (Sec. 2.1), but no such g is tested. Our bandit can go back to blurry arms. Nothing in the paper covers that.
3. Arms change the images; the paper changes which images are used. The theory (Sec. 4) reweights a fixed example set, so "same global optimum" does not carry over to a blur or contrast augmented distribution. The optimum for the augmented distribution can differ.
4. Nested pool versus exclusive arms. In the paper, easy examples stay in the pool after release. A bandit arm may show only one condition at a time.
5. Learning-rate confound. Pacing acts as an effective learning-rate change (Sec. 3, p. 4). Our bandit will change the effective data distribution too. The learning-rate schedule must be identical for the bandit and for every baseline, or tuned the same way.
6. The paper tests only clean accuracy. It gives no support to the claim that a curriculum improves robustness or shape bias. We cannot cite it for that. It also gives no evidence that a network picks an infant-like order by itself, because every curriculum in it is imposed from outside.
7. Early-training effect. Sec. 3.3 says most of the benefit is at the start. If a self-selected blur-first curriculum exists, the first part of training is where we should look for it (my suggestion).

What we CAN cite: the scoring and pacing split (Sec. 2.1-2.3); the order-versus-pace controls (anti, random, pacing variants); the sensitivity to learning rate (Sec. 3, App. B); the covariance view (Eq. 4) as a loose argument for why a prior that is positively correlated with utility can help, with the caveats in Item 8.
