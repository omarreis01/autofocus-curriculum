# Graves et al. (2017) - Automated Curriculum Learning for Neural Networks

ICML 2017, arXiv 1704.03003 (v1 only). Authors: Graves, Bellemare, Menick, Munos, Kavukcuoglu.

## 0. Source and confidence (read this first)

- The arXiv PDF could not be read. WebFetch returned raw compressed PDF bytes.
- All text below comes from the ar5iv HTML copy (https://ar5iv.labs.arxiv.org/html/1704.03003), read through WebFetch. WebFetch passes the page through a small summariser model. So formulas and quotes are second-hand.
- I ran 6 fetches on the ar5iv page, with different prompts. Facts that appear the same in 3 or more of them are marked as stable. Everything else says so.
- Side effect: WebFetch saved a copy of the PDF in its own tool-results folder on its own. I did not open it, because the brief said no downloads.
- Before you cite a formula in the thesis, check it once against the typeset PDF. The signs and dropped constants in GVCG and GL2G matter most.
- Equation numbering in the paper: only Eq. (1) to (8) are numbered. PG, GPG, SPG, TPG, MPG, VCG and GVCG are unnumbered, so I give section plus the run-in heading name as the location.
- Section outline (confirmed): 1 Introduction; 2 Background (2.1 Curriculum Learning, 2.2 Adversarial Multi-Armed Bandits, 2.3 Reward Scaling); 3 Learning Progress Signals (3.1 Loss-driven, 3.2 Complexity-driven, 3.3 Prediction Gain Bias); 4 Experiments (4.1 N-Gram, 4.2 Repeat Copy, 4.3 bAbI); 5 Conclusion. There are no tables. There are 5 figures.

## 1. Main claim (Abstract, Sec. 1)

- Choosing the order of training data can be treated as a non-stationary multi-armed bandit. Each arm is a task. The reward is a measure of learning progress made on the last sample. The bandit output is a stochastic syllabus.
- The authors test two families of progress signals: gains in prediction accuracy (loss-driven) and gains in network complexity (MDL-driven).
- On LSTMs and three curricula, a good signal speeds up learning. The abstract says it "in some cases" halves the time to reach a good level. Section 4 gives one concrete "about twice as fast" case (see item 7).

## 2. How they tested it (Sec. 4)

Common setup (Sec. 4):
- Model: stacked unidirectional LSTM, cross-entropy loss (softmax for categorical targets, sigmoid for Bernoulli targets).
- Optimiser: RMSProp, momentum 0.9, learning rate 1e-5 unless stated (3e-5 for bAbI).
- Bandit: Exp3.S with eta = 1e-3, beta = 0, epsilon = 0.05.
- Each experiment: 10 random seeds. Maximum likelihood (ML) nets and variational (VI) nets are separate sets.
- Time axis: total input steps processed.
- Baselines: uniform policy over all tasks; training directly on the target task where there is one.
- Losses and error rates are measured on samples not used for training or for the reward.
- L2-gain nets use L2 regularisation with alpha = 1e-4.

| Task | Sec. | Tasks (arms) | Model | Notes |
|---|---|---|---|---|
| N-gram char modelling | 4.1 | 11 sources, n = 0..10, Kneser-Ney models trained on King James Bible; 150-char sequences, first 50 are burn-in, predict next 100 | 2 x 512 LSTM, batch 32 | Authors call the curriculum superfluous here. |
| Repeat copy | 4.2 | 169 tasks = sequence length 1..13 x repeats 1..13. Target is length 13, 13 repeats | 1 x 512 LSTM, batch 32 | Data made online, so S for the VI loss was set to 169M (1M per task). |
| bAbI | 4.3 | 20 bAbI tasks, 1M generated stories each | 2 x 512 LSTM, batch 16 | Task counts as done at under 5% test errors. Only training performance reported. |

Main numbers stated in the text:
- Repeat copy, VI, GVCG: solves the target "about twice as fast" as uniform. No step counts are given in the text I could read (Sec. 4.2).
- Repeat copy, ML: PG, SPG, TPG somewhat faster than uniform, mainly early. Training on the target alone "failed to learn at all" (Sec. 4.2).
- bAbI: PG clearly beats uniform in speed and number of tasks completed. No numeric speed-up is given (Sec. 4.3).

## 3. Figure 1 (Sec. 4.1)

- Caption (stable): policies over the n-gram tasks for each gain signal, truncated at 2e8 steps, averaged over 10 runs.
- Text reading: the complexity gains (L2G, GL2G, GVCG) move quickly through the lower-n tasks, then put most mass on the 10-gram. GVCG later returns briefly to the 0-gram. VCG follows a similar path but with less confidence. The loss-based gains (PG, GPG, SPG, TPG) move to higher n more slowly and less surely, and first favour lower n.
- NOT CONFIRMED: axis labels, and whether it has one panel per signal. The fetched text did not describe the layout.

## 4. Conclusion (Sec. 5)

- A stochastic syllabus that maximises learning progress can make curriculum learning more efficient, if the progress signal is suitable.
- Uniform sampling is "a surprisingly strong benchmark". The authors guess that learning is dominated by gradients from the tasks with the fastest progress. This gives an "implicit curriculum" with some waste on unneeded samples.
- ML training: PG was the most consistent signal. VI training: GVCG did best.
- Both of these are instantaneous (only need the training sample). The authors argue this is cheaper and works where an outside evaluation is hard. Learning progress is best judged locally.

## 5. The learning progress signals (Sec. 3) - the key part

Notation (Sec. 3): theta = parameters before the update on batch x. theta' = parameters after one update on x. L(x, theta) = loss of parameters theta on sample or batch x. D_k = data of task k. D_N = target task (the last task). x' = a second, fresh sample.

Algorithm 1 (Sec. 3), one step:
1. Sample task k from the policy pi.
2. Sample a whole batch x from D_k. One task per batch, to reduce noise in the gain signal.
3. Train on x.
4. Compute the progress signal nu (any signal below).
5. Raw reward r_hat = nu / tau(x). Here tau(x) is the length of the longest input sequence in x. The paper says other tasks may need a more complex tau.
6. Rescale r_hat to r in [-1, 1] with Eq. (2). Update Exp3.S with Eq. (1).

### 5.1 Loss-driven signals (Sec. 3.1)

| Signal | Formula (plain text) | Where | Loss measured on | Before / after |
|---|---|---|---|---|
| PG | nu_PG = L(x, theta) - L(x, theta') | Sec. 3.1, "Prediction gain", unnumbered | the same batch x used for the update | before vs after the update on x |
| GPG | nu_GPG = grad L(x, theta)^T grad L(x, theta) = \|\|grad L(x, theta)\|\|_2^2 | Sec. 3.1, "Gradient prediction gain", unnumbered. Derived from Eq. (7) and Eq. (8) | the same batch x | at theta, before the update only. No second evaluation |
| SPG | nu_SPG = L(x', theta) - L(x', theta'), x' ~ D_k | Sec. 3.1, "Self prediction gain", unnumbered | fresh sample x' from the same task | before vs after the update on x |
| TPG | nu_TPG = L(x', theta) - L(x', theta'), x' ~ D_N | Sec. 3.1, "Target prediction gain", unnumbered | sample from the target task N | before vs after the update on x |
| MPG | nu_MPG = L(x', theta) - L(x', theta'), x' ~ D_k, k ~ Uniform(1..N) | Sec. 3.1, "Mean prediction gain", unnumbered | sample from a uniformly chosen task | before vs after the update on x |

Supporting equations for GPG (Sec. 3.1 and 3.3):
- Eq. (7): L(x, theta') is approximately L(x, theta) + grad L(x, theta)^T * Delta_theta (first-order Taylor).
- Eq. (8): Delta_theta = - alpha * grad L(x, theta) (plain SGD step).
- Together they give PG roughly alpha * \|\|grad L\|\|^2. GPG drops alpha.

Stated trade-offs (paraphrased):
- PG: instantaneous. It only needs the training batch, and it does not need a task ordering or knowledge of the goal. Computing it needs one extra forward pass (after the update); the paper's sentence is in Sec. 3.1, next to the GPG motivation, and the exact count is NOT STATED beyond "an additional forward pass". For a Bayesian mixture model, PG upper-bounds the information gain (citing Bellemare et al., 2016). Bias: see Sec. 3.3 below. In expectation PG = true progress + gradient variance.
- GPG: cheapest, because it re-uses the training gradient. It is a biased estimate of progress, it favours tasks with higher gradient variance, and it adds Taylor-approximation error on top.
- SPG: unbiased for expected progress, E_x[nu_SPG] = nu (Sec. 3.3). It costs an extra sample and two extra evaluations, and has more variance than PG because it uses two samples. The authors call PG vs SPG a bias-variance trade-off.
- TPG: closest to what you finally care about, but it "tends to suffer from high variance". Early in training the model cannot improve on the hard target task, so the signal is weak when it is most needed.
- MPG: the multi-task version of TPG. It has extra variance from sampling the evaluation task.

### 5.2 Sec. 3.3 bias analysis (unnumbered formulas, assumes Eq. 7 and Eq. 8)

- True expected progress: nu = E_{x'~D}[ L(theta) - L(theta') ] = alpha * \|\| E_{x~D} grad L(x, theta) \|\|^2.
- Expected PG: nu_pg = alpha * E_{x~D} \|\| grad L(x, theta) \|\|^2.
- Decomposition: nu_pg = nu + Var(grad L(x, theta)), where Var(g) = E \|\| g - E[g'] \|\|^2. Where the factor alpha sits in this last line is NOT CONFIRMED (the fetched text dropped it); it follows from the two lines above if you carry alpha through.
- Consequence, in the authors' words: "for equal learning progress, a prediction gain-based curriculum maximizes variance". (Quote 1 of 2.)

### 5.3 Complexity-driven signals (Sec. 3.2)

Set-up: variational inference with diagonal Gaussian posterior P_phi and prior Q_psi. The prior is adapted by gradient descent together with phi.
- Eq. (3), per-sample VI loss: L_VI(x, phi, psi) = (1/S) * KL(P_phi || Q_psi) + E_{theta ~ P_phi} L(x, theta). S = total number of samples in the curriculum (sum of the sizes of all D_k). For generated data S is a large constant that matches the expected sample count.
- KL between diagonal Gaussians: KL = ((mu_phi - mu_psi)^2 + sigma_phi^2 - sigma_psi^2) / (2 sigma_psi^2) + ln(sigma_psi / sigma_phi).
- Expected-loss gradient: reparameterisation trick, one Monte Carlo sample. SoftPlus keeps variances positive.

| Signal | Formula (plain text) | Where | Measured on | Before / after |
|---|---|---|---|---|
| VCG | nu_VCG = KL(P_phi' \|\| Q_psi') - KL(P_phi \|\| Q_psi) | Sec. 3.2, "Variational complexity gain", unnumbered | complexity of the net, after the update on x. The loss does not enter directly | before vs after one update of phi and psi on x |
| GVCG | nu_GVCG = [grad_{phi,psi} KL(P_phi \|\| Q_psi)]^T grad_phi E_{theta~P_phi} L(x, theta) | Sec. 3.2, "Gradient variational complexity gain", unnumbered | gradient of the KL w.r.t. phi and psi, dotted with the loss gradient on x w.r.t. phi | at current parameters, before the update only |
| L2G | nu_L2G = \|\|theta'\|\|_2^2 - \|\|theta\|\|_2^2 | Sec. 3.2, Eq. (5). Regularised loss is Eq. (4): L_L2(x, theta) = L(x, theta) + (alpha/2) \|\|theta\|\|_2^2, with alpha = 1e-4 | weight norm after the update on x | before vs after |
| GL2G | nu_GL2G = theta^T grad_theta L(x, theta) | Sec. 3.2, Eq. (6) | the loss gradient on x | at theta, before the update only |

Details and trade-offs:
- VCG: complexity change after one step. The paper gives no bias analysis. In experiments it was low-confidence and noisy (Sec. 4.1) and stayed near uniform on repeat copy (Sec. 4.2).
- GVCG: first-order version of VCG. Derivation starts from nu_VCG approx C - [grad KL]^T grad_phi E L(x, theta). The gain as defined drops the minus sign and the constant C, which does not depend on x. The authors say they trust the linear approximation more than for PG, because complexity has less curvature than the loss surface. This is a belief; it is not tested. It is cheaper than VCG (no second evaluation). The paper links it to VIME (KL between posteriors before and after a step), and says the relative merits need further study.
- L2G: the alpha/2 factor is dropped because rewards are rescaled anyway. The authors hypothesise that a rising regulariser cost shows rising complexity. For networks without regularisation they found it "unreliable" (Sec. 3.2 / 4), presumably because the net has no push to cut complexity on uninformative data.
- GL2G: first-order version of L2G. No minus sign, no alpha. It inherits the gradient-approximation bias of GPG.
- NOT CONFIRMED: the exact sign convention in GVCG and GL2G as typeset. Check the PDF.

## 6. The bandit (Sec. 2.2, 2.3, 4)

Exp3.S (Sec. 2.2, Eq. 1). N arms. Reward r_t in [-1, 1] after scaling.
- Base Exp3 policy: pi_t(i) = exp(w_{t,i}) / sum_j exp(w_{t,j}). Base Exp3 weights: w_{t,i} = eta * sum_{s<t} r_tilde_{s,i}.
- Importance-weighted reward with offset beta (Eq. 1 area): r_tilde^beta_{s,i} = ( r_s * 1[a_s = i] + beta ) / pi_s(i).
- Eq. (1), sampling policy: pi_t(i) = (1 - epsilon) * exp(w^S_{t,i}) / sum_j exp(w^S_{t,j}) + epsilon / N. So every arm keeps at least epsilon/N probability.
- Eq. (1), Fixed-Share weight update: w^S_{t,i} = log[ (1 - alpha_t) * exp( w^S_{t-1,i} + eta * r_tilde^beta_{t-1,i} ) + (alpha_t / (N-1)) * sum_{j != i} exp( w^S_{t-1,j} + eta * r_tilde^beta_{t-1,j} ) ].
- Initial weights w^S_{1,i} = 0. Share rate alpha_t = 1/t.
- Hyperparameters (Sec. 4): eta = 1e-3, beta = 0, epsilon = 0.05. They are stated once, for all experiments. NOT STATED: how they were tuned, or how eta should scale with N or training length.

Reward rescaling (Sec. 2.3, Eq. 2):
- History R_t = { r_hat_i : i < t } of all raw (unscaled) rewards, over all arms.
- q_lo_t = 20th percentile of R_t. q_hi_t = 80th percentile of R_t.
- Eq. (2): r_t = -1 if r_hat_t < q_lo_t; r_t = +1 if r_hat_t > q_hi_t; otherwise r_t = 2 * (r_hat_t - q_lo_t) / (q_hi_t - q_lo_t) - 1.
- The full history is not stored. Reservoir sampling keeps a representative sample, and quantiles are read from it. The paper says this costs logarithmic time in the sample size (as relayed by the fetch; verify).
- Reservoir size: NOT STATED in the text I could read.

## 7. Which signals worked where, and vs uniform (Sec. 4, 5)

- ML training: PG most consistent (Sec. 5). Repeat copy: PG, SPG, TPG somewhat faster than uniform, mostly early. GPG, L2G, GL2G "much worse than uniform". VCG near uniform. bAbI: PG clearly better than uniform; SPG same direction but weaker; other signals roughly equal to or worse than uniform.
- VI training: GVCG best. Repeat copy: about 2x faster than uniform, and it reaches higher network complexity than uniform, which supports the authors' hypothesis (Fig. 2). bAbI: GVCG faster than uniform at first, then slightly worse. VCG poor for reasons "unclear" to the authors. VI as a whole "appeared to hamper progress" on bAbI.
- N-gram (Sec. 4.1): curriculum not needed; direct training on the 10-gram is the best way to learn the 10-gram. Complexity gains still homed in on the 10-gram.
- Uniform: "a surprisingly strong benchmark" (Sec. 5). (Quote 2 of 2.) Curriculum beat it clearly only in some settings.
- Repeat copy syllabus found by GVCG (Fig. 3): short sequences with many repeats first, then long sequences with few repeats (Sec. 4.2).
- bAbI syllabus (Fig. 5): PG and GVCG both go early to "Time Reasoning". PG stays on "Path Finding" until it is solved. The order from one-supporting-fact to three-supporting-facts tasks appears (Sec. 4.3).
- Figures: Fig. 1 n-gram policies; Fig. 2 repeat copy (target loss, policy entropy, complexity; truncated 1.1e9 steps); Fig. 3 GVCG policy and loss over five time bins (first 4e8 steps); Fig. 4 bAbI completion and entropy (3.5e8 steps); Fig. 5 per-task bAbI policy and error (2e8 steps, 5% line). All are averages over 10 runs.

## 8. Limitations and caveats the authors state

- Strong uniform baseline (Sec. 5). The curriculum often gives little over it.
- N-gram curriculum is superfluous (Sec. 4.1).
- PG is biased toward high gradient variance (Sec. 3.3). GPG adds Taylor error. SPG trades bias for variance. TPG is high variance. MPG adds evaluation-task variance (Sec. 3.1).
- VCG is noisy and unreliable (Sec. 4.1 to 4.3). Reason on bAbI is unknown. L2G is unreliable without a regulariser (Sec. 3.2).
- On repeat copy the authors suggest gradient-approximation bias may "pernicious[ly]" hurt learning for some signals (Sec. 4.2; exact wording not confirmed).
- S (dataset size in the VI loss) is undefined for generated data and was set by hand (Sec. 3.2, 4.2).
- tau(x) was just the longest sequence length. Other tasks may need a more complex cost term (Sec. 3, Alg. 1).
- bAbI: only training performance is reported (Sec. 4.3).
- Targets are assumed to be from a finite set. The authors say the framework extends to continuous targets (Sec. 2).
- Not stated by the authors: any discussion of the bandit getting stuck, of forgetting, or of future work. Exp3.S is motivated by a "piecewise stationary" best arm (Sec. 2.2), and epsilon/N is a floor on arm probability, but the paper does not call it a safeguard.

## 9. What matters for our blur/contrast bandit (my inference, not claims of the paper)

1. Start with PG. Our CNN uses plain maximum-likelihood cross-entropy, and PG was the paper's most consistent ML signal (Sec. 5). It needs no fresh batch. The cost is one extra forward pass on the same batch after the weight update. Skip GPG and GL2G first (they were much worse than uniform on repeat copy). Skip the complexity gains (they need a variational net). Add SPG later as an unbiased check: it needs a fresh batch from the same arm and two extra evaluations per step.
2. Beware the PG bias. For equal true progress, PG prefers arms with noisy gradients (Sec. 3.3). Blur and contrast levels also change loss scale and gradient noise. A blurry-first schedule from PG could come from gradient variance and not from learning value. Controls to run: uniform sampling over the same arm grid (the paper shows it is strong), SPG, and an arm-shuffled run. Keep robustness and shape bias out of the reward (as planned), and measure them only at the end.
3. Retune and log. The paper used 11 to 169 arms, batches of 16 to 32, and horizons up to 1e9 input steps. eta = 1e-3 and epsilon = 0.05 come from there. A 2-D blur x contrast grid with N around 25 would give epsilon/N around 0.002 per arm, and our run will have far fewer bandit updates. Expect to retune eta. Log policy entropy per arm over time (as Figs. 2 and 4 do) to see if the policy collapses. Use one arm per batch (Alg. 1). For fixed-size images, tau(x) is constant, so r_hat = nu.
4. The Eq. (2) rescaling uses one global reward history over all arms. It makes rewards scale-free, but arms that always give large loss drops will look good in every window.
