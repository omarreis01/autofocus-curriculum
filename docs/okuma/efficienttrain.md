# Reading note: EfficientTrain (A) and EfficientTrain++ (B)

A = Wang et al., ICCV 2023, arXiv 2211.09703 (venue from the arXiv comments field: ICCV 2023; v1 17 Nov 2022, v3 16 Aug 2023).
B = Wang et al., TPAMI, arXiv 2405.08768 (arXiv comments field says it is accepted by TPAMI and is the journal version of A; v1 14 May 2024). Code: github.com/LeapLabTHU/EfficientTrain.

## Source and reliability (read this first)

- The PDF fetch of A returned binary data. It was not used.
- Text came from: ar5iv HTML for A (127,815 chars, both halves read), arxiv.org/html for B (207,080 chars; first 200,000 read; last ~7,000 chars, probably references, NOT read).
- The WebFetch tool runs a small summarizer model over the page. Facts and keyword scans below come from its answers, not from my own reading of the raw text. Table cells in A are partly garbled (see item 5, Table 4). Check the numbers against the PDF before citing them in the thesis.
- Section numbering: A uses Arabic numbers (Sec 4.3). B uses Roman numbers (Sec V-A). Appendix and table numbers are as the extraction showed them. A has duplicate table labels in the extraction (two tables labelled Table 7).
- Only 3 verbatim quotes are used in this file. They are marked with quotation marks.

---

## 1. Main claim

**A (Abstract, Sec 1).**
- Curriculum learning can be generalized. Do not pick easier samples. Use all samples in every epoch. But show only the easy patterns of each sample early, and add harder patterns as training goes on.
- Easy patterns = low-frequency components (Fourier cropping) and unaugmented image content (weak data augmentation).
- Result: more than 1.5x less wall-time on ImageNet-1K/22K with no accuracy loss. Also helps MAE (self-supervised).

**B (Abstract, Sec I).**
- Same idea. B calls the curriculum a "soft-selection function" over patterns inside each example.
- Schedules come from a search algorithm, now with a compute constraint.
- Result: 1.5-3.0x less training time on ImageNet-1K/22K with no accuracy loss. Also helps MAE.

## 2. How they tested it

**A.**
- Datasets: ImageNet-1K and ImageNet-22K (Sec 5, App A.1-A.2). Transfer: CIFAR, Flowers-102, Stanford Dogs (Sec 5.3, Table 11); MS COCO detection and segmentation (Table 12).
- Architectures: ResNet, ConvNeXt, DeiT, PVT, Swin, CSWin; MAE for self-supervised (Sec 5.1-5.2, Tables 6, 9, 10).
- Baselines: the same recipe at fixed full resolution, re-run by the authors, plus numbers from the original papers (Table 6, App A.1). Table 9 also compares with Progressive Learning, Auto Progressive Learning, Budgeted ViT and others.
- Recipe (App Table 13): 300 epochs, AdamW, LR 4e-3, cosine, batch 4096, RandAug(9, 0.5), Mixup 0.8, CutMix 1.0, random erasing 0.25, label smoothing 0.1.
- Headline numbers: Table 6 (ImageNet-1K, 300 ep). Wall-time speed-up about 1.44x (ResNet-50) to 1.56x (PVT-Small). Accuracy equal or slightly higher, for example ResNet-50 79.4 vs 78.8, CSWin-Small 83.6 vs 83.4. Table 7 (ImageNet-22K): CSWin-Large 469.5 to 307.7 GPU-days, about 1.53x. DeiT-S at 100 ep: 76.4 vs 75.5 (Table 7a).
- Speed is measured on NVIDIA 3090 GPUs (Table 6/7 captions).
- Minor point, NOT CONFIRMED: the abstract says at least 1.5x, but Table 6 shows 1.44x for ResNet-50 wall-time. I could not tell which measure the abstract uses.

**B.**
- Datasets: ImageNet-1K and 22K (Sec VI). Transfer: ADE20K, MS COCO, CIFAR, Flowers-102, Stanford Dogs, NABirds, Food-101 (Tables XVI-XVIII).
- Architectures: ResNet-50, ConvNeXt-T/S/B(/L), DeiT-T/S, PVT-T/S/M/L, Swin-T/S/B, CSWin-T/S/B(/L), CAFormer-S18/S36/M36 (Table VII). Plus MAE (Table XV).
- Baselines: standard fixed 224x224 training. Comparison with other efficient methods in Table XI: Progressive Learning, Auto Progressive Learning, Budgeted ViT, Budgeted Training, InfoBatch, DIH/Minimax/CurriculumNet/Self-paced curricula, FixRes.
- Headline numbers:
  - Table VII (200 equivalent epochs, ImageNet-1K, wall-time on 3090s): DeiT-T 72.5 to 73.7 (1.64x); Swin-T 81.3 to 81.6 (1.49x); ResNet-50 78.8 to 79.6 (1.45x); CAFormer-S18 83.5 to 83.4 (1.59x).
  - Table XII (ImageNet-22K pre-training): CSWin-Base and CSWin-Large about 3.0x with accuracy equal to the baseline (for example CSWin-L 86.8/87.8 vs 86.8/87.9 at 224/384). ConvNeXt-B/L about 1.5x.
  - Table XV (MAE + HOG, 400 epochs with ET++): 83.7 top-1, equal to the 1600-epoch baseline. The exact speed-up column is NOT CONFIRMED (the extraction was ambiguous: 3.98x vs 4.00x).
  - Largest speed-up claimed: 3.0x (Table XII). An ablation (Table XXI(d)) shows 3.8x at 86.0%, but the text does not present it as the main result.
- Search cost (Sec VI-D2, Table XIX): about 2-3x the cost of training Swin-T once. Table XIX lists 1088.1 GPU-hours for ET vs 425.8 for ET++. What exactly those two numbers measure is NOT CONFIRMED (probably search cost).

## 3. Figure 1

**A (Sec 1, Fig 1).** (a) Sample-wise curriculum: a discrete keep-or-drop decision for each example. (b) Generalized curriculum: a continuous function T_t applied to every example. It shows easy patterns (for example low frequencies) early. It adds harder patterns as training goes on.

**B (Sec I, Fig 1).** Same two panels, same message. B's Fig 1(b) names T_t(.) as the function and points to Sec IV for what easier-to-learn means.

## 4. Conclusion

**A (Sec 6).** Always use all data. Show easy patterns first, add hard patterns step by step. Improves training efficiency of strong networks on ImageNet-1K/22K, in supervised and self-supervised learning.

**B (Sec VII).**
- Same story, plus a compute-constrained sequential search to design the schedule.
- 1.5-3.0x less wall-time with no accuracy loss and no new tunable hyper-parameters.
- Future work listed: a more general or learnable transformation (meta-learning is named), separating easy and hard spatial-temporal regions, tabular data via feature masking.

## 5. Mechanism: how images become easier

Both papers use the same two knobs. Neither uses blur or contrast.

**Knob 1: low-pass filtering by cropping the Fourier spectrum.**
- A, Sec 4.1, Eq. 1: X_c = IDFT( crop_BxB( DFT(X) ) ). Keep the central BxB patch of the 2D spectrum, then invert. The output is a BxB image. B < H, W. This is lossless for the kept frequencies and makes the input smaller (so it is cheaper).
- Proposition 1 (A Sec 4.1, proof in App C; B Sec IV-A): cropping depends only on low frequencies. Plain down-sampling also depends on high frequencies.
- Analysis tools: a circular low-pass filter with radius r (A Fig 2/3, Table 2; B Fig 2/3, Table II). Table I (B; A has the same kind of table): a model trained on normal images does well on low-pass validation images first. At epoch 20 the numbers are 46.9 vs 23.9 (low-pass vs high-pass). Both reach 71.3 at the end. So networks learn low frequencies first.
- Low-pass for the first 50-75% of training leaves final accuracy mostly intact (B Table II, DeiT-S, baseline 80.3).
- The paper never uses the word blur. It says low-pass filtering. My inference: cropping + resizing is close to an ideal low-pass filter plus down-sampling. It also changes image size, which a Gaussian blur does not.

**Knob 2: weaker-to-stronger augmentation.**
- RandAugment magnitude m grows with time: m = (t/T) x m0, m0 = 9, other augmentations unchanged (A Sec 4.3; B Sec IV-C).
- A Table 4 (Sec 4.1) tests weaker RandAug early. Values: m = 1, 3, 5, 7, 9 give about 80.2-80.7 vs 80.3 baseline in one setting. Layout is garbled, so the mapping is tentative.

**Schedule form.**
- A, Table 5 (300 epochs): B = 160 for epochs 1-180, B = 192 for 181-240, B = 224 for 241-300. RandAug m rises linearly 0 to 9. A staircase approximation of a continuous B(t) (Sec 4.3; App B.2). For other final sizes gamma: B = 160, (160+gamma)/2, gamma (Sec 5.1).
- B, Table VI: schedule is defined over cumulative compute, not epochs. B = 96 for 0-20% of the compute budget, B = 160 for 20-60%, B = 224 for 60-100%. RandAug m rises linearly 0 to 9. Number of update steps changes with B so each stage keeps its compute share (Sec V-A).
- Implementation (B Sec V-C): large batch early with square-root LR scaling, Eq. 2; a replay buffer that reuses pre-processed mini-batches. ViTs need position-bias resizing when the input size changes (A App A.1). ConvNeXt needed B = 160 instead of 96 (B App A-A).
- Whether B schedules augmentations other than RandAug: NOT CONFIRMED (visible text does not mention Mixup, CutMix, random erasing in the schedule).

## 6. How the schedule is FOUND

**A: offline greedy search, then fixed.**
- Location: Sec 4.3, Algorithm 1.
- Procedure: split training into N stages (text says N = 5). Start with B = 224 in every stage. Go from stage N-1 backwards to stage 1. For each stage, make B as small as possible while validation accuracy stays at or above a0, the baseline accuracy with B = 224. (The prose says the search begins at the last stage. The algorithm loop starts at N-1, so stage N stays at 224.)
- Objective: minimize compute (via B) under an accuracy constraint.
- When: offline, once, before the real training. Not adaptive.
- Scope: run once on Swin-Tiny, ImageNet-1K, 300 epochs, about 7 training runs. The result is reused for all other models and settings with no re-tuning (Sec 4.3).
- Per-dataset or per-model: no. One schedule for all.
- Open point: Sec 4.3 says N = 5 but Table 5 shows 3 stages. The paper does not explain this (NOT CONFIRMED why). App B.2 / Table 15 compares N = 3, 4, 5, 10: speed-up 1.34-1.63x, accuracy about 81.3-81.4 on Swin-T. More stages (toward continuous) give only a limited gain.
- Open point: Validation accuracy is the constraint. A separate held-out split is not described (NOT CONFIRMED).

**B: offline compute-constrained sequential search, then fixed.**
- Location: Sec V-A, Algorithm 2, Table VI.
- Procedure: training is split into N stages. Each stage gets 1/N of the compute budget. Overall cost is a fixed fraction beta of the baseline (beta = 2/3, T0 = 300, T_ft = 10; my reading: 2/3 of baseline compute, about 1.5x theoretical saving). Stages are searched one after another. For each candidate B, train from the previous stage's checkpoint, fine-tune 10 epochs at 224x224, and keep the B with the best validation accuracy. Update steps vary with B to hold the stage budget. The loop runs i = 1 to N-1, so the last stage is not searched (Table VI sets B = 224 for the last 40%).
- Objective: maximize validation accuracy under a fixed compute budget. (A did the reverse: minimize compute under an accuracy floor.)
- When: offline, once, on Swin-Tiny / ImageNet-1K. Reused for other models and settings: the text says to "directly adopt the curriculum in Table VI for other models" (Sec V-A).
- Per-model search gives minor gains for notable cost (Sec VI-D2). Transfer across networks, datasets and data scales is tested in Table XXI.
- Value of N and number of candidates M for Algorithm 2: NOT CONFIRMED in the visible text.
- Cost: Algorithm 1 needs about N+M-1 full trainings; Algorithm 2 is cheaper (Sec V-A; Sec VI-D2 says about 2-3x one Swin-T training).

## 7. Robustness, shape bias, OOD

- **A: NOT MEASURED.** No ImageNet-C, adversarial, shape-bias or OOD evaluation. Evidence: keyword scan of the whole body text (corruption, adversarial, shape, texture, out-of-distribution: none in the body; adversarial only in reference titles). Sec 2 (frequency paragraph) says earlier work [55] studied ConvNet robustness through frequency and says this paper targets training efficiency instead. Sec 5.3 / Tables 11-12 are transfer-accuracy tests, not robustness tests.
- **B: NOT MEASURED.** Same result. Evidence: Sec II says the same about robustness work [61]. The word robustness in Sec VI-D2 means that the same schedule works across different architectures. It does not mean robustness to corruptions. The word adversarial appears only in reference [85]. corruption, ImageNet-C, shape, OOD: none. Transfer tests (Tables XVI-XVIII) are accuracy only.
- Caveat for both: scan was done by the fetch tool's summarizer. The last ~7,000 characters of B were not read.

## 8. Blur, contrast, infant vision, learning progress, bandits

| Term | A | B |
|---|---|---|
| blur | NOT FOUND (low-pass filtering used instead; Sec 4.1, App A.4, App C) | NOT FOUND (low-pass filtering; Sec IV-A, V-B) |
| contrast | NOT FOUND | NOT FOUND |
| infant / baby / child / developmental / biological vision | NOT FOUND | NOT FOUND |
| human learning (general only) | Sec 2: curriculum learning is "inspired by the organized learning order of examples in human curricula"; the starting-small idea (Elman, ref [23]) cited. Sec 4.1 cites Campbell and Robson [71] for Fourier analysis of visibility, not for development. | Sec II, Sec III: same human-curricula and starting-small wording. |
| learning progress (Graves-style) | NOT FOUND | NOT FOUND as a concept (only a generic phrase about learning progressing over time occurs once in Sec III; it is not the Graves concept) |
| bandit / reinforcement learning / meta-learning | NOT FOUND | NOT FOUND in the body. Meta-learning appears only as future work (Sec VII). |
| adaptive / online / dynamic | Sec 1 uses the word adaptive for the function that extracts easy patterns, but the function is fixed and hand-designed. The word dynamically means applied on the fly during training (Sec 3, Sec 4). No online adaptation of the schedule. | Same wording (Sec I, Sec III). Schedule is fixed. The word automated appears only as part of a baseline name (Auto Progressive Learning, Table XI). |

## 9. What B adds over A

From B Sec I (contribution bullets) and Sec V:
1. New schedule search (Sec V-A, Algorithm 2). Budget-constrained and sequential, instead of A's greedy search with an accuracy floor. Much cheaper search. Schedule is defined over cumulative compute. Starts at B = 96 instead of 160 (Table VI vs Table 5).
2. Efficient low-frequency down-sampling (Sec V-B). B x B low-pass filter then B x B down-sampling, done with a Lanczos-style resize on CPU to cut CPU-GPU I/O. (B says the filter is equal to sinc convolution, shown in App D. App D was not visible, so NOT CONFIRMED.)
3. Two implementation tricks (Sec V-C): early large batch with square-root LR scaling, and a replay buffer.
4. Larger experiments: more architectures (CAFormer added), ImageNet-22K up to 3.0x, MAE up to about 4x, more transfer sets, comparison with more methods (Table XI), more ablations (Tables XIII, XIV, XIX-XXII).
5. Headline speed-up goes from 1.5x (A) to 1.5-3.0x (B).
6. Not added: any robustness test, any adaptive or online schedule, any contrast axis.

## 10. Difference table: EfficientTrain / EfficientTrain++ vs our project

| Row | EfficientTrain (A) and EfficientTrain++ (B) | Our project |
|---|---|---|
| What is scheduled | Input bandwidth B (image size after Fourier cropping) and RandAugment magnitude. Both rise together from easy to hard. | Blur level and contrast level of the training image (two image-quality axes). |
| Who decides the schedule | The authors' search algorithm, run once on Swin-Tiny. A human then fixes a 3-stage staircase. The direction (easy to hard) is built in. | A non-stationary multi-armed bandit. The direction is not built in. The network's own learning signal decides. |
| When it is decided | Offline, before training (A Sec 4.3, Alg. 1; B Sec V-A, Alg. 2). Fixed during training. Reused for other models. | Online, during training, at every step or block. Adapts to the current model. |
| Objective / reward | A: minimize compute subject to validation accuracy >= baseline. B: maximize validation accuracy under a fixed compute budget. Goal = cheaper training. | Reward = learning progress only (Graves et al. 2017 style). No accuracy-at-end term, no compute term, no robustness term. |
| Measured at the end | Top-1 accuracy, wall-time or GPU time, transfer accuracy, MAE fine-tuning accuracy. Robustness, shape bias, OOD: NOT MEASURED. | Common-corruption robustness and shape bias (measured only after training), plus clean accuracy. Also what curriculum the bandit chose. |
| Motivation | Make training of large vision backbones faster. Networks learn low frequencies first, so skip hard detail early. General human-curricula / starting-small analogy only. | Test whether a network picks a blurry, low-contrast-first curriculum by itself, like infant vision, and whether this makes it more robust. |
| Form of the easy-to-hard order | An assumption plus evidence from a one-way test (low-pass vs high-pass validation accuracy, A Table 1/Fig 4, B Table I). | A question to answer. The order is an outcome, and could differ from low-frequency-first. |
| Contrast axis | None. | Yes. |
| Image size | Changes (B x B input, cheaper). | Same resolution (assumption; confirm in our design). |

## How to write our difference clearly (short draft points)

1. EfficientTrain proves a fixed, hand-built, resolution-based easy-to-hard schedule saves compute without losing accuracy. We ask a different question. We do not impose the order. We let a learning-progress bandit choose, and then ask if the choice looks like infant vision and helps robustness.
2. Their schedule search uses validation accuracy and compute. Our reward uses neither.
3. They never measure robustness or shape bias (A Sec 2 and Sec 5.3; B Sec II and Sec VI-D2). We measure it at the end only, so it cannot leak into the reward.
4. They use one schedule from Swin-Tiny for all models (A Sec 4.3; B Sec V-A). Ours adapts online to each model and run.
5. Use their findings as support, not as competition: networks learn low frequencies first (A Table 1; B Table I), and early low-pass training does not hurt final accuracy (B Table II). A fixed EfficientTrain-style staircase (A Table 5 or B Table VI) is a natural non-adaptive baseline arm for our experiments.
6. Risk: a reviewer may say it is EfficientTrain with a bandit. The reply is the table above: online vs offline, reward vs search objective, blur and contrast vs bandwidth and augmentation, robustness measured vs not.
7. Do not claim they never mention human learning. They do, in one general sentence (A Sec 2; B Sec II). They never mention infant vision.

## Open items to check in the PDFs before the thesis cites them

- A: N = 5 vs 3 stages (Sec 4.3 vs Table 5); abstract (at least 1.5x) vs 1.44x for ResNet-50 (Table 6); exact Table 4 values.
- B: N and M for Algorithm 2; the meaning of the Table XIX columns; the exact MAE speed-up in Table XV; App D (sinc / Lanczos); the last ~7,000 characters (not read).
- Everything here was read through a summarizer. Confirm numbers you plan to quote.
