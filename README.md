# autofocus-curriculum

Biological vision starts blurry and sharpens over development, and training deep networks on a similar blur-to-sharp schedule is known to improve robustness and shape bias. This project asks whether that schedule has to be hand-crafted. We let a learning-progress-driven selector (a non-stationary multi-armed bandit) pick among degradation levels during training and test (1) whether a coarse-to-fine ordering emerges on its own, (2) how the emergent schedule compares with fixed biomimetic and standard training on robustness (CIFAR-10-C, ImageNet-100-C, adversarial) and shape bias, and (3) under which conditions it fails to emerge.

Senior project, Boğaziçi University, Department of Computer Engineering, 2026–27. Advisor: Prof. Şuayb S. Arslan.
