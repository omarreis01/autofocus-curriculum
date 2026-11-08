import numpy as np
from sklearn.mixture import GaussianMixture

from .schedule import Schedule


class AlpGmm(Schedule):
    # ALP-GMM (Portelas et al. 2019) over the box [min, max] of sigmas x contrasts, scaled to [0, 1]^2.
    # The trainer renders grid arms, so each continuous sample is quantized to the nearest arm.
    # Competence = -(training-batch loss before the update): free, but noisy.
    # ALP = |competence now - competence at the last visit of the nearest visited arm|. In rendered space the
    # nearest earlier task is the same arm once it has been visited, so ALP is mostly progress on that arm over time.
    # The GMM is fit on the continuous samples (not the 16 grid points) plus their ALP.
    def __init__(self, sigmas, contrasts, total, p_rnd=0.2, fit_rate=250, k_max=10, seed=0):
        self.lo = np.array([min(sigmas), min(contrasts)])
        self.span = np.array([max(sigmas), max(contrasts)]) - self.lo
        self.grid = (np.array([(s, c) for s in sigmas for c in contrasts]) - self.lo) / self.span
        self.p_rnd, self.fit_rate, self.k_max, self.seed = p_rnd, fit_rate, k_max, seed
        self.rng = np.random.default_rng(seed)
        self.u = np.zeros((total, 2))  # continuous samples, scaled
        self.alp = np.zeros(total)
        self.comp = np.zeros(total)
        self.last = np.full(len(self.grid), np.nan)  # latest competence per arm
        self.gmm = None  # (means, covariances) of the last fit

    def next_arm(self, step):
        if self.gmm is None or self.rng.random() < self.p_rnd:
            u = self.rng.random(2)
        else:
            means, covs = self.gmm
            p = means[:, 2].clip(0)
            p = p / p.sum() if p.sum() > 0 else np.full(len(p), 1 / len(p))
            k = self.rng.choice(len(p), p=p)
            u = self.rng.multivariate_normal(means[k, :2], covs[k, :2, :2]).clip(0, 1)
        self.u[step] = u
        return int(((self.grid - u) ** 2).sum(1).argmin())

    def update(self, arm, info):
        step, c = info["step"], -info["loss"].item()
        seen = np.flatnonzero(~np.isnan(self.last))
        if len(seen):
            j = seen[((self.grid[seen] - self.grid[arm]) ** 2).sum(1).argmin()]
            self.alp[step] = abs(c - self.last[j])
        self.comp[step], self.last[arm] = c, c
        if (step + 1) % self.fit_rate == 0:
            self.fit(step + 1)

    def fit(self, n):
        # Paper: refit every fit_rate samples on the last fit_rate (task, ALP) points; k in 2..k_max by AIC.
        x = np.column_stack([self.u[n - self.fit_rate : n], self.alp[n - self.fit_rate : n]])
        fits = [GaussianMixture(k, covariance_type="full", random_state=self.seed * 1_000_003 + n).fit(x)
                for k in range(2, min(self.k_max, len(x)) + 1)]
        best = min(fits, key=lambda g: g.aic(x))
        self.gmm = (best.means_, best.covariances_)

    def state_dict(self):
        return {"rng": self.rng.bit_generator.state, "u": self.u.copy(), "alp": self.alp.copy(),
                "comp": self.comp.copy(), "last": self.last.copy(), "gmm": self.gmm}

    def load_state_dict(self, st):
        self.rng.bit_generator.state = st["rng"]
        self.u, self.alp, self.comp, self.last = st["u"].copy(), st["alp"].copy(), st["comp"].copy(), st["last"].copy()
        self.gmm = st["gmm"]

    def save(self, out_dir):
        np.savez(out_dir / "teacher.npz", tasks=self.lo + self.u * self.span, alp=self.alp, competence=self.comp)
