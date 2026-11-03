import numpy as np

from .schedule import Schedule


class RewardScaler:
    # Graves 2017 Eq. 2: map a raw reward to [-1, 1] by the 20th/80th percentiles of recent raw rewards.
    def __init__(self, window):
        self.buf = np.zeros(window)
        self.n = 0

    def __call__(self, raw):
        hist = self.buf[: min(self.n, len(self.buf))]
        r = 0.0
        if len(hist):
            lo, hi = np.percentile(hist, [20, 80])
            if hi > lo:
                r = float(np.clip(2 * (raw - lo) / (hi - lo) - 1, -1, 1))
        self.buf[self.n % len(self.buf)] = raw
        self.n += 1
        return r


class Exp3S:
    # Graves 2017 Eq. 1 with beta = 0; log-weights keep it stable over long runs.
    def __init__(self, n_arms, eta, eps, alpha, seed):
        self.w = np.zeros(n_arms)
        self.eta, self.eps, self.alpha = eta, eps, alpha
        self.t = 0
        self.rng = np.random.default_rng(seed)

    def probs(self):
        e = np.exp(self.w - self.w.max())
        return (1 - self.eps) * e / e.sum() + self.eps / len(self.w)

    def sample(self):
        return int(self.rng.choice(len(self.w), p=self.probs()))

    def update(self, arm, r):
        self.t += 1
        n = len(self.w)
        alpha = 1 / self.t if self.alpha is None else self.alpha
        v = self.w.copy()
        v[arm] += self.eta * r / self.probs()[arm]
        e = np.exp(v - v.max())
        w = np.log((1 - alpha) * e + alpha / (n - 1) * (e.sum() - e))
        self.w = w - w.max()


class Teacher(Schedule):
    def __init__(self, n_arms, total, signal="pg", eta=0.01, eps=0.05, alpha=None, reward_window=1000, seed=0):
        if signal not in ("pg", "spg"):
            raise ValueError(f"unknown signal: {signal}")
        self.signal = signal
        self.wants_loss_after = signal == "pg"
        self.wants_fresh_batch = signal == "spg"
        self.bandit = Exp3S(n_arms, eta, eps, alpha, seed)
        self.scaler = RewardScaler(reward_window)
        self.rewards = np.zeros(total, dtype=np.float32)
        self.probs = np.zeros((total, n_arms), dtype=np.float32)

    def next_arm(self, step):
        self.probs[step] = self.bandit.probs()
        return self.bandit.sample()

    def update(self, arm, info):
        if self.signal == "pg":
            raw = (info["loss"] - info["loss_after"]).item()  # the only per-step GPU sync
        else:
            raw = (info["fresh_before"] - info["fresh_after"]).item()
        self.rewards[info["step"]] = raw
        self.bandit.update(arm, self.scaler(raw))

    def state_dict(self):
        b, s = self.bandit, self.scaler
        return {"w": b.w.copy(), "t": b.t, "rng": b.rng.bit_generator.state, "buf": s.buf.copy(), "n": s.n,
                "rewards": self.rewards.copy(), "probs": self.probs.copy()}

    def load_state_dict(self, st):
        b, s = self.bandit, self.scaler
        b.w, b.t, b.rng.bit_generator.state = st["w"].copy(), st["t"], st["rng"]
        s.buf, s.n = st["buf"].copy(), st["n"]
        self.rewards, self.probs = st["rewards"].copy(), st["probs"].copy()

    def save(self, out_dir):
        np.savez(out_dir / "teacher.npz", rewards=self.rewards, probs=self.probs)
