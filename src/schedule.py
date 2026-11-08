import numpy as np

from .degrade import sharp_arm


class Schedule:
    # When True, the trainer adds the post-update loss on the same batch to
    # reward_info["loss_after"] (PG reward).
    wants_loss_after = False
    # When True, the trainer draws a fresh batch from the chosen arm and adds its loss before and
    # after the update as reward_info["fresh_before"] / ["fresh_after"] (SPG reward).
    wants_fresh_batch = False
    # When True, the trainer degrades each image with its own arm from image_arms(step).
    per_image = False

    def next_arm(self, step):
        raise NotImplementedError

    def update(self, arm, reward_info):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass

    def save(self, out_dir):
        pass


class FixedArm(Schedule):
    def __init__(self, arm):
        self.arm = arm

    def next_arm(self, step):
        return self.arm


class SharpOnly(FixedArm):
    def __init__(self, arms):
        super().__init__(sharp_arm(arms))


class Sequence(Schedule):
    # Precomputed arm per step; stateless, so resume needs nothing.
    def __init__(self, seq):
        self.seq = np.asarray(seq, dtype=np.int64)

    def next_arm(self, step):
        return int(self.seq[step])


class PerImageRandom(Schedule):
    # Uniform random arm per image, a pure function of (seed, step) so resume needs nothing.
    # next_arm (the logged arm) is the first image's arm: a uniform sample of the batch's arms.
    per_image = True

    def __init__(self, n_arms, batch_size, seed):
        self.n_arms, self.batch_size, self.seed = n_arms, batch_size, seed

    def image_arms(self, step):
        return np.random.default_rng([self.seed, step]).integers(0, self.n_arms, self.batch_size)

    def next_arm(self, step):
        return int(self.image_arms(step)[0])


def staircase(n_sigma, n_contrast, total, frac):
    # Diagonal of the grid from most degraded to sharp, 4 equal stages over the first frac of steps.
    diag = [i * n_contrast + min(i, n_contrast - 1) for i in reversed(range(n_sigma))]
    seq = np.full(total, diag[-1])
    cut = int(total * frac)
    for k, arm in enumerate(diag):
        seq[k * cut // len(diag) : (k + 1) * cut // len(diag)] = arm
    return seq


def make_schedule(cfg, arms, total):
    c, seed = cfg["condition"], cfg["seed"]
    n_s, n_c = len(cfg["sigmas"]), len(cfg["contrasts"])
    if c == "sharp":
        return SharpOnly(arms)
    if c == "fixed":
        return FixedArm(cfg["arm"])
    if c == "random":
        return Sequence(np.random.default_rng(seed).integers(0, len(arms), total))
    if c == "random_image":
        return PerImageRandom(len(arms), cfg["batch_size"], seed)
    if c in ("random_blur", "random_contrast"):
        axis = [i for i, (s, k) in enumerate(arms) if (k == 1.0 if c == "random_blur" else s == 0.0)]
        return Sequence(np.random.default_rng(seed).choice(axis, total))
    if c == "staircase":
        return Sequence(staircase(n_s, n_c, total, cfg["staircase_frac"]))
    if c == "reverse":
        return Sequence(staircase(n_s, n_c, total, cfg["staircase_frac"])[::-1])
    if c == "shuffled":
        src = np.load(cfg["arms_path"])
        if len(src) != total or src.min() < 0 or src.max() >= len(arms):
            raise ValueError(f"{cfg['arms_path']}: need {total} complete arm entries")
        return Sequence(np.random.default_rng(seed).permutation(src))
    if c == "teacher":
        from .teacher import Teacher

        return Teacher(len(arms), total, cfg["signal"], cfg["eta"], cfg["eps"], cfg["alpha"],
                       cfg["reward_window"], seed, cfg.get("importance", True))
    if c == "alp":
        from .teacher_alp import AlpGmm

        return AlpGmm(cfg["sigmas"], cfg["contrasts"], total, cfg.get("p_rnd", 0.2), cfg.get("fit_rate", 250),
                      cfg.get("k_max", 10), seed)
    raise ValueError(f"unknown condition: {c}")
