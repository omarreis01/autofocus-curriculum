from .degrade import sharp_arm


class Schedule:
    # When True, the trainer adds the post-update loss on the same batch to
    # reward_info["loss_after"] (needed by the PG reward in F2).
    wants_loss_after = False

    def next_arm(self, step):
        raise NotImplementedError

    def update(self, arm, reward_info):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


class FixedArm(Schedule):
    def __init__(self, arm):
        self.arm = arm

    def next_arm(self, step):
        return self.arm


class SharpOnly(FixedArm):
    def __init__(self, arms):
        super().__init__(sharp_arm(arms))


def make_schedule(cfg, arms):
    if cfg["condition"] == "sharp":
        return SharpOnly(arms)
    if cfg["condition"] == "fixed":
        return FixedArm(cfg["arm"])
    raise ValueError(f"unknown condition: {cfg['condition']}")
