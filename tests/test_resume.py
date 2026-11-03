import json

import numpy as np
import pytest
import torch

from src.data import fake_cifar10
from src.schedule import FixedArm
from src.train import load_config, train


class Crash(Exception):
    pass


class CrashAt(FixedArm):
    def __init__(self, arm, step):
        super().__init__(arm)
        self.crash_step = step

    def next_arm(self, step):
        if step == self.crash_step:
            raise Crash
        return self.arm


def test_resume_after_crash_matches_uninterrupted_run(tmp_path):
    cfg = load_config(None) | {"epochs": 4, "batch_size": 8, "ckpt_every": 2, "lr": 0.05}
    data = fake_cifar10(16, 16, torch.device("cpu"))  # 2 steps per epoch

    ref = train(cfg, data, FixedArm(5), tmp_path / "ref")

    run = tmp_path / "run"
    with pytest.raises(Crash):
        # Dies in epoch 4: epochs 1-3 are logged, the last checkpoint is from epoch 2.
        train(cfg, data, CrashAt(5, step=6), run)
    resumed = train(cfg, data, FixedArm(5), run, resume=True)

    for k, v in ref.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[k], v, rtol=0, atol=0, msg=k)
    np.testing.assert_array_equal(np.load(run / "arms.npy"), np.load(tmp_path / "ref" / "arms.npy"))
    rows = [json.loads(r)["epoch"] for r in (run / "metrics.jsonl").read_text().splitlines()]
    assert rows == [1, 2, 3, 4]
