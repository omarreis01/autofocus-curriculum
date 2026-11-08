import json

import numpy as np
import pytest
import torch

from src.data import fake_cifar10
from src.degrade import arm_grid
from src.schedule import FixedArm, make_schedule
from src.train import load_config, train


class Crash(Exception):
    pass


def crash_at(schedule, step):
    next_arm = schedule.next_arm

    def crashing(s):
        if s == step:
            raise Crash
        return next_arm(s)

    schedule.next_arm = crashing
    return schedule


@pytest.mark.parametrize("condition", ["fixed", "fixed_gn", "teacher_pg", "teacher_spg", "teacher_rpg", "alp"])
def test_resume_after_crash_matches_uninterrupted_run(tmp_path, condition):
    cfg = load_config(None) | {"epochs": 4, "batch_size": 8, "ckpt_every": 2, "lr": 0.05}
    if condition.startswith("teacher"):
        cfg |= {"condition": "teacher", "signal": condition[8:], "eta": 0.5}  # large eta: the policy moves
    elif condition == "alp":
        cfg |= {"condition": "alp", "fit_rate": 3}  # refits at steps 3 and 6: resume restores a fitted GMM
    elif condition == "fixed_gn":
        cfg |= {"condition": "fixed", "arm": 5, "norm": "gn"}
    else:
        cfg |= {"condition": condition, "arm": 5}
    data = fake_cifar10(16, 16, torch.device("cpu"))  # 2 steps per epoch
    new = lambda: make_schedule(cfg, arm_grid(cfg["sigmas"], cfg["contrasts"]), 8)

    ref = train(cfg, data, new(), tmp_path / "ref")

    run = tmp_path / "run"
    with pytest.raises(Crash):
        # Dies in epoch 4: epochs 1-3 are logged, the last checkpoint is from epoch 2.
        train(cfg, data, crash_at(new(), 6), run)
    resumed = train(cfg, data, new(), run, resume=True)

    for k, v in ref.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[k], v, rtol=0, atol=0, msg=k)
    np.testing.assert_array_equal(np.load(run / "arms.npy"), np.load(tmp_path / "ref" / "arms.npy"))
    if not condition.startswith("fixed"):
        a, b = np.load(run / "teacher.npz"), np.load(tmp_path / "ref" / "teacher.npz")
        for k in a.files:
            np.testing.assert_array_equal(a[k], b[k])
    has_bn = any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in resumed.modules())
    assert has_bn == (cfg["norm"] == "bn")
    rows = [json.loads(r)["epoch"] for r in (run / "metrics.jsonl").read_text().splitlines()]
    assert rows == [1, 2, 3, 4]
