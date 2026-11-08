import numpy as np
import pytest
import torch

from src.data import fake_cifar10
from src.degrade import arm_grid
from src.schedule import FixedArm, make_schedule
from src.teacher import Teacher
from src.teacher_alp import AlpGmm
from src.train import load_config, train

CFG = load_config(None) | {"epochs": 2, "batch_size": 8, "lr": 0.05}
DATA = fake_cifar10(16, 16, torch.device("cpu"))


class MeasuredFixedArm(FixedArm):
    wants_loss_after = True
    wants_fresh_batch = True


def test_reward_measurements_do_not_change_training(tmp_path):
    # BN running stats and the data stream must not see the extra PG/SPG forwards.
    ref = train(CFG, DATA, FixedArm(5), tmp_path / "ref")
    measured = train(CFG, DATA, MeasuredFixedArm(5), tmp_path / "measured")
    for k, v in ref.state_dict().items():
        torch.testing.assert_close(measured.state_dict()[k], v, rtol=0, atol=0, msg=k)


@pytest.mark.parametrize("signal", ["pg", "spg", "rpg", "rspg"])
def test_network_that_does_not_learn_gives_no_reward_and_uniform_policy(tmp_path, signal):
    cfg = CFG | {"lr": 0.0, "condition": "teacher", "signal": signal}
    train(cfg, DATA, make_schedule(cfg, arm_grid(cfg["sigmas"], cfg["contrasts"]), 4), tmp_path)
    log = np.load(tmp_path / "teacher.npz")
    assert (log["rewards"] == 0).all()
    np.testing.assert_allclose(log["probs"], 1 / 16)


def run_synthetic(mean_reward, steps=2910, seed=0):
    # Default hyperparameters over the short-budget horizon; raw reward = per-arm mean + unit noise.
    teacher = Teacher(16, steps, seed=seed)
    rng = np.random.default_rng(seed + 1)
    for t in range(steps):
        arm = teacher.next_arm(t)
        raw = mean_reward[arm] + rng.normal()
        teacher.update(arm, {"step": t, "loss": torch.tensor(raw), "loss_after": torch.tensor(0.0)})
    return teacher.probs[-steps // 10 :].mean(0)


def test_teacher_concentrates_on_arm_with_higher_reward():
    mean = np.zeros(16)
    mean[11] = 1.0
    probs = run_synthetic(mean)
    assert probs.argmax() == 11 and probs[11] > 0.5


def test_teacher_stays_near_uniform_when_all_arms_reward_alike():
    assert run_synthetic(np.zeros(16)).max() < 0.3


def test_alp_gmm_concentrates_on_arm_with_steady_progress():
    # Arm 5's loss falls 0.1 per visit; every other arm's loss is flat noise. Uniform sampling hits arm 5 ~10%.
    # ALP-GMM keeps exploring (20% random, ALP-proportional choice over noise Gaussians), so expect a clear
    # majority arm, not a collapse onto it.
    cfg, steps = load_config(None), 3000
    alp = AlpGmm(cfg["sigmas"], cfg["contrasts"], steps, seed=0)
    rng, visits, arms = np.random.default_rng(1), 0, []
    for t in range(steps):
        arm = alp.next_arm(t)
        visits += arm == 5
        loss = 100 - 0.1 * visits if arm == 5 else 1 + 0.01 * rng.normal()
        alp.update(arm, {"step": t, "loss": torch.tensor(loss)})
        arms.append(arm)
    share = np.bincount(arms[-steps // 5 :], minlength=16) / (steps // 5)
    assert share[5] > 0.3 and share[5] > 3 * np.delete(share, 5).max()
