import numpy as np
import pytest

from src.degrade import arm_grid, sharp_arm
from src.make_configs import condition_configs
from src.schedule import make_schedule
from src.train import load_config

ARMS = arm_grid(load_config(None)["sigmas"], load_config(None)["contrasts"])
MOST_DEGRADED = ARMS.index((3.0, 0.2))


def arm_sequence(name, total, tmp_path):
    cfg = load_config(None) | condition_configs("short", 0)[name]
    if name == "shuffled":
        cfg["arms_path"] = tmp_path / "teacher_arms.npy"
        np.save(cfg["arms_path"], np.arange(total) % len(ARMS) // 3)
    s = make_schedule(cfg, ARMS, total)
    return np.array([s.next_arm(t) for t in range(total)])


@pytest.mark.parametrize("name", list(condition_configs("short", 0)))
def test_every_condition_gives_a_valid_arm_for_every_step(tmp_path, name):
    arms = arm_sequence(name, 2910, tmp_path)
    assert ((arms >= 0) & (arms < len(ARMS))).all()


def test_staircase_goes_blurry_to_sharp_and_reverse_mirrors_it(tmp_path):
    total = 2910
    stair = arm_sequence("staircase", total, tmp_path)
    rev = arm_sequence("reverse", total, tmp_path)
    assert stair[0] == MOST_DEGRADED and (stair[total // 2 :] == sharp_arm(ARMS)).all()
    path = [ARMS[a] for a in stair[np.r_[0, np.flatnonzero(np.diff(stair)) + 1]]]
    assert path == [(3.0, 0.2), (1.5, 0.35), (0.75, 0.6), (0.0, 1.0)]
    np.testing.assert_array_equal(rev, stair[::-1])


def test_shuffled_replays_teacher_arm_counts_in_new_order(tmp_path):
    total = 2910
    src = np.arange(total) % len(ARMS) // 3
    shuffled = arm_sequence("shuffled", total, tmp_path)
    np.testing.assert_array_equal(np.bincount(shuffled, minlength=16), np.bincount(src, minlength=16))
    assert (shuffled != src).mean() > 0.5
