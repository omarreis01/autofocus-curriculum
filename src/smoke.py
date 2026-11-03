import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .data import CORRUPTIONS, fake_cifar10, load_cifar10
from .degrade import arm_grid
from .eval import evaluate
from .schedule import FixedArm
from .train import load_config, train


def write_fake_cifar10c(c_dir, per_severity):
    rng = np.random.default_rng(0)
    c_dir.mkdir(parents=True)
    n = 5 * per_severity
    np.save(c_dir / "labels.npy", rng.integers(0, 10, n).astype(np.uint8))
    for name in CORRUPTIONS:
        np.save(c_dir / f"{name}.npy", rng.integers(0, 256, (n, 32, 32, 3), dtype=np.uint8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=None)
    p.add_argument("--fake-data", action="store_true")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--c-limit", type=int, default=4, help="CIFAR-10-C images per corruption x severity")
    a = p.parse_args()
    if not a.fake_data and not a.data_dir:
        p.error("--data-dir is required unless --fake-data")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} device {device}", end="")
    print(f" gpu {torch.cuda.get_device_name()}" if device.type == "cuda" else "")

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if a.fake_data:
            data = fake_cifar10(a.steps * a.batch_size, 500, device)
            c_dir = tmp / "CIFAR-10-C"
            write_fake_cifar10c(c_dir, a.c_limit)
        else:
            full = load_cifar10(a.data_dir, device)
            n = a.steps * a.batch_size
            data = {"x_train": full["x_train"][:n], "y_train": full["y_train"][:n],
                    "x_test": full["x_test"][:500], "y_test": full["y_test"][:500]}
            c_dir = Path(a.data_dir) / "CIFAR-10-C"
        print(f"data ready {time.time() - t0:.1f}s")

        cfg = load_config(None) | {"epochs": 1, "batch_size": a.batch_size, "ckpt_every": 1}
        arms = arm_grid(cfg["sigmas"], cfg["contrasts"])
        t1 = time.time()
        model = train(cfg, data, FixedArm(len(arms) - 1), tmp / "run")
        print(f"train {a.steps} steps on most degraded arm {arms[-1]}: {time.time() - t1:.1f}s")
        assert (tmp / "run" / "last.pt").exists() and (tmp / "run" / "arms.npy").exists()

        t2 = time.time()
        res = evaluate(model, data, c_dir, limit=a.c_limit)
        print(f"eval clean {res['clean']:.3f} seen {res['seen_mean']:.3f} unseen {res['unseen_mean']:.3f}: {time.time() - t2:.1f}s")
    print(f"smoke ok, total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
