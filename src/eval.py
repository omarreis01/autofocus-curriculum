import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import CORRUPTIONS, SEEN, augment, load_cifar10, load_cifar10c
from .train import accuracy, amp_dtype, autocast, build_model, prepare


@torch.no_grad()
def recalibrate_bn(model, data, n_batches=100, batch=512, seed=0):
    # Training batches each come from one arm, so the running stats mix arms and do not fit clean
    # images. Re-estimate them on clean training images (no test data) before every eval.
    device = data["x_train"].device
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None
    model.train()
    g = torch.Generator(device=device).manual_seed(seed)
    dtype = amp_dtype(device)
    for _ in range(n_batches):
        idx = torch.randint(0, len(data["y_train"]), (batch,), device=device, generator=g)
        with autocast(dtype):
            model(prepare(augment(data["x_train"][idx], g), 0.0, 1.0))


def evaluate(model, data, c_dir, limit=None, batch=2000):
    device = data["x_test"].device
    dtype = amp_dtype(device)
    res = {"clean": accuracy(model, data["x_test"], data["y_test"], dtype, batch), "cifar10c": {}}
    for name in CORRUPTIONS:
        res["cifar10c"][name] = {}
        for sev in range(1, 6):
            x, y = load_cifar10c(c_dir, name, sev, limit)
            x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(device)
            y = torch.from_numpy(y).long().to(device)
            res["cifar10c"][name][str(sev)] = accuracy(model, x, y, dtype, batch)
    mean = {n: float(np.mean(list(v.values()))) for n, v in res["cifar10c"].items()}
    res["corruption_mean"] = mean
    res["seen_mean"] = float(np.mean([mean[n] for n in SEEN]))
    res["unseen_mean"] = float(np.mean([mean[n] for n in CORRUPTIONS if n not in SEEN]))
    res["all_mean"] = float(np.mean(list(mean.values())))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None, help="images per corruption x severity")
    p.add_argument("--batch", type=int, default=2000)
    p.add_argument("--no-recal", action="store_true", help="keep the BN running stats from training")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    model = build_model(device, ck["cfg"].get("norm", "bn"))
    model.load_state_dict(ck["model"])
    t0 = time.time()
    data = load_cifar10(a.data_dir, device)
    if not a.no_recal:
        recalibrate_bn(model, data)
    res = evaluate(model, data, Path(a.data_dir) / "CIFAR-10-C", a.limit, a.batch)
    res["bn_recal"] = not a.no_recal
    res["ckpt"] = a.ckpt
    res["sec"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"clean {res['clean']:.4f} seen {res['seen_mean']:.4f} unseen {res['unseen_mean']:.4f} ({res['sec']}s)")


if __name__ == "__main__":
    main()
