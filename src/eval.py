import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import CORRUPTIONS, SEEN, load_cifar10, load_cifar10c
from .train import accuracy, amp_dtype, build_model


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
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device, weights_only=False)["model"])
    t0 = time.time()
    res = evaluate(model, load_cifar10(a.data_dir, device), Path(a.data_dir) / "CIFAR-10-C", a.limit, a.batch)
    res["ckpt"] = a.ckpt
    res["sec"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"clean {res['clean']:.4f} seen {res['seen_mean']:.4f} unseen {res['unseen_mean']:.4f} ({res['sec']}s)")


if __name__ == "__main__":
    main()
