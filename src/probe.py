import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import MEAN, load_cifar10, normalize, to_float
from .degrade import gaussian_blur
from .eval import recalibrate_bn
from .train import amp_dtype, autocast, build_model

SIGMAS = (1.0, 2.0, 3.0)
EPS = {"1/255": 1 / 255, "2/255": 2 / 255, "4/255": 4 / 255}


def forward(model, x):
    # x: float images in [0,1] pixel space; the model sees them normalized, like prepare().
    x = normalize(x)
    if x.is_cuda:
        x = x.contiguous(memory_format=torch.channels_last)
    return model(x)


@torch.no_grad()
def predict(model, x, dtype, batch=1000):
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        with autocast(dtype):
            out.append(forward(model, x[i : i + batch]).argmax(1))
    return torch.cat(out)


def highpass(x, sigma):
    # Wang et al. 2020 (HFC) feed the zero-mean high-frequency residual; we use a Gaussian instead of a
    # Fourier mask and add the dataset channel mean back, so the normalized input is centred on 0 like
    # clean images (a zero-mean residual would land at -mean/std, about -2). No clamping.
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    return x - gaussian_blur(x, sigma) + mean


def hybrid(a, b, sigma):
    # lowpass(A) + highpass(B) (Oliva-style hybrid, no gray offset, no clamping).
    return gaussian_blur(a, sigma) + b - gaussian_blur(b, sigma)


def pair_indices(y, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    yc = y.cpu()
    a = torch.randint(0, len(yc), (3 * n,), generator=g)
    b = torch.randint(0, len(yc), (3 * n,), generator=g)
    keep = yc[a] != yc[b]
    return a[keep][:n].to(y.device), b[keep][:n].to(y.device)


def shape_bias(model, x, y, a, b, sigma, dtype, batch=1000):
    pred = predict(model, hybrid(x[a], x[b], sigma), dtype, batch)
    pa = (pred == y[a]).float().mean().item()
    pb = (pred == y[b]).float().mean().item()
    return {"bias": pa / (pa + pb) if pa + pb > 0 else None, "p_low": pa, "p_high": pb}


def pgd(model, x, y, eps, steps, step_size, rand_start, generator=None):
    # L_inf attack in [0,1] pixel space, fp32, eval-mode BN. steps=1, step_size=eps, no start = FGSM.
    model.eval()
    delta = torch.zeros_like(x)
    if rand_start:
        delta = (torch.rand(x.shape, device=x.device, generator=generator) * 2 - 1) * eps
    delta = (x + delta).clamp(0, 1) - x
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.cross_entropy(forward(model, x + delta).float(), y, reduction="sum")
        (grad,) = torch.autograd.grad(loss, delta)
        delta = (delta.detach() + step_size * grad.sign()).clamp(-eps, eps)
        delta = (x + delta).clamp(0, 1) - x
    return (x + delta).detach()


def adv_accuracy(model, x, y, eps, steps, batch=500, seed=0):
    g = torch.Generator(device=x.device).manual_seed(seed)
    correct = 0
    for i in range(0, len(y), batch):
        xb, yb = x[i : i + batch], y[i : i + batch]
        if steps == 1:
            xa = pgd(model, xb, yb, eps, 1, eps, False)
        else:
            xa = pgd(model, xb, yb, eps, steps, eps / 4, True, g)
        correct += (predict(model, xa, None, batch) == yb).sum().item()
    return correct / len(y)


def probe(model, data, n_hybrid=10000, n_pgd=2000, batch=1000, seed=0):
    x, y = to_float(data["x_test"]), data["y_test"]
    dtype = amp_dtype(x.device)
    acc = lambda xs: (predict(model, xs, dtype, batch) == y).float().mean().item()
    res, sec = {"clean": acc(x)}, {}

    t = time.time()
    res["lowpass"] = {str(s): acc(gaussian_blur(x, s)) for s in SIGMAS}
    res["highpass"] = {str(s): acc(highpass(x, s)) for s in SIGMAS}
    sec["filters"] = time.time() - t

    t = time.time()
    a, b = pair_indices(y, n_hybrid, seed)
    res["hybrid"] = {str(s): shape_bias(model, x, y, a, b, s, dtype, batch) for s in SIGMAS}
    sec["hybrid"] = time.time() - t

    t = time.time()
    res["fgsm"] = {k: adv_accuracy(model, x, y, e, 1, batch // 2, seed) for k, e in EPS.items()}
    sec["fgsm"] = time.time() - t

    t = time.time()
    xs, ys = x[:n_pgd], y[:n_pgd]
    res["pgd_clean_fp32"] = adv_accuracy(model, xs, ys, 0.0, 1, batch // 2, seed)
    res["pgd10"] = {k: adv_accuracy(model, xs, ys, e, 10, batch // 2, seed) for k, e in EPS.items()}
    sec["pgd10"] = time.time() - t

    res["n"] = {"test": len(y), "hybrid": len(a), "pgd": len(ys)}
    res["sec"] = {k: round(v, 1) for k, v in sec.items()}
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-hybrid", type=int, default=10000)
    p.add_argument("--n-pgd", type=int, default=2000)
    p.add_argument("--batch", type=int, default=1000)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    model = build_model(device, ck.get("cfg", {}).get("norm", "bn"))
    model.load_state_dict(ck["model"])
    t0 = time.time()
    data = load_cifar10(a.data_dir, device)
    recalibrate_bn(model, data)
    res = probe(model, data, a.n_hybrid, a.n_pgd, a.batch)
    res["ckpt"] = a.ckpt
    res["sec"]["total"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    hb = {s: round(v["bias"], 3) if v["bias"] is not None else None for s, v in res["hybrid"].items()}
    print(f"clean {res['clean']:.4f} bias {hb} pgd10 {res['pgd10']} ({res['sec']['total']}s)")


if __name__ == "__main__":
    main()
