import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import augment, fake_cifar10, load_cifar10, normalize, to_float
from .degrade import arm_grid, degrade
from .model import ResNet18
from .schedule import make_schedule

DEFAULTS = {
    "condition": "sharp",
    "arm": None,
    "seed": 0,
    "epochs": 30,
    "batch_size": 512,
    "lr": 0.2,
    "warmup_frac": 0.15,
    "weight_decay": 5e-4,
    "sigmas": [0.0, 0.75, 1.5, 3.0],
    "contrasts": [1.0, 0.6, 0.35, 0.2],
    "ckpt_every": 5,
    # teacher (Exp3.S); eta is 10x Graves' 1e-3 so the policy can move within 3k-10k steps
    "signal": "pg",
    "eta": 0.01,
    "eps": 0.05,
    "alpha": None,  # None: Graves' 1/t share rate
    "reward_window": 1000,
    "staircase_frac": 0.5,
    "arms_path": None,
}


def load_config(path):
    cfg = dict(DEFAULTS)
    if path:
        cfg.update(json.loads(Path(path).read_text()))
    return cfg


def lr_at(step, total, peak, warmup_frac):
    # Linear warmup then cosine to zero; a pure function of step so resume needs no scheduler state.
    warm = max(1, int(total * warmup_frac))
    if step < warm:
        return peak * (step + 1) / warm
    return 0.5 * peak * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def amp_dtype(device):
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast(dtype):
    return torch.autocast("cuda", dtype=dtype) if dtype else contextlib.nullcontext()


def prepare(x_uint8, sigma, c):
    x = normalize(degrade(to_float(x_uint8), sigma, c))
    if x.is_cuda:
        x = x.contiguous(memory_format=torch.channels_last)
    return x


@torch.no_grad()
def measure_loss(net, model, x, y, dtype):
    # Train-mode forward (batch statistics, like the training loss) that leaves BN running stats untouched.
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.track_running_stats = False
    try:
        with autocast(dtype):
            return F.cross_entropy(net(x), y).float()
    finally:
        for m in bns:
            m.track_running_stats = True


@torch.no_grad()
def accuracy(net, x_uint8, y, dtype, batch=1000):
    net.eval()
    correct = torch.zeros((), device=y.device)
    for i in range(0, len(y), batch):
        with autocast(dtype):
            logits = net(prepare(x_uint8[i : i + batch], 0.0, 1.0))
        correct += (logits.argmax(1) == y[i : i + batch]).sum()
    return correct.item() / len(y)


def train_step(net, opt, scaler, x, y, dtype):
    with autocast(dtype):
        loss = F.cross_entropy(net(x), y)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    return loss.detach().float()


def build_model(device):
    model = ResNet18().to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model


def save_atomic(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(cfg, data, schedule, out_dir, resume=False, compile=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = data["x_train"].device
    torch.manual_seed(cfg["seed"])
    arms = arm_grid(cfg["sigmas"], cfg["contrasts"])
    model = build_model(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.0, momentum=0.9, nesterov=True, weight_decay=cfg["weight_decay"])
    dtype = amp_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    n, bs = len(data["y_train"]), cfg["batch_size"]
    steps_per_epoch = n // bs
    total = steps_per_epoch * cfg["epochs"]
    arm_log = np.full(total, -1, dtype=np.int16)
    start_epoch = 0
    ckpt_path = out_dir / "last.pt"
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        schedule.load_state_dict(ck["schedule"])
        arm_log = ck["arms"]
        start_epoch = ck["epoch"]
        print(f"resumed from {ckpt_path} at epoch {start_epoch}", flush=True)
    metrics = out_dir / "metrics.jsonl"
    if metrics.exists():  # drop rows logged after the checkpoint we restart from
        rows = [r for r in metrics.read_text().splitlines() if json.loads(r)["epoch"] <= start_epoch]
        metrics.write_text("".join(r + "\n" for r in rows))

    def checkpoint(epoch):
        save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                     "schedule": schedule.state_dict(), "arms": arm_log, "epoch": epoch, "cfg": cfg}, ckpt_path)

    net = torch.compile(model) if compile else model
    for epoch in range(start_epoch, cfg["epochs"]):
        t0 = time.time()
        net.train()
        # Per-epoch generator: data order and augmentation are reproducible after resume.
        g = torch.Generator(device=device).manual_seed(cfg["seed"] * 100_003 + epoch)
        perm = torch.randperm(n, device=device, generator=g)
        if schedule.wants_fresh_batch:  # separate stream so the training data order is unchanged
            g2 = torch.Generator(device=device).manual_seed(10**9 + cfg["seed"] * 100_003 + epoch)
            perm2 = torch.randperm(n, device=device, generator=g2)
        loss_sum = torch.zeros((), device=device)
        for i in range(steps_per_epoch):
            step = epoch * steps_per_epoch + i
            idx = perm[i * bs : (i + 1) * bs]
            arm = schedule.next_arm(step)
            arm_log[step] = arm
            x = prepare(augment(data["x_train"][idx], g), *arms[arm])
            y = data["y_train"][idx]
            for group in opt.param_groups:
                group["lr"] = lr_at(step, total, cfg["lr"], cfg["warmup_frac"])
            info = {"step": step}
            if schedule.wants_fresh_batch:
                idx2 = perm2[i * bs : (i + 1) * bs]
                x2, y2 = prepare(augment(data["x_train"][idx2], g2), *arms[arm]), data["y_train"][idx2]
                info["fresh_before"] = measure_loss(net, model, x2, y2, dtype)
            loss = info["loss"] = train_step(net, opt, scaler, x, y, dtype)
            if schedule.wants_loss_after:
                info["loss_after"] = measure_loss(net, model, x, y, dtype)
            if schedule.wants_fresh_batch:
                info["fresh_after"] = measure_loss(net, model, x2, y2, dtype)
            schedule.update(arm, info)
            loss_sum += loss
        train_loss = loss_sum.item() / steps_per_epoch
        train_sec = time.time() - t0
        test_acc = accuracy(net, data["x_test"], data["y_test"], dtype)
        row = {"epoch": epoch + 1, "train_loss": train_loss, "test_acc": test_acc, "sec": round(train_sec, 2)}
        with open(metrics, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"epoch {epoch + 1}/{cfg['epochs']} loss {train_loss:.4f} acc {test_acc:.4f} {train_sec:.1f}s/epoch", flush=True)
        if (epoch + 1) % cfg["ckpt_every"] == 0 or epoch + 1 == cfg["epochs"]:
            checkpoint(epoch + 1)
    np.save(out_dir / "arms.npy", arm_log)
    schedule.save(out_dir)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--fake-data", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"device {device} {torch.cuda.get_device_name() if device.type == 'cuda' else ''}", flush=True)
    data = fake_cifar10(2048, 512, device) if a.fake_data else load_cifar10(a.data_dir, device)
    total = len(data["y_train"]) // cfg["batch_size"] * cfg["epochs"]
    schedule = make_schedule(cfg, arm_grid(cfg["sigmas"], cfg["contrasts"]), total)
    train(cfg, data, schedule, a.out_dir, resume=a.resume, compile=a.compile)


if __name__ == "__main__":
    main()
