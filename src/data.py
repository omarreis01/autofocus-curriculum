import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "gaussian_blur",
    "snow", "frost", "fog", "spatter",
    "brightness", "contrast", "saturate",
    "elastic_transform", "pixelate", "jpeg_compression",
]
SEEN = ["gaussian_blur", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "contrast"]
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def download_cifar10(data_dir):
    import os

    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    from huggingface_hub import hf_hub_download
    from torchvision.datasets import CIFAR10

    # exact original tar; torchvision md5-checks it before extracting
    hf_hub_download("liangnanying/cifar-10-python", "cifar-10-python.tar.gz", repo_type="dataset", local_dir=data_dir)
    for train in (False, True):
        CIFAR10(data_dir, train=train, download=True)


def _decode_png(b):
    import io

    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


def download_cifar10c(data_dir):
    import os
    from multiprocessing import Pool

    data_dir = Path(data_dir)
    out = data_dir / "CIFAR-10-C"
    if (out / "labels.npy").exists() and all((out / f"{c}.npy").exists() for c in CORRUPTIONS):
        print(f"CIFAR-10-C present in {out}")
        return
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    from datasets import Image, load_from_disk
    from huggingface_hub import snapshot_download
    from torchvision.datasets import CIFAR10

    ref = np.array(CIFAR10(data_dir, train=False, download=True).targets)
    out.mkdir(parents=True, exist_ok=True)
    tmp = data_dir / "_cifar10c_hf"
    # unofficial arrow mirror: PNG images + corruption_name, corruption_level (1-5), label
    p = snapshot_download("robro/cifar10-c-parquet", repo_type="dataset", local_dir=tmp)
    ds = load_from_disk(p).cast_column("image", Image(decode=False))
    names = np.array(ds["corruption_name"])
    lv = np.array(ds["corruption_level"])
    lab = np.array(ds["label"])
    if sorted(set(names)) != sorted(CORRUPTIONS):
        raise RuntimeError(f"unexpected corruptions in mirror: {sorted(set(names))}")
    raw = [d["bytes"] for d in ds["image"]]
    with Pool(os.cpu_count()) as pool:
        imgs = np.stack(pool.map(_decode_png, raw, chunksize=2000))
    labels = np.tile(ref, 5)
    for c in CORRUPTIONS:
        idx = np.concatenate([np.where((names == c) & (lv == s))[0] for s in range(1, 6)])
        if len(idx) != 50000 or not (lab[idx] == labels).all():
            raise RuntimeError(f"{c}: mirror order does not match original CIFAR-10-C")
        np.save(out / f"{c}.npy", imgs[idx])
    np.save(out / "labels.npy", labels.astype(np.uint8))
    shutil.rmtree(tmp)


def load_cifar10(data_dir, device):
    from torchvision.datasets import CIFAR10

    def to_device(ds):
        x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous().to(device)
        y = torch.tensor(ds.targets, dtype=torch.long, device=device)
        return x, y

    x_train, y_train = to_device(CIFAR10(data_dir, train=True, download=False))
    x_test, y_test = to_device(CIFAR10(data_dir, train=False, download=False))
    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}


def fake_cifar10(n_train, n_test, device, seed=0):
    g = torch.Generator().manual_seed(seed)

    def make(n):
        x = torch.randint(0, 256, (n, 3, 32, 32), dtype=torch.uint8, generator=g)
        return x.to(device), torch.randint(0, 10, (n,), generator=g).to(device)

    x_train, y_train = make(n_train)
    x_test, y_test = make(n_test)
    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}


def load_cifar10c(c_dir, corruption, severity, limit=None):
    # Each <corruption>.npy holds 5 severities x 10k images (NHWC uint8); labels.npy is shared.
    c_dir = Path(c_dir)
    x = np.load(c_dir / f"{corruption}.npy", mmap_mode="r")
    y = np.load(c_dir / "labels.npy", mmap_mode="r")
    n = len(x) // 5
    lo, hi = (severity - 1) * n, (severity - 1) * n + (n if limit is None else min(limit, n))
    return np.array(x[lo:hi]), np.array(y[lo:hi])


def augment(x, generator):
    # Random crop (zero padding 4) + horizontal flip, on the batch's device. x: uint8 NCHW.
    b, _, h, w = x.shape
    dev = x.device
    xp = torch.nn.functional.pad(x, (4, 4, 4, 4))
    oy = torch.randint(0, 9, (b, 1), device=dev, generator=generator)
    ox = torch.randint(0, 9, (b, 1), device=dev, generator=generator)
    flip = torch.rand(b, 1, device=dev, generator=generator) < 0.5
    ar = torch.arange(w, device=dev)
    rows = oy + torch.arange(h, device=dev)
    cols = ox + torch.where(flip, w - 1 - ar, ar)
    out = xp[torch.arange(b, device=dev)[:, None, None], :, rows[:, :, None], cols[:, None, :]]
    return out.permute(0, 3, 1, 2)


def to_float(x):
    return x.float().div_(255)


def normalize(x):
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--skip-c", action="store_true")
    a = p.parse_args()
    download_cifar10(a.data_dir)
    if not a.skip_c:
        download_cifar10c(a.data_dir)


if __name__ == "__main__":
    main()
