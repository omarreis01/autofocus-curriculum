import math

import torch
import torch.nn.functional as F

SIGMAS = (0.0, 0.75, 1.5, 3.0)
CONTRASTS = (1.0, 0.6, 0.35, 0.2)


def arm_grid(sigmas=SIGMAS, contrasts=CONTRASTS):
    # arm index = i_sigma * len(contrasts) + i_contrast
    return [(float(s), float(c)) for s in sigmas for c in contrasts]


def sharp_arm(arms):
    return arms.index((0.0, 1.0))


def gaussian_blur(x, sigma):
    if sigma <= 0:
        return x
    r = max(1, math.ceil(3 * sigma))
    t = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-(t**2) / (2 * sigma**2))
    k = (k / k.sum()).to(x.dtype)
    ch = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).expand(ch, 1, 1, -1), groups=ch)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).expand(ch, 1, -1, 1), groups=ch)
    return x


def reduce_contrast(x, c):
    if c >= 1:
        return x
    m = x.mean(dim=(2, 3), keepdim=True)
    return m + c * (x - m)


def degrade(x, sigma, c):
    return reduce_contrast(gaussian_blur(x, sigma), c)
