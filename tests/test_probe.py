import pytest
import torch
import torch.nn as nn

from src.data import MEAN, STD, fake_cifar10
from src.probe import adv_accuracy, pair_indices, pgd, predict, probe, shape_bias


def coded_images(n=200, seed=0):
    # Class k: flat gray level 0.1 + 0.08k (low frequency) plus a period-2 checkerboard of amplitude
    # 0.01(k+1) (high frequency only), so each frequency band alone tells the class.
    y = torch.randint(0, 10, (n,), generator=torch.Generator().manual_seed(seed))
    ar = torch.arange(32)
    checker = ((ar[:, None] + ar[None, :]) % 2 * 2 - 1).float()
    x = (0.1 + 0.08 * y).view(-1, 1, 1, 1) + (0.01 * (y + 1)).view(-1, 1, 1, 1) * checker
    return x.expand(-1, 3, -1, -1).contiguous(), y


class Decoder(nn.Module):
    def __init__(self, band):
        super().__init__()
        self.band = band

    def forward(self, x):
        x = x * torch.tensor(STD).view(1, 3, 1, 1) + torch.tensor(MEAN).view(1, 3, 1, 1)
        if self.band == "low":
            k = (x.mean(dim=(1, 2, 3)) - 0.1) / 0.08
        else:
            k = (x[..., :, 1:] - x[..., :, :-1]).abs().mean(dim=(1, 2, 3)) / 0.02 - 1
        return -(torch.arange(10).float() - k[:, None]) ** 2


@pytest.mark.parametrize("band, sigma, expected", [("low", 2.0, 1.0), ("high", 2.0, 0.0), ("high", 0.0, 1.0)])
def test_hybrid_bias_follows_the_band_the_model_reads(band, sigma, expected):
    # sigma=0: highpass(B) is zero, so the hybrid is A and even a high-frequency reader picks A.
    x, y = coded_images()
    a, b = pair_indices(y, 100)
    assert (y[a] != y[b]).all()
    r = shape_bias(Decoder(band), x, y, a, b, sigma, None)
    assert r["bias"] == pytest.approx(expected)
    assert r["p_low" if expected else "p_high"] == pytest.approx(1.0)


def tiny_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(128, 10)).eval()


def test_attacks_stay_in_the_box_and_hurt_only_with_budget():
    model = tiny_model()
    x = torch.rand(64, 3, 32, 32, generator=torch.Generator().manual_seed(1))
    y = predict(model, x, None)  # labels = clean predictions, so clean accuracy is 1
    assert adv_accuracy(model, x, y, 0.0, 1) == 1.0
    eps = 8 / 255
    xa = pgd(model, x, y, eps, 10, eps / 4, True, torch.Generator().manual_seed(0))
    assert (xa - x).abs().max() <= eps + 1e-6 and xa.min() >= 0 and xa.max() <= 1
    assert adv_accuracy(model, x, y, eps, 10) < 0.5


def test_probe_reports_every_measure_on_fake_data():
    data = fake_cifar10(8, 64, torch.device("cpu"))
    res = probe(tiny_model(), data, n_hybrid=32, n_pgd=16, batch=32)
    sig, eps = {"1.0", "2.0", "3.0"}, {"1/255", "2/255", "4/255"}
    assert set(res["lowpass"]) == set(res["highpass"]) == set(res["hybrid"]) == sig
    assert set(res["fgsm"]) == set(res["pgd10"]) == eps
    assert set(res["hybrid"]["1.0"]) == {"bias", "p_low", "p_high"}
    assert res["n"] == {"test": 64, "hybrid": 32, "pgd": 16}
