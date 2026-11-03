import pytest
import torch

from src.data import augment
from src.degrade import arm_grid, degrade, gaussian_blur, reduce_contrast
from src.schedule import SharpOnly


def images(n=4, seed=0):
    return torch.rand(n, 3, 32, 32, generator=torch.Generator().manual_seed(seed))


def high_freq_energy(x):
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return (dx**2).mean() + (dy**2).mean()


def test_sharp_arm_is_identity():
    x = images()
    assert torch.equal(degrade(x, 0.0, 1.0), x)


@pytest.mark.parametrize("c", [0.6, 0.35, 0.2])
def test_contrast_keeps_channel_mean_and_scales_std(c):
    x = images()
    y = reduce_contrast(x, c)
    torch.testing.assert_close(y.mean(dim=(2, 3)), x.mean(dim=(2, 3)))
    torch.testing.assert_close(y.std(dim=(2, 3)), c * x.std(dim=(2, 3)))


def test_blur_removes_more_high_freq_energy_as_sigma_grows():
    x = images()
    energies = [high_freq_energy(gaussian_blur(x, s)).item() for s in (0.0, 0.75, 1.5, 3.0)]
    assert all(a > b for a, b in zip(energies, energies[1:])), energies
    out = gaussian_blur(x, 3.0)
    assert out.shape == x.shape
    torch.testing.assert_close(out.mean(dim=(2, 3)), x.mean(dim=(2, 3)), atol=0.02, rtol=0)


def test_arm_grid_mapping():
    arms = arm_grid()
    assert len(arms) == 16
    assert arms[0] == (0.0, 1.0)
    assert arms[1 * 4 + 2] == (0.75, 0.35)
    assert arms[15] == (3.0, 0.2)
    custom = arm_grid([1.5, 0.0], [0.5, 1.0])
    assert custom[SharpOnly(custom).next_arm(0)] == (0.0, 1.0)


def test_augment_returns_padded_crop_or_its_mirror():
    x = torch.randint(1, 256, (16, 3, 32, 32), dtype=torch.uint8)
    out = augment(x, torch.Generator().manual_seed(0))
    xp = torch.nn.functional.pad(x, (4, 4, 4, 4))
    flips = 0
    for i in range(len(x)):
        crops = [xp[i, :, oy : oy + 32, ox : ox + 32] for oy in range(9) for ox in range(9)]
        if any(torch.equal(out[i], c) for c in crops):
            continue
        assert any(torch.equal(out[i], c.flip(-1)) for c in crops), i
        flips += 1
    assert 0 < flips < len(x)
