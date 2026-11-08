import torch.nn as nn
import torch.nn.functional as F


def norm_layer(kind, ch):
    if kind == "bn":
        return nn.BatchNorm2d(ch)
    if kind == "gn":
        return nn.GroupNorm(32, ch)
    raise ValueError(f"unknown norm: {kind}")


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride, norm="bn"):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = norm_layer(norm, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = norm_layer(norm, cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), norm_layer(norm, cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    # CIFAR variant: 3x3 stem, no max-pool.
    def __init__(self, num_classes=10, norm="bn"):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), norm_layer(norm, 64), nn.ReLU(inplace=True))
        layers, cin = [], 64
        for cout, stride in [(64, 1), (128, 2), (256, 2), (512, 2)]:
            layers += [BasicBlock(cin, cout, stride, norm), BasicBlock(cout, cout, 1, norm)]
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))
