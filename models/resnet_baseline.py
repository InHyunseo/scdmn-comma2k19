"""
Plain ResNet18 regression baseline for comma2k19 (224x224, 1-output + tanh).
Ignores ctx_label; trained on all contexts mixed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


def _make_stage(in_planes, planes, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for s in strides:
        layers.append(BasicBlock(in_planes, planes, s))
        in_planes = planes * BasicBlock.expansion
    return nn.Sequential(*layers), in_planes


class ResNet18Reg(nn.Module):
    """ResNet18 with ImageNet stem, single scalar tanh output."""
    def __init__(self):
        super().__init__()
        ch = [64, 128, 256, 512]
        self.conv1 = nn.Conv2d(3, ch[0], 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(ch[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1, c1 = _make_stage(ch[0], ch[0], 2, stride=1)
        self.layer2, c2 = _make_stage(c1,    ch[1], 2, stride=2)
        self.layer3, c3 = _make_stage(c2,    ch[2], 2, stride=2)
        self.layer4, c4 = _make_stage(c3,    ch[3], 2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c4, 1)

    def forward(self, x, ctx_label=None):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out).flatten(1)
        return torch.tanh(self.fc(out))
