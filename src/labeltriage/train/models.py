"""Backbones: a CIFAR-style ResNet-18 trained from scratch, and a Hugging Face text classifier."""

from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    """ResNet-18 with a 3x3 stem and no max-pool, the usual variant for 32x32 inputs."""

    def __init__(self, num_classes: int = 10, width: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU()
        )
        layers, in_planes = [], width
        for i, stride in enumerate((1, 2, 2, 2)):
            planes = width * 2**i
            layers += [BasicBlock(in_planes, planes, stride), BasicBlock(planes, planes)]
            in_planes = planes
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(in_planes, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(self.stem(x))
        return self.head(torch.flatten(nn.functional.adaptive_avg_pool2d(out, 1), 1))


def text_classifier(name: str, num_classes: int):
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_classes)
