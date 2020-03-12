from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import models


def one_hot(labels: torch.Tensor, target: torch.Tensor) \
        -> torch.Tensor:
    return (labels - target.view(1, -1)).abs().argmin(0)


class Calib(nn.Module):
    def __init__(self, focal_shape: int = 300, size: Tuple[int, int] = (300, 300)):
        super(Calib, self).__init__()
        self.focal_shape = focal_shape
        self.size = size
        self.feature = models.inception_v3(pretrained=True)
        self.feature.fc = nn.Identity(2048)

        self.fc1 = nn.Linear(2048 + focal_shape, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        features = self.backbone(x)
        torch.cat((features, ))
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
