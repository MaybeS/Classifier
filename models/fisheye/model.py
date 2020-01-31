from typing import Union

import torch
import torch.nn as nn
from torchvision import models

from .layers import SweepLayer


class FisheyeNet(nn.Module):
    def __init__(self, backbone):
        super(FisheyeNet, self).__init__()
        self.features = self._backbone(backbone)
        self.sweep = SweepLayer()

    def forward(self, x: torch.Tensor) \
            -> torch.Tensor:
        x = self.features(x)
        x = self.sweep(x)

        return x

    @staticmethod
    def _backbone(base: Union[str, nn.Module] = None, **kwargs) \
            -> nn.Module:

        base = base or 'resnet101'

        if not kwargs:
            kwargs = {'pretrained': True}

        try:
            backbone = getattr(models, base, None)
            backbone = base if backbone is None else backbone(**kwargs).features

        except RuntimeError as e:
            raise NotImplementedError('Not Implemented')

        return backbone
