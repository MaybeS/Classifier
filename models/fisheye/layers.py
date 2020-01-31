import torch
import torch.nn as nn
import torch.nn.functional as F


class SweepLayer(nn.Module):
    def __init__(self, channels: int = 32, grid=None):
        super(SweepLayer, self).__init__()
        self.grid = grid
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> \
            torch.Tensor:
        x = F.grid_sample(x, self.grid)
        x = self.conv1(x)
        return x
