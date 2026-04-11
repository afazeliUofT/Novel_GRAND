from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


class TabularMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Iterable[int], out_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        dims: List[int] = [in_dim] + list(hidden_dims)
        layers = []
        for d0, d1 in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d0, d1))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
