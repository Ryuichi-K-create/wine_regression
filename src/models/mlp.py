# src/models/mlp.py
from __future__ import annotations
import torch
from torch import nn

class MLPRegressor(nn.Module):
    """
    シンプルな全結合回帰ネットワーク。
    [Linear -> ReLU -> Dropout] x N -> Linear(→1)
    """
    def __init__(self, in_features: int, hidden_dims=(128, 64), dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, in_features) -> (B,)
        return self.net(x).squeeze(1)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
