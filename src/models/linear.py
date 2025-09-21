# src/models/linear.py
from __future__ import annotations
import torch
from torch import nn


class LinearRegressor(nn.Module):
    """
    シンプルな線形回帰モデル（PyTorch実装）
    
    Args:
        in_features: 入力特徴量の数
        dropout: ドロップアウト率（線形層前に適用、デフォルト=0.0で無効）
    """
    
    def __init__(self, in_features: int, dropout: float = 0.0):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.linear = nn.Linear(in_features, 1)
        
        # 重みの初期化（Xavier uniform）
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: 入力テンソル [batch_size, in_features]
            
        Returns:
            回帰予測値 [batch_size, 1]
        """
        x = self.dropout(x)
        x = self.linear(x)
        return x.squeeze(-1)  # [batch_size, 1] -> [batch_size]
    
    @property
    def n_params(self) -> int:
        """総パラメータ数を返す"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_feature_importance(self) -> torch.Tensor:
        """
        線形層の重みを特徴量重要度として返す
        
        Returns:
            特徴量重要度 [in_features]
        """
        return self.linear.weight.squeeze(0).detach()