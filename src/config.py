# src/config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class Config:
    # paths
    data_path: str = "data/winequality-red.csv"
    results_dir: str = "results"
    target_col: str = "quality"

    # split
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42

    # training
    epochs: int = 300
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 20
    num_workers: int = 0

    # model
    hidden_dims: List[int] = (128, 64)  # ReLU+Dropout
    dropout: float = 0.10

    # device
    use_cuda_if_available: bool = True
