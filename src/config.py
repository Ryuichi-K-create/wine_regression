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
    model_type: str = "linear"  # "mlp", "linear", or "svr"
    hidden_dims: List[int] = (128, 64)  # ReLU+Dropout (MLPのみ)
    dropout: float = 0.10  # MLP/Linearのみ

    # SVR params
    svr_kernel: str = "rbf"  # "rbf", "linear", "poly", "sigmoid"
    svr_C: float = 1.0
    svr_epsilon: float = 0.1
    svr_gamma: str = "scale"
    svr_grid_search: bool = True
    svr_cv: int = 5

    # device
    use_cuda_if_available: bool = True
