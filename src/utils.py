# src/utils.py
from __future__ import annotations
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .config import Config

# ---------- misc ----------
def set_seed(seed: int) -> None:
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def make_run_dir(results_dir: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(results_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figs").mkdir(parents=True, exist_ok=True)
    return run_dir

# ---------- data ----------
def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";") if csv_path.endswith(".csv") else pd.read_csv(csv_path)
    # UCI/Kaggleの赤ワインCSVはセミコロン区切り
    return df

def split_and_scale(
    df: pd.DataFrame, cfg: Config
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    X = df.drop(columns=[cfg.target_col]).values.astype(np.float32)
    y = df[cfg.target_col].values.astype(np.float32)

    # testを先に確保 → 残りからval
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )
    val_ratio_wrt_train = cfg.val_size / (1.0 - cfg.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio_wrt_train, random_state=cfg.random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

def make_loaders(
    X_train, y_train, X_val, y_val, X_test, y_test, cfg: Config, device: torch.device
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    def to_tensor(x, y):
        x = torch.from_numpy(x).to(device=device, dtype=torch.float32)
        y = torch.from_numpy(y).to(device=device, dtype=torch.float32)
        return TensorDataset(x, y)

    train_ds = to_tensor(X_train, y_train)
    val_ds   = to_tensor(X_val, y_val)
    test_ds  = to_tensor(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))
    return train_loader, val_loader, test_loader

# ---------- metrics ----------
def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

# ---------- plotting ----------
def plot_pred_scatter(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str) -> None:
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.6, s=20)
    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("True quality")
    plt.ylabel("Predicted quality")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_residual_hist(residuals: np.ndarray, path: Path, title: str) -> None:
    plt.figure()
    plt.hist(residuals, bins=20)
    plt.xlabel("Residual (y - y_hat)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_learning_curve(history: pd.DataFrame, path: Path) -> None:
    plt.figure()
    plt.plot(history["epoch"], history["train_rmse"], label="train RMSE")
    plt.plot(history["epoch"], history["val_rmse"], label="val RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("Learning Curve (RMSE)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ---------- saving ----------
def save_config(cfg: Config, run_dir: Path) -> None:
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

def save_metrics_table(metrics: Dict[str, Dict[str, float]], run_dir: Path) -> None:
    """
    metrics: {"train": {...}, "val": {...}, "test": {...}}
    """
    df = pd.DataFrame(metrics).T[["RMSE", "MAE", "R2"]]
    df.to_csv(run_dir / "metrics.csv", index=True)

def save_predictions(y_true: np.ndarray, y_pred: np.ndarray, run_dir: Path, split_name: str) -> None:
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "residual": y_true - y_pred})
    df.to_csv(run_dir / f"predictions_{split_name}.csv", index=False)
