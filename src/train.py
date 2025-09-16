# src/train.py
from __future__ import annotations
import argparse
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from pathlib import Path
import pandas as pd

from .config import Config
from .models.mlp import MLPRegressor
from . import utils


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    n = 0
    for xb, yb in loader:
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        n += xb.size(0)

    mse = running_loss / max(n, 1)
    rmse = np.sqrt(mse)
    return rmse  # RMSEで返すと直感的


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        pred = model(xb)
        ys.append(yb.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden", type=str, default=None, help="例: 128,64")
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None: cfg = Config(epochs=args.epochs)
    if args.lr is not None:     cfg = Config(epochs=cfg.epochs, lr=args.lr)
    if args.hidden:
        hidden = tuple(int(x.strip()) for x in args.hidden.split(","))
        cfg = Config(epochs=cfg.epochs, lr=cfg.lr, hidden_dims=hidden)

    # device
    device = torch.device("cuda" if (cfg.use_cuda_if_available and torch.cuda.is_available()) else "cpu")

    # 再現性
    utils.set_seed(cfg.random_state)

    # 出力ディレクトリ
    run_dir = utils.make_run_dir(cfg.results_dir)
    figs_dir = run_dir / "figs"

    # 設定保存
    utils.save_config(cfg, run_dir)

    # データ
    df = utils.load_dataframe(cfg.data_path)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = utils.split_and_scale(df, cfg)
    in_features = X_train.shape[1]

    # DataLoader
    train_loader, val_loader, test_loader = utils.make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, cfg, device
    )

    # モデル
    model = MLPRegressor(in_features=in_features, hidden_dims=cfg.hidden_dims, dropout=cfg.dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    print(f"[INFO] device={device}, params={model.n_params:,}")
    print(f"[INFO] train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")

    # 学習ループ（早期終了: val RMSE）
    best_val_rmse = float("inf")
    patience = 0
    history_rows = []

    for epoch in range(1, cfg.epochs + 1):
        train_rmse = train_one_epoch(model, train_loader, criterion, optimizer, device)
        yv, pv = evaluate(model, val_loader, device)  # y_true, y_pred
        val_metrics = utils.compute_regression_metrics(yv, pv)
        val_rmse = val_metrics["RMSE"]

        history_rows.append({"epoch": epoch, "train_rmse": train_rmse, "val_rmse": val_rmse})

        improved = val_rmse < best_val_rmse - 1e-6
        if improved:
            best_val_rmse = val_rmse
            patience = 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        else:
            patience += 1

        if epoch % 10 == 0 or improved:
            print(f"Epoch {epoch:03d} | train RMSE={train_rmse:.4f} | val RMSE={val_rmse:.4f} | best={best_val_rmse:.4f}")

        if patience >= cfg.early_stopping_patience:
            print(f"[INFO] Early stopping at epoch {epoch} (best val RMSE={best_val_rmse:.4f})")
            break

    # 学習履歴保存 & 図
    hist_df = pd.DataFrame(history_rows)
    hist_df.to_csv(run_dir / "history.csv", index=False)
    utils.plot_learning_curve(hist_df, figs_dir / "learning_curve.png")

    # ベストモデルで再評価
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))

    # train/val/test の指標と予測ファイル
    metrics_all = {}
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        y_true, y_pred = evaluate(model, loader, device)
        metrics = utils.compute_regression_metrics(y_true, y_pred)
        metrics_all[split_name] = metrics
        utils.save_predictions(y_true, y_pred, run_dir, split_name)
        utils.plot_pred_scatter(y_true, y_pred, figs_dir / f"scatter_{split_name}.png",
                                title=f"Pred vs True ({split_name})")
        utils.plot_residual_hist(y_true - y_pred, figs_dir / f"residual_{split_name}.png",
                                 title=f"Residuals ({split_name})")

    utils.save_metrics_table(metrics_all, run_dir)

    print("\n=== Summary (RMSE / MAE / R2) ===")
    for k, v in metrics_all.items():
        print(f"{k:>5s}: RMSE={v['RMSE']:.4f} | MAE={v['MAE']:.4f} | R2={v['R2']:.4f}")
    print(f"\nArtifacts saved to: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
