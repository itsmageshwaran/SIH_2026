"""
PyTorch GRU Iceberg Trajectory Forecasting Model
================================================
Trains a recurrent sequence model (GRU) to forecast multi-horizon iceberg
displacements (24h and 48h) from historical kinematics and environmental features.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import create_train_val_test_datasets, FEATURE_COLS, TrajectoryDataPipeline


class IcebergGRU(nn.Module):
    """
    GRU Recurrent Neural Network for multi-horizon polar trajectory forecasting.
    Combines GRU temporal encoding with dense multi-horizon projection heads.
    """
    def __init__(
        self,
        input_dim: int = len(FEATURE_COLS),
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 4,  # [dlat_24h, dlon_24h, dlat_48h, dlon_48h]
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Input feature projection
        self.input_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Multi-layer GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        # Multi-horizon regression head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            deltas: (batch_size, output_dim)
        """
        proj = self.input_fc(x)
        out, h_n = self.gru(proj)  # out: (batch_size, seq_len, hidden_dim)

        # Temporal pooling: concatenate last hidden state and mean hidden state
        last_step = out[:, -1, :]
        mean_pool = out.mean(dim=1)
        context = torch.cat([last_step, mean_pool], dim=-1)

        deltas = self.head(context)
        return deltas


def get_device() -> torch.device:
    """Select best available acceleration device (MPS on Apple Silicon, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(
    csv_path: str = "data/processed/iceberg_tracks_clean.csv",
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    hidden_dim: int = 128,
    num_layers: int = 2,
    seq_len: int = 14,
    model_save_path: str = "models/gru_iceberg.pt",
    scaler_save_path: str = "models/feature_scaler.pkl",
    history_save_path: str = "models/train_history.json"
) -> Dict:
    """
    Main training routine for the PyTorch GRU trajectory forecaster.
    """
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    device = get_device()
    print(f"[DEVICE] Training on device: {device}")

    # Build datasets
    train_dataset, val_dataset, test_dataset, pipeline, test_icebergs = create_train_val_test_datasets(
        csv_path=csv_path,
        seq_len=seq_len,
        forecast_horizons=(1, 2)
    )

    # Save fitted scaler
    pipeline.save_scaler(scaler_save_path)
    print(f"[SAVED] Feature scaler saved to {scaler_save_path}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"[DATA] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = IcebergGRU(
        input_dim=len(pipeline.feature_cols),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=4,  # 2 horizons * 2 coords
        dropout=0.2
    ).to(device)

    criterion = nn.SmoothL1Loss(beta=0.1)  # Robust Huber loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "best_epoch": 0}

    print("\n[TRAINING] Commencing model training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch, _, _ in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val, _, _ in val_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                val_preds = model(X_val)
                val_loss += criterion(val_preds, y_val).item()

        val_loss /= len(val_loader)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_epoch"] = epoch
            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "input_dim": len(pipeline.feature_cols),
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "seq_len": seq_len,
                "feature_cols": pipeline.feature_cols,
                "forecast_horizons": (1, 2),
                "test_icebergs": test_icebergs
            }
            torch.save(checkpoint, model_save_path)

    print(f"\n[SUCCESS] Training complete. Best Val Loss: {best_val_loss:.5f} at epoch {history['best_epoch']}.")
    print(f"[SAVED] Best model checkpoint saved to: {model_save_path}")

    with open(history_save_path, "w") as f:
        json.dump(history, f, indent=2)

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GRU Iceberg Trajectory Model")
    parser.add_argument("--csv_path", type=str, default="data/processed/iceberg_tracks_clean.csv")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=14)
    args = parser.parse_args()

    train_model(
        csv_path=args.csv_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        seq_len=args.seq_len
    )
