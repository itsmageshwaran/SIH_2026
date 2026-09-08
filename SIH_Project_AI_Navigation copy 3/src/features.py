"""
Feature Engineering & Dataset Utilities for Trajectory Modeling
================================================================
Extracts sliding sequence windows, applies cyclic temporal & angular encodings,
normalizes features, and structures PyTorch Dataset & DataLoaders.
"""

import math
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Optional


FEATURE_COLS = [
    "lat", "lon", "vx_km_day", "vy_km_day", "speed_knots",
    "sin_heading", "cos_heading", "sin_doy", "cos_doy",
    "wind_u", "wind_v", "curr_u", "curr_v"
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trigonometric cyclic features and environmental placeholders."""
    df = df.copy()

    # Heading cyclic encoding
    rad = np.radians(df["heading_deg"].fillna(0.0))
    df["sin_heading"] = np.sin(rad)
    df["cos_heading"] = np.cos(rad)

    # Day of Year seasonal encoding
    doy = df["day_of_year"].fillna(1)
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    # Environmental feature placeholders (Antarctic Circumpolar Current & Polar Westerlies defaults)
    # Allows seamless integration of real ERA5 / Copernicus reanalysis grids if available
    # Latitude-dependent polar easterlies/westerlies approximation
    lats = df["lat"].values
    df["wind_u"] = np.where(lats < -65.0, -5.0, 8.0) # Polar easterlies south of 65S, Westerlies north
    df["wind_v"] = np.sin(np.radians(df["lon"])) * 2.0
    df["curr_u"] = np.where(lats < -65.0, -0.2, 0.5) # Coastal current westward, ACC eastward
    df["curr_v"] = np.cos(np.radians(df["lon"])) * 0.1

    return df


class IcebergSequenceDataset(Dataset):
    """
    PyTorch Dataset yielding sequence input windows of length T_in
    and trajectory delta targets for horizons T_out.
    """
    def __init__(
        self,
        X_seqs: np.ndarray,
        y_deltas: np.ndarray,
        target_coords: np.ndarray,
        base_coords: np.ndarray
    ):
        self.X_seqs = torch.tensor(X_seqs, dtype=torch.float32)
        self.y_deltas = torch.tensor(y_deltas, dtype=torch.float32)
        self.target_coords = torch.tensor(target_coords, dtype=torch.float32)
        self.base_coords = torch.tensor(base_coords, dtype=torch.float32)

    def __len__(self):
        return len(self.X_seqs)

    def __getitem__(self, idx):
        return (
            self.X_seqs[idx],
            self.y_deltas[idx],
            self.target_coords[idx],
            self.base_coords[idx]
        )


class TrajectoryDataPipeline:
    """
    Handles sliding window sequence extraction, feature scaling,
    and train/val/test splits grouped by iceberg ID.
    """
    def __init__(
        self,
        seq_len: int = 14,
        forecast_horizons: Tuple[int, ...] = (1, 2),  # +24h (1 day), +48h (2 days)
        feature_cols: Optional[List[str]] = None
    ):
        self.seq_len = seq_len
        self.forecast_horizons = forecast_horizons
        self.max_horizon = max(forecast_horizons)
        self.feature_cols = feature_cols or FEATURE_COLS
        self.scaler = StandardScaler()
        self.is_fitted = False

    def build_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract sliding windows of historical inputs and multi-horizon target deltas.
        Returns:
            X: (N, seq_len, num_features)
            y_deltas: (N, len(horizons) * 2) -> [dlat_1, dlon_1, dlat_2, dlon_2, ...]
            target_coords: (N, len(horizons) * 2) -> [lat_1, lon_1, lat_2, lon_2, ...]
            base_coords: (N, 2) -> [lat_t, lon_t]
        """
        df = add_engineered_features(df)
        icebergs = df["iceberg_id"].unique()

        X_list, y_deltas_list, target_coords_list, base_coords_list = [], [], [], []

        for berg_id in icebergs:
            berg_df = df[df["iceberg_id"] == berg_id].sort_values("timestamp").reset_index(drop=True)
            if len(berg_df) < (self.seq_len + self.max_horizon):
                continue

            feature_arr = berg_df[self.feature_cols].values
            lat_arr = berg_df["lat"].values
            lon_arr = berg_df["lon"].values

            n_samples = len(berg_df) - self.seq_len - self.max_horizon + 1
            for i in range(n_samples):
                # Input sequence
                seq_x = feature_arr[i : i + self.seq_len]

                # Base coordinate at current timestep t
                curr_lat = lat_arr[i + self.seq_len - 1]
                curr_lon = lon_arr[i + self.seq_len - 1]

                # Targets for horizons
                deltas = []
                coords = []
                for h in self.forecast_horizons:
                    tgt_idx = i + self.seq_len - 1 + h
                    future_lat = lat_arr[tgt_idx]
                    future_lon = lon_arr[tgt_idx]
                    # Handle antimeridian wrap-around for longitude delta
                    d_lon = future_lon - curr_lon
                    if d_lon > 180.0:
                        d_lon -= 360.0
                    elif d_lon < -180.0:
                        d_lon += 360.0

                    d_lat = future_lat - curr_lat
                    deltas.extend([d_lat, d_lon])
                    coords.extend([future_lat, future_lon])

                X_list.append(seq_x)
                y_deltas_list.append(deltas)
                target_coords_list.append(coords)
                base_coords_list.append([curr_lat, curr_lon])

        if not X_list:
            raise ValueError("No sequences could be constructed. Check track lengths.")

        return (
            np.array(X_list, dtype=np.float32),
            np.array(y_deltas_list, dtype=np.float32),
            np.array(target_coords_list, dtype=np.float32),
            np.array(base_coords_list, dtype=np.float32)
        )

    def fit_scaler(self, X_train: np.ndarray):
        """Fit StandardScaler on 2D reshaped training sequence data."""
        N, T, D = X_train.shape
        self.scaler.fit(X_train.reshape(-1, D))
        self.is_fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted scaler to 3D sequence array."""
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before transforming data.")
        N, T, D = X.shape
        scaled_2d = self.scaler.transform(X.reshape(-1, D))
        return scaled_2d.reshape(N, T, D)

    def save_scaler(self, path: str = "models/feature_scaler.pkl"):
        """Serialize scaler to disk."""
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "feature_cols": self.feature_cols}, f)

    def load_scaler(self, path: str = "models/feature_scaler.pkl"):
        """Load serialized scaler from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.scaler = data["scaler"]
            self.feature_cols = data["feature_cols"]
            self.is_fitted = True


def create_train_val_test_datasets(
    csv_path: str = "data/processed/iceberg_tracks_clean.csv",
    seq_len: int = 14,
    forecast_horizons: Tuple[int, ...] = (1, 2),
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[IcebergSequenceDataset, IcebergSequenceDataset, IcebergSequenceDataset, TrajectoryDataPipeline, List[str]]:
    """
    Splits icebergs by ID (to prevent data leakage) into train/val/test sets,
    scales features, and produces PyTorch Datasets.
    """
    df = pd.read_csv(csv_path)
    all_icebergs = df["iceberg_id"].unique()
    np.random.seed(random_seed)
    shuffled_icebergs = np.random.permutation(all_icebergs)

    n_total = len(shuffled_icebergs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = shuffled_icebergs[:n_train]
    val_ids = shuffled_icebergs[n_train : n_train + n_val]
    test_ids = shuffled_icebergs[n_train + n_val :]

    print(f"[DATA] Iceberg split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test.")

    train_df = df[df["iceberg_id"].isin(train_ids)]
    val_df = df[df["iceberg_id"].isin(val_ids)]
    test_df = df[df["iceberg_id"].isin(test_ids)]

    pipeline = TrajectoryDataPipeline(seq_len=seq_len, forecast_horizons=forecast_horizons)

    X_train, y_train_del, y_train_coords, base_train = pipeline.build_sequences(train_df)
    X_val, y_val_del, y_val_coords, base_val = pipeline.build_sequences(val_df)
    X_test, y_test_del, y_test_coords, base_test = pipeline.build_sequences(test_df)

    pipeline.fit_scaler(X_train)

    X_train_scaled = pipeline.transform(X_train)
    X_val_scaled = pipeline.transform(X_val)
    X_test_scaled = pipeline.transform(X_test)

    train_dataset = IcebergSequenceDataset(X_train_scaled, y_train_del, y_train_coords, base_train)
    val_dataset = IcebergSequenceDataset(X_val_scaled, y_val_del, y_val_coords, base_val)
    test_dataset = IcebergSequenceDataset(X_test_scaled, y_test_del, y_test_coords, base_test)

    return train_dataset, val_dataset, test_dataset, pipeline, test_ids.tolist()
