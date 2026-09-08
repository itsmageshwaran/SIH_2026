"""
Constant Velocity Baseline Model
================================
Implements spherical dead reckoning projection as a benchmark for trajectory forecasting.
Projects future positions across horizons (e.g. +24h, +48h) assuming constant speed and heading.
"""

import math
import numpy as np
from typing import Tuple, List, Union


class ConstantVelocityBaseline:
    """
    Spherical Dead Reckoning Constant Velocity model.
    Takes recent trajectory history or current state (lat, lon, speed, heading)
    and forecasts future positions using great-circle spherical kinematic equations.
    """
    def __init__(self, earth_radius_km: float = 6371.0):
        self.R = earth_radius_km

    def predict_point(
        self,
        lat: float,
        lon: float,
        speed_km_day: float,
        heading_deg: float,
        horizon_days: float
    ) -> Tuple[float, float]:
        """
        Projects a single coordinate forward by horizon_days using spherical dead reckoning.
        """
        distance_km = speed_km_day * horizon_days
        delta = distance_km / self.R  # Angular distance in radians

        phi1 = math.radians(lat)
        lam1 = math.radians(lon)
        theta = math.radians(heading_deg)

        # Great circle destination point formula
        sin_phi2 = math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(theta)
        phi2 = math.asin(max(-1.0, min(1.0, sin_phi2)))

        y = math.sin(theta) * math.sin(delta) * math.cos(phi1)
        x = math.cos(delta) - math.sin(phi1) * math.sin(phi2)
        lam2 = lam1 + math.atan2(y, x)

        pred_lat = math.degrees(phi2)
        pred_lon = (math.degrees(lam2) + 540.0) % 360.0 - 180.0  # Normalize to [-180, 180]

        return pred_lat, pred_lon

    def predict_batch(
        self,
        base_coords: np.ndarray,
        recent_speeds: np.ndarray,
        recent_headings: np.ndarray,
        horizons: Tuple[int, ...] = (1, 2)
    ) -> np.ndarray:
        """
        Generates predictions for a batch of trajectories across multiple forecast horizons.
        Args:
            base_coords: (N, 2) array of [lat, lon] at time t
            recent_speeds: (N,) array of speed in km/day
            recent_headings: (N,) array of heading in degrees
            horizons: tuple of horizons in days, e.g. (1, 2) for +24h, +48h
        Returns:
            pred_coords: (N, len(horizons) * 2) array of [lat_h1, lon_h1, lat_h2, lon_h2, ...]
        """
        N = len(base_coords)
        preds = np.zeros((N, len(horizons) * 2), dtype=np.float32)

        for i in range(N):
            lat = float(base_coords[i, 0])
            lon = float(base_coords[i, 1])
            speed = float(recent_speeds[i])
            heading = float(recent_headings[i])

            for h_idx, h in enumerate(horizons):
                p_lat, p_lon = self.predict_point(lat, lon, speed, heading, float(h))
                preds[i, h_idx * 2] = p_lat
                preds[i, h_idx * 2 + 1] = p_lon

        return preds

    def predict_from_sequences(
        self,
        X_seqs_unscaled: np.ndarray,
        base_coords: np.ndarray,
        horizons: Tuple[int, ...] = (1, 2),
        speed_idx: int = 2,  # vx_km_day
        heading_sin_idx: int = 5,
        heading_cos_idx: int = 6
    ) -> np.ndarray:
        """
        Convenience wrapper extracting velocity & heading from raw unscaled input sequence.
        """
        N, T, D = X_seqs_unscaled.shape
        # Average recent speed and heading from the last 3 days to smooth sensor jitter
        recent_3 = X_seqs_unscaled[:, -3:, :]

        vx = recent_3[:, :, 2].mean(axis=1) # vx_km_day
        vy = recent_3[:, :, 3].mean(axis=1) # vy_km_day
        speeds = np.sqrt(vx**2 + vy**2)
        headings = (np.degrees(np.arctan2(vx, vy)) + 360.0) % 360.0

        return self.predict_batch(base_coords, speeds, headings, horizons)
