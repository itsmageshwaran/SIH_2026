"""
Antarctic Iceberg Dataset Preprocessing Pipeline
================================================
Automated ingestion, parsing, quality filtering, and feature extraction
for the BYU/NIC Antarctic Iceberg Tracking Database.

Features:
- Automated download/unzip from BYU Scatterometer Climate Record Pathfinder if not local.
- Converts YYYYDDD (ordinal dates) to ISO timestamps (YYYY-MM-DD).
- Filters out noise, outliers, corrupted coordinates, and low-quality short tracks.
- Computes spherical Haversine displacement, velocity (km/day, knots), bearing angle (0-360 deg),
  and Cartesian velocity components (vx, vy).
- Exports the top ~75 high-fidelity tracks to 'data/processed/iceberg_tracks_clean.csv'.
"""

import os
import sys
import glob
import math
import json
import zipfile
import argparse
import requests
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np

# BYU/NIC official archive URLs
BYU_STATS_URL = "https://www.scp.byu.edu/data/iceberg/stats_database_v7.1.zip"
BYU_CONSOLIDATED_URL = "https://www.scp.byu.edu/data/iceberg/consolidated_database_v8.0.zip"


def download_and_extract_byu(dest_dir: str = "data/raw", url: str = BYU_STATS_URL) -> str:
    """Download and extract the BYU iceberg database zip if not already present."""
    os.makedirs(dest_dir, exist_ok=True)
    zip_name = os.path.basename(url)
    zip_path = os.path.join(dest_dir, zip_name)
    extract_folder = os.path.join(dest_dir, os.path.splitext(zip_name)[0])

    if os.path.isdir(extract_folder) and len(glob.glob(os.path.join(extract_folder, "*.csv"))) > 10:
        print(f"[INFO] Raw database already present at: {extract_folder}")
        return extract_folder

    print(f"[INFO] Downloading BYU Iceberg Database from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (1024 * 1024) == 0:
                        sys.stdout.write(f"\r  Progress: {downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB")
                        sys.stdout.flush()
        print(f"\n[INFO] Download completed: {zip_path}")

        print(f"[INFO] Extracting archive to {extract_folder}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_folder)
        print(f"[INFO] Extraction completed successfully.")
        return extract_folder
    except Exception as e:
        print(f"[WARNING] Automated download failed: {e}. Checking local alternatives...")
        return dest_dir


def parse_yyyyddd(date_val) -> Optional[datetime]:
    """Parse ordinal YYYYDDD (e.g. 2021232) into a Python datetime object."""
    try:
        s = str(int(float(date_val))).strip()
        if len(s) == 7:
            return datetime.strptime(s, "%Y%j")
        elif len(s) == 8: # Sometimes YYYYMMDD
            return datetime.strptime(s, "%Y%m%d")
        return None
    except Exception:
        return None


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees).
    """
    R = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate initial compass bearing in degrees (0 to 360) from point 1 to point 2.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


def process_single_iceberg(csv_path: str, min_obs: int = 100) -> Optional[pd.DataFrame]:
    """
    Parse, clean, and enrich a single iceberg CSV track.
    Returns None if the track does not meet quality criteria.
    """
    iceberg_id = os.path.splitext(os.path.basename(csv_path))[0].upper()
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    df.columns = [c.strip().lower() for c in df.columns]
    required_cols = {"date", "lat", "lon"}
    if not required_cols.issubset(df.columns):
        return None

    # Parse dates
    df["timestamp"] = df["date"].apply(parse_yyyyddd)
    df = df.dropna(subset=["timestamp", "lat", "lon"])
    if len(df) < min_obs:
        return None

    # Clean coordinates (Antarctica bounds: lat <= -45.0, lon in [-180, 180])
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    df = df[(df["lat"] >= -90.0) & (df["lat"] <= -45.0) & (df["lon"] >= -180.0) & (df["lon"] <= 180.0)]

    # Sort strictly by timestamp and drop duplicate timestamps
    df = df.sort_values(by="timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    if len(df) < min_obs:
        return None

    # Compute time gaps (days)
    df["dt_days"] = (df["timestamp"].diff().dt.total_seconds() / 86400.0).fillna(1.0)
    # Clip large gaps or drop points following gaps > 30 days to maintain track coherence
    df = df[df["dt_days"] <= 30.0].reset_index(drop=True)
    if len(df) < min_obs:
        return None

    # Compute spherical displacements, speeds, and bearings
    displacements_km = [0.0]
    bearings_deg = [0.0]

    lats = df["lat"].values
    lons = df["lon"].values
    dt_days = df["dt_days"].values

    for i in range(1, len(df)):
        d = haversine_distance(lats[i-1], lons[i-1], lats[i], lons[i])
        b = calculate_bearing(lats[i-1], lons[i-1], lats[i], lons[i])
        displacements_km.append(d)
        bearings_deg.append(b)

    df["displacement_km"] = displacements_km
    df["displacement_nm"] = df["displacement_km"] / 1.852

    # Speed: km/day and knots (nautical miles per hour)
    safe_dt = np.where(dt_days > 0, dt_days, 1.0)
    df["speed_km_day"] = df["displacement_km"] / safe_dt
    df["speed_knots"] = (df["displacement_nm"] / (safe_dt * 24.0)).clip(upper=15.0)  # Clip physical anomalies

    df["heading_deg"] = bearings_deg
    df["heading_rad"] = np.radians(df["heading_deg"])

    # Velocity components (Cartesian representation)
    df["vx_km_day"] = df["speed_km_day"] * np.sin(df["heading_rad"])
    df["vy_km_day"] = df["speed_km_day"] * np.cos(df["heading_rad"])

    # Size if available
    if "size" in df.columns:
        df["size_sq_km"] = pd.to_numeric(df["size"], errors="coerce").fillna(0.0)
    else:
        df["size_sq_km"] = 0.0

    # Add metadata columns
    df["iceberg_id"] = iceberg_id
    df["iso_date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["year"] = df["timestamp"].dt.year

    # Final column ordering
    clean_cols = [
        "iceberg_id", "timestamp", "iso_date", "year", "day_of_year",
        "lat", "lon", "displacement_km", "displacement_nm",
        "speed_km_day", "speed_knots", "heading_deg", "vx_km_day", "vy_km_day",
        "size_sq_km", "dt_days"
    ]
    return df[clean_cols]


def run_preprocessing_pipeline(
    raw_dir: Optional[str] = None,
    output_path: str = "data/processed/iceberg_tracks_clean.csv",
    top_n: int = 75,
    min_obs: int = 100
) -> Tuple[pd.DataFrame, Dict]:
    """
    Executes the end-to-end data pipeline:
    1. Locates or downloads raw BYU datasets.
    2. Parses all CSV files and filters low-quality tracks.
    3. Ranks icebergs and selects top ~75 high-fidelity tracks.
    4. Outputs clean merged CSV and metadata summary.
    """
    # Look for candidate directories
    candidate_dirs = []
    if raw_dir and os.path.exists(raw_dir):
        candidate_dirs.append(raw_dir)
    candidate_dirs.extend([
        "stats_database_v7.1",
        "data/raw/stats_database_v7.1",
        "data/raw"
    ])

    search_dir = None
    for d in candidate_dirs:
        if os.path.exists(d) and len(glob.glob(os.path.join(d, "*.csv"))) > 10:
            search_dir = d
            break

    if search_dir is None:
        print("[INFO] Local dataset not detected in default paths. Initiating BYU download...")
        search_dir = download_and_extract_byu(dest_dir="data/raw")

    csv_files = glob.glob(os.path.join(search_dir, "*.csv"))
    print(f"[INFO] Found {len(csv_files)} iceberg track files in {search_dir}")

    processed_tracks = []
    track_stats = []

    for idx, c in enumerate(csv_files):
        track_df = process_single_iceberg(c, min_obs=min_obs)
        if track_df is not None:
            berg_id = track_df["iceberg_id"].iloc[0]
            obs_count = len(track_df)
            start_date = track_df["iso_date"].iloc[0]
            end_date = track_df["iso_date"].iloc[-1]
            avg_speed = float(track_df["speed_knots"].mean())
            total_dist_km = float(track_df["displacement_km"].sum())

            track_stats.append({
                "iceberg_id": berg_id,
                "file": os.path.basename(c),
                "obs_count": obs_count,
                "start_date": start_date,
                "end_date": end_date,
                "avg_speed_knots": round(avg_speed, 3),
                "total_dist_km": round(total_dist_km, 1),
                "df": track_df
            })

    # Sort tracks by observation count & continuity
    track_stats.sort(key=lambda x: x["obs_count"], reverse=True)
    selected_tracks = track_stats[:top_n]
    print(f"[INFO] Selected top {len(selected_tracks)} high-quality tracks (threshold: min {min_obs} obs).")

    if not selected_tracks:
        raise ValueError("No tracks satisfied the quality filtering criteria!")

    # Merge into a single unified dataframe
    merged_dfs = [item["df"] for item in selected_tracks]
    final_df = pd.concat(merged_dfs, ignore_index=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Exported {len(final_df)} clean trajectory points to {output_path}")

    # Build metadata summary
    summary = {
        "dataset_name": "BYU/NIC Antarctic Iceberg Tracking Database",
        "processed_timestamp": datetime.utcnow().isoformat(),
        "total_icebergs_selected": len(selected_tracks),
        "total_trajectory_points": len(final_df),
        "lat_min": float(final_df["lat"].min()),
        "lat_max": float(final_df["lat"].max()),
        "lon_min": float(final_df["lon"].min()),
        "lon_max": float(final_df["lon"].max()),
        "avg_speed_knots": round(float(final_df["speed_knots"].mean()), 3),
        "max_speed_knots": round(float(final_df["speed_knots"].max()), 3),
        "icebergs": [
            {
                "iceberg_id": t["iceberg_id"],
                "observations": t["obs_count"],
                "start_date": t["start_date"],
                "end_date": t["end_date"],
                "avg_speed_knots": t["avg_speed_knots"],
                "total_dist_km": t["total_dist_km"]
            }
            for t in selected_tracks
        ]
    }

    summary_path = os.path.join(os.path.dirname(output_path), "metadata_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[SUCCESS] Exported dataset summary to {summary_path}")

    return final_df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Antarctic Iceberg Dataset Preprocessing Pipeline")
    parser.add_argument("--raw_dir", type=str, default="stats_database_v7.1", help="Path to raw BYU CSV files directory")
    parser.add_argument("--output", type=str, default="data/processed/iceberg_tracks_clean.csv", help="Target output CSV path")
    parser.add_argument("--top_n", type=int, default=75, help="Number of top high-quality tracks to retain")
    parser.add_argument("--min_obs", type=int, default=100, help="Minimum observations per track")
    args = parser.parse_args()

    run_preprocessing_pipeline(
        raw_dir=args.raw_dir,
        output_path=args.output,
        top_n=args.top_n,
        min_obs=args.min_obs
    )
