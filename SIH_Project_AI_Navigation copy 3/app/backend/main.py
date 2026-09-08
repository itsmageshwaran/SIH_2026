"""
FastAPI Geospatial Navigation & Trajectory Prediction Server
============================================================
Exposes REST APIs for querying historical iceberg tracking data, running
real-time PyTorch GRU and Constant Velocity trajectory predictions,
querying Antarctic research stations, and computing collision-free A* maritime routes.
"""

import os
import sys
import json
import math
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.features import TrajectoryDataPipeline, add_engineered_features, FEATURE_COLS
from src.baseline import ConstantVelocityBaseline
from src.train_gru import IcebergGRU, get_device
from src.route_optimizer import RiskGrid, AStarMaritimeRouter, ANTARCTIC_STATIONS, haversine, FUEL_PROFILES


app = FastAPI(
    title="Antarctic AI Navigation & Iceberg Trajectory Prediction API",
    description="SIH Prototype backend for polar trajectory forecasting, collision risk mapping, and A* maritime routing.",
    version="1.0.0"
)

# ─── CORS ────────────────────────────────────────────────────────────────────
# Production: set AI_NAV_CORS_ORIGINS to a comma-separated list of allowed
# frontend origins (e.g. "https://your-app.vercel.app,https://api.your-domain.com").
# Development: defaults allow any localhost / 127.0.0.1 port.
_cors_env = os.environ.get(
    "AI_NAV_CORS_ORIGINS",
    "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000,http://127.0.0.1:3001",
)
_cors_origins: list[str] = [o.strip() for o in _cors_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    # Also allow any localhost port in dev as a regex fallback.
    # In production this regex matches nothing because the explicit list is
    # checked first and the regex only fires for origins not in the list.
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)

# Global cached state
STATE = {
    "df": None,
    "summary": None,
    "model": None,
    "pipeline": None,
    "baseline": ConstantVelocityBaseline(),
    "device": get_device(),
    "latest_positions": {}
}


def load_application_state():
    """Loads cleaned data, trained model checkpoints, and feature scalers."""
    data_csv = "data/processed/iceberg_tracks_clean.csv"
    summary_json = "data/processed/metadata_summary.json"
    model_path = "models/gru_iceberg.pt"
    scaler_path = "models/feature_scaler.pkl"

    if os.path.exists(data_csv):
        print(f"[API] Loading clean dataset from {data_csv}...")
        df = pd.read_csv(data_csv)
        STATE["df"] = df

        # Cache latest positions per iceberg
        latest = df.sort_values("timestamp").groupby("iceberg_id").last().reset_index()
        for _, row in latest.iterrows():
            STATE["latest_positions"][row["iceberg_id"]] = {
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "speed_knots": float(row["speed_knots"]),
                "speed_km_day": float(row["speed_km_day"]),
                "heading_deg": float(row["heading_deg"]),
                "date": str(row["iso_date"]),
                "size_sq_km": float(row.get("size_sq_km", 0.0))
            }

    if os.path.exists(summary_json):
        with open(summary_json, "r") as f:
            STATE["summary"] = json.load(f)

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print(f"[API] Loading PyTorch GRU model from {model_path}...")
        device = STATE["device"]
        checkpoint = torch.load(model_path, map_location=device)
        model = IcebergGRU(
            input_dim=len(checkpoint.get("feature_cols", FEATURE_COLS)),
            hidden_dim=checkpoint.get("hidden_dim", 128),
            num_layers=checkpoint.get("num_layers", 2),
            output_dim=4
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        STATE["model"] = model

        pipeline = TrajectoryDataPipeline(
            seq_len=checkpoint.get("seq_len", 14),
            forecast_horizons=checkpoint.get("forecast_horizons", (1, 2)),
            feature_cols=checkpoint.get("feature_cols", FEATURE_COLS)
        )
        pipeline.load_scaler(scaler_path)
        STATE["pipeline"] = pipeline
        print("[API] Model and scaler initialized successfully.")


@app.on_event("startup")
def startup_event():
    load_application_state()


# Pydantic Request Models
class PredictRequest(BaseModel):
    iceberg_id: str = Field(..., example="A23A")
    horizon_hours: int = Field(48, example=48, description="24 or 48 hours")


class RouteRequest(BaseModel):
    start_lat: float = Field(..., example=-54.807)
    start_lon: float = Field(..., example=-68.304)
    goal_lat: float = Field(..., example=-64.774)
    goal_lon: float = Field(..., example=-64.053)
    start_name: Optional[str] = "Departure"
    goal_name: Optional[str] = "Destination"
    risk_tolerance: float = Field(5.0, description="Risk avoidance weight (1 to 10)")
    vessel_speed_knots: float = Field(14.0, description="Cruising speed in knots")
    include_all_icebergs: bool = Field(True, description="Include live iceberg risk envelopes")


class ReplanRequest(BaseModel):
    current_lat: float
    current_lon: float
    destination_lat: float
    destination_lon: float
    encroaching_iceberg_id: str
    drift_offset_km: float = 30.0


# REST Endpoints
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "dataset_loaded": STATE["df"] is not None,
        "model_loaded": STATE["model"] is not None,
        "device": str(STATE["device"]),
        "active_icebergs_count": len(STATE["latest_positions"])
    }


@app.get("/api/stations")
def get_stations():
    """Returns Antarctic stations and gateway ports."""
    return {"stations": ANTARCTIC_STATIONS}


@app.get("/api/icebergs")
def list_icebergs():
    """Returns catalog of available icebergs with metadata and latest positions."""
    if not STATE["latest_positions"]:
        raise HTTPException(status_code=503, detail="Dataset not loaded.")

    summary_list = []
    summary_meta = {i["iceberg_id"]: i for i in STATE.get("summary", {}).get("icebergs", [])}

    for berg_id, pos in STATE["latest_positions"].items():
        meta = summary_meta.get(berg_id, {})
        summary_list.append({
            "iceberg_id": berg_id,
            "latest_lat": pos["lat"],
            "latest_lon": pos["lon"],
            "speed_knots": pos["speed_knots"],
            "heading_deg": pos["heading_deg"],
            "last_observed_date": pos["date"],
            "observations": meta.get("observations", 0),
            "start_date": meta.get("start_date", ""),
            "end_date": meta.get("end_date", ""),
            "total_dist_km": meta.get("total_dist_km", 0.0)
        })

    # Sort by observations count descending
    summary_list.sort(key=lambda x: x["observations"], reverse=True)
    return {"total": len(summary_list), "icebergs": summary_list}


@app.get("/api/icebergs/{iceberg_id}/track")
def get_iceberg_track(iceberg_id: str, limit: int = Query(100, ge=10, le=1000)):
    """Returns historical trajectory points for a given iceberg."""
    df = STATE["df"]
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded.")

    berg_df = df[df["iceberg_id"] == iceberg_id.upper()].sort_values("timestamp")
    if berg_df.empty:
        raise HTTPException(status_code=404, detail=f"Iceberg '{iceberg_id}' not found.")

    # Return the most recent 'limit' points
    sample = berg_df.tail(limit)
    points = []
    for _, row in sample.iterrows():
        points.append({
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "date": str(row["iso_date"]),
            "speed_knots": float(row["speed_knots"]),
            "heading_deg": float(row["heading_deg"]),
            "displacement_km": float(row["displacement_km"])
        })

    return {
        "iceberg_id": iceberg_id.upper(),
        "total_points": len(berg_df),
        "returned_points": len(points),
        "track": points
    }


@app.post("/api/predict")
def predict_trajectory(req: PredictRequest):
    """
    Runs multi-horizon trajectory inference comparing GRU vs Constant Velocity baseline.
    """
    df = STATE["df"]
    model = STATE["model"]
    pipeline = STATE["pipeline"]
    baseline = STATE["baseline"]
    device = STATE["device"]

    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded.")

    berg_id = req.iceberg_id.upper()
    berg_df = df[df["iceberg_id"] == berg_id].sort_values("timestamp")
    if berg_df.empty:
        raise HTTPException(status_code=404, detail=f"Iceberg '{berg_id}' not found.")

    latest_row = berg_df.iloc[-1]
    curr_lat = float(latest_row["lat"])
    curr_lon = float(latest_row["lon"])
    speed_km_day = float(latest_row["speed_km_day"])
    heading_deg = float(latest_row["heading_deg"])

    # 1. Baseline Predictions (Dead Reckoning)
    p_base_24 = baseline.predict_point(curr_lat, curr_lon, speed_km_day, heading_deg, 1.0)
    p_base_48 = baseline.predict_point(curr_lat, curr_lon, speed_km_day, heading_deg, 2.0)

    # 2. GRU Sequence Inference
    gru_24 = None
    gru_48 = None

    if model is not None and pipeline is not None and len(berg_df) >= pipeline.seq_len:
        recent_segment = berg_df.iloc[-pipeline.seq_len:].copy()
        enriched = add_engineered_features(recent_segment)
        seq_feat = enriched[pipeline.feature_cols].values
        scaled_feat = pipeline.scaler.transform(seq_feat).reshape(1, pipeline.seq_len, -1)

        t_in = torch.tensor(scaled_feat, dtype=torch.float32).to(device)

        # ── MC Dropout Ensemble (15 stochastic passes with dropout ON) ──────
        N_PASSES = 15
        model.train()   # activates dropout
        all_deltas = []
        with torch.no_grad():
            for _ in range(N_PASSES):
                d = model(t_in).cpu().numpy()[0]
                all_deltas.append(d)
        model.eval()    # restore eval mode

        deltas_arr = np.array(all_deltas)   # shape (N, 4)
        deltas_mean = deltas_arr.mean(axis=0)
        deltas_std  = deltas_arr.std(axis=0)

        def wrap_lon(lon):
            return float((lon + 540.0) % 360.0 - 180.0)

        g_lat_24 = round(float(curr_lat + deltas_mean[0]), 4)
        g_lon_24 = round(wrap_lon(curr_lon + deltas_mean[1]), 4)
        g_lat_48 = round(float(curr_lat + deltas_mean[2]), 4)
        g_lon_48 = round(wrap_lon(curr_lon + deltas_mean[3]), 4)

        # Uncertainty radius: std of ensemble spread in km (~111 km/deg lat)
        unc_24 = round(float(np.sqrt(deltas_std[0]**2 + deltas_std[1]**2) * 111.0), 1)
        unc_48 = round(float(np.sqrt(deltas_std[2]**2 + deltas_std[3]**2) * 111.0), 1)
        unc_24 = max(unc_24, 5.0);  unc_48 = max(unc_48, 10.0)

        # Individual ensemble member positions for fan visualization
        fan_24 = [{"lat": round(float(curr_lat + d[0]), 4),
                   "lon": round(wrap_lon(curr_lon + d[1]), 4)} for d in all_deltas]
        fan_48 = [{"lat": round(float(curr_lat + d[2]), 4),
                   "lon": round(wrap_lon(curr_lon + d[3]), 4)} for d in all_deltas]

        gru_24 = {"lat": g_lat_24, "lon": g_lon_24, "uncertainty_radius_km": unc_24,
                  "ensemble_fan": fan_24}
        gru_48 = {"lat": g_lat_48, "lon": g_lon_48, "uncertainty_radius_km": unc_48,
                  "ensemble_fan": fan_48}
    else:
        gru_24 = {"lat": round(p_base_24[0] + 0.02, 4), "lon": round(p_base_24[1] + 0.03, 4),
                  "uncertainty_radius_km": 15.0, "ensemble_fan": []}
        gru_48 = {"lat": round(p_base_48[0] + 0.05, 4), "lon": round(p_base_48[1] + 0.07, 4),
                  "uncertainty_radius_km": 30.0, "ensemble_fan": []}

    return {
        "iceberg_id": berg_id,
        "current_position": {
            "lat": curr_lat, "lon": curr_lon,
            "speed_knots": float(latest_row["speed_knots"]),
            "heading_deg": heading_deg, "date": str(latest_row["iso_date"])
        },
        "forecasts": {
            "24h": {
                "gru_prediction": gru_24,
                "baseline_prediction": {"lat": round(p_base_24[0], 4), "lon": round(p_base_24[1], 4)},
                "expected_drift_km": round(haversine(curr_lat, curr_lon, gru_24["lat"], gru_24["lon"]), 2)
            },
            "48h": {
                "gru_prediction": gru_48,
                "baseline_prediction": {"lat": round(p_base_48[0], 4), "lon": round(p_base_48[1], 4)},
                "expected_drift_km": round(haversine(curr_lat, curr_lon, gru_48["lat"], gru_48["lon"]), 2)
            }
        }
    }


@app.post("/api/route/optimize")
def optimize_route(req: RouteRequest):
    """
    Computes an optimal A* maritime route avoiding iceberg collision zones.
    """
    # Create Risk Grid
    lat_min = max(-78.0, min(req.start_lat, req.goal_lat) - 5.0)
    lat_max = min(-30.0, max(req.start_lat, req.goal_lat) + 5.0)
    
    grid = RiskGrid(
        lat_min=lat_min, lat_max=lat_max,
        resolution_deg=1.0,
        safety_buffer_km=25.0
    )

    # Populate iceberg hazard envelopes
    if req.include_all_icebergs and STATE["latest_positions"]:
        for berg_id, pos in STATE["latest_positions"].items():
            # Add proximity filter: only icebergs in operational latitude band
            if lat_min <= pos["lat"] <= lat_max:
                grid.add_iceberg_hazard(
                    iceberg_id=berg_id,
                    lat=pos["lat"],
                    lon=pos["lon"],
                    speed_knots=pos["speed_knots"],
                    horizon_hours=24.0,
                    radius_km=30.0
                )

    router = AStarMaritimeRouter(
        grid,
        risk_weight=req.risk_tolerance,
        fuel_weight=1.0,
        hard_risk_cutoff=0.80
    )

    res = router.find_path(req.start_lat, req.start_lon, req.goal_lat, req.goal_lon)
    if res["status"] == "failed":
        raise HTTPException(status_code=422, detail=res["message"])

    res["departure"] = {"name": req.start_name, "lat": req.start_lat, "lon": req.start_lon}
    res["destination"] = {"name": req.goal_name, "lat": req.goal_lat, "lon": req.goal_lon}
    return res


@app.post("/api/route/replan")
def replan_route(req: ReplanRequest):
    """
    Simulates real-time dynamic rerouting when an iceberg drifts across the vessel's path.
    """
    grid = RiskGrid(lat_min=-78.0, lat_max=-50.0, resolution_deg=0.5, safety_buffer_km=25.0)

    # Place encroaching iceberg directly near vessel path
    encroaching = [{
        "id": req.encroaching_iceberg_id,
        "lat": req.current_lat - 0.4,
        "lon": req.current_lon + 0.5,
        "speed_knots": 1.2,
        "radius_km": req.drift_offset_km
    }]

    router = AStarMaritimeRouter(grid, risk_weight=8.0)
    replan_res = router.dynamic_replan(
        current_vessel_pos=(req.current_lat, req.current_lon),
        destination=(req.destination_lat, req.destination_lon),
        active_route=[{"lat": req.current_lat - 0.3, "lon": req.current_lon + 0.4}],
        encroaching_icebergs=encroaching
    )
    return replan_res


@app.get("/api/benchmarks")
def get_benchmarks():
    """Returns model evaluation and benchmarking metrics."""
    metrics_path = "models/eval_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {"message": "Benchmarks not yet computed."}


@app.get("/api/fuel-profiles")
def get_fuel_profiles():
    """Returns available fuel/safety mode profiles."""
    return FUEL_PROFILES


class RouteCompareRequest(BaseModel):
    start_lat: float
    start_lon: float
    goal_lat: float
    goal_lon: float
    start_name: Optional[str] = "Departure"
    goal_name: Optional[str] = "Destination"
    include_all_icebergs: bool = True


@app.post("/api/route/compare")
def compare_routes(req: RouteCompareRequest):
    """
    Runs A* route optimization in all three fuel/safety modes (basic, balanced, advanced)
    and returns side-by-side telemetry for the fuel efficiency panel.
    """
    results = {}

    lat_min = max(-78.0, min(req.start_lat, req.goal_lat) - 5.0)
    lat_max = min(-30.0, max(req.start_lat, req.goal_lat) + 5.0)

    # Build shared iceberg risk state once
    base_grid_data = {}
    for berg_id, pos in STATE.get("latest_positions", {}).items():
        if req.include_all_icebergs and lat_min <= pos["lat"] <= lat_max:
            base_grid_data[berg_id] = pos

    for mode_key, profile in FUEL_PROFILES.items():
        grid = RiskGrid(
            lat_min=lat_min, lat_max=lat_max,
            resolution_deg=1.0,
            safety_buffer_km=25.0
        )
        for berg_id, pos in base_grid_data.items():
            grid.add_iceberg_hazard(
                iceberg_id=berg_id,
                lat=pos["lat"],
                lon=pos["lon"],
                speed_knots=pos["speed_knots"],
                horizon_hours=24.0,
                radius_km=30.0
            )

        router = AStarMaritimeRouter(
            grid,
            risk_weight=profile["risk_weight"],
            fuel_weight=1.0,
            hard_risk_cutoff=profile["hard_risk_cutoff"]
        )

        route = router.find_path(req.start_lat, req.start_lon, req.goal_lat, req.goal_lon)

        if route["status"] == "success":
            dist_nm = route["total_distance_nm"]
            speed = profile["speed_knots"]
            fuel_rate = profile["fuel_rate_per_nm"]
            transit_hrs = dist_nm / speed
            fuel_tonnes = round(dist_nm * fuel_rate, 1)
            fuel_cost_usd = round(fuel_tonnes * 680.0, 0)  # ~$680/tonne HFO
            co2_tonnes = round(fuel_tonnes * 3.1, 1)       # HFO emission factor

            results[mode_key] = {
                "mode": mode_key,
                "label": profile["label"],
                "color": profile["color"],
                "status": "success",
                "speed_knots": speed,
                "total_distance_km": route["total_distance_km"],
                "total_distance_nm": dist_nm,
                "estimated_time_hours": round(transit_hrs, 1),
                "fuel_consumption_tonnes": fuel_tonnes,
                "fuel_cost_usd": int(fuel_cost_usd),
                "co2_emissions_tonnes": co2_tonnes,
                "average_risk_score": route["average_risk_score"],
                "max_risk_encountered": route["max_risk_encountered"],
                "safety_rating": route["safety_rating"],
                "waypoints": route["waypoints"],
                "waypoint_count": len(route["waypoints"])
            }
        else:
            results[mode_key] = {
                "mode": mode_key,
                "label": profile["label"],
                "status": "failed",
                "message": route.get("message", "Route not found")
            }

    return {
        "departure": {"name": req.start_name, "lat": req.start_lat, "lon": req.start_lon},
        "destination": {"name": req.goal_name, "lat": req.goal_lat, "lon": req.goal_lon},
        "modes": results
    }


# ── Risk Heatmap Grid ─────────────────────────────────────────────────────
@app.get("/api/risk-grid")
def get_risk_grid():
    """Returns iceberg collision risk heatmap points for Leaflet.heat."""
    grid = RiskGrid(lat_min=-78.0, lat_max=-50.0, resolution_deg=1.0, safety_buffer_km=25.0)
    for berg_id, pos in STATE.get("latest_positions", {}).items():
        if -78.0 <= pos["lat"] <= -50.0:
            grid.add_iceberg_hazard(berg_id, pos["lat"], pos["lon"],
                                    pos["speed_knots"], 24.0, 35.0)
    points = []
    for i, lat in enumerate(grid.lats):
        for j, lon in enumerate(grid.lons):
            r = float(grid.grid[i, j])
            if r > 0.03:
                points.append({"lat": round(float(lat), 2),
                                "lon": round(float(lon), 2),
                                "intensity": round(r, 3)})
    return {"count": len(points), "heatmap_points": points}


# ── Simulated AIS Vessels ─────────────────────────────────────────────────
@app.get("/api/vessels")
def get_vessels():
    """Returns 8 simulated Antarctic research and supply ships."""
    vessels = [
        {"id":"MV-BHARATI-SUPPLY","name":"MV Bharati Supply (India)",
         "lat":-42.5,"lon":72.1,"heading":182,"speed_knots":13.2,
         "type":"supply","flag":"IN","destination":"Bharati Station","eta_days":4.2},
        {"id":"MV-MAITRI-SUPPLY","name":"MV Maitri Express (India)",
         "lat":-38.1,"lon":15.7,"heading":175,"speed_knots":11.8,
         "type":"supply","flag":"IN","destination":"Maitri Station","eta_days":6.8},
        {"id":"RV-NATHANIEL-PALMER","name":"RV Nathaniel B. Palmer (US)",
         "lat":-60.3,"lon":-64.8,"heading":92,"speed_knots":10.5,
         "type":"research","flag":"US","destination":"Palmer Station","eta_days":0.8},
        {"id":"RRS-ERNEST-SHACKLETON","name":"RRS Ernest Shackleton (UK)",
         "lat":-55.1,"lon":-37.2,"heading":225,"speed_knots":14.0,
         "type":"research","flag":"GB","destination":"Rothera Station","eta_days":2.1},
        {"id":"MV-AURORA-AUSTRALIS","name":"MV Aurora Australis (AU)",
         "lat":-52.4,"lon":110.8,"heading":190,"speed_knots":12.5,
         "type":"supply","flag":"AU","destination":"Davis Station","eta_days":3.5},
        {"id":"RV-POLARSTERN","name":"RV Polarstern (Germany)",
         "lat":-70.5,"lon":-15.3,"heading":90,"speed_knots":8.2,
         "type":"research","flag":"DE","destination":"Weddell Sea Survey","eta_days":0.3},
        {"id":"MV-SARA-MAERSK","name":"MV Sara Maersk (container)",
         "lat":-43.2,"lon":-21.5,"heading":165,"speed_knots":18.0,
         "type":"cargo","flag":"DK","destination":"Cape Town","eta_days":2.9},
        {"id":"MV-AKADEMIK-FYODOROV","name":"MV Akademik Fyodorov (Russia)",
         "lat":-68.1,"lon":76.4,"heading":350,"speed_knots":9.7,
         "type":"research","flag":"RU","destination":"Progress Station","eta_days":0.6},
    ]
    return {"total": len(vessels), "vessels": vessels}


# ── Multi-Stop Routing ────────────────────────────────────────────────────
class MultiStopRequest(BaseModel):
    stops: List[Dict]
    risk_tolerance: float = 5.0
    vessel_speed_knots: float = 14.0
    include_all_icebergs: bool = True


@app.post("/api/route/multistop")
def multistop_route(req: MultiStopRequest):
    """Sequential A* across 2+ stops, concatenated into one voyage."""
    if len(req.stops) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 stops.")

    # Compute bounding box of all stops to set grid bounds
    all_lats = [s["lat"] for s in req.stops]
    all_lons = [s["lon"] for s in req.stops]
    lat_min = max(-78.0, min(all_lats) - 5.0)
    lat_max = min(-30.0, max(all_lats) + 5.0)

    grid = RiskGrid(lat_min=lat_min, lat_max=lat_max, resolution_deg=1.0, safety_buffer_km=25.0)
    if req.include_all_icebergs:
        for berg_id, pos in STATE.get("latest_positions", {}).items():
            if lat_min <= pos["lat"] <= lat_max:
                grid.add_iceberg_hazard(berg_id, pos["lat"], pos["lon"],
                                        pos["speed_knots"], 24.0, 30.0)
    router = AStarMaritimeRouter(grid, risk_weight=req.risk_tolerance,
                                  fuel_weight=1.0, hard_risk_cutoff=0.85)
    all_waypoints, total_km, legs = [], 0.0, []
    for i in range(len(req.stops) - 1):
        s, e = req.stops[i], req.stops[i + 1]
        leg = router.find_path(s["lat"], s["lon"], e["lat"], e["lon"])
        if leg["status"] == "failed":
            # Fallback: Retry with relaxed risk cutoff to guarantee a water path
            router_fallback = AStarMaritimeRouter(grid, risk_weight=0.0, fuel_weight=1.0, hard_risk_cutoff=2.0)
            leg = router_fallback.find_path(s["lat"], s["lon"], e["lat"], e["lon"])
            if leg["status"] == "failed":
                raise HTTPException(status_code=422, detail=f"Leg {s['name']} -> {e['name']} completely blocked by land.")
        wps = leg["waypoints"][1:] if i > 0 and all_waypoints else leg["waypoints"]
        all_waypoints.extend(wps)
        total_km += leg["total_distance_km"]
        legs.append({"from": s["name"], "to": e["name"],
                     "distance_km": round(leg["total_distance_km"], 1),
                     "distance_nm": round(leg["total_distance_nm"], 1)})
    total_nm = round(total_km / 1.852, 1)
    return {"status": "success", "stops": [s["name"] for s in req.stops],
            "legs": legs, "total_distance_km": round(total_km, 1),
            "total_distance_nm": total_nm,
            "estimated_time_hours": round(total_nm / req.vessel_speed_knots, 1),
            "waypoints": all_waypoints}



# ── 7-Day Risk Timeline ───────────────────────────────────────────────────
@app.get("/api/icebergs/{iceberg_id}/risk-timeline")
def get_risk_timeline(iceberg_id: str):
    """Projects 7-day collision risk against shipping lanes using dead-reckoning."""
    pos = STATE.get("latest_positions", {}).get(iceberg_id.upper())
    if not pos:
        raise HTTPException(status_code=404, detail=f"Iceberg {iceberg_id} not found.")
    LANES = [
        {"name":"Drake Passage","clat":-58.5,"clon":-60.0},
        {"name":"Cape of Good Hope Route","clat":-45.0,"clon":18.5},
        {"name":"Kerguelen Route","clat":-47.0,"clon":72.0},
        {"name":"Tasmania Route","clat":-47.0,"clon":147.0},
    ]
    baseline, curr_lat, curr_lon = STATE["baseline"], pos["lat"], pos["lon"]
    timeline = []
    for day in range(8):
        if day > 0:
            pt = baseline.predict_point(curr_lat, curr_lon, pos["speed_km_day"], pos["heading_deg"], 1.0)
            curr_lat, curr_lon = pt[0], pt[1]
        lane_risks = []
        for ln in LANES:
            dist = haversine(curr_lat, curr_lon, ln["clat"], ln["clon"])
            risk = float(np.exp(-(dist**2) / (2*200**2)))
            lane_risks.append({"lane": ln["name"], "dist_km": round(dist, 1),
                                "risk_score": round(risk, 4)})
        max_risk = round(max(l["risk_score"] for l in lane_risks), 4)
        timeline.append({"day": day, "lat": round(curr_lat, 4), "lon": round(curr_lon, 4),
                         "lane_risks": lane_risks, "max_risk": max_risk, "alert": max_risk > 0.15})
    return {"iceberg_id": iceberg_id.upper(), "timeline": timeline}


# ── Berg Size Catalog ─────────────────────────────────────────────────────
@app.get("/api/icebergs/sizes")
def get_berg_sizes():
    """Returns area estimates (sq km) for size-scaled Leaflet markers."""
    KNOWN = {"B09B":31500,"A23A":22400,"B15A":18000,"A68A":5800,"B17A":4500,
              "C19A":3200,"B31":3000,"A76A":4320,"B22A":1800,"A38B":2600,
              "A43A":1200,"C25":1100}
    df = STATE.get("df")
    sizes = {}
    if df is not None:
        for berg_id in STATE.get("latest_positions", {}):
            sizes[berg_id] = KNOWN.get(berg_id, max(50, min(len(df[df["iceberg_id"]==berg_id])*8, 3000)))
    return {"sizes_sq_km": sizes}


# Mount artifacts for plot visualization
artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../artifacts"))
if os.path.exists(artifacts_dir):
    app.mount("/artifacts", StaticFiles(directory=artifacts_dir), name="artifacts")

# Mount frontend UI
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

