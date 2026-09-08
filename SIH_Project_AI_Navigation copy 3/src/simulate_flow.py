import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import TrajectoryDataPipeline, FEATURE_COLS, add_engineered_features
from src.train_gru import IcebergGRU, get_device
from src.route_optimizer import RiskGrid, AStarMaritimeRouter, ANTARCTIC_STATIONS, FUEL_PROFILES

def run_simulation():
    print("======================================================================")
    print(" 🚢 SIH: ADVANCED AI MARITIME NAVIGATION & COLLISION AVOIDANCE SIMULATION")
    print("======================================================================")
    
    # ---------------------------------------------------------
    # STEP 1: LOAD LIVE DATA
    # ---------------------------------------------------------
    print("\n[1] DATA INGESTION: Loading BYU/NIC Iceberg Database...")
    data_path = "data/processed/iceberg_tracks_clean.csv"
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = add_engineered_features(df)
    
    # Pick a specific iceberg that is active in the Drake Passage
    # Let's find one near Palmer Station (lat ~ -64, lon ~ -64)
    target_berg = "a23a" # Example major iceberg
    berg_df = df[df['iceberg_id'] == target_berg].sort_values('timestamp').tail(14)
    if len(berg_df) < 14:
        berg_df = df.groupby('iceberg_id').filter(lambda x: len(x) >= 14).sort_values('timestamp').tail(14)
        target_berg = berg_df['iceberg_id'].iloc[0]
        
    print(f"✅ Tracking Target Iceberg: {target_berg.upper()}")
    print(f"   Last Known Position: {berg_df.iloc[-1]['lat']:.3f}°S, {berg_df.iloc[-1]['lon']:.3f}°W")
    print(f"   Current Speed: {berg_df.iloc[-1]['speed_knots']:.1f} kts")

    # ---------------------------------------------------------
    # STEP 2: ML DRIFT FORECASTING (PyTorch GRU)
    # ---------------------------------------------------------
    print("\n[2] AI DRIFT FORECASTING: PyTorch Multi-Horizon GRU...")
    model_path = "models/gru_iceberg.pt"
    scaler_path = "models/feature_scaler.pkl"
    device = get_device()
    checkpoint = torch.load(model_path, map_location=device)
    
    model = IcebergGRU(
        input_dim=len(checkpoint.get("feature_cols", FEATURE_COLS)),
        hidden_dim=checkpoint.get("hidden_dim", 128),
        num_layers=checkpoint.get("num_layers", 2),
        output_dim=4
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    pipeline = TrajectoryDataPipeline(seq_len=14, forecast_horizons=(1, 2))
    pipeline.load_scaler(scaler_path)
    
    # Prepare sequence
    features = berg_df[FEATURE_COLS].values
    scaled_features = pipeline.scaler.transform(features)
    seq_tensor = torch.FloatTensor(scaled_features).unsqueeze(0).to(device)
    
    with torch.no_grad():
        preds = model(seq_tensor).cpu().numpy()[0]
    
    # Inverse transform to get coordinate deltas
    base_lat = berg_df.iloc[-1]['lat']
    base_lon = berg_df.iloc[-1]['lon']
    pred_24h_lat = base_lat + preds[0]
    pred_24h_lon = base_lon + preds[1]
    pred_48h_lat = base_lat + preds[2]
    pred_48h_lon = base_lon + preds[3]
    
    print(f"✅ +24h Forecast: {pred_24h_lat:.3f}°S, {pred_24h_lon:.3f}°W (Uncertainty ±15km)")
    print(f"✅ +48h Forecast: {pred_48h_lat:.3f}°S, {pred_48h_lon:.3f}°W (Uncertainty ±30km)")

    # ---------------------------------------------------------
    # STEP 3: DYNAMIC RISK ENGINE
    # ---------------------------------------------------------
    print("\n[3] RISK ENGINE: Generating Gaussian Hazard Heatmaps...")
    start = ANTARCTIC_STATIONS["Ushuaia Port (Argentina)"]
    goal = ANTARCTIC_STATIONS["Palmer Station (US)"]
    
    lat_min, lat_max = min(start["lat"], goal["lat"]) - 5.0, max(start["lat"], goal["lat"]) + 5.0
    lon_min, lon_max = min(start["lon"], goal["lon"]) - 10.0, max(start["lon"], goal["lon"]) + 10.0
    
    grid = RiskGrid(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max, resolution_deg=0.5)
    
    # Inject Iceberg Hazards with expanding uncertainty cones
    grid.add_iceberg_hazard(target_berg, base_lat, base_lon, radius_km=10.0) # Current
    grid.add_iceberg_hazard(f"{target_berg}_24h", pred_24h_lat, pred_24h_lon, radius_km=15.0) # 24h
    grid.add_iceberg_hazard(f"{target_berg}_48h", pred_48h_lat, pred_48h_lon, radius_km=30.0) # 48h
    print("✅ Spatial Risk Grid populated with predictive hazard envelopes.")

    # ---------------------------------------------------------
    # STEP 4: A* MARITIME ROUTE OPTIMIZATION
    # ---------------------------------------------------------
    print("\n[4] A* ROUTING: Computing Optimal Collision-Free Trajectory...")
    profile = FUEL_PROFILES["advanced"]
    print(f"   Mission: Ushuaia 🇦🇷 ➔ Palmer Station 🇺🇸")
    print(f"   Profile: {profile['label']} ({profile['speed_knots']} kts, Risk Weight {profile['risk_weight']})")
    
    router = AStarMaritimeRouter(grid, risk_weight=profile["risk_weight"])
    route = router.find_path(start["lat"], start["lon"], goal["lat"], goal["lon"])
    
    if not route or route['status'] != 'success':
        print("❌ Route calculation failed.")
        return
        
    print("\n[RESULTS & TELEMETRY]")
    print(f"  Distance:        {route['total_distance_km']:.1f} km ({route['total_distance_nm']:.1f} NM)")
    print(f"  Transit Time:    {route['estimated_time_hours']:.1f} hours")
    print(f"  Est. Fuel Burn:  {route['total_distance_nm'] * profile['fuel_rate_per_nm']:.1f} tonnes")
    print(f"  Max Risk Score:  {route['max_risk_encountered']:.4f}")
    print(f"  Safety Rating:   {route['safety_rating']}")
    
    # ---------------------------------------------------------
    # STEP 5: VISUALIZATION (REALISTIC INTERACTIVE MAP)
    # ---------------------------------------------------------
    print("\n[5] VISUALIZATION: Generating Realistic Geospatial Map...")
    import folium
    
    # Initialize Map centered on the Drake Passage
    m = folium.Map(location=[(start['lat'] + goal['lat'])/2, (start['lon'] + goal['lon'])/2], 
                   zoom_start=5, 
                   tiles='Esri.OceanBasemap')
    
    # Plot Iceberg History
    h_lons = berg_df['lon'].values
    h_lats = berg_df['lat'].values
    hist_points = list(zip(h_lats, h_lons))
    folium.PolyLine(hist_points, color="black", weight=2, dash_array="5, 10", tooltip=f"{target_berg.upper()} History").add_to(m)
    folium.Marker(location=[base_lat, base_lon], 
                  popup=f"Iceberg {target_berg.upper()} (Now)",
                  icon=folium.Icon(color='red', icon='snowflake')).add_to(m)
                  
    # Plot GRU Predictions (24h and 48h)
    folium.Marker([pred_24h_lat, pred_24h_lon], popup="+24h Forecast", icon=folium.Icon(color='orange', icon='forward')).add_to(m)
    folium.Marker([pred_48h_lat, pred_48h_lon], popup="+48h Forecast", icon=folium.Icon(color='red', icon='fast-forward')).add_to(m)
    
    # Draw uncertainty radius circles
    folium.Circle(location=[pred_24h_lat, pred_24h_lon], radius=15000, color='orange', fill=True, fillOpacity=0.3, tooltip="24h Hazard Zone").add_to(m)
    folium.Circle(location=[pred_48h_lat, pred_48h_lon], radius=30000, color='red', fill=True, fillOpacity=0.2, tooltip="48h Hazard Zone").add_to(m)

    # Plot Route
    r_points = [(w['lat'], w['lon']) for w in route['waypoints']]
    folium.PolyLine(r_points, color="#0ea5e9", weight=4, opacity=0.8, tooltip="A* Optimized Maritime Route").add_to(m)
    
    # Plot Ports
    folium.Marker([start['lat'], start['lon']], popup="Ushuaia Port", icon=folium.Icon(color='green', icon='anchor')).add_to(m)
    folium.Marker([goal['lat'], goal['lon']], popup="Palmer Station", icon=folium.Icon(color='green', icon='flag')).add_to(m)

    output_html = "artifacts/simulation_map.html"
    os.makedirs("artifacts", exist_ok=True)
    m.save(output_html)
    
    print(f"✅ Realistic Interactive Simulation Map saved to: {output_html}")
    print("======================================================================")

if __name__ == "__main__":
    run_simulation()
