import os
import sys
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Wedge

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.features import TrajectoryDataPipeline, FEATURE_COLS, add_engineered_features
from src.train_gru import IcebergGRU, get_device
from src.route_optimizer import RiskGrid, AStarMaritimeRouter, ANTARCTIC_STATIONS, FUEL_PROFILES

def haversine_dist(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(p1) * np.cos(p2) * np.sin(dl/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def interpolate_route(waypoints, speed_knots, time_step_hours=1.0):
    """Interpolate route to get vessel position at each time step."""
    speed_kmh = speed_knots * 1.852
    dist_per_step = speed_kmh * time_step_hours
    
    times = []
    positions = []
    
    current_time = 0.0
    current_idx = 0
    
    if len(waypoints) < 2:
        return [0.0], [(waypoints[0]['lat'], waypoints[0]['lon'])]
        
    current_pos = (waypoints[0]['lat'], waypoints[0]['lon'])
    positions.append(current_pos)
    times.append(current_time)
    
    while current_idx < len(waypoints) - 1:
        next_wp = waypoints[current_idx + 1]
        next_pos = (next_wp['lat'], next_wp['lon'])
        
        dist_to_next = haversine_dist(current_pos[0], current_pos[1], next_pos[0], next_pos[1])
        
        if dist_to_next < dist_per_step:
            # Move to next waypoint
            current_pos = next_pos
            current_idx += 1
            # Adjust time proportionally
            current_time += (dist_to_next / speed_kmh)
            positions.append(current_pos)
            times.append(current_time)
        else:
            # Interpolate step
            ratio = dist_per_step / dist_to_next
            lat_step = current_pos[0] + (next_pos[0] - current_pos[0]) * ratio
            lon_step = current_pos[1] + (next_pos[1] - current_pos[1]) * ratio
            
            current_pos = (lat_step, lon_step)
            current_time += time_step_hours
            positions.append(current_pos)
            times.append(current_time)
            
    return times, positions

def create_radar_animation():
    print("Initializing Radar Simulation...")
    # 1. Load Data
    data_path = "data/processed/iceberg_tracks_clean.csv"
    df = pd.read_csv(data_path)
    df = add_engineered_features(df)
    
    # 2. Get Icebergs
    # Pick a cluster of icebergs. 
    # Let's get the last known positions for all icebergs that have >= 14 days history.
    iceberg_histories = {}
    for berg_id, group in df.groupby('iceberg_id'):
        if len(group) >= 14:
            iceberg_histories[berg_id] = group.sort_values('timestamp').tail(14)
            
    print(f"Loaded {len(iceberg_histories)} icebergs for tracking.")

    # 3. Load Model for Predictions
    device = get_device()
    checkpoint = torch.load("models/gru_iceberg.pt", map_location=device)
    model = IcebergGRU(input_dim=len(FEATURE_COLS)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    pipeline = TrajectoryDataPipeline()
    pipeline.load_scaler("models/feature_scaler.pkl")

    # Generate predictions for all icebergs
    predictions = {}
    with torch.no_grad():
        for berg_id, hist in iceberg_histories.items():
            features = hist[FEATURE_COLS].values
            scaled = pipeline.scaler.transform(features)
            seq = torch.FloatTensor(scaled).unsqueeze(0).to(device)
            preds = model(seq).cpu().numpy()[0]
            
            base_lat = hist.iloc[-1]['lat']
            base_lon = hist.iloc[-1]['lon']
            predictions[berg_id] = {
                'now': (base_lat, base_lon),
                'p24': (base_lat + preds[0], base_lon + preds[1]),
                'p48': (base_lat + preds[2], base_lon + preds[3]),
                'speed': hist.iloc[-1]['speed_knots'],
                'heading': hist.iloc[-1]['heading_deg']
            }

    # 4. Compute Vessel Route
    start = ANTARCTIC_STATIONS["Ushuaia Port (Argentina)"]
    goal = ANTARCTIC_STATIONS["Palmer Station (US)"]
    
    grid = RiskGrid(lat_min=-68.0, lat_max=-50.0, lon_min=-75.0, lon_max=-55.0, resolution_deg=0.5)
    
    # Add real iceberg hazards to grid to compute a safe route
    for berg_id, pred in predictions.items():
        if -68 <= pred['now'][0] <= -50 and -75 <= pred['now'][1] <= -55:
            grid.add_iceberg_hazard(berg_id, pred['now'][0], pred['now'][1], radius_km=15.0)
            
    router = AStarMaritimeRouter(grid, risk_weight=5.0)
    route = router.find_path(start["lat"], start["lon"], goal["lat"], goal["lon"])
    
    # 5. Interpolate Route
    times, positions = interpolate_route(route['waypoints'], speed_knots=16.0, time_step_hours=1.0)
    
    # 6. Create Animation
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
    ax.set_facecolor('black')
    
    # Static elements
    ax.set_xlim(-75, -55)
    ax.set_ylim(-68, -50)
    ax.set_title("Vessel AI Radar System - Drake Passage", color='cyan', fontsize=14, pad=20)
    ax.set_xlabel("Longitude", color='cyan')
    ax.set_ylabel("Latitude", color='cyan')
    ax.tick_params(colors='cyan')
    
    # Draw land masses using our is_land_or_shelf function
    lons = np.linspace(-75, -55, 100)
    lats = np.linspace(-68, -50, 100)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    land_mask = np.zeros_like(lon_grid, dtype=float)
    
    from src.route_optimizer import is_land_or_shelf
    for i in range(len(lats)):
        for j in range(len(lons)):
            if is_land_or_shelf(lats[i], lons[j]):
                land_mask[i, j] = 1.0
                
    # Plot land as dark grey/green
    ax.contourf(lon_grid, lat_grid, land_mask, levels=[0.5, 1.5], colors=['#1a3320'], alpha=0.8, zorder=0)
    
    # Add grid rings (Radar effect)
    for r in range(2, 20, 2):
        circle = Circle((-65, -62), r, color='#003300', fill=False, lw=1, zorder=1)
        ax.add_patch(circle)
        
    # Draw route
    r_lons = [p[1] for p in positions]
    r_lats = [p[0] for p in positions]
    ax.plot(r_lons, r_lats, color='#004444', lw=2, linestyle='--', zorder=2)
    
    # Dynamic elements
    vessel_marker, = ax.plot([], [], 'o', color='cyan', markersize=10, zorder=5)
    radar_sweep = Wedge((0,0), 0, 0, 0, color='cyan', alpha=0.2, zorder=3)
    ax.add_patch(radar_sweep)
    
    radar_radius_deg = 3.0 # ~330km radar range
    
    hud_text = ax.text(-74.5, -51.5, "", color='cyan', fontfamily='monospace', fontsize=10, 
                       bbox=dict(facecolor='black', alpha=0.7, edgecolor='cyan'), zorder=6)

    # Make icebergs white to look like icebergs
    berg_scatter = ax.scatter([], [], color='white', s=50, marker='s', edgecolors='cyan', zorder=5)
    pred_scatter = ax.scatter([], [], color='orange', s=20, marker='x', zorder=5)
    
    # To store lines for drift vectors
    vector_lines = []

    def init():
        vessel_marker.set_data([], [])
        berg_scatter.set_offsets(np.empty((0, 2)))
        pred_scatter.set_offsets(np.empty((0, 2)))
        hud_text.set_text("")
        return vessel_marker, radar_sweep, hud_text, berg_scatter, pred_scatter

    def update(frame):
        # Clear old vectors
        for line in vector_lines:
            line.remove()
        vector_lines.clear()
        
        pos = positions[frame]
        vessel_marker.set_data([pos[1]], [pos[0]])
        
        # Radar sweep animation
        angle = (frame * 20) % 360
        radar_sweep.set_center((pos[1], pos[0]))
        radar_sweep.set_radius(radar_radius_deg)
        radar_sweep.set_theta1(angle)
        radar_sweep.set_theta2(angle + 60)
        
        # Find icebergs in radar range
        in_range_bergs = []
        for bid, pred in predictions.items():
            dist = haversine_dist(pos[0], pos[1], pred['now'][0], pred['now'][1])
            if dist < (radar_radius_deg * 111):
                in_range_bergs.append((bid, pred, dist))
                
        # Update Iceberg points
        if in_range_bergs:
            b_coords = [[b[1]['now'][1], b[1]['now'][0]] for b in in_range_bergs]
            p_coords = []
            
            for b in in_range_bergs:
                p24 = b[1]['p24']
                p48 = b[1]['p48']
                now = b[1]['now']
                p_coords.extend([[p24[1], p24[0]], [p48[1], p48[0]]])
                
                # Draw prediction line
                line, = ax.plot([now[1], p24[1], p48[1]], [now[0], p24[0], p48[0]], 
                                color='orange', lw=1, linestyle='-', alpha=0.7)
                vector_lines.append(line)
                
                # Draw velocity vector
                dx = np.sin(np.radians(b[1]['heading'])) * 0.5
                dy = np.cos(np.radians(b[1]['heading'])) * 0.5
                arr = ax.arrow(now[1], now[0], dx, dy, color='red', head_width=0.1, alpha=0.5)
                vector_lines.append(arr)
            
            berg_scatter.set_offsets(b_coords)
            pred_scatter.set_offsets(p_coords)
        else:
            berg_scatter.set_offsets(np.empty((0, 2)))
            pred_scatter.set_offsets(np.empty((0, 2)))
            
        hud_info = f"TIME: T+{times[frame]:.1f}h\n"
        hud_info += f"VESSEL: {pos[0]:.2f}S, {pos[1]:.2f}W\n"
        hud_info += f"RADAR CONTACTS: {len(in_range_bergs)}\n"
        hud_info += "-"*20 + "\n"
        for i, b in enumerate(in_range_bergs[:5]): # Top 5 contacts
            hud_info += f"[{b[0]}] D:{b[2]:.0f}km S:{b[1]['speed']:.1f}kt\n"
            
        hud_text.set_text(hud_info)
        
        return vessel_marker, radar_sweep, hud_text, berg_scatter, pred_scatter

    print("Generating Animation frames...")
    ani = animation.FuncAnimation(fig, update, frames=len(positions), init_func=init, blit=False, interval=200)
    
    out_file = "artifacts/radar_view.html"
    os.makedirs("artifacts", exist_ok=True)
    
    print("Saving to HTML5 (JSAnimation)...")
    with open(out_file, "w") as f:
        f.write(ani.to_jshtml())
        
    print(f"✅ Radar Animation Saved to {out_file}")

if __name__ == "__main__":
    create_radar_animation()
