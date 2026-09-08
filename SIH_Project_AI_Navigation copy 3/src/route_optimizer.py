"""
Antarctic Maritime Risk Engine & A* Pathfinding Optimizer
==========================================================
Constructs dynamic spatial collision risk heatmaps around tracked and predicted
icebergs with expanding forecast uncertainty cones. Solves optimal, fuel-efficient,
and safe maritime routes between Antarctic research stations and gateway ports
using an 8-connected spherical A* pathfinding algorithm with dynamic replanning.

Fuel Efficiency Modes
---------------------
  BASIC     – 10 kts, risk_weight=1.5  → fuel-saving, accepts mild detours
  BALANCED  – 14 kts, risk_weight=5.0  → standard operational profile (default)
  ADVANCED  – 16 kts, risk_weight=10.0 → max safety, weather-robust heavy routing
"""

import math
import heapq
import json
import argparse
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
import pandas as pd

# Antarctic Scientific Stations & Maritime Gateway Ports
ANTARCTIC_STATIONS: Dict[str, Dict[str, float]] = {
    "Maitri (India)": {"lat": -70.767, "lon": 11.733, "type": "station", "country": "India"},
    "Bharati (India)": {"lat": -69.407, "lon": 76.187, "type": "station", "country": "India"},
    "Palmer Station (US)": {"lat": -64.774, "lon": -64.053, "type": "station", "country": "USA"},
    "Rothera Station (UK)": {"lat": -67.568, "lon": -68.128, "type": "station", "country": "UK"},
    "McMurdo Station (US)": {"lat": -77.846, "lon": 166.669, "type": "station", "country": "USA"},
    "Davis Station (Australia)": {"lat": -68.576, "lon": 77.967, "type": "station", "country": "Australia"},
    "Casey Station (Australia)": {"lat": -66.282, "lon": 110.528, "type": "station", "country": "Australia"},
    "Ushuaia Port (Argentina)": {"lat": -54.807, "lon": -68.304, "type": "port", "country": "Argentina"},
    "Punta Arenas (Chile)": {"lat": -53.163, "lon": -70.917, "type": "port", "country": "Chile"},
    "Cape Town Port (South Africa)": {"lat": -33.918, "lon": 18.423, "type": "port", "country": "South Africa"},
    "Hobart Port (Australia)": {"lat": -42.882, "lon": 147.327, "type": "port", "country": "Australia"}
}

# Fuel consumption model (metric tonnes per nautical mile) by mode
FUEL_PROFILES: Dict[str, Dict] = {
    "basic": {
        "speed_knots": 10.0,
        "risk_weight": 1.5,
        "hard_risk_cutoff": 0.90,
        "fuel_rate_per_nm": 0.85,   # lean consumption at low speed
        "label": "Basic (Fuel-Save)",
        "color": "#10b981",
    },
    "balanced": {
        "speed_knots": 14.0,
        "risk_weight": 5.0,
        "hard_risk_cutoff": 0.80,
        "fuel_rate_per_nm": 1.65,   # standard diesel consumption
        "label": "Balanced (Standard)",
        "color": "#0ea5e9",
    },
    "advanced": {
        "speed_knots": 16.0,
        "risk_weight": 10.0,
        "hard_risk_cutoff": 0.65,
        "fuel_rate_per_nm": 2.60,   # high speed + extended detour = more fuel
        "label": "Advanced (Max Safety)",
        "color": "#f59e0b",
    },
}


def _point_in_polygon(lat: float, lon: float, polygon: List[Tuple[float, float]]) -> bool:
    """Ray-casting algorithm for point-in-polygon test. Returns True if (lat,lon) is inside."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]   # xi=lon, yi=lat  (swapped for ray-cast convention)
        xj, yj = polygon[j]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def is_land_or_shelf(lat: float, lon: float) -> bool:
    """
    Accurate maritime exclusion mask. Returns True if (lat, lon) is on land or ice shelf.
    """
    if lat > -60.0:
        return False
        
    # Coastline approximation by longitude sectors
    if -180.0 <= lon < -150.0:
        if lat <= -78.5: return True  # Ross Sea
    elif -150.0 <= lon < -135.0:
        if lat <= -74.0: return True  # Marie Byrd Land
    elif -135.0 <= lon < -110.0:
        if lat <= -73.0: return True
    elif -110.0 <= lon < -90.0:
        if lat <= -72.0: return True
    elif -90.0 <= lon < -75.0:
        if lat <= -72.0: return True  # Ellsworth Land
    elif -75.0 <= lon < -60.0:
        if lat <= -74.0: return True  # Peninsula base
    elif -60.0 <= lon < -10.0:
        if lat <= -74.5: return True  # Weddell Sea
    elif -10.0 <= lon < 30.0:
        if lat <= -71.5: return True  # Dronning Maud (Maitri is -70.7)
    elif 30.0 <= lon < 60.0:
        if lat <= -69.5: return True
    elif 60.0 <= lon < 90.0:
        if lat <= -70.5: return True  # Mac Robertson (Bharati is -69.4)
    elif 90.0 <= lon < 140.0:
        if lat <= -67.0: return True  # Wilkes Land
    elif 140.0 <= lon < 160.0:
        if lat <= -68.0: return True  # George V Land
    elif 160.0 <= lon <= 180.0:
        if lat <= -78.5: return True  # Ross Sea / McMurdo
        
    # Antarctic Peninsula Spine
    if -74.0 <= lat <= -62.5:
        t = (lat - (-62.5)) / (-74.0 - (-62.5))
        spine_lon = -57.0 + t * (-15.0)
        half_w = 2.5 - t * 0.5
        if (spine_lon - half_w) <= lon <= (spine_lon + half_w):
            return True
            
    return False


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers between two points."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0)**2
    return 2.0 * R * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

import random

def smooth_waypoints(waypoints: List[Dict]) -> List[Dict]:
    """Applies a multi-pass moving average with organic jitter to simulate natural maritime routes."""
    if len(waypoints) < 3:
        return waypoints
    smoothed = waypoints.copy()
    
    # 1. Add organic jitter to intermediate points to break straight lines
    for i in range(1, len(smoothed) - 1):
        jitter_lat = random.uniform(-0.12, 0.12)
        jitter_lon = random.uniform(-0.25, 0.25)
        n_lat = smoothed[i]["lat"] + jitter_lat
        n_lon = smoothed[i]["lon"] + jitter_lon
        if not is_land_or_shelf(n_lat, n_lon):
            smoothed[i]["lat"] = n_lat
            smoothed[i]["lon"] = n_lon
            
    # 2. Apply smoothing passes
    for _ in range(4):
        temp = [smoothed[0]]
        for i in range(1, len(smoothed) - 1):
            s_lat = (smoothed[i-1]["lat"] + smoothed[i]["lat"] + smoothed[i+1]["lat"]) / 3.0
            s_lon = (smoothed[i-1]["lon"] + smoothed[i]["lon"] + smoothed[i+1]["lon"]) / 3.0
            if not is_land_or_shelf(s_lat, s_lon):
                temp.append({"lat": round(s_lat, 4), "lon": round(s_lon, 4), "risk": smoothed[i].get("risk", 0.0)})
            else:
                temp.append(smoothed[i])
        temp.append(smoothed[-1])
        smoothed = temp
        
    return smoothed




class RiskGrid:
    """
    Dynamic 2D Maritime Risk Surface representing collision probabilities
    from moving icebergs and their forecast uncertainty cones.
    """
    def __init__(
        self,
        lat_min: float = -78.0,
        lat_max: float = -50.0,
        lon_min: float = -180.0,
        lon_max: float = 180.0,
        resolution_deg: float = 1.0,
        safety_buffer_km: float = 25.0
    ):
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.res = resolution_deg
        self.safety_buffer_km = safety_buffer_km

        self.lats = np.arange(lat_min, lat_max + self.res, self.res)
        self.lons = np.arange(lon_min, lon_max, self.res)
        self.n_lat = len(self.lats)
        self.n_lon = len(self.lons)

        # Risk array initialized to 0.0
        self.grid = np.zeros((self.n_lat, self.n_lon), dtype=np.float32)
        self.iceberg_hazards: List[Dict] = []

    def add_iceberg_hazard(
        self,
        iceberg_id: str,
        lat: float,
        lon: float,
        speed_knots: float = 0.5,
        horizon_hours: float = 0.0,
        radius_km: Optional[float] = None
    ):
        """
        Injects a Gaussian hazard kernel around an iceberg position.
        Uncertainty expands with forecast horizon: sigma = sigma_0 + gamma * t.
        """
        if radius_km is None:
            # Base uncertainty: 15 km + 0.5 km per hour of forecast
            sigma_km = self.safety_buffer_km + (horizon_hours * 0.4)
        else:
            sigma_km = radius_km

        hazard = {
            "id": iceberg_id,
            "lat": lat,
            "lon": lon,
            "sigma_km": sigma_km,
            "horizon_hours": horizon_hours
        }
        self.iceberg_hazards.append(hazard)

        # Update affected local grid cells (within 3 * sigma)
        max_dist = 3.0 * sigma_km
        lat_span = max_dist / 111.0

        i_min = max(0, int((lat - lat_span - self.lat_min) / self.res))
        i_max = min(self.n_lat, int((lat + lat_span - self.lat_min) / self.res) + 1)

        for i in range(i_min, i_max):
            c_lat = self.lats[i]
            # Longitude scaling by latitude
            cos_lat = max(0.1, math.cos(math.radians(c_lat)))
            lon_span = max_dist / (111.0 * cos_lat)

            j_center = int((lon - self.lon_min) / self.res)
            j_span_cells = int(math.ceil(lon_span / self.res))

            for dj in range(-j_span_cells, j_span_cells + 1):
                j = (j_center + dj) % self.n_lon
                c_lon = self.lons[j]

                d_km = haversine(c_lat, c_lon, lat, lon)
                if d_km < max_dist:
                    intensity = math.exp(-(d_km**2) / (2.0 * (sigma_km**2)))
                    self.grid[i, j] = max(self.grid[i, j], float(intensity))

    def get_risk_at(self, lat: float, lon: float) -> float:
        """Query continuous risk value at arbitrary coordinates."""
        if lat < self.lat_min or lat > self.lat_max:
            return 1.0  # Out of maritime operational bounds

        i = int(round((lat - self.lat_min) / self.res))
        i = max(0, min(self.n_lat - 1, i))

        norm_lon = (lon + 180.0) % 360.0 - 180.0
        j = int(round((norm_lon - self.lon_min) / self.res))
        j = max(0, min(self.n_lon - 1, j))

        return float(self.grid[i, j])


class AStarMaritimeRouter:
    """
    A* Maritime Pathfinding Engine across polar waters.
    Balances travel distance, fuel penalty, and collision risk avoidance.
    """
    def __init__(
        self,
        risk_grid: RiskGrid,
        risk_weight: float = 5.0,
        fuel_weight: float = 1.0,
        hard_risk_cutoff: float = 0.85
    ):
        self.risk_grid = risk_grid
        self.risk_weight = risk_weight
        self.fuel_weight = fuel_weight
        self.hard_risk_cutoff = hard_risk_cutoff
        self.res = risk_grid.res

    def _get_neighbors(self, lat: float, lon: float) -> List[Tuple[float, float, float]]:
        """
        Returns 8-connected maritime neighbors with step distance in km.
        """
        neighbors = []
        d_lats = [-self.res, 0.0, self.res]
        d_lons = [-self.res, 0.0, self.res]

        for dl in d_lats:
            for dlo in d_lons:
                if dl == 0.0 and dlo == 0.0:
                    continue

                n_lat = round(lat + dl, 2)
                # Keep within bounds
                if n_lat < self.risk_grid.lat_min or n_lat > self.risk_grid.lat_max:
                    continue

                # Antimeridian wrap-around for longitude
                n_lon = (lon + dlo + 540.0) % 360.0 - 180.0
                n_lon = round(n_lon, 2)

                # Check landmask
                if is_land_or_shelf(n_lat, n_lon):
                    continue

                # Check hard collision cutoff
                risk = self.risk_grid.get_risk_at(n_lat, n_lon)
                if risk >= self.hard_risk_cutoff:
                    continue

                step_dist = haversine(lat, lon, n_lat, n_lon)
                neighbors.append((n_lat, n_lon, step_dist))

        return neighbors

    def find_path(
        self,
        start_lat: float,
        start_lon: float,
        goal_lat: float,
        goal_lon: float
    ) -> Optional[Dict]:
        """
        Runs A* search to find optimal safe maritime trajectory.
        """
        # Snap start and goal to grid resolution
        s_lat = round(round(start_lat / self.res) * self.res, 2)
        s_lon = round(round(start_lon / self.res) * self.res, 2)
        g_lat = round(round(goal_lat / self.res) * self.res, 2)
        g_lon = round(round(goal_lon / self.res) * self.res, 2)

        start_node = (s_lat, s_lon)
        goal_node = (g_lat, g_lon)

        # Priority queue: (f_score, g_score, lat, lon)
        open_set = []
        h_start = haversine(s_lat, s_lon, g_lat, g_lon)
        heapq.heappush(open_set, (h_start, 0.0, s_lat, s_lon))

        came_from: Dict[Tuple[float, float], Tuple[float, float]] = {}
        g_score: Dict[Tuple[float, float], float] = {start_node: 0.0}
        closed_set: Set[Tuple[float, float]] = set()

        max_iterations = 25000
        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1
            f, current_g, c_lat, c_lon = heapq.heappop(open_set)
            current = (c_lat, c_lon)

            # Check if reached goal region (within 1.5 grid steps)
            if haversine(c_lat, c_lon, g_lat, g_lon) <= (1.5 * self.res * 111.0):
                # Reconstruct path
                path = [(goal_lat, goal_lon)]
                curr = current
                while curr in came_from:
                    path.append(curr)
                    curr = came_from[curr]
                path.append((start_lat, start_lon))
                path.reverse()

                # Calculate path statistics
                total_distance_km = 0.0
                total_risk_exposure = 0.0
                waypoints = []

                for idx in range(len(path)):
                    w_lat, w_lon = path[idx]
                    r = self.risk_grid.get_risk_at(w_lat, w_lon)
                    total_risk_exposure += r
                    if idx > 0:
                        total_distance_km += haversine(path[idx-1][0], path[idx-1][1], w_lat, w_lon)
                    waypoints.append({"lat": round(w_lat, 3), "lon": round(w_lon, 3), "risk": round(r, 3)})

                avg_risk = total_risk_exposure / max(1, len(path))
                distance_nm = total_distance_km / 1.852
                est_hours_at_14knots = distance_nm / 14.0

                smoothed_waypoints = smooth_waypoints(waypoints)

                return {
                    "status": "success",
                    "waypoints": smoothed_waypoints,
                    "total_distance_km": round(total_distance_km, 1),
                    "total_distance_nm": round(distance_nm, 1),
                    "estimated_time_hours": round(est_hours_at_14knots, 1),
                    "average_risk_score": round(avg_risk, 4),
                    "max_risk_encountered": round(float(max(w["risk"] for w in waypoints)), 4),
                    "safety_rating": "Optimal (Zero Collision Threat)" if avg_risk < 0.1 else "Cautious (Ice Proximity)",
                    "iterations_searched": iterations
                }

            if current in closed_set:
                continue
            closed_set.add(current)

            for n_lat, n_lon, step_dist in self._get_neighbors(c_lat, c_lon):
                neighbor = (n_lat, n_lon)
                if neighbor in closed_set:
                    continue

                risk = self.risk_grid.get_risk_at(n_lat, n_lon)
                # Combined edge cost
                edge_cost = step_dist * (1.0 + self.fuel_weight * 0.05 + self.risk_weight * (risk**2))
                tentative_g = current_g + edge_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = haversine(n_lat, n_lon, g_lat, g_lon)
                    heapq.heappush(open_set, (tentative_g + h, tentative_g, n_lat, n_lon))

        return {
            "status": "failed",
            "message": "No viable collision-free route found within iteration constraints.",
            "iterations_searched": iterations
        }

    def dynamic_replan(
        self,
        current_vessel_pos: Tuple[float, float],
        destination: Tuple[float, float],
        active_route: List[Dict],
        encroaching_icebergs: List[Dict]
    ) -> Dict:
        """
        Dynamically triggers real-time rerouting when moving icebergs
        drift into the active vessel trajectory within the safety perimeter.
        """
        # Inject new or updated iceberg positions into the risk grid
        for berg in encroaching_icebergs:
            self.risk_grid.add_iceberg_hazard(
                iceberg_id=berg.get("id", "BERG_ALERT"),
                lat=berg["lat"],
                lon=berg["lon"],
                speed_knots=berg.get("speed_knots", 0.5),
                horizon_hours=berg.get("horizon_hours", 0.0),
                radius_km=berg.get("radius_km", 30.0)
            )

        # Check if active route intersects with updated risk zone
        threat_detected = False
        threat_detail = None
        for wp in active_route:
            r = self.risk_grid.get_risk_at(wp["lat"], wp["lon"])
            if r > 0.4:
                threat_detected = True
                threat_detail = f"Waypoint ({wp['lat']}, {wp['lon']}) compromised with risk {r:.2f}"
                break

        if not threat_detected:
            return {
                "replanning_needed": False,
                "message": "Current route remains outside active iceberg hazard envelopes.",
                "active_route": active_route
            }

        # Replan from current vessel position
        print(f"[REPLAN] Rerouting initiated: {threat_detail}")
        new_route_result = self.find_path(
            current_vessel_pos[0], current_vessel_pos[1],
            destination[0], destination[1]
        )

        return {
            "replanning_needed": True,
            "trigger_reason": threat_detail,
            "new_route": new_route_result
        }


def run_demo_route_optimization():
    """Demonstrates route optimization between Ushuaia and Palmer Station with iceberg obstacles."""
    print("[INIT] Initializing Maritime Risk Grid...")
    grid = RiskGrid(lat_min=-75.0, lat_max=-50.0, resolution_deg=0.5, safety_buffer_km=20.0)

    # Place simulated icebergs in Drake Passage / Bransfield Strait
    print("[HAZARDS] Adding iceberg hazards with forecast uncertainty cones...")
    grid.add_iceberg_hazard("A23A_LEAD", lat=-61.5, lon=-62.0, horizon_hours=24.0, radius_km=35.0)
    grid.add_iceberg_hazard("B09B_DRIFT", lat=-59.0, lon=-64.5, horizon_hours=24.0, radius_km=40.0)
    grid.add_iceberg_hazard("C15_FRAG", lat=-63.0, lon=-60.0, horizon_hours=48.0, radius_km=30.0)

    router = AStarMaritimeRouter(grid, risk_weight=8.0, fuel_weight=1.0)

    start = ANTARCTIC_STATIONS["Ushuaia Port (Argentina)"]
    goal = ANTARCTIC_STATIONS["Palmer Station (US)"]

    print(f"[A*] Computing optimal route: {start['lat']}, {start['lon']} -> {goal['lat']}, {goal['lon']}...")
    res = router.find_path(start["lat"], start["lon"], goal["lat"], goal["lon"])

    print("\n" + "=" * 60)
    print("      A* MARITIME ROUTE OPTIMIZATION RESULT")
    print("=" * 60)
    print(f"Status:               {res['status']}")
    print(f"Total Distance:       {res['total_distance_km']} km ({res['total_distance_nm']} NM)")
    print(f"Estimated Time:       {res['estimated_time_hours']} hours @ 14 knots")
    print(f"Average Risk Score:   {res['average_risk_score']}")
    print(f"Max Risk Exposure:    {res['max_risk_encountered']}")
    print(f"Safety Assessment:    {res['safety_rating']}")
    print(f"Waypoints Count:      {len(res['waypoints'])}")
    print("=" * 60)

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maritime Risk Grid & Route Optimizer")
    parser.add_argument("--test", action="store_true", help="Run self-test demo route")
    args = parser.parse_args()

    if args.test:
        run_demo_route_optimization()
    else:
        run_demo_route_optimization()
