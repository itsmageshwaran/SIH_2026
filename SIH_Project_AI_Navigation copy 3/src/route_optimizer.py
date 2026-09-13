"""
Antarctic Maritime Risk Engine & A* Pathfinding Optimizer
==========================================================
Constructs dynamic spatial collision risk heatmaps around tracked icebergs
with expanding forecast uncertainty cones. Solves optimal candidate maritime
trajectories between Antarctic stations and gateway ports using continuous
segment clearance validation against Natural Earth 1:50m geometries.

Terminal Statuses:
  SCREENED_COARSE_REGIONAL_CONSTRAINTS – Clears coarse 1:50m screening (DEMO PLAYBACK ONLY)
  ROUTE_BLOCKED                        – Segment intersects land/shelf or severe iceberg envelope
  REQUIRES_REPLANNING_HAZARD_INTERCEPT – Active trajectory intercepted by drifting iceberg cone
  SAFETY_UNVERIFIED_DATA_INSUFFICIENT  – Missing data, out of envelope, or unconfirmed endpoint
"""

import math
import heapq
import json
import argparse
from typing import List, Tuple, Dict, Optional, Set, Any
import numpy as np
import pandas as pd

from src.geospatial_obstacles import obstacle_engine, haversine_km, get_obstacle_engine

# Audited Scientific Stations & Gateway Ports
ANTARCTIC_STATIONS: Dict[str, Dict[str, Any]] = {
    "Princess Astrid Staging Point (Illustrative)": {
        "lat": -69.8500, "lon": 11.9000,
        "type": "illustrative_staging", "country": "India",
        "marine_accessible": True,
        "requires_opt_in": True,
        "overland_transit_estimate_km": 101.0,
        "notes": "Illustrative demonstration fast-ice staging coordinate along Princess Astrid Coast (NOT OFFICIAL / DYNAMIC). Requires operator opt-in or manual coordinate entry. Overland transit to Maitri Station is an unverified estimate (~100 km geodesic across ice sheet) requiring seasonal field survey."
    },
    "Maitri Base (India)": {
        "lat": -70.7658, "lon": 11.7358,
        "official_dms": "70S 11E",
        "authority": "NCPOR / COMNAP Station Catalogue (Schirmacher Oasis, Queen Maud Land)",
        "type": "inland_base", "country": "India",
        "marine_accessible": False,
        "notes": "Inland bedrock facility in Schirmacher Oasis (~90-100 km from ocean). Direct maritime navigation impossible.",
        "marine_landing": {
            "name": "India Bay (Princess Astrid Coast fast ice staging)",
            "lat": -69.8500, "lon": 11.9000,
            "distance_km": 101.0
        }
    },
    "Bharati Station (India)": {
        "lat": -69.4078, "lon": 76.1872,
        "official_dms": "69°24'S, 76°11'E",
        "authority": "NCPOR Scientific Expedition Reports (Quilty Bay / Thala Fjord, Larsemann Hills)",
        "type": "station_anchorage", "country": "India",
        "marine_accessible": True,
        "notes": "Coastal anchorage ~500m offshore with ship barge offload.",
        "marine_landing": {
            "name": "Thala Fjord / Quilty Bay Landing Site",
            "lat": -69.4078, "lon": 76.1872,
            "distance_km": 0.5
        }
    },
    "Palmer Station (US)": {
        "lat": -64.8000, "lon": -64.1500,
        "authority": "USAP Palmer Station Operations Manual (Biscoe Bay Approach, Anvers Island)",
        "type": "station_pier", "country": "USA",
        "marine_accessible": True,
        "notes": "Biscoe Bay seaward approach off Anvers Island."
    },
    "Rothera Station (UK)": {
        "lat": -67.6000, "lon": -68.3000,
        "authority": "UKHO Admiralty Chart 3570 / BAS Directory (Adelaide Island Seaward Approach)",
        "type": "station_wharf", "country": "UK",
        "marine_accessible": True,
        "notes": "Adelaide Island seaward approach."
    },
    "McMurdo Station (US)": {
        "lat": -77.850, "lon": 166.667,
        "authority": "USAP / NSF Polar Programs Manual (Winter Quarters Bay, Ross Island)",
        "type": "station_ice_pier", "country": "USA",
        "marine_accessible": True
    },
    "Davis Station (Australia)": {
        "lat": -68.576, "lon": 77.967,
        "authority": "Australian Antarctic Division (Anchorage Cove, Vestfold Hills)",
        "type": "station_anchorage", "country": "Australia",
        "marine_accessible": True
    },
    "Casey Station (Australia)": {
        "lat": -66.282, "lon": 110.528,
        "authority": "Australian Antarctic Division (Newcomb Bay, Bailey Peninsula)",
        "type": "station_anchorage", "country": "Australia",
        "marine_accessible": True
    },
    "Ushuaia Port (Argentina)": {
        "lat": -55.2000, "lon": -66.2000,
        "authority": "NGA World Port Index Pub 150 (#61530) (Eastern Beagle Channel Seaward Roadstead)",
        "type": "port", "country": "Argentina",
        "marine_accessible": True,
        "notes": "Eastern Beagle Channel seaward roadstead. Inner Beagle Channel wharf pilotage is tactical and unmodeled at 1:50m regional scale."
    },
    "Punta Arenas (Chile)": {
        "lat": -53.167, "lon": -70.908,
        "authority": "NGA World Port Index Pub 150 (#61470) (Strait of Magellan)",
        "type": "port", "country": "Chile",
        "marine_accessible": True
    },
    "Cape Town Port (South Africa)": {
        "lat": -33.8800, "lon": 18.4400,
        "authority": "NGA World Port Index Pub 150 (#46270) (Table Bay Roadstead)",
        "type": "port", "country": "South Africa",
        "marine_accessible": True,
        "notes": "Table Bay seaward roadstead. Duncan Dock inner harbor basin is unmodeled at 1:50m regional scale."
    },
    "Hobart Port (Australia)": {
        "lat": -43.1500, "lon": 147.6000,
        "authority": "NGA World Port Index Pub 150 (#55060) (Storm Bay Seaward Approach)",
        "type": "port", "country": "Australia",
        "marine_accessible": True,
        "notes": "Storm Bay seaward approach."
    },
    "Maitri (India)": {
        "lat": -70.7658, "lon": 11.7358,
        "official_dms": "70S 11E",
        "authority": "NCPOR / COMNAP Station Catalogue (Schirmacher Oasis, Queen Maud Land)",
        "type": "inland_base", "country": "India",
        "marine_accessible": False,
        "notes": "Inland bedrock facility in Schirmacher Oasis (~90-100 km from ocean). Direct maritime navigation impossible.",
        "marine_landing": {
            "name": "India Bay (Princess Astrid Coast fast ice staging)",
            "lat": -69.8500, "lon": 11.9000,
            "distance_km": 101.0
        }
    },
    "Bharati (India)": {
        "lat": -69.4078, "lon": 76.1872,
        "official_dms": "69°24'S, 76°11'E",
        "authority": "NCPOR Scientific Expedition Reports (Quilty Bay / Thala Fjord, Larsemann Hills)",
        "type": "station_anchorage", "country": "India",
        "marine_accessible": True,
        "notes": "Coastal anchorage ~500m offshore with ship barge offload.",
        "marine_landing": {
            "name": "Thala Fjord / Quilty Bay Landing Site",
            "lat": -69.4078, "lon": 76.1872,
            "distance_km": 0.5
        }
    }
}

FUEL_PROFILES: Dict[str, Dict] = {
    "basic": {
        "speed_knots": 10.0,
        "risk_weight": 1.5,
        "hard_risk_cutoff": 0.90,
        "fuel_rate_per_nm": 0.85,
        "label": "Basic (Fuel-Save)",
        "color": "#10b981",
    },
    "balanced": {
        "speed_knots": 14.0,
        "risk_weight": 5.0,
        "hard_risk_cutoff": 0.80,
        "fuel_rate_per_nm": 1.65,
        "label": "Balanced (Standard)",
        "color": "#0ea5e9",
    },
    "advanced": {
        "speed_knots": 16.0,
        "risk_weight": 10.0,
        "hard_risk_cutoff": 0.65,
        "fuel_rate_per_nm": 2.60,
        "label": "Advanced (Max Safety)",
        "color": "#f59e0b",
    },
}

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return haversine_km(lat1, lon1, lat2, lon2)

def is_land_or_shelf(lat: float, lon: float) -> bool:
    return obstacle_engine.is_land_or_shelf(lat, lon)

def smooth_waypoints(waypoints: List[Dict]) -> List[Dict]:
    """
    Applies moving average to candidate waypoints followed by an immutable
    Post-Smoothing Validation Gate. If any smoothed segment violates land clearance,
    smoothing is discarded and the original path is preserved.
    """
    if len(waypoints) < 3:
        return waypoints

    smoothed = [waypoints[0]]
    for i in range(1, len(waypoints) - 1):
        s_lat = (waypoints[i-1]["lat"] + waypoints[i]["lat"] + waypoints[i+1]["lat"]) / 3.0
        s_lon = (waypoints[i-1]["lon"] + waypoints[i]["lon"] + waypoints[i+1]["lon"]) / 3.0
        smoothed.append({
            "lat": round(s_lat, 4),
            "lon": round(s_lon, 4),
            "risk": waypoints[i].get("risk", 0.0)
        })
    smoothed.append(waypoints[-1])

    # Gate 5: Post-Smoothing Validation Gate
    for i in range(len(smoothed) - 1):
        is_term = (i == 0 or i == len(smoothed) - 2)
        seg_res = obstacle_engine.check_segment_clearance(
            smoothed[i]["lat"], smoothed[i]["lon"],
            smoothed[i+1]["lat"], smoothed[i+1]["lon"],
            min_standoff_km=0.0 if is_term else 6.0
        )
        if not seg_res["is_clear"]:
            return waypoints  # Revert to unsmoothed candidate path

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
            sigma_km = self.safety_buffer_km + (horizon_hours * 0.40)
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

        max_dist = 3.0 * sigma_km
        lat_span = max_dist / 111.0

        i_min = max(0, int((lat - lat_span - self.lat_min) / self.res))
        i_max = min(self.n_lat, int((lat + lat_span - self.lat_min) / self.res) + 1)

        for i in range(i_min, i_max):
            c_lat = self.lats[i]
            cos_lat = max(0.1, math.cos(math.radians(c_lat)))
            lon_span = max_dist / (111.0 * cos_lat)

            j_center = int((lon - self.lon_min) / self.res)
            j_span_cells = int(math.ceil(lon_span / self.res))

            for dj in range(-j_span_cells, j_span_cells + 1):
                j = (j_center + dj) % self.n_lon
                c_lon = self.lons[j]

                d_km = haversine_km(c_lat, c_lon, lat, lon)
                if d_km < max_dist:
                    intensity = math.exp(-(d_km**2) / (2.0 * (sigma_km**2)))
                    self.grid[i, j] = max(self.grid[i, j], float(intensity))

    def get_risk_at(self, lat: float, lon: float) -> float:
        if lat < self.lat_min or lat > self.lat_max:
            return 1.0

        i = int(round((lat - self.lat_min) / self.res))
        i = max(0, min(self.n_lat - 1, i))

        norm_lon = (lon + 180.0) % 360.0 - 180.0
        j = int(round((norm_lon - self.lon_min) / self.res))
        j = max(0, min(self.n_lon - 1, j))

        return float(self.grid[i, j])


class AStarMaritimeRouter:
    """
    A* Maritime Pathfinding Engine with Continuous Segment Obstacle Screening.
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
        neighbors = []
        d_lats = [-self.res, 0.0, self.res]
        d_lons = [-self.res, 0.0, self.res]

        for dl in d_lats:
            for dlo in d_lons:
                if dl == 0.0 and dlo == 0.0:
                    continue

                n_lat = round(lat + dl, 2)
                if n_lat < self.risk_grid.lat_min or n_lat > self.risk_grid.lat_max:
                    continue

                n_lon = (lon + dlo + 540.0) % 360.0 - 180.0
                n_lon = round(n_lon, 2)

                # 1. Point check against land/ice-shelf
                if is_land_or_shelf(n_lat, n_lon):
                    continue

                # 2. Continuous segment clearance check (guarantees no diagonal cutting across land)
                seg_res = obstacle_engine.check_segment_clearance(lat, lon, n_lat, n_lon)
                if not seg_res["is_clear"]:
                    continue

                # 3. Hard collision cutoff against iceberg risk surface
                risk = self.risk_grid.get_risk_at(n_lat, n_lon)
                if risk >= self.hard_risk_cutoff:
                    continue

                step_dist = haversine_km(lat, lon, n_lat, n_lon)
                neighbors.append((n_lat, n_lon, step_dist))

        return neighbors

    def find_path(
        self,
        start_lat: float,
        start_lon: float,
        goal_lat: float,
        goal_lon: float,
        start_name: str = "Origin",
        goal_name: str = "Destination",
        vessel_speed_knots: float = 14.0,
        operator_opt_in: bool = False
    ) -> Dict[str, Any]:
        """
        Runs A* search and passes candidate trajectory through the 5-Gate Validation Pipeline.
        """
        # Gate 1: Operational Envelopes
        val_start, reason_start = obstacle_engine.validate_operational_envelope(start_lat, start_lon)
        val_goal, reason_goal = obstacle_engine.validate_operational_envelope(goal_lat, goal_lon)
        if not val_start or not val_goal:
            return {
                "status": "SAFETY_UNVERIFIED_DATA_INSUFFICIENT",
                "demo_playback_only": True,
                "message": f"Endpoint outside documented operational screening coverage: {reason_start if not val_start else reason_goal}",
                "waypoints": [],
                "total_distance_km": 0.0,
                "total_distance_nm": 0.0,
                "bathymetry": {"status": "UNVERIFIED_NOT_MODELED", "under_keel_clearance_checked": False},
                "safety_disclaimer": "Safety cannot be determined. Requested coordinates leave documented screening coverage."
            }

        # Gate 2: Endpoint Operational Validity
        def _resolve_roadstead(lat: float, lon: float, name: str) -> Tuple[float, float, bool]:
            n = (name or "").lower()
            if "ushuaia" in n or (abs(lat - (-54.807)) < 0.5 and abs(lon - (-68.304)) < 1.0):
                return -55.2000, -66.2000, True
            if "punta" in n or (abs(lat - (-53.16)) < 0.5 and abs(lon - (-70.91)) < 1.0):
                return -52.8000, -67.5000, True
            if "palmer" in n or (abs(lat - (-64.774)) < 0.2 and abs(lon - (-64.053)) < 0.2):
                return -64.8000, -64.1500, True
            if "cape town" in n or (abs(lat - (-33.91)) < 0.2 and abs(lon - 18.43) < 0.2):
                return -33.8800, 18.4400, True
            if "hobart" in n or (abs(lat - (-42.88)) < 0.2 and abs(lon - 147.33) < 0.2):
                return -43.1500, 147.6000, True
            return lat, lon, False

        start_lat, start_lon, _ = _resolve_roadstead(start_lat, start_lon, start_name)
        goal_lat, goal_lon, _ = _resolve_roadstead(goal_lat, goal_lon, goal_name)

        # Check inland station targeting
        is_maitri_inland_start = (abs(start_lat - (-70.7658)) < 0.05 and abs(start_lon - 11.7358) < 0.05)
        is_maitri_inland_goal = (abs(goal_lat - (-70.7658)) < 0.05 and abs(goal_lon - 11.7358) < 0.05)
        if (is_maitri_inland_start or is_maitri_inland_goal) and not operator_opt_in:
            return {
                "status": "SAFETY_UNVERIFIED_DATA_INSUFFICIENT",
                "demo_playback_only": True,
                "message": "Maitri Base (-70.7658° S, 11.7358° E) is an inland facility in Schirmacher Oasis with no maritime access. Select 'Princess Astrid Staging Point (Illustrative)' with operator opt-in confirmation or supply custom surveyed fast-ice coordinates.",
                "waypoints": [],
                "total_distance_km": 0.0,
                "total_distance_nm": 0.0,
                "bathymetry": {"status": "UNVERIFIED_NOT_MODELED", "under_keel_clearance_checked": False}
            }

        # Check if endpoints are on land
        if is_land_or_shelf(start_lat, start_lon):
            return {
                "status": "ROUTE_BLOCKED",
                "demo_playback_only": True,
                "message": f"Departure point ({start_lat}, {start_lon}) is located on land or permanent ice shelf.",
                "waypoints": [],
                "total_distance_km": 0.0,
                "bathymetry": {"status": "UNVERIFIED_NOT_MODELED", "under_keel_clearance_checked": False}
            }
        if is_land_or_shelf(goal_lat, goal_lon):
            return {
                "status": "ROUTE_BLOCKED",
                "demo_playback_only": True,
                "message": f"Destination point ({goal_lat}, {goal_lon}) is located on land or permanent ice shelf.",
                "waypoints": [],
                "total_distance_km": 0.0,
                "bathymetry": {"status": "UNVERIFIED_NOT_MODELED", "under_keel_clearance_checked": False}
            }

        # Snap start and goal to grid resolution, ensuring grid nodes are in navigable water
        # and have collision-free line-of-sight to the endpoint
        s_lat = round(round(start_lat / self.res) * self.res, 2)
        s_lon = round(round(start_lon / self.res) * self.res, 2)
        best_start_node = None
        best_start_d = 999999.0
        for dl in [0.0, self.res, -self.res, 2*self.res, -2*self.res]:
            for dlo in [0.0, self.res, -self.res, 2*self.res, -2*self.res]:
                c_lat = round(s_lat + dl, 2)
                c_lon = round((s_lon + dlo + 540.0) % 360.0 - 180.0, 2)
                if not is_land_or_shelf(c_lat, c_lon):
                    chk = obstacle_engine.check_segment_clearance(start_lat, start_lon, c_lat, c_lon, min_standoff_km=0.0)
                    if chk["is_clear"]:
                        d = haversine_km(start_lat, start_lon, c_lat, c_lon)
                        if d < best_start_d:
                            best_start_d = d
                            best_start_node = (c_lat, c_lon)
            if best_start_node is not None:
                break

        if best_start_node is None:
            return {
                "status": "ROUTE_BLOCKED",
                "demo_playback_only": True,
                "message": f"Departure point ({start_lat}, {start_lon}) cannot reach open navigable water without intersecting land/ice barriers.",
                "waypoints": [],
                "total_distance_km": 0.0,
                "bathymetry": {"status": "UNVERIFIED_NOT_MODELED", "under_keel_clearance_checked": False}
            }
        s_lat, s_lon = best_start_node

        g_lat = round(round(goal_lat / self.res) * self.res, 2)
        g_lon = round(round(goal_lon / self.res) * self.res, 2)
        best_goal_node = None
        best_goal_d = 999999.0
        for dl in [0.0, self.res, -self.res, 2*self.res, -2*self.res]:
            for dlo in [0.0, self.res, -self.res, 2*self.res, -2*self.res]:
                c_lat = round(g_lat + dl, 2)
                c_lon = round((g_lon + dlo + 540.0) % 360.0 - 180.0, 2)
                if not is_land_or_shelf(c_lat, c_lon):
                    chk = obstacle_engine.check_segment_clearance(c_lat, c_lon, goal_lat, goal_lon, min_standoff_km=0.0)
                    if chk["is_clear"]:
                        d = haversine_km(c_lat, c_lon, goal_lat, goal_lon)
                        if d < best_goal_d:
                            best_goal_d = d
                            best_goal_node = (c_lat, c_lon)
            if best_goal_node is not None:
                break

        if best_goal_node is None:
            return {
                "status": "ROUTE_BLOCKED",
                "demo_playback_only": True,
                "message": f"Destination point ({goal_lat}, {goal_lon}) cannot be reached from open navigable water without intersecting land/ice barriers.",
                "waypoints": [],
                "total_distance_km": 0.0,
                "bathymetry": {"status": "UNVERIFIED_NOT_MODELED", "under_keel_clearance_checked": False}
            }
        g_lat, g_lon = best_goal_node

        start_node = (s_lat, s_lon)
        goal_node = (g_lat, g_lon)

        open_set = []
        h_start = haversine_km(s_lat, s_lon, goal_lat, goal_lon)
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

            # Check if reached goal region
            if haversine_km(c_lat, c_lon, g_lat, g_lon) <= (1.5 * self.res * 111.0):
                # Verify segment from current node to actual goal coordinates
                goal_seg_check = obstacle_engine.check_segment_clearance(c_lat, c_lon, goal_lat, goal_lon, min_standoff_km=0.0)
                if goal_seg_check["is_clear"]:
                    raw_path = [(goal_lat, goal_lon)]
                    curr = current
                    while curr is not None:
                        raw_path.append(curr)
                        curr = came_from.get(curr)
                    raw_path.append((start_lat, start_lon))
                    raw_path.reverse()

                    # Deduplicate consecutive identical/near-identical points
                    path = [raw_path[0]]
                    for pt in raw_path[1:]:
                        if haversine_km(path[-1][0], path[-1][1], pt[0], pt[1]) > 0.5:
                            path.append(pt)
                    if len(path) == 1:
                        path.append((goal_lat, goal_lon))

                    # Calculate path metrics and minimum clearance
                    total_distance_km = 0.0
                    total_risk_exposure = 0.0
                    min_land_clearance_km = 9999.0
                    waypoints = []
                    path_valid = True

                    for idx in range(len(path)):
                        w_lat, w_lon = path[idx]
                        r = self.risk_grid.get_risk_at(w_lat, w_lon)
                        total_risk_exposure += r
                        if idx > 0:
                            seg_d = haversine_km(path[idx-1][0], path[idx-1][1], w_lat, w_lon)
                            total_distance_km += seg_d
                            is_terminal = (idx == 1 or idx == len(path) - 1)
                            # Segment clearance check
                            chk = obstacle_engine.check_segment_clearance(
                                path[idx-1][0], path[idx-1][1], w_lat, w_lon,
                                min_standoff_km=0.0 if is_terminal else 6.0
                            )
                            if not chk["is_clear"]:
                                path_valid = False
                                break
                            min_land_clearance_km = min(min_land_clearance_km, chk.get("min_clearance_km", 9999.0))
                        waypoints.append({"lat": round(w_lat, 3), "lon": round(w_lon, 3), "risk": round(r, 3)})

                    if path_valid:
                        avg_risk = total_risk_exposure / max(1, len(path))
                        distance_nm = total_distance_km / 1.852
                        est_hours = distance_nm / max(1.0, vessel_speed_knots)

                        # Gate 5: Post-Smoothing Gate
                        smoothed_waypoints = smooth_waypoints(waypoints)

                        return {
                            "route_found": True,
                            "status": "SCREENED_COARSE_REGIONAL_CONSTRAINTS",
                            "status_label": "DEMO PLAYBACK ONLY (SCREENED AGAINST COARSE 1:50M CONSTRAINTS)",
                            "label": "DEMO PLAYBACK ONLY (SCREENED AGAINST COARSE 1:50M CONSTRAINTS)",
                            "demo_playback_only": True,
                            "waypoints": smoothed_waypoints,
                            "total_distance_km": round(total_distance_km, 1),
                            "total_distance_nm": round(distance_nm, 1),
                            "estimated_time_hours": round(est_hours, 1),
                            "average_risk_score": round(avg_risk, 4),
                            "max_risk_encountered": round(float(max(w["risk"] for w in waypoints)), 4),
                            "min_land_clearance_km": round(min_land_clearance_km, 1),
                            "clearance_summary": {
                                "min_land_clearance_km": round(min_land_clearance_km, 1),
                                "coastal_proximity_warning": min_land_clearance_km < 25.0
                            },
                            "bathymetry": {
                                "status": "UNVERIFIED_NOT_MODELED",
                                "under_keel_clearance_checked": False,
                                "disclaimer": "Water depth and under-keel clearance are NOT modeled. Grounding risk is unverified."
                            },
                            "safety_disclaimer": "Screened against Natural Earth 1:50m regional geometries and BYU/NIC tracked icebergs. NOT PROOF OF NAVIGABILITY OR REAL-WORLD SAFETY.",
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
                edge_cost = step_dist * (1.0 + self.fuel_weight * 0.05 + self.risk_weight * (risk**2))
                tentative_g = current_g + edge_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = haversine_km(n_lat, n_lon, goal_lat, goal_lon)
                    heapq.heappush(open_set, (tentative_g + h, tentative_g, n_lat, n_lon))

        return {
            "status": "ROUTE_BLOCKED",
            "demo_playback_only": True,
            "message": "No collision-free maritime route found around modeled land/ice obstacles within iteration constraints.",
            "waypoints": [],
            "total_distance_km": 0.0,
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
        Dynamically triggers rerouting when moving icebergs breach active trajectory.
        """
        for berg in encroaching_icebergs:
            self.risk_grid.add_iceberg_hazard(
                iceberg_id=berg.get("id", "BERG_ALERT"),
                lat=berg["lat"],
                lon=berg["lon"],
                speed_knots=berg.get("speed_knots", 0.5),
                horizon_hours=berg.get("horizon_hours", 0.0),
                radius_km=berg.get("radius_km", 30.0)
            )

        threat_detected = False
        threat_detail = None
        for wp in active_route:
            r = self.risk_grid.get_risk_at(wp["lat"], wp["lon"])
            if r > 0.40:
                threat_detected = True
                threat_detail = f"Waypoint ({wp['lat']}, {wp['lon']}) compromised with risk {r:.2f}"
                break

        if not threat_detected:
            return {
                "replanning_needed": False,
                "status": "SCREENED_COARSE_REGIONAL_CONSTRAINTS",
                "message": "Current route remains outside active iceberg hazard envelopes.",
                "active_route": active_route
            }

        new_route_result = self.find_path(
            current_vessel_pos[0], current_vessel_pos[1],
            destination[0], destination[1]
        )

        return {
            "replanning_needed": True,
            "status": "REQUIRES_REPLANNING_HAZARD_INTERCEPT",
            "trigger_reason": threat_detail,
            "new_route": new_route_result
        }


class AntarcticRouteOptimizer:
    """
    High-level Antarctic Maritime Route Optimizer and Safety Gate Verifier.
    Wraps AStarMaritimeRouter and provides post-processing and programmatic safety verification.
    """
    def __init__(self, obstacle_engine=None):
        self.engine = obstacle_engine or get_obstacle_engine()

    def optimize_route(
        self,
        start_lat: float,
        start_lon: float,
        goal_lat: float,
        goal_lon: float,
        start_name: str = "Origin",
        goal_name: str = "Destination",
        vessel_speed_knots: float = 14.0,
        operator_opt_in: bool = False,
        risk_weight: float = 5.0
    ) -> Dict[str, Any]:
        lat_min = max(-78.0, min(start_lat, goal_lat) - 5.0)
        lat_max = min(-30.0, max(start_lat, goal_lat) + 5.0)
        grid = RiskGrid(lat_min=lat_min, lat_max=lat_max, resolution_deg=1.0)
        router = AStarMaritimeRouter(grid, risk_weight=risk_weight)
        res = router.find_path(
            start_lat, start_lon, goal_lat, goal_lon,
            start_name=start_name, goal_name=goal_name,
            vessel_speed_knots=vessel_speed_knots,
            operator_opt_in=operator_opt_in
        )
        res["route_found"] = (res.get("status") == "SCREENED_COARSE_REGIONAL_CONSTRAINTS" and len(res.get("waypoints", [])) > 0)
        res["label"] = res.get("status_label", "DEMO PLAYBACK ONLY (SCREENED AGAINST COARSE 1:50M CONSTRAINTS)")
        return res

    def post_process_route_smoothing(
        self,
        route_coords: List[Any],
        min_standoff_km: float = 0.0
    ) -> List[Any]:
        """
        Gate 5: Line-of-sight shortcutting / smoothing.
        Strictly verifies that candidate shortcuts do not intersect obstacles.
        """
        if len(route_coords) <= 2:
            return route_coords

        is_dict = isinstance(route_coords[0], dict)
        coords = [(c["lat"], c["lon"]) if is_dict else (c[0], c[1]) for c in route_coords]

        smoothed = [coords[0]]
        curr_idx = 0

        while curr_idx < len(coords) - 1:
            furthest_idx = curr_idx + 1
            for next_idx in range(len(coords) - 1, curr_idx, -1):
                p1 = coords[curr_idx]
                p2 = coords[next_idx]
                blocked, _ = self.engine.check_segment_collision(p1, p2, min_standoff_km=min_standoff_km)
                if not blocked:
                    furthest_idx = next_idx
                    break
            smoothed.append(coords[furthest_idx])
            curr_idx = furthest_idx

        if is_dict:
            return [{"lat": p[0], "lon": p[1]} for p in smoothed]
        return smoothed
