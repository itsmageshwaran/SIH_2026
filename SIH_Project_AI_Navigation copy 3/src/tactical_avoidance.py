import math
import heapq
from typing import List, Dict, Any, Tuple

try:
    from src.geospatial_obstacles import get_obstacle_engine
except ImportError:
    get_obstacle_engine = None

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def get_point_at_distance_bearing(lat: float, lon: float, dist_km: float, bearing_deg: float) -> Tuple[float, float]:
    R = 6371.0
    lat_r, lon_r = math.radians(lat), math.radians(lon)
    bearing_r = math.radians(bearing_deg)
    new_lat = math.asin(math.sin(lat_r) * math.cos(dist_km / R) +
                        math.cos(lat_r) * math.sin(dist_km / R) * math.cos(bearing_r))
    new_lon = lon_r + math.atan2(math.sin(bearing_r) * math.sin(dist_km / R) * math.cos(lat_r),
                                 math.cos(dist_km / R) - math.sin(lat_r) * math.sin(new_lat))
    return math.degrees(new_lat), math.degrees(new_lon)

def point_to_segment_distance_haversine(lat: float, lon: float, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    avg_lat = math.radians((lat1 + lat2 + lat) / 3.0)
    cos_lat = math.cos(avg_lat)
    
    x = lon * cos_lat * 111.0
    y = lat * 111.0
    x1 = lon1 * cos_lat * 111.0
    y1 = lat1 * 111.0
    x2 = lon2 * cos_lat * 111.0
    y2 = lat2 * 111.0
    
    l2 = (x2 - x1)**2 + (y2 - y1)**2
    if l2 == 0: return math.hypot(x - x1, y - y1)
    
    t = max(0.0, min(1.0, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / l2))
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)
    
    return math.hypot(x - proj_x, y - proj_y)

def segment_intersects_circle(lat1: float, lon1: float, lat2: float, lon2: float, c_lat: float, c_lon: float, radius_km: float) -> bool:
    d = point_to_segment_distance_haversine(c_lat, c_lon, lat1, lon1, lat2, lon2)
    return d <= radius_km

def generate_tactical_avoidance(
    ship_lat: float, ship_lon: float,
    current_route: List[List[float]],
    icebergs: List[Dict[str, Any]],
    detection_radius_km: float = 110.0,
    safety_margin_km: float = 10.0,
    ship_safety_radius_km: float = 2.0
) -> Dict[str, Any]:
    
    obstacles = []
    for berg in icebergs:
        b_lat, b_lon = berg['lat'], berg['lon']
        dist_to_ship = haversine(ship_lat, ship_lon, b_lat, b_lon)
        if dist_to_ship > detection_radius_km:
            continue
            
        sz = berg.get('size_sq_km', 25.0)
        br = math.sqrt(sz / math.pi)
        
        speed_kts = berg.get('speed_knots', 0)
        drift_km_per_hr = speed_kts * 1.852
        
        eff_radius = br + safety_margin_km + ship_safety_radius_km + (drift_km_per_hr * 1.5)
        
        obstacles.append({
            'id': berg.get('id', 'UNKNOWN'),
            'lat': b_lat,
            'lon': b_lon,
            'eff_radius': eff_radius,
            'dist_to_ship': dist_to_ship
        })

    if len(current_route) < 2:
        return {"status": "safe", "route": current_route, "reason": "Route too short", "safe_clearance": -1}
        
    min_dist = float('inf')
    closest_wp_idx = 0
    for i, wp in enumerate(current_route):
        d = haversine(ship_lat, ship_lon, wp[0], wp[1])
        if d < min_dist:
            min_dist = d
            closest_wp_idx = i
            
    if closest_wp_idx < len(current_route) - 1:
        d_to_next = haversine(ship_lat, ship_lon, current_route[closest_wp_idx+1][0], current_route[closest_wp_idx+1][1])
        if d_to_next < min_dist:
            closest_wp_idx += 1
            
    active_path = [[ship_lat, ship_lon]] + current_route[closest_wp_idx:]
    
    red_threats = []
    for obs in obstacles:
        is_threat = False
        min_clear = float('inf')
        for i in range(len(active_path) - 1):
            p1 = active_path[i]
            p2 = active_path[i+1]
            d = point_to_segment_distance_haversine(obs['lat'], obs['lon'], p1[0], p1[1], p2[0], p2[1])
            if d <= obs['eff_radius']:
                is_threat = True
            min_clear = min(min_clear, d)
        
        if is_threat:
            obs['clearance'] = min_clear
            red_threats.append(obs)
            
    if not red_threats:
        return {"status": "safe", "route": current_route, "reason": "No path intersection", "safe_clearance": -1}
        
    engine = None
    if get_obstacle_engine:
        engine = get_obstacle_engine()
        
    def is_segment_safe(lat1, lon1, lat2, lon2):
        for obs in obstacles:
            if segment_intersects_circle(lat1, lon1, lat2, lon2, obs['lat'], obs['lon'], obs['eff_radius']):
                return False
        if engine:
            hit, _ = engine.check_segment_collision((lat1, lon1), (lat2, lon2))
            if hit:
                return False
        return True

    nodes = [(ship_lat, ship_lon)]
    
    end_nodes = []
    # An end node is ONLY valid if the ENTIRE route segment AFTER it is safe.
    for i in range(closest_wp_idx, len(current_route)):
        wp = current_route[i]
        
        # Check if wp itself is safe
        wp_safe = True
        for obs in obstacles:
            if haversine(wp[0], wp[1], obs['lat'], obs['lon']) <= obs['eff_radius']:
                wp_safe = False
                break
        
        # Check if the rest of the route from this wp is safe
        rest_safe = True
        if wp_safe:
            rest_path = [wp] + current_route[i+1:]
            for k in range(len(rest_path) - 1):
                p1 = rest_path[k]
                p2 = rest_path[k+1]
                for obs in obstacles:
                    if segment_intersects_circle(p1[0], p1[1], p2[0], p2[1], obs['lat'], obs['lon'], obs['eff_radius']):
                        rest_safe = False
                        break
                if not rest_safe: break
                
        if wp_safe and rest_safe:
            nodes.append(tuple(wp))
            end_nodes.append(tuple(wp))
            
    if not end_nodes:
        # If no valid end nodes, at least add the destination, even if risky, so A* has a target.
        dest_wp = current_route[-1]
        nodes.append(tuple(dest_wp))
        end_nodes.append(tuple(dest_wp))
        
    for obs in obstacles:
        pad_radius = obs['eff_radius'] * 1.25 # increased padding for maneuverability
        for angle in range(0, 360, 45):
            pt = get_point_at_distance_bearing(obs['lat'], obs['lon'], pad_radius, angle)
            pt_safe = True
            for obs2 in obstacles:
                if haversine(pt[0], pt[1], obs2['lat'], obs2['lon']) <= obs2['eff_radius']:
                    pt_safe = False
                    break
            if pt_safe and engine:
                if engine.is_point_in_obstacle(pt[0], pt[1])[0]:
                    pt_safe = False
            if pt_safe:
                nodes.append(pt)
                
    adj = {i: [] for i in range(len(nodes))}
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            n1 = nodes[i]
            n2 = nodes[j]
            dist = haversine(n1[0], n1[1], n2[0], n2[1])
            if dist > 250.0: # relaxed max segment
                continue
            if is_segment_safe(n1[0], n1[1], n2[0], n2[1]):
                adj[i].append((j, dist))
                adj[j].append((i, dist))
                
    dest = end_nodes[-1]
    pq = [(0 + haversine(nodes[0][0], nodes[0][1], dest[0], dest[1]), 0, 0, [0])]
    visited = {}
    
    best_path = None
    best_end_node_idx = -1
    
    while pq:
        f, g, u, path = heapq.heappop(pq)
        if u in visited and visited[u] <= g:
            continue
        visited[u] = g
        
        if nodes[u] in end_nodes and u != 0:
            best_path = path
            best_end_node_idx = u
            break
            
        for v, weight in adj[u]:
            new_g = g + weight
            if v not in visited or new_g < visited[v]:
                h = haversine(nodes[v][0], nodes[v][1], dest[0], dest[1])
                if nodes[v] in end_nodes:
                    h *= 0.1 # Heavily prefer reconnecting to the route as soon as safe
                heapq.heappush(pq, (new_g + h, new_g, v, path + [v]))
                
    if not best_path:
        return {"status": "blocked", "route": current_route, "reason": "No safe path around active hazards"}
        
    avoidance_pts = [nodes[i] for i in best_path]
    reconnect_wp = nodes[best_end_node_idx]
    
    reconnect_idx = -1
    for i in range(closest_wp_idx, len(current_route)):
        if tuple(current_route[i]) == reconnect_wp:
            reconnect_idx = i
            break
            
    if reconnect_idx == -1:
        reconnect_idx = len(current_route) - 1
        
    final_route = [list(pt) for pt in avoidance_pts] + current_route[reconnect_idx+1:]
    
    red_ids = [t['id'] for t in red_threats]
    min_clear = min(t['clearance'] for t in red_threats)
    
    return {
        "status": "avoidance_required",
        "threats": red_ids,
        "avoidance_side": "dynamic",
        "original_route": current_route,
        "avoidance_route": final_route,
        "safe_clearance": min_clear,
        "reason": f"Projected intersection with {', '.join(red_ids)}"
    }
