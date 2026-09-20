import math
from typing import List, Dict, Any, Tuple

def generate_tactical_avoidance(
    ship_lat: float, ship_lon: float,
    current_route: List[List[float]],
    icebergs: List[Dict[str, Any]],
    detection_radius_km: float = 110.0,
    safety_margin_km: float = 10.0,
    ship_safety_radius_km: float = 2.0
) -> Dict[str, Any]:
    cos_lat = math.cos(math.radians(ship_lat))
    def to_km(lat, lon):
        return (lon - ship_lon) * 111.0 * cos_lat, (lat - ship_lat) * 111.0
    def to_deg(x, y):
        return ship_lat + y / 111.0, ship_lon + x / (111.0 * cos_lat)

    route_km = [to_km(p[0], p[1]) for p in current_route]
    
    def point_to_segment_distance(px, py, ax, ay, bx, by):
        l2 = (bx - ax)**2 + (by - ay)**2
        if l2 == 0: return math.hypot(px - ax, py - ay)
        t = max(0, min(1, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / l2))
        proj_x = ax + t * (bx - ax)
        proj_y = ay + t * (by - ay)
        return math.hypot(px - proj_x, py - proj_y)
    
    threats = []
    for berg in icebergs:
        bx, by = to_km(berg['lat'], berg['lon'])
        dist_to_ship = math.hypot(bx, by)
        if dist_to_ship > detection_radius_km:
            continue
            
        sz = berg.get('size_sq_km', 25.0)
        br = math.sqrt(sz / math.pi)
        eff_radius = br + safety_margin_km + ship_safety_radius_km
        
        is_threat = False
        min_clearance = float('inf')
        threat_seg_idx = -1
        
        for i in range(len(route_km) - 1):
            ax, ay = route_km[i]
            bx_seg, by_seg = route_km[i+1]
            
            d_ship_a = math.hypot(ax, ay)
            if d_ship_a > detection_radius_km + 50.0:
                continue
                
            d = point_to_segment_distance(bx, by, ax, ay, bx_seg, by_seg)
            if d < eff_radius:
                is_threat = True
                if d < min_clearance:
                    min_clearance = d
                    threat_seg_idx = i
                
        if is_threat:
            threats.append({
                "iceberg": berg,
                "bx": bx, "by": by,
                "eff_radius": eff_radius,
                "seg_idx": threat_seg_idx,
                "clearance": min_clearance,
                "dist_to_ship": dist_to_ship
            })
            
    if not threats:
        return {"status": "safe", "route": current_route, "reason": "No path intersection", "safe_clearance": -1}
        
    threats.sort(key=lambda t: t['dist_to_ship'])
    primary = threats[0]

    def is_point_safe(pt_x, pt_y):
        for berg in icebergs:
            bx_b, by_b = to_km(berg['lat'], berg['lon'])
            sz = berg.get('size_sq_km', 25.0)
            br = math.sqrt(sz / math.pi)
            eff_radius = br + safety_margin_km + ship_safety_radius_km
            if math.hypot(pt_x - bx_b, pt_y - by_b) <= eff_radius:
                return False
        return True

    # Search outward for safe waypoints
    first_safe_idx_before = primary['seg_idx']
    while first_safe_idx_before > 0:
        if is_point_safe(*route_km[first_safe_idx_before]):
            # Also check if it's sufficiently far from the primary
            if math.hypot(route_km[first_safe_idx_before][0] - primary['bx'], 
                          route_km[first_safe_idx_before][1] - primary['by']) > primary['eff_radius'] * 2.0:
                break
        first_safe_idx_before -= 1

    first_safe_idx_after = primary['seg_idx'] + 1
    while first_safe_idx_after < len(route_km) - 1:
        if is_point_safe(*route_km[first_safe_idx_after]):
            if math.hypot(route_km[first_safe_idx_after][0] - primary['bx'], 
                          route_km[first_safe_idx_after][1] - primary['by']) > primary['eff_radius'] * 2.0:
                break
        first_safe_idx_after += 1

    if first_safe_idx_after <= first_safe_idx_before:
        first_safe_idx_after = min(len(route_km)-1, first_safe_idx_before + 1)
        
    ax, ay = route_km[first_safe_idx_before]
    bx_seg, by_seg = route_km[first_safe_idx_after]
    
    dx = bx_seg - ax
    dy = by_seg - ay
    l = math.hypot(dx, dy)
    
    if l < 0.1:
        return {"status": "blocked", "route": current_route, "reason": "Too close to resolve"}
        
    nx, ny = dx/l, dy/l
    px, py = -ny, nx
    
    def validate_curve(curve_pts: List[Tuple[float, float]], ax, ay, bx_seg, by_seg) -> bool:
        pts = [(ax, ay)] + curve_pts + [(bx_seg, by_seg)]
        for berg in icebergs:
            bx_b, by_b = to_km(berg['lat'], berg['lon'])
            sz = berg.get('size_sq_km', 25.0)
            br = math.sqrt(sz / math.pi)
            eff_radius = br + safety_margin_km + ship_safety_radius_km
            for i in range(len(pts) - 1):
                px1, py1 = pts[i]
                px2, py2 = pts[i+1]
                dist = point_to_segment_distance(bx_b, by_b, px1, py1, px2, py2)
                if dist <= eff_radius:
                    return False
        return True

    # Iteratively expand the curve outwards until safe
    base_push = primary['eff_radius'] * 1.5
    for push_multiplier in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        required_push = base_push * push_multiplier
        
        def generate_curve(side_px, side_py) -> List[Tuple[float, float]]:
            steps = 8
            curve = []
            for i in range(1, steps):
                frac = i / steps
                base_x = ax + frac * dx
                base_y = ay + frac * dy
                offset = math.sin(frac * math.pi) * required_push
                curve_x = base_x + side_px * offset
                curve_y = base_y + side_py * offset
                curve.append((curve_x, curve_y))
            return curve

        curve_left = generate_curve(px, py)
        c1_safe = validate_curve(curve_left, ax, ay, bx_seg, by_seg)
        
        curve_right = generate_curve(-px, -py)
        c2_safe = validate_curve(curve_right, ax, ay, bx_seg, by_seg)
        
        if c1_safe and not c2_safe:
            chosen_curve = curve_left
            side = "left"
            break
        elif c2_safe and not c1_safe:
            chosen_curve = curve_right
            side = "right"
            break
        elif c1_safe and c2_safe:
            chosen_curve = curve_left 
            side = "left"
            break
    else:
        return {"status": "blocked", "route": current_route, "reason": "No safe path around iceberg"}
        
    new_route_km = route_km[:first_safe_idx_before+1]
    new_route_km.extend(chosen_curve)
    new_route_km.extend(route_km[first_safe_idx_after:])
    
    new_route_deg = [list(to_deg(x, y)) for x, y in new_route_km]
    
    return {
        "status": "avoidance_required",
        "threats": [t['iceberg']['id'] for t in threats],
        "avoidance_side": side,
        "original_route": current_route,
        "avoidance_route": new_route_deg,
        "safe_clearance": primary['eff_radius'],
        "reason": f"Projected intersection with {primary['iceberg']['id']}"
    }
