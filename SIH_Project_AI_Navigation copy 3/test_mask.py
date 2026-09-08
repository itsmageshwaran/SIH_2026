def is_land(lat, lon):
    # Antarctica Coastline Approximation
    if lat > -60.0:
        return False
        
    if -180.0 <= lon < -150.0:
        if lat <= -76.0: return True
    elif -150.0 <= lon < -135.0:
        if lat <= -74.0: return True
    elif -135.0 <= lon < -110.0:
        if lat <= -73.0: return True
    elif -110.0 <= lon < -90.0:
        if lat <= -72.0: return True
    elif -90.0 <= lon < -75.0:
        if lat <= -72.0: return True
    elif -75.0 <= lon < -60.0:
        if lat <= -74.0: return True
    elif -60.0 <= lon < -10.0:
        if lat <= -74.5: return True # Weddell
    elif -10.0 <= lon < 30.0:
        if lat <= -71.5: return True # Maitri is -70.7
    elif 30.0 <= lon < 60.0:
        if lat <= -69.5: return True
    elif 60.0 <= lon < 90.0:
        if lat <= -70.5: return True # Bharati is -69.4
    elif 90.0 <= lon < 140.0:
        if lat <= -67.0: return True
    elif 140.0 <= lon < 160.0:
        if lat <= -68.0: return True
    elif 160.0 <= lon <= 180.0:
        if lat <= -71.0: return True
        
    # Peninsula Spine
    if -74.0 <= lat <= -62.5:
        t = (lat - (-62.5)) / (-74.0 - (-62.5))
        spine_lon = -57.0 + t * (-15.0)
        half_w = 2.5 - t * 0.5
        if (spine_lon - half_w) <= lon <= (spine_lon + half_w):
            return True
            
    return False

stations = [
    ("Maitri", -70.767, 11.733),
    ("Bharati", -69.407, 76.187),
    ("Palmer", -64.774, -64.053),
    ("Rothera", -67.568, -68.128),
    ("McMurdo", -77.846, 166.669),
    ("Davis", -68.576, 77.967),
    ("Casey", -66.282, 110.528),
    ("Ushuaia", -54.807, -68.304),
    ("Cape Town", -33.918, 18.423)
]

for name, lat, lon in stations:
    print(f"{name}: {is_land(lat, lon)}")
