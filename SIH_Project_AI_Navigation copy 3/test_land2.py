import sys
sys.path.append('.')
from src.route_optimizer import is_land_or_shelf
print(f"(-70, 40) deep in East Antarctica: {is_land_or_shelf(-70, 40)}")
print(f"(-71, 0) deep in Dronning Maud Land: {is_land_or_shelf(-71, 0)}")
