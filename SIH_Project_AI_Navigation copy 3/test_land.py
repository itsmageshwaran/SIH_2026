import sys
import os
sys.path.append('.')
from src.route_optimizer import is_land_or_shelf
print(f"Ushuaia (-54.8, -68.3): {is_land_or_shelf(-54.8, -68.3)}")
print(f"Palmer (-64.77, -64.05): {is_land_or_shelf(-64.77, -64.05)}")
print(f"Maitri (-70.76, 11.73): {is_land_or_shelf(-70.76, 11.73)}")
print(f"Cape Town (-33.91, 18.42): {is_land_or_shelf(-33.91, 18.42)}")
