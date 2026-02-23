"""Core foundation classes for terrain analysis and geodesic calculations.

This module provides the mathematical backbone for ski resort planning:
- GeoCalculator: Geodesic calculations (distances, bearings, destinations)
- DEMService: Digital Elevation Model access (singleton pattern)
- TerrainAnalyzer: Gradient and orientation analysis
- PathTracer: Path generation algorithms (import directly from path_tracer module)

Mathematical details documented in DETAILS.md.
"""

from skiresort_planner.core.dem_service import DEMService
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.core.terrain_analyzer import (
    SideSlope,
    TerrainAnalyzer,
    TerrainGradient,
    TerrainOrientation,
)

# PathTracer and TracedPath have circular import with model.path_point
# Import directly: from skiresort_planner.core.path_tracer import PathTracer

__all__ = [
    # Geo calculator
    "GeoCalculator",
    # DEM service
    "DEMService",
    # Terrain analyzer
    "TerrainAnalyzer",
    "TerrainGradient",
    "TerrainOrientation",
    "SideSlope",
    # Path tracer
    "PathTracer",
    "TracedPath",
]
