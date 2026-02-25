"""PathPoint - The fundamental geometry atom for ski resort planning.

A PathPoint represents a single GPS coordinate with elevation.
It is the single source of truth for location throughout the system.

Used by:
- Node (wraps a PathPoint for its location)
- SlopeSegment (contains list of PathPoints for geometry)
- ProposedSlopeSegment (path proposals before committing)

Reference: DETAILS.md
"""

from dataclasses import dataclass

import numpy as np

from skiresort_planner.core.geo_calculator import GeoCalculator


@dataclass
class PathPoint:
    """A point on a path with GPS coordinates and elevation.

    The geometry atom - represents a single location in 3D space.
    Used throughout the system for all location data.

    Attributes:
        lon: Longitude in decimal degrees (WGS84)
        lat: Latitude in decimal degrees (WGS84)
        elevation: Elevation in meters above sea level

    Example:
        point = PathPoint(lon=10.295, lat=46.985, elevation=2400.0)
    """

    lon: float
    lat: float
    elevation: float

    @property
    def lat_lon(self) -> tuple[float, float]:
        """Return (lat, lon) tuple - standard geographic order."""
        return (self.lat, self.lon)

    @property
    def lon_lat(self) -> tuple[float, float]:
        """Return (lon, lat) tuple - GeoJSON/Pydeck order."""
        return (self.lon, self.lat)

    def __post_init__(self) -> None:
        """Validate data after initialization."""
        if np.isnan(self.elevation):
            raise ValueError(f"PathPoint cannot have NaN elevation at ({self.lon}, {self.lat})")

    def distance_to(self, other: "PathPoint") -> float:
        """Calculate haversine distance to another point in meters.

        Args:
            other: Another PathPoint to measure distance to

        Returns:
            Distance in meters using great-circle calculation.
        """
        return GeoCalculator.haversine_distance_m(
            lat1=self.lat,
            lon1=self.lon,
            lat2=other.lat,
            lon2=other.lon,
        )

    def __repr__(self) -> str:
        return f"PathPoint(lon={self.lon:.5f}, lat={self.lat:.5f}, elev={self.elevation:.1f}m)"
