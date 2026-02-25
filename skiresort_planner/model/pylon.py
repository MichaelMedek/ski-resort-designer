"""Pylon - Support structure for ski lifts.

A Pylon represents a single support tower along a lift line.
Positions are calculated based on catenary simulation considering
terrain clearance and maximum span constraints.

Reference: DETAILS.md
"""

from dataclasses import dataclass


@dataclass
class Pylon:
    """A support pylon for a ski lift.

    Attributes:
        index: Position index along the sampled terrain points
        distance_m: Distance from lift start in meters
        lat: Latitude of pylon base
        lon: Longitude of pylon base
        ground_elevation_m: Terrain elevation at pylon base
        height_m: Pylon height from ground to top
    """

    index: int
    distance_m: float
    lat: float
    lon: float
    ground_elevation_m: float
    height_m: float

    @property
    def top_elevation_m(self) -> float:
        """Elevation of pylon top (cable attachment point)."""
        return self.ground_elevation_m + self.height_m

    @property
    def lat_lon(self) -> tuple[float, float]:
        """Return (lat, lon) tuple - standard geographic order."""
        return (self.lat, self.lon)

    @property
    def lon_lat(self) -> tuple[float, float]:
        """Return (lon, lat) tuple - GeoJSON/Pydeck order."""
        return (self.lon, self.lat)

    def __repr__(self) -> str:
        return f"Pylon(idx={self.index}, {self.distance_m:.0f}m, {self.height_m:.0f}m tall)"
