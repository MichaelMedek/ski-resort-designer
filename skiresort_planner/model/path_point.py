"""PathPoint - The fundamental geometry atom for ski resort planning.

A PathPoint represents a single GPS coordinate with elevation.
It is the single source of truth for location throughout the system.

Used by:
- Node (wraps a PathPoint for its location)
- PathSegment (contains list of PathPoints for geometry)
- ProposedPathSegment (path proposals before committing)

Reference: DETAILS.md
"""

import bisect
from collections.abc import Sequence
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

    @property
    def lon_lat_elev(self) -> tuple[float, float, float]:
        """Return (lon, lat, elevation) — the drawable 3-tuple polylines/routes are built from."""
        return (self.lon, self.lat, self.elevation)

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

    @staticmethod
    def cumulative_distances(points: Sequence["PathPoint"]) -> list[float]:
        """Cumulative along-path distance (m) at each point, starting at 0.

        Single source for polyline arc length (was duplicated in path_smoothing + bottom_chart).
        """
        cum = [0.0]
        for i in range(1, len(points)):
            cum.append(cum[-1] + points[i - 1].distance_to(other=points[i]))
        return cum

    @staticmethod
    def total_length_m(points: Sequence["PathPoint"]) -> float:
        """Total polyline length (m) — 0 for fewer than two points. Single source for path length."""
        return PathPoint.cumulative_distances(points)[-1] if len(points) >= 2 else 0.0

    @staticmethod
    def interpolate_at_distance(
        points: Sequence["PathPoint"], distances: Sequence[float], target_m: float
    ) -> "PathPoint":
        """Interpolate a PathPoint at arc distance `target_m` along the polyline. Single source for
        distance-based lookup (bracket by cumulative distance + lerp), correct for NON-uniform spacing.

        distances: cumulative_distances(points), passed in so callers reuse it. Clamps to the endpoints
        outside [0, total].
        """
        idx_high = bisect.bisect_left(distances, target_m)
        if idx_high <= 0:
            return points[0]
        if idx_high >= len(points):
            return points[-1]
        idx_low = idx_high - 1
        seg = distances[idx_high] - distances[idx_low]
        frac = (target_m - distances[idx_low]) / seg if seg > 0 else 0.0
        low, high = points[idx_low], points[idx_high]
        return PathPoint(
            lon=low.lon + (high.lon - low.lon) * frac,
            lat=low.lat + (high.lat - low.lat) * frac,
            elevation=low.elevation + (high.elevation - low.elevation) * frac,
        )

    @staticmethod
    def interpolate_at_fraction(points: Sequence["PathPoint"], fraction: float) -> "PathPoint":
        """The PathPoint at normalized arc-length `fraction` (0..1) along `points`. Clamps to [0, 1].

        Thin wrapper over interpolate_at_distance (the ONE interpolation kernel) — fraction × total length
        → absolute distance. A polyline's arc-length interpolation IS piecewise-linear between vertices,
        so this matches the old Shapely LineString.interpolate without building a geometry per call.
        """
        assert len(points) >= 2, f"interpolate_at_fraction needs >=2 points, got {len(points)}"
        distances = PathPoint.cumulative_distances(points)
        target_m = max(0.0, min(1.0, fraction)) * distances[-1]
        return PathPoint.interpolate_at_distance(points=points, distances=distances, target_m=target_m)

    def __repr__(self) -> str:
        return f"PathPoint(lon={self.lon:.5f}, lat={self.lat:.5f}, elev={self.elevation:.1f}m)"


def endpoints_match(
    pair_a: tuple["PathPoint", "PathPoint"],
    pair_b: tuple["PathPoint", "PathPoint"],
    tol_m: float,
) -> bool:
    """True if the two entities span the same pair of endpoints (each within tol_m, either order).

    A direct geometric comparison of two endpoint pairs — no rounding, no shared node registry, so
    it stays correct even where many runs cluster around one junction. Used to detect that a run
    (from OSM re-import or built by hand) already exists: import snaps endpoints onto junction
    nodes, so tol_m is the snap distance (STEP_SIZE_M) to absorb that shift.
    """
    (a1, a2), (b1, b2) = pair_a, pair_b
    return (a1.distance_to(other=b1) <= tol_m and a2.distance_to(other=b2) <= tol_m) or (
        a1.distance_to(other=b2) <= tol_m and a2.distance_to(other=b1) <= tol_m
    )
