"""Geodesic calculations on Earth's surface.

Provides geographic helper functions for ski resort planning:
- Distance calculation (Haversine formula)
- Bearing calculation (initial heading between points)
- Destination calculation (endpoint from start, bearing, distance)
- Bearing interpolation and averaging (circular math)

All calculations use WGS84 spherical Earth approximation (R = 6,371 km).

Reference: DETAILS.md Section 1
"""

from math import atan2, cos, degrees, radians, sin

import numpy as np
import numpy.typing as npt

from skiresort_planner.constants import MapConfig


class GeoCalculator:
    """Static methods for geodesic calculations on Earth's surface.

    All methods use WGS84 spherical Earth model (R = 6,371 km).
    Coordinates are in decimal degrees (WGS84).
    Bearings are in degrees clockwise from North (0-360).
    Distances are in meters.
    """

    @staticmethod
    def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great-circle distance between two points using Haversine formula.

        Args:
            lat1: Latitude of first point (decimal degrees)
            lon1: Longitude of first point (decimal degrees)
            lat2: Latitude of second point (decimal degrees)
            lon2: Longitude of second point (decimal degrees)

        Returns:
            Distance in meters.
        """
        return float(GeoCalculator.haversine_vec(lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2))

    @staticmethod
    def initial_bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """Calculate initial bearing from point 1 to point 2.

        The bearing is the compass direction to travel from start to end,
        measured clockwise from true North.

        Args:
            lon1: Longitude of start point (decimal degrees)
            lat1: Latitude of start point (decimal degrees)
            lon2: Longitude of end point (decimal degrees)
            lat2: Latitude of end point (decimal degrees)

        Returns:
            Bearing in degrees (0-360, clockwise from North).
        """
        lon1_rad, lat1_rad = radians(lon1), radians(lat1)
        lon2_rad, lat2_rad = radians(lon2), radians(lat2)
        dlon = lon2_rad - lon1_rad
        y = sin(dlon) * cos(lat2_rad)
        x = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(dlon)
        return (degrees(atan2(y, x)) + 360) % 360

    @staticmethod
    def normalize_bearing_diff(diff_deg: float) -> float:
        """Wrap a bearing difference into (-180, 180] so ±350° reads as the short ∓10° turn."""
        return (diff_deg + 180) % 360 - 180

    @staticmethod
    def meters_per_degree(lat: float) -> tuple[float, float]:
        """Local flat-earth scale at `lat`: (metres per degree longitude, per degree latitude).

        Longitude degrees shrink by cos(lat); latitude degrees are constant. The single source for
        the lon/lat→metre projection used by every local-frame builder (no per-site cos(lat) copies).

        Returns:
            (m_per_deg_lon, m_per_deg_lat).
        """
        m_per_deg_lat = MapConfig.METERS_PER_DEGREE_EQUATOR
        return m_per_deg_lat * cos(radians(lat)), m_per_deg_lat

    @staticmethod
    def destination(
        lon: float,
        lat: float,
        bearing_deg: float,
        distance_m: float,
    ) -> tuple[float, float]:
        """Calculate destination point given start, bearing, and distance.

        Uses the formula for finding a point at given distance and bearing
        from a starting point on a sphere.

        Args:
            lon: Longitude of start point (decimal degrees)
            lat: Latitude of start point (decimal degrees)
            bearing_deg: Bearing in degrees (clockwise from North)
            distance_m: Distance to travel in meters

        Returns:
            Tuple (lon, lat) of destination point in decimal degrees.
        """
        lon2, lat2 = GeoCalculator.destination_vec(lon=lon, lat=lat, bearing_deg=bearing_deg, distance_m=distance_m)
        return float(lon2), float(lat2)

    @staticmethod
    def haversine_vec(
        lat1: "npt.NDArray[np.float64] | float",
        lon1: "npt.NDArray[np.float64] | float",
        lat2: "npt.NDArray[np.float64] | float",
        lon2: "npt.NDArray[np.float64] | float",
    ) -> "npt.NDArray[np.float64]":
        """Vectorized `haversine_distance_m` — numpy mirror, same formula/operand order, elementwise
        over broadcastable arrays/scalars. Returns great-circle distances in metres.
        """
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
        return MapConfig.EARTH_RADIUS_M * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))  # type: ignore[no-any-return]

    @staticmethod
    def destination_vec(
        lon: "npt.NDArray[np.float64] | float",
        lat: "npt.NDArray[np.float64] | float",
        bearing_deg: "npt.NDArray[np.float64] | float",
        distance_m: "npt.NDArray[np.float64] | float",
    ) -> "tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]":
        """Vectorized `destination` — numpy mirror, same formula/operand order, broadcasting over
        array start-coords and/or distances. Returns (lon, lat) arrays in decimal degrees.
        """
        brng = np.radians(bearing_deg)
        lat1 = np.radians(lat)
        lon1 = np.radians(lon)
        d_R = distance_m / MapConfig.EARTH_RADIUS_M
        lat2 = np.arcsin(np.sin(lat1) * np.cos(d_R) + np.cos(lat1) * np.sin(d_R) * np.cos(brng))
        lon2 = lon1 + np.arctan2(
            np.sin(brng) * np.sin(d_R) * np.cos(lat1),
            np.cos(d_R) - np.sin(lat1) * np.sin(lat2),
        )
        return np.degrees(lon2), np.degrees(lat2)
