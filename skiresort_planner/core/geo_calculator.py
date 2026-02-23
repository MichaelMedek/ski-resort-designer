"""Geodesic calculations on Earth's surface.

Provides geographic helper functions for ski resort planning:
- Distance calculation (Haversine formula)
- Bearing calculation (initial heading between points)
- Destination calculation (endpoint from start, bearing, distance)
- Bearing interpolation and averaging (circular math)

All calculations use WGS84 spherical Earth approximation (R = 6,371 km).

Reference: DETAILS.md Section 1
"""

from math import asin, atan2, cos, degrees, radians, sin, sqrt

# Earth's radius in meters (WGS84 spherical approximation)
EARTH_RADIUS_M = 6_371_000


class GeoCalculator:
    """Static methods for geodesic calculations on Earth's surface.

    All methods use WGS84 spherical Earth model (R = 6,371 km).
    Coordinates are in decimal degrees (WGS84).
    Bearings are in degrees clockwise from North (0-360).
    Distances are in meters.
    """

    EARTH_RADIUS_M = EARTH_RADIUS_M

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
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        return EARTH_RADIUS_M * 2 * atan2(sqrt(a), sqrt(1 - a))

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
        brng = radians(bearing_deg)
        lat1 = radians(lat)
        lon1 = radians(lon)
        d_R = distance_m / EARTH_RADIUS_M

        lat2 = asin(sin(lat1) * cos(d_R) + cos(lat1) * sin(d_R) * cos(brng))
        lon2 = lon1 + atan2(
            sin(brng) * sin(d_R) * cos(lat1),
            cos(d_R) - sin(lat1) * sin(lat2),
        )
        return degrees(lon2), degrees(lat2)

    @staticmethod
    def lerp_bearing(
        bearing_a: float,
        bearing_b: float,
        alpha: float = 0.5,
    ) -> float:
        """Circular linear interpolation between two bearings.

        Handles the wraparound at 0°/360° correctly, always taking the shorter arc.
        Uses vector averaging weighted by alpha.

        Special case: alpha=0.5 gives the circular mean of two bearings.

        Args:
            bearing_a: Starting bearing in degrees (0-360)
            bearing_b: Target bearing in degrees (0-360)
            alpha: Interpolation weight (0 = all bearing_a, 1 = all bearing_b, 0.5 = mean)

        Returns:
            Interpolated bearing in degrees (0-360).
        """
        alpha = max(0.0, min(1.0, alpha))
        rad_a = radians(bearing_a)
        rad_b = radians(bearing_b)
        x = (1 - alpha) * sin(rad_a) + alpha * sin(rad_b)
        y = (1 - alpha) * cos(rad_a) + alpha * cos(rad_b)
        return (degrees(atan2(x, y)) + 360) % 360
