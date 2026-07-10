"""Unit tests for GeoCalculator geodesic functions (core/geo_calculator.py).

Pure deterministic functions — distances, bearings, destinations on WGS84.
"""

from skiresort_planner.core.geo_calculator import GeoCalculator


class TestGeoCalculator:
    """Tests for geodesic calculations."""

    def test_haversine_and_bearing_cardinal_directions(self) -> None:
        """Haversine distance and bearing for cardinal directions.

        Tests:
        - 1° latitude = ~111km
        - 1° longitude at 46°N = ~77km
        - Bearing north = 0°, south = 180°, east = 90°
        - destination() roundtrip consistent with haversine()
        """
        # 1 degree latitude should be ~111km
        dist_lat = GeoCalculator.haversine_distance_m(lat1=46.0, lon1=10.0, lat2=47.0, lon2=10.0)
        assert 110_000 < dist_lat < 112_000, "1° lat should be ~111km"

        # 1 degree longitude at 46°N should be ~77km (cos(46°) factor)
        dist_lon = GeoCalculator.haversine_distance_m(lat1=46.0, lon1=10.0, lat2=46.0, lon2=11.0)
        assert 76_000 < dist_lon < 78_000, "1° lon at 46°N should be ~77km"

        # North bearing should be ~0°
        bearing_north = GeoCalculator.initial_bearing_deg(lon1=10.0, lat1=46.0, lon2=10.0, lat2=47.0)
        assert bearing_north < 1 or bearing_north > 359, "North should be ~0°"

        # South bearing should be ~180°
        bearing_south = GeoCalculator.initial_bearing_deg(lon1=10.0, lat1=47.0, lon2=10.0, lat2=46.0)
        assert 179 < bearing_south < 181, "South should be ~180°"

        # East bearing should be ~90°
        bearing_east = GeoCalculator.initial_bearing_deg(lon1=10.0, lat1=46.0, lon2=11.0, lat2=46.0)
        assert 89 < bearing_east < 91, "East should be ~90°"

    def test_destination_roundtrip_consistency(self) -> None:
        """destination() should be consistent with haversine() distance."""
        lon_end, lat_end = GeoCalculator.destination(lon=10.0, lat=46.0, bearing_deg=45.0, distance_m=1000.0)
        dist_check = GeoCalculator.haversine_distance_m(lat1=46.0, lon1=10.0, lat2=lat_end, lon2=lon_end)
        assert abs(dist_check - 1000) < 10, "Roundtrip should be within 10m tolerance"
