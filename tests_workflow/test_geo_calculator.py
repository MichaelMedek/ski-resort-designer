"""Unit tests for GeoCalculator geodesic functions (core/geo_calculator.py).

Pure deterministic functions — distances, bearings, destinations on WGS84.
"""

import math

import pytest

from skiresort_planner.constants import MapConfig
from skiresort_planner.core.geo_calculator import GeoCalculator


class TestGeoCalculator:
    """Tests for geodesic calculations."""

    def test_haversine_and_bearing_cardinal_directions(self) -> None:
        """Haversine distance and bearing for cardinal directions.

        Tests:
        - 1° latitude = ~111km
        - 1° longitude at 46°N = ~77km
        - Bearing north = 0°, south = 180°, east = 90°
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
        """destination() should be consistent in distance AND direction."""
        lon_end, lat_end = GeoCalculator.destination(lon=10.0, lat=46.0, bearing_deg=45.0, distance_m=1000.0)
        dist_check = GeoCalculator.haversine_distance_m(lat1=46.0, lon1=10.0, lat2=lat_end, lon2=lon_end)
        assert abs(dist_check - 1000) < 10, "Roundtrip should be within 10m tolerance"

        # Bearing back to the endpoint must equal the requested 45° (direction preserved)
        bearing_back = GeoCalculator.initial_bearing_deg(lon1=10.0, lat1=46.0, lon2=lon_end, lat2=lat_end)
        assert abs(bearing_back - 45.0) < 1.0, "Bearing to endpoint should be ~45°"

        # 45° is NE: endpoint must move both north and east (catches lat/lon sign swaps)
        assert lat_end > 46.0, "NE bearing should increase latitude"
        assert lon_end > 10.0, "NE bearing should increase longitude"

    def test_haversine_equator_degree_is_exact_closed_form(self) -> None:
        """One degree along the equator equals R·π/180 to full float precision.

        This pins the spherical-Earth constant, not a ~111km ballpark: a wrong radius
        or a degrees/radians slip would move this number, and the ±band tests wouldn't
        catch a few-percent error. Both a lat step and a lon step at the equator span
        the same great-circle degree, so both must equal the closed form.
        """
        expected = MapConfig.EARTH_RADIUS_M * math.pi / 180.0  # 111194.9266… m
        along_lat = GeoCalculator.haversine_distance_m(lat1=0.0, lon1=0.0, lat2=1.0, lon2=0.0)
        along_lon = GeoCalculator.haversine_distance_m(lat1=0.0, lon1=0.0, lat2=0.0, lon2=1.0)
        assert along_lat == pytest.approx(expected, rel=1e-9), "1° of latitude at the equator = R·π/180"
        assert along_lon == pytest.approx(expected, rel=1e-9), "1° of longitude at the equator = R·π/180"

    def test_meters_per_degree_constant_is_a_sane_nominal_degree(self) -> None:
        """MapConfig.METERS_PER_DEGREE_EQUATOR is the single lat/lon↔metre constant used codebase-wide.

        It's a round NOMINAL value (111320 m), not the exact spherical degree — tests build synthetic
        geometry from it, so it only needs to be a realistic ~111 km/°. Guard it stays within 0.2% of
        the true geodesic degree so a typo can't silently distort every test's coordinates.
        """
        from skiresort_planner.constants import MapConfig

        one_degree_m = GeoCalculator.haversine_distance_m(lat1=0.0, lon1=0.0, lat2=1.0, lon2=0.0)
        assert pytest.approx(one_degree_m, rel=2e-3) == MapConfig.METERS_PER_DEGREE_EQUATOR, (
            "the nominal metres-per-degree constant must stay within 0.2% of one geodesic degree"
        )

    def test_meters_per_degree_scales_longitude_by_cos_lat(self) -> None:
        """meters_per_degree(lat) → (m_lon, m_lat): the ONE local-frame projection helper.

        Latitude degrees are constant (the equator nominal); longitude degrees shrink by cos(lat).
        At the equator lon == lat; at 60°N a lon degree is half a lat degree.
        """
        from skiresort_planner.constants import MapConfig

        eq_lon, eq_lat = GeoCalculator.meters_per_degree(lat=0.0)
        assert eq_lon == pytest.approx(MapConfig.METERS_PER_DEGREE_EQUATOR)
        assert eq_lat == MapConfig.METERS_PER_DEGREE_EQUATOR

        lon60, lat60 = GeoCalculator.meters_per_degree(lat=60.0)
        assert lat60 == MapConfig.METERS_PER_DEGREE_EQUATOR, "latitude scale is lat-independent"
        assert lon60 == pytest.approx(MapConfig.METERS_PER_DEGREE_EQUATOR * 0.5, rel=1e-9), "cos(60°)=0.5"

    def test_haversine_zero_distance_and_symmetry(self) -> None:
        """Distance to self is exactly 0, and haversine(A,B) == haversine(B,A)."""
        assert GeoCalculator.haversine_distance_m(lat1=46.0, lon1=10.0, lat2=46.0, lon2=10.0) == 0.0
        ab = GeoCalculator.haversine_distance_m(lat1=10.0, lon1=20.0, lat2=30.0, lon2=40.0)
        ba = GeoCalculator.haversine_distance_m(lat1=30.0, lon1=40.0, lat2=10.0, lon2=20.0)
        assert ab == pytest.approx(ba, rel=1e-12), "great-circle distance is symmetric"

    def test_destination_north_by_one_degree_arc_lands_on_exact_latitude(self) -> None:
        """Walking due north from the equator by exactly one degree of arc lands at lat 1.0°.

        distance = R·π/180 (the length of 1° of arc), bearing 0°, so the destination
        must be (lon 0°, lat 1.0°) with the longitude unchanged. A closed-form anchor for
        destination(), independent of the round-trip test's tolerance band.
        """
        one_degree_arc_m = MapConfig.EARTH_RADIUS_M * math.pi / 180.0
        lon_end, lat_end = GeoCalculator.destination(lon=0.0, lat=0.0, bearing_deg=0.0, distance_m=one_degree_arc_m)
        assert lon_end == pytest.approx(0.0, abs=1e-9), "due-north travel keeps longitude fixed"
        assert lat_end == pytest.approx(1.0, rel=1e-9), "one degree of arc north lands at latitude 1.0°"

    def test_destination_zero_distance_returns_start(self) -> None:
        """Zero-distance travel returns the start point unchanged, for any bearing."""
        lon_end, lat_end = GeoCalculator.destination(lon=10.0, lat=46.0, bearing_deg=137.0, distance_m=0.0)
        assert lon_end == pytest.approx(10.0, abs=1e-12)
        assert lat_end == pytest.approx(46.0, abs=1e-12)
