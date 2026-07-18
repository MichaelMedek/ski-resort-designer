"""Unit tests for validator functions (ui/validators.py).

Validators return None when valid or a typed Message carrying the offending
values. Tests assert BOTH the message type AND its fields, and pin the exact
MIN_DROP_M / SEGMENT_LENGTH_MAX_M boundaries (imported, not hardcoded).
"""

from skiresort_planner.constants import ConnectionConfig, PathConfig
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.model.message import (
    SameNodeLiftMessage,
    TargetNotDownhillMessage,
    TargetTooFarMessage,
)
from skiresort_planner.ui.validators import (
    validate_custom_target_distance,
    validate_custom_target_downhill,
    validate_lift_stations_differ,
)

M = 111320.0  # metres per degree near the equator


class TestLiftStationsDiffer:
    def test_distinct_points_valid(self) -> None:
        # Orientation is decided by elevation later, so distinct coordinates always pass here —
        # even a downhill or equal-elevation second point (that just flips which end is bottom).
        assert validate_lift_stations_differ(first_lon=0.0, first_lat=0.0, second_lon=0.0, second_lat=1.0) is None

    def test_coincident_points_rejected(self) -> None:
        result = validate_lift_stations_differ(first_lon=5.0, first_lat=6.0, second_lon=5.0, second_lat=6.0)
        assert isinstance(result, SameNodeLiftMessage)


class TestCustomTargetDownhill:
    def test_sufficient_drop_valid(self) -> None:
        assert validate_custom_target_downhill(start_elevation=2500.0, target_elevation=2400.0, may_climb=False) is None

    def test_at_min_drop_boundary_is_valid(self) -> None:
        # Exactly MIN_DROP_M of drop is allowed (the reject is drop < MIN_DROP_M).
        start, target = 2500.0, 2500.0 - ConnectionConfig.MIN_DROP_M
        assert validate_custom_target_downhill(start_elevation=start, target_elevation=target, may_climb=False) is None

    def test_just_below_min_drop_is_rejected_with_fields(self) -> None:
        start = 2500.0
        target = 2500.0 - (ConnectionConfig.MIN_DROP_M - 0.5)  # one notch short of the boundary
        result = validate_custom_target_downhill(start_elevation=start, target_elevation=target, may_climb=False)
        assert isinstance(result, TargetNotDownhillMessage)
        assert result.start_elevation_m == start
        assert result.target_elevation_m == target

    def test_uphill_is_rejected(self) -> None:
        assert isinstance(
            validate_custom_target_downhill(start_elevation=2400.0, target_elevation=2500.0, may_climb=False),
            TargetNotDownhillMessage,
        )

    def test_may_climb_skips_the_check(self) -> None:
        # Roads (may_climb=True) route uphill freely — an uphill target must NOT be rejected.
        assert validate_custom_target_downhill(start_elevation=2400.0, target_elevation=2500.0, may_climb=True) is None

    def test_uphill_message_explains_target_is_above(self) -> None:
        # Target 100m ABOVE start: the message must explain it's above the current
        # point, NOT just print a confusing negative drop with no explanation.
        msg = TargetNotDownhillMessage(
            start_elevation_m=2400.0, target_elevation_m=2500.0, min_drop_m=ConnectionConfig.MIN_DROP_M
        ).message
        assert "100m above" in msg

    def test_too_little_drop_message_keeps_need_wording(self) -> None:
        # Target below start but not enough drop: keep the "need at least Xm" wording,
        # with no "above" explainer.
        msg = TargetNotDownhillMessage(
            start_elevation_m=2500.0, target_elevation_m=2498.0, min_drop_m=ConnectionConfig.MIN_DROP_M
        ).message
        assert "above" not in msg
        assert f"{ConnectionConfig.MIN_DROP_M:.0f}m" in msg


class TestCustomTargetDistance:
    def test_within_range_valid(self) -> None:
        # ~500m south of start.
        assert (
            validate_custom_target_distance(start_lat=0.0, start_lon=0.0, target_lat=-500 / M, target_lon=0.0) is None
        )

    def test_at_max_distance_boundary_is_valid(self) -> None:
        # Exactly SEGMENT_LENGTH_MAX_M is allowed (reject is distance > max).
        at_max = PathConfig.SEGMENT_LENGTH_MAX_M / M
        assert validate_custom_target_distance(start_lat=0.0, start_lon=0.0, target_lat=-at_max, target_lon=0.0) is None

    def test_beyond_max_is_rejected_with_fields(self) -> None:
        # Well beyond the cap.
        far_lat = -(PathConfig.SEGMENT_LENGTH_MAX_M + 500) / M
        result = validate_custom_target_distance(start_lat=0.0, start_lon=0.0, target_lat=far_lat, target_lon=0.0)
        assert isinstance(result, TargetTooFarMessage)
        assert result.max_distance_m == PathConfig.SEGMENT_LENGTH_MAX_M
        # The reported distance matches the real great-circle distance.
        expected = GeoCalculator.haversine_distance_m(lat1=0.0, lon1=0.0, lat2=far_lat, lon2=0.0)
        assert result.distance_m == expected
