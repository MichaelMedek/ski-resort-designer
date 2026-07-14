"""Unit tests for validator functions (ui/validators.py).

Validators return None when valid or a typed Message carrying the offending
values. Tests assert BOTH the message type AND its fields, and pin the exact
MIN_DROP_M / SEGMENT_LENGTH_MAX_M boundaries (imported, not hardcoded).
"""

from skiresort_planner.constants import ConnectionConfig, PathConfig
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.model.message import (
    LiftMustGoUphillMessage,
    SameNodeLiftMessage,
    TargetNotDownhillMessage,
    TargetTooFarMessage,
)
from skiresort_planner.ui.validators import (
    validate_custom_target_distance,
    validate_custom_target_downhill,
    validate_lift_different_nodes,
    validate_lift_goes_uphill,
)

M = 111320.0  # metres per degree near the equator


class TestLiftGoesUphill:
    def test_uphill_is_valid(self) -> None:
        assert validate_lift_goes_uphill(start_elevation=1500.0, end_elevation=2000.0) is None

    def test_downhill_reports_both_elevations(self) -> None:
        result = validate_lift_goes_uphill(start_elevation=2000.0, end_elevation=1500.0)
        assert isinstance(result, LiftMustGoUphillMessage)
        assert result.start_elevation_m == 2000.0
        assert result.end_elevation_m == 1500.0

    def test_equal_elevation_is_rejected(self) -> None:
        # Boundary: end == start is NOT uphill.
        assert isinstance(
            validate_lift_goes_uphill(start_elevation=2000.0, end_elevation=2000.0), LiftMustGoUphillMessage
        )


class TestLiftDifferentNodes:
    def test_different_nodes_valid(self) -> None:
        assert validate_lift_different_nodes(start_node_id="N1", end_node_id="N2") is None

    def test_same_node_rejected(self) -> None:
        assert isinstance(validate_lift_different_nodes(start_node_id="N1", end_node_id="N1"), SameNodeLiftMessage)


class TestCustomTargetDownhill:
    def test_sufficient_drop_valid(self) -> None:
        assert validate_custom_target_downhill(start_elevation=2500.0, target_elevation=2400.0) is None

    def test_at_min_drop_boundary_is_valid(self) -> None:
        # Exactly MIN_DROP_M of drop is allowed (the reject is drop < MIN_DROP_M).
        start, target = 2500.0, 2500.0 - ConnectionConfig.MIN_DROP_M
        assert validate_custom_target_downhill(start_elevation=start, target_elevation=target) is None

    def test_just_below_min_drop_is_rejected_with_fields(self) -> None:
        start = 2500.0
        target = 2500.0 - (ConnectionConfig.MIN_DROP_M - 0.5)  # one notch short of the boundary
        result = validate_custom_target_downhill(start_elevation=start, target_elevation=target)
        assert isinstance(result, TargetNotDownhillMessage)
        assert result.start_elevation_m == start
        assert result.target_elevation_m == target

    def test_uphill_is_rejected(self) -> None:
        assert isinstance(
            validate_custom_target_downhill(start_elevation=2400.0, target_elevation=2500.0), TargetNotDownhillMessage
        )

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
