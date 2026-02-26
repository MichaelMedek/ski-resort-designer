"""Unit tests for validator functions.

Tests all validation functions with valid and invalid inputs.
"""

from skiresort_planner.model.message import (
    LiftMustGoUphillMessage,
    SameNodeLiftMessage,
    TargetNotDownhillMessage,
    TargetTooFarMessage,
)
from skiresort_planner.model.node import Node
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.ui.validators import (
    validate_custom_target_distance,
    validate_custom_target_downhill,
    validate_lift_different_nodes,
    validate_lift_goes_uphill,
)


class TestLiftValidators:
    """Tests for lift placement validators."""

    def test_lift_must_go_uphill_valid(self) -> None:
        """validate_lift_goes_uphill returns None when end is higher."""
        start = Node(id="N1", location=PathPoint(lon=10.0, lat=46.0, elevation=1500.0))
        end = Node(id="N2", location=PathPoint(lon=10.0, lat=46.01, elevation=2000.0))

        result = validate_lift_goes_uphill(start_node=start, end_node=end)

        assert result is None, "Valid lift should return None"

    def test_lift_must_go_uphill_invalid_downhill(self) -> None:
        """validate_lift_goes_uphill returns error when end is lower."""
        start = Node(id="N1", location=PathPoint(lon=10.0, lat=46.0, elevation=2000.0))
        end = Node(id="N2", location=PathPoint(lon=10.0, lat=46.01, elevation=1500.0))

        result = validate_lift_goes_uphill(start_node=start, end_node=end)

        assert isinstance(result, LiftMustGoUphillMessage), "Should return error message"
        assert result.start_elevation_m == 2000.0
        assert result.end_elevation_m == 1500.0

    def test_lift_must_go_uphill_invalid_same_elevation(self) -> None:
        """validate_lift_goes_uphill returns error when elevations are equal."""
        start = Node(id="N1", location=PathPoint(lon=10.0, lat=46.0, elevation=2000.0))
        end = Node(id="N2", location=PathPoint(lon=10.0, lat=46.01, elevation=2000.0))

        result = validate_lift_goes_uphill(start_node=start, end_node=end)

        assert result is not None, "Equal elevation should be invalid"

    def test_lift_different_nodes_valid(self) -> None:
        """validate_lift_different_nodes returns None for different nodes."""
        result = validate_lift_different_nodes(start_node_id="N1", end_node_id="N2")

        assert result is None, "Different nodes should be valid"

    def test_lift_different_nodes_invalid_same(self) -> None:
        """validate_lift_different_nodes returns error for same node."""
        result = validate_lift_different_nodes(start_node_id="N1", end_node_id="N1")

        assert isinstance(result, SameNodeLiftMessage), "Same node should return error"


class TestCustomTargetValidators:
    """Tests for custom target connection validators."""

    def test_custom_target_downhill_valid(self) -> None:
        """validate_custom_target_downhill returns None when drop is sufficient."""
        result = validate_custom_target_downhill(
            start_elevation=2500.0,
            target_elevation=2400.0,  # 100m drop
        )

        assert result is None, "Sufficient drop should be valid"

    def test_custom_target_downhill_invalid_uphill(self) -> None:
        """validate_custom_target_downhill returns error when target is uphill."""
        result = validate_custom_target_downhill(
            start_elevation=2400.0,
            target_elevation=2500.0,  # Going uphill
        )

        assert isinstance(result, TargetNotDownhillMessage), "Uphill target should be invalid"

    def test_custom_target_downhill_invalid_too_small_drop(self) -> None:
        """validate_custom_target_downhill returns error when drop is too small."""
        result = validate_custom_target_downhill(
            start_elevation=2500.0,
            target_elevation=2496.0,  # Only 4m drop, less than MIN_DROP_M (5m)
        )

        assert isinstance(result, TargetNotDownhillMessage), "Small drop should be invalid"

    def test_custom_target_distance_valid(self) -> None:
        """validate_custom_target_distance returns None when distance is acceptable."""
        result = validate_custom_target_distance(
            start_lat=46.0,
            start_lon=10.0,
            target_lat=46.005,  # ~500m away
            target_lon=10.0,
        )

        assert result is None, "Acceptable distance should be valid"

    def test_custom_target_distance_invalid_too_far(self) -> None:
        """validate_custom_target_distance returns error when target is too far."""
        result = validate_custom_target_distance(
            start_lat=46.0,
            start_lon=10.0,
            target_lat=46.1,  # ~11km away
            target_lon=10.1,
        )

        assert isinstance(result, TargetTooFarMessage), "Far target should be invalid"
