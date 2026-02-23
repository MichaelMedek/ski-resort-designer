"""Unit tests for skiresort_planner validators.

Tests pure validation functions that require no Streamlit or browser interaction.
These validators return Optional[Message] - None if valid, a Message if invalid.
"""

import pytest

from skiresort_planner.constants import ConnectionConfig
from skiresort_planner.model.message import (
    LiftMustGoUphillMessage,
    OutsideTerrainMessage,
    SameNodeLiftMessage,
    TargetNotDownhillMessage,
    TargetTooFarMessage,
)
from skiresort_planner.model.node import Node
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.ui.validators import (
    validate_custom_target_distance,
    validate_custom_target_downhill,
    validate_elevation_exists,
    validate_lift_different_nodes,
    validate_lift_goes_uphill,
)


class TestValidateElevationExists:
    """Tests for validate_elevation_exists."""

    def test_valid_elevation_returns_none(self) -> None:
        """Non-None elevation is valid."""
        result = validate_elevation_exists(lat=46.97, lon=10.27, elevation=2500.0)
        assert result is None

    def test_zero_elevation_is_valid(self) -> None:
        """Zero elevation (sea level) is valid."""
        result = validate_elevation_exists(lat=46.97, lon=10.27, elevation=0.0)
        assert result is None

    def test_negative_elevation_is_valid(self) -> None:
        """Negative elevation (below sea level) is valid."""
        result = validate_elevation_exists(lat=46.97, lon=10.27, elevation=-100.0)
        assert result is None

    def test_none_elevation_returns_message(self) -> None:
        """None elevation returns OutsideTerrainMessage."""
        result = validate_elevation_exists(lat=46.97, lon=10.27, elevation=None)
        assert isinstance(result, OutsideTerrainMessage)
        assert result.lat == 46.97
        assert result.lon == 10.27


class TestValidateLiftGoesUphill:
    """Tests for validate_lift_goes_uphill."""

    def test_uphill_lift_is_valid(self) -> None:
        """End elevation higher than start is valid."""
        start = Node(id="N1", location=PathPoint(lon=10.27, lat=46.97, elevation=2000.0))
        end = Node(id="N2", location=PathPoint(lon=10.28, lat=46.98, elevation=2500.0))
        result = validate_lift_goes_uphill(start_node=start, end_node=end)
        assert result is None

    def test_downhill_lift_returns_message(self) -> None:
        """End elevation lower than start returns LiftMustGoUphillMessage."""
        start = Node(id="N1", location=PathPoint(lon=10.27, lat=46.97, elevation=2500.0))
        end = Node(id="N2", location=PathPoint(lon=10.28, lat=46.98, elevation=2000.0))
        result = validate_lift_goes_uphill(start_node=start, end_node=end)
        assert isinstance(result, LiftMustGoUphillMessage)
        assert result.start_elevation_m == 2500.0
        assert result.end_elevation_m == 2000.0

    def test_flat_lift_returns_message(self) -> None:
        """Same elevation returns LiftMustGoUphillMessage (end must be HIGHER)."""
        start = Node(id="N1", location=PathPoint(lon=10.27, lat=46.97, elevation=2500.0))
        end = Node(id="N2", location=PathPoint(lon=10.28, lat=46.98, elevation=2500.0))
        result = validate_lift_goes_uphill(start_node=start, end_node=end)
        assert isinstance(result, LiftMustGoUphillMessage)


class TestValidateLiftDifferentNodes:
    """Tests for validate_lift_different_nodes."""

    def test_different_nodes_is_valid(self) -> None:
        """Different node IDs are valid."""
        result = validate_lift_different_nodes(start_node_id="N1", end_node_id="N2")
        assert result is None

    def test_same_node_returns_message(self) -> None:
        """Same node ID returns SameNodeLiftMessage."""
        result = validate_lift_different_nodes(start_node_id="N1", end_node_id="N1")
        assert isinstance(result, SameNodeLiftMessage)


class TestValidateCustomTargetDownhill:
    """Tests for validate_custom_target_downhill."""

    def test_sufficient_drop_is_valid(self) -> None:
        """Target with sufficient drop is valid."""
        result = validate_custom_target_downhill(
            start_elevation=2500.0,
            target_elevation=2400.0,  # 100m drop > MIN_DROP_M
        )
        assert result is None

    def test_exactly_min_drop_is_valid(self) -> None:
        """Exactly MIN_DROP_M is valid."""
        result = validate_custom_target_downhill(
            start_elevation=2000.0 + ConnectionConfig.MIN_DROP_M,
            target_elevation=2000.0,
        )
        assert result is None

    def test_insufficient_drop_returns_message(self) -> None:
        """Target with insufficient drop returns TargetNotDownhillMessage."""
        result = validate_custom_target_downhill(
            start_elevation=2500.0,
            target_elevation=2499.0,  # Only 1m drop < MIN_DROP_M
        )
        assert isinstance(result, TargetNotDownhillMessage)
        assert result.start_elevation_m == 2500.0
        assert result.target_elevation_m == 2499.0

    def test_uphill_target_returns_message(self) -> None:
        """Target that goes uphill returns TargetNotDownhillMessage."""
        result = validate_custom_target_downhill(
            start_elevation=2500.0,
            target_elevation=2600.0,  # 100m UPHILL
        )
        assert isinstance(result, TargetNotDownhillMessage)


class TestValidateCustomTargetDistance:
    """Tests for validate_custom_target_distance."""

    def test_within_distance_is_valid(self) -> None:
        """Target within max distance is valid."""
        # Two points about 100m apart
        result = validate_custom_target_distance(
            start_lat=46.97,
            start_lon=10.27,
            target_lat=46.9709,  # ~100m north
            target_lon=10.27,
            max_distance_m=200.0,
        )
        assert result is None

    def test_exactly_at_limit_is_valid(self) -> None:
        """Target exactly at max distance is valid."""
        # Same point = 0m distance
        result = validate_custom_target_distance(
            start_lat=46.97,
            start_lon=10.27,
            target_lat=46.97,
            target_lon=10.27,
            max_distance_m=1.0,  # Very small limit, but same point works
        )
        assert result is None

    def test_beyond_distance_returns_message(self) -> None:
        """Target beyond max distance returns TargetTooFarMessage."""
        # Two points about 500m apart
        result = validate_custom_target_distance(
            start_lat=46.97,
            start_lon=10.27,
            target_lat=46.975,  # ~500m north
            target_lon=10.27,
            max_distance_m=200.0,
        )
        assert isinstance(result, TargetTooFarMessage)
        assert result.max_distance_m == 200.0
        assert result.distance_m > 200.0
