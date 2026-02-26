"""Unit tests for validator functions using parametrize with indirect fixtures.

Tests lift and custom target validators with data-driven test cases.
"""

import pytest

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


# =============================================================================
# INDIRECT FIXTURE FOR NODE PAIRS
# =============================================================================


@pytest.fixture
def node_pair(request: pytest.FixtureRequest) -> tuple[Node, Node]:
    """Create start/end Node pair from elevation tuple (start_elev, end_elev)."""
    start_elev, end_elev = request.param
    start = Node(id="N1", location=PathPoint(lon=10.0, lat=46.0, elevation=start_elev))
    end = Node(id="N2", location=PathPoint(lon=10.0, lat=46.01, elevation=end_elev))
    return start, end


# =============================================================================
# LIFT VALIDATOR TESTS
# =============================================================================


class TestLiftValidators:
    """Parametrized tests for lift placement validators."""

    @pytest.mark.parametrize(
        "node_pair,expected_type",
        [
            pytest.param((1500.0, 2000.0), None, id="valid_uphill"),
            pytest.param((2000.0, 1500.0), LiftMustGoUphillMessage, id="invalid_downhill"),
            pytest.param((2000.0, 2000.0), LiftMustGoUphillMessage, id="invalid_same_elevation"),
        ],
        indirect=["node_pair"],
    )
    def test_lift_goes_uphill(self, node_pair: tuple[Node, Node], expected_type: type | None) -> None:
        """validate_lift_goes_uphill returns None for valid, error message for invalid."""
        start, end = node_pair
        result = validate_lift_goes_uphill(start_node=start, end_node=end)
        if expected_type is None:
            assert result is None
        else:
            assert isinstance(result, expected_type)

    @pytest.mark.parametrize(
        "start_id,end_id,expected_type",
        [
            pytest.param("N1", "N2", None, id="different_nodes"),
            pytest.param("N1", "N1", SameNodeLiftMessage, id="same_node"),
        ],
    )
    def test_lift_different_nodes(self, start_id: str, end_id: str, expected_type: type | None) -> None:
        """validate_lift_different_nodes returns None for valid, error for same node."""
        result = validate_lift_different_nodes(start_node_id=start_id, end_node_id=end_id)
        if expected_type is None:
            assert result is None
        else:
            assert isinstance(result, expected_type)


# =============================================================================
# CUSTOM TARGET VALIDATOR TESTS
# =============================================================================


class TestCustomTargetValidators:
    """Parametrized tests for custom target connection validators."""

    @pytest.mark.parametrize(
        "start_elev,target_elev,expected_type",
        [
            pytest.param(2500.0, 2400.0, None, id="valid_100m_drop"),
            pytest.param(2400.0, 2500.0, TargetNotDownhillMessage, id="invalid_uphill"),
            pytest.param(2500.0, 2496.0, TargetNotDownhillMessage, id="invalid_small_drop"),
        ],
    )
    def test_custom_target_downhill(self, start_elev: float, target_elev: float, expected_type: type | None) -> None:
        """validate_custom_target_downhill returns None for valid, error for invalid."""
        result = validate_custom_target_downhill(start_elevation=start_elev, target_elevation=target_elev)
        if expected_type is None:
            assert result is None
        else:
            assert isinstance(result, expected_type)

    @pytest.mark.parametrize(
        "target_lat,target_lon,expected_type",
        [
            pytest.param(46.005, 10.0, None, id="valid_500m"),
            pytest.param(46.1, 10.1, TargetTooFarMessage, id="invalid_11km"),
        ],
    )
    def test_custom_target_distance(self, target_lat: float, target_lon: float, expected_type: type | None) -> None:
        """validate_custom_target_distance returns None for valid, error for too far."""
        result = validate_custom_target_distance(
            start_lat=46.0, start_lon=10.0, target_lat=target_lat, target_lon=target_lon
        )
        if expected_type is None:
            assert result is None
        else:
            assert isinstance(result, expected_type)
