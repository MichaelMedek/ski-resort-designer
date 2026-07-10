"""Unit tests for Node (model/node.py) — distance to coordinates."""

from skiresort_planner.model.node import Node
from skiresort_planner.model.path_point import PathPoint


class TestNodeDistanceCalculation:
    """Tests for Node distance calculation."""

    def test_node_distance_to_point(self) -> None:
        """Node.distance_to() calculates correct distance to coordinates."""
        node = Node(id="N1", location=PathPoint(lon=10.0, lat=46.0, elevation=2000.0))

        assert node.distance_to(lon=10.0, lat=46.0) == 0.0
        dist_nearby = node.distance_to(lon=10.0001, lat=46.0001)
        assert 0 < dist_nearby < 100
