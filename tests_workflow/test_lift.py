"""Unit tests for the Lift model (model/lift.py)."""

import pytest

from skiresort_planner.model.lift import Lift
from skiresort_planner.model.node import Node
from skiresort_planner.model.path_point import PathPoint

_PTS = [
    PathPoint(lon=10.0, lat=46.0, elevation=2000.0),
    PathPoint(lon=10.001, lat=46.001, elevation=2100.0),
]


def _valid_lift() -> Lift:
    """A minimal structurally-valid lift (2 terrain + 2 cable points, valid type)."""
    return Lift(
        id="L1",
        name="Test Lift",
        start_node_id="N1",
        end_node_id="N2",
        lift_type="chairlift",
        terrain_points=list(_PTS),
        pylons=[],
        cable_points=list(_PTS),
    )


class TestLiftPostInitValidation:
    def test_valid_lift_constructs(self) -> None:
        assert _valid_lift().lift_type == "chairlift"

    def test_invalid_lift_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid lift_type"):
            Lift(
                id="L1",
                name="x",
                start_node_id="N1",
                end_node_id="N2",
                lift_type="rocket",
                terrain_points=list(_PTS),
                pylons=[],
                cable_points=list(_PTS),
            )

    def test_too_few_terrain_points_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 terrain_points"):
            Lift(
                id="L1",
                name="x",
                start_node_id="N1",
                end_node_id="N2",
                lift_type="chairlift",
                terrain_points=_PTS[:1],
                pylons=[],
                cable_points=list(_PTS),
            )

    def test_too_few_cable_points_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 cable_points"):
            Lift(
                id="L1",
                name="x",
                start_node_id="N1",
                end_node_id="N2",
                lift_type="chairlift",
                terrain_points=list(_PTS),
                pylons=[],
                cable_points=_PTS[:1],
            )


class TestLiftNodeLookups:
    def test_get_vertical_rise_missing_node_raises(self) -> None:
        with pytest.raises(ValueError, match="Start or end node not found"):
            _valid_lift().get_vertical_rise(nodes={})

    def test_get_length_m_missing_node_raises(self) -> None:
        with pytest.raises(ValueError, match="Start or end node not found"):
            _valid_lift().get_length_m(nodes={})

    def test_get_vertical_rise_computes_when_nodes_present(self) -> None:
        nodes = {
            "N1": Node(id="N1", location=PathPoint(lon=10.0, lat=46.0, elevation=2000.0)),
            "N2": Node(id="N2", location=PathPoint(lon=10.001, lat=46.001, elevation=2100.0)),
        }
        assert _valid_lift().get_vertical_rise(nodes=nodes) == pytest.approx(100.0)


class TestUpdateTypeValidation:
    def test_update_type_invalid_raises(self) -> None:
        node_a = Node(id="N1", location=PathPoint(lon=10.0, lat=46.0, elevation=2000.0))
        node_b = Node(id="N2", location=PathPoint(lon=10.001, lat=46.001, elevation=2100.0))
        with pytest.raises(ValueError, match="Invalid lift_type"):
            _valid_lift().update_type(new_type="rocket", start_node=node_a, end_node=node_b)


class TestCalculateStaticmethodValidation:
    def test_calculate_pylons_too_few_points_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 points"):
            Lift.calculate_pylons(terrain_points=_PTS[:1], lift_type="chairlift", total_distance_m=500.0)

    def test_calculate_pylons_nonpositive_distance_raises(self) -> None:
        with pytest.raises(ValueError, match="total_distance_m must be positive"):
            Lift.calculate_pylons(terrain_points=list(_PTS), lift_type="chairlift", total_distance_m=0.0)

    def test_calculate_cable_points_too_few_points_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 points"):
            Lift.calculate_cable_points(
                terrain_points=_PTS[:1],
                pylons=[],
                start_elevation=2000.0,
                end_elevation=2100.0,
                lift_type="chairlift",
                total_distance_m=500.0,
            )

    def test_calculate_cable_points_nonpositive_distance_raises(self) -> None:
        with pytest.raises(ValueError, match="total_distance_m must be positive"):
            Lift.calculate_cable_points(
                terrain_points=list(_PTS),
                pylons=[],
                start_elevation=2000.0,
                end_elevation=2100.0,
                lift_type="chairlift",
                total_distance_m=-5.0,
            )


class TestLiftNaming:
    """generate_name / number_from_id. RNG seeded by the autouse _deterministic_rng fixture."""

    def test_summit_name_for_large_rise(self) -> None:
        # Vertical rise above SUMMIT_RISE_M (500m) → "Summit" lift name.
        name = Lift.generate_name(
            lift_type="chairlift", lift_id="L1", length_m=800.0, vertical_rise_m=600.0, avg_bearing=0.0
        )
        assert "Summit" in name

    def test_name_embeds_lift_number(self) -> None:
        name = Lift.generate_name(
            lift_type="gondola", lift_id="L7", length_m=800.0, vertical_rise_m=100.0, avg_bearing=0.0
        )
        assert name.startswith("7 (")

    @pytest.mark.parametrize(
        "id_str,expected",
        [("L1", 1), ("L7", 7), ("L99", 99)],
    )
    def test_number_from_id(self, id_str: str, expected: int) -> None:
        assert Lift.number_from_id(lift_id=id_str) == expected
