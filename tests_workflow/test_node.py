"""Unit tests for Node (model/node.py) — distance, properties, and from_dict."""

from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.model.node import Node
from skiresort_planner.model.path_point import PathPoint


class TestNodeProperties:
    def test_properties_delegate_to_location(self) -> None:
        node = Node(id="N1", location=PathPoint(lon=10.5, lat=46.5, elevation=2100.0))
        assert node.lon == 10.5
        assert node.lat == 46.5
        assert node.elevation == 2100.0
        assert node.lon_lat == (10.5, 46.5)
        assert node.lat_lon == (46.5, 10.5)


class TestNodeDistanceCalculation:
    def test_distance_matches_haversine_exactly(self) -> None:
        """Node.distance_to() equals the haversine distance to the coordinates."""
        node = Node(id="N1", location=PathPoint(lon=10.0, lat=46.0, elevation=2000.0))

        assert node.distance_to(lon=10.0, lat=46.0) == 0.0
        expected = GeoCalculator.haversine_distance_m(lat1=46.0, lon1=10.0, lat2=46.0001, lon2=10.0001)
        assert node.distance_to(lon=10.0001, lat=46.0001) == expected


class TestNodeFromDict:
    """from_dict drives all graph deserialization — verify it reconstructs fully."""

    def test_from_dict_reconstructs_id_and_location(self) -> None:
        data: dict[str, object] = {"id": "N7", "location": {"lon": 10.25, "lat": 46.98, "elevation": 2345.6}}
        node = Node.from_dict(data=data)

        assert node.id == "N7"
        assert node.lon == 10.25
        assert node.lat == 46.98
        assert node.elevation == 2345.6

    def test_from_dict_round_trips_coordinates(self) -> None:
        """A Node's coordinates survive a dict → Node reconstruction."""
        original = Node(id="N3", location=PathPoint(lon=10.1, lat=46.2, elevation=2222.0))
        data: dict[str, object] = {
            "id": original.id,
            "location": {"lon": original.lon, "lat": original.lat, "elevation": original.elevation},
        }
        restored = Node.from_dict(data=data)

        assert restored.id == original.id
        assert restored.lon == original.lon
        assert restored.lat == original.lat
        assert restored.elevation == original.elevation
