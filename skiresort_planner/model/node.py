"""Node - Junction point in the resort graph.

A Node represents a connection point where slope segments can meet.
It wraps a PathPoint for its location (single source of truth).

Nodes are created automatically when paths are committed.
Multiple segments can share the same node, creating junctions.

Reference: DETAILS.md
"""

from dataclasses import dataclass
from typing import Any

from skiresort_planner.model.path_point import PathPoint


@dataclass
class Node:
    """A junction point in the resort graph.

    Wraps a PathPoint for location - this is the single source of truth
    for the node's geometry. The ID provides topological identity.

    Attributes:
        id: Unique identifier (e.g., "N1", "N2", ...)
        location: PathPoint containing the geographic coordinates

    Example:
        location = PathPoint(lon=10.295, lat=46.985, elevation=2400.0)
        node = Node(id="N1", location=location)
        print(node.elevation)  # 2400.0
    """

    id: str
    location: PathPoint

    @property
    def lon(self) -> float:
        """Longitude delegated from location."""
        return self.location.lon

    @property
    def lat(self) -> float:
        """Latitude delegated from location."""
        return self.location.lat

    @property
    def elevation(self) -> float:
        """Elevation delegated from location."""
        return self.location.elevation

    @property
    def lat_lon(self) -> tuple[float, float]:
        """Return (lat, lon) tuple - standard geographic order."""
        return self.location.lat_lon

    @property
    def lon_lat(self) -> tuple[float, float]:
        """Return (lon, lat) tuple - GeoJSON/Pydeck order."""
        return self.location.lon_lat

    def distance_to(self, lon: float, lat: float) -> float:
        """Calculate distance to given coordinates in meters.

        Args:
            lon: Target longitude in decimal degrees
            lat: Target latitude in decimal degrees

        Returns:
            Distance in meters using great-circle calculation.
        """
        # Elevation doesn't affect horizontal distance
        target = PathPoint(lon=lon, lat=lat, elevation=0.0)
        return self.location.distance_to(other=target)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Node":
        """Create Node from dictionary."""
        return cls(
            id=data["id"],
            location=PathPoint(**data["location"]),
        )

    def __repr__(self) -> str:
        return f"Node({self.id}, {self.location})"
