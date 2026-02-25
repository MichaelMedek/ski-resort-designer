"""BaseSlopePath - Base class for slope path geometry with computed metrics.

Provides shared functionality for both proposed and committed slope segments:
- Points storage
- Computed metrics (length, drop, slope percentage, difficulty)

Reference: DETAILS.md
"""

from dataclasses import dataclass
from typing import Optional

from skiresort_planner.model.path_point import PathPoint


@dataclass
class BaseSlopePath:
    """Base class for slope paths with computed geometric metrics.

    Stores path points and computes metrics on-the-fly from the point data.
    Both ProposedSlopeSegment and SlopeSegment inherit from this.

    Attributes:
        points: Path points (source of truth for geometry)

    Computed Properties:
        start: First point
        end: Last point
        total_drop_m: Elevation difference between endpoints
        length_m: Sum of distances between consecutive points
        avg_slope_pct: (total_drop / length) * 100
        difficulty: Classification based on avg_slope_pct
    """

    points: list[PathPoint]

    @property
    def start(self) -> Optional[PathPoint]:
        """First point of the path."""
        return self.points[0] if self.points else None

    @property
    def end(self) -> Optional[PathPoint]:
        """Last point of the path."""
        return self.points[-1] if self.points else None

    @property
    def total_drop_m(self) -> float:
        """Total vertical drop in meters (computed from endpoints)."""
        if len(self.points) < 2:
            return 0.0
        assert self.start is not None and self.end is not None
        return self.start.elevation - self.end.elevation

    @property
    def length_m(self) -> float:
        """Total path length in meters (computed from point distances)."""
        if len(self.points) < 2:
            return 0.0
        return sum(self.points[i].distance_to(other=self.points[i + 1]) for i in range(len(self.points) - 1))

    @property
    def avg_slope_pct(self) -> float:
        """Average slope percentage (computed from drop/length)."""
        if self.length_m <= 0:
            return 0.0
        return (self.total_drop_m / self.length_m) * 100

    @property
    def difficulty(self) -> str:
        """Classified difficulty based on average slope."""
        from skiresort_planner.core.terrain_analyzer import (
            TerrainAnalyzer,
        )

        return TerrainAnalyzer.classify_difficulty(slope_pct=self.avg_slope_pct)
