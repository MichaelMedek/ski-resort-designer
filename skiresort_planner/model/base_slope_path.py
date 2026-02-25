"""BaseSlopePath - Base class for slope path geometry with computed metrics.

Provides shared functionality for both proposed and committed slope segments:
- Points storage
- Computed metrics (length, drop, slope percentage, difficulty)

Reference: DETAILS.md
"""

from dataclasses import dataclass
from typing import Optional

from skiresort_planner.constants import SlopeConfig
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
        max_slope_pct: Maximum slope in any rolling window
        difficulty: Classification based on max_slope_pct (steepest section)
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
    def max_slope_pct(self) -> float:
        """Maximum slope percentage within any rolling window.

        Walks point by point, extending window until it exceeds ROLLING_WINDOW_M.
        Returns the steepest section found (determines difficulty rating).

        Algorithm:
        1. Start with avg_slope_pct as initial max
        2. For each starting point, walk forward point by point
        3. When cumulative distance exceeds window, calculate section slope
        4. Track the maximum slope found across all windows

        Returns:
            Maximum slope percentage. Returns avg_slope_pct if path is shorter
            than the rolling window size.
        """
        window_m = SlopeConfig.ROLLING_WINDOW_M
        max_slope = self.avg_slope_pct  # Initial guess

        # Short paths: just return average
        if self.length_m < window_m or len(self.points) < 2:
            return max_slope

        # Build cumulative distances for efficient lookup
        cum_dist = [0.0]
        for i in range(len(self.points) - 1):
            cum_dist.append(cum_dist[-1] + self.points[i].distance_to(other=self.points[i + 1]))

        # Roll window starting from each point
        for start_idx in range(len(self.points) - 1):
            start_dist = cum_dist[start_idx]

            # Walk forward until we exceed window size
            for end_idx in range(start_idx + 1, len(self.points)):
                window_dist = cum_dist[end_idx] - start_dist
                if window_dist > window_m:
                    # This point is beyond window - calculate slope of previous section
                    if end_idx > start_idx + 1:
                        section_dist = cum_dist[end_idx - 1] - start_dist
                        section_drop = self.points[start_idx].elevation - self.points[end_idx - 1].elevation
                        if section_dist > 0:
                            section_slope = abs(section_drop / section_dist) * 100
                            max_slope = max(max_slope, section_slope)
                    break

        return max_slope

    @property
    def difficulty(self) -> str:
        """Classified difficulty based on maximum slope (steepest section)."""
        from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer

        return TerrainAnalyzer.classify_difficulty(slope_pct=self.max_slope_pct)
