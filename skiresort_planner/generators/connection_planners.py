"""Connection Path Planner - Grid-based Dijkstra for generating paths to specific target points.

This module provides a domain-agnostic algorithm for generating a path that connects
a start point to a user-specified target point while holding a target grade. Callers
supply the target grade and whether climbing is allowed; the planner knows nothing
about pistes or roads.

Uses SciPy's optimized sparse graph Dijkstra for performance, followed by
cubic spline smoothing to eliminate grid artifacts.

Reference: DETAILS.md Section 7 for algorithm details.
"""

import logging
import math
from dataclasses import dataclass
from enum import StrEnum
from math import exp

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from skiresort_planner.constants import GeometricTuningConfig, PlannerConfig
from skiresort_planner.core.dem_service import DEMService
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
from skiresort_planner.enum_utils import enum_eq
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_smoothing import resample_cubic_spline
from skiresort_planner.model.proposed_path import ProposedPathSegment

logger = logging.getLogger(__name__)


class GradientMode(StrEnum):
    """Which way a single segment is allowed to run along its length."""

    DOWNHILL = "downhill"  # net-descends; climbing is penalized (skiable pistes + descending roads)
    UPHILL = "uphill"  # net-climbs; descending is penalized (climbing roads)


@dataclass(frozen=True)
class GridNode:
    """A node in the search grid."""

    row: int
    col: int

    def __lt__(self, other: "GridNode") -> bool:
        """Comparison for sorting and debugging."""
        return (self.row, self.col) < (other.row, other.col)


class LeastCostPathPlanner:
    """Connection planner using grid-based Dijkstra search on terrain.

    Algorithm Overview:
    1. Create a grid covering the area between start and target (+ buffer)
    2. Build sparse graph with 8-connectivity, edges weighted by slope cost
    3. Run SciPy's C-optimized Dijkstra to find minimum-cost path
    4. Smooth the grid path using cubic spline interpolation
    5. Resample at regular intervals with DEM elevation lookups

    Cost Function:
        cost(edge) = distance × exp(|actual_grade - target_grade| / σ) × against_penalty

    Where:
    - σ (COST_SIGMA) controls sensitivity (smaller = stricter grade matching)
    - against_penalty = 1.0 when the edge runs the segment's way, else exp(|grade|/σ):
      DOWNHILL mode penalizes climbing, UPHILL mode penalizes descending. That one-way
      monotonicity keeps the path from looping.

    This exponential cost heavily penalizes grade deviations, causing the algorithm to
    prefer longer traverses over steep shortcuts — so a gentle target grade serpentines
    on steep ground instead of exceeding it.

    Configuration: See GeometricTuningConfig in constants.py for tunable parameters.
    """

    def __init__(
        self,
        dem_service: DEMService,
        terrain_analyzer: TerrainAnalyzer,
    ) -> None:
        """Initialize the planner with terrain services."""
        self.dem = dem_service
        self.terrain_analyzer = terrain_analyzer

    def plan(
        self,
        start_lon: float,
        start_lat: float,
        start_elevation: float,
        target_lon: float,
        target_lat: float,
        target_elevation: float,
        target_grade_pct: float,
        gradient_mode: GradientMode = GradientMode.DOWNHILL,
    ) -> ProposedPathSegment | None:
        """Plan a least-cost path that holds a target grade from start to target.

        Args:
            start_lon/lat/elevation, target_lon/lat/elevation: the two endpoints.
            target_grade_pct: signed grade the path aims to hold (+ = descending).
            gradient_mode: DOWNHILL (default) forces net-descent and penalizes climbing
                (skiable pistes); UPHILL forces net-ascent and penalizes descending. The
                monotonicity is what prevents the path from looping.

        Returns:
            ProposedPathSegment if path found, None otherwise.
        """
        # The segment must actually run in its mode's direction (net drop / net climb).
        net_drop = start_elevation - target_elevation
        if enum_eq(a=gradient_mode, b=GradientMode.DOWNHILL) and net_drop <= 0:
            return None
        if enum_eq(a=gradient_mode, b=GradientMode.UPHILL) and net_drop >= 0:
            return None

        direct_distance_m = GeoCalculator.haversine_distance_m(
            lat1=start_lat, lon1=start_lon, lat2=target_lat, lon2=target_lon
        )

        if direct_distance_m < GeometricTuningConfig.STEP_SIZE_M:
            return None

        # Build the search grid
        grid_data = self._build_grid(
            start_lon=start_lon,
            start_lat=start_lat,
            target_lon=target_lon,
            target_lat=target_lat,
            direct_distance=direct_distance_m,
        )

        if grid_data is None:
            return None

        elevations, lons, lats, start_node, target_node = grid_data

        # Run fast SciPy Dijkstra on the exact same graph
        path_nodes, _, _ = self._graph_dijkstra(
            elevations=elevations,
            start=start_node,
            target=target_node,
            target_grade_pct=target_grade_pct,
            lons=lons,
            lats=lats,
            gradient_mode=gradient_mode,
        )

        if path_nodes is None:
            return None

        # Convert grid path to PathPoints
        raw_points = self._path_to_points(
            path_nodes=path_nodes,
            elevations=elevations,
            lons=lons,
            lats=lats,
        )

        # Smooth the grid path using spline interpolation
        points = self._smooth_path_spline(
            points=raw_points,
            target_grade_pct=target_grade_pct,
        )

        return ProposedPathSegment(
            points=points,
            target_slope_pct=target_grade_pct,
            is_connector=True,
        )

    def _build_grid(
        self,
        start_lon: float,
        start_lat: float,
        target_lon: float,
        target_lat: float,
        direct_distance: float,
    ) -> tuple[list[list[float]], list[list[float]], list[list[float]], GridNode, GridNode] | None:
        """Build elevation grid covering the search area."""
        # Calculate grid bounds with buffer
        # TODO(serpentine): this buffer (0.5×direct) is too tight for switchbacks — a
        # zig-zag needs much more lateral room than a near-straight line. When a gentle
        # target grade forces serpentining on steep ground, widen this buffer so those
        # switchbacks physically fit in the search grid instead of being clipped.
        buffer_m = direct_distance * GeometricTuningConfig.GRID_BUFFER_FACTOR
        total_extent = direct_distance + 2 * buffer_m

        # Grid dimensions
        n_cells = int(total_extent / GeometricTuningConfig.GRID_RESOLUTION_M) + 1
        n_cells = min(n_cells, int(GeometricTuningConfig.MAX_GRID_SIZE))  # Cap grid size for performance

        # Center point
        center_lon = (start_lon + target_lon) / 2
        center_lat = (start_lat + target_lat) / 2

        # Bearing from center to target for grid orientation
        bearing = GeoCalculator.initial_bearing_deg(lon1=center_lon, lat1=center_lat, lon2=target_lon, lat2=target_lat)

        # Grid origin (top-left corner)
        origin_lon, origin_lat = GeoCalculator.destination(
            lon=center_lon,
            lat=center_lat,
            bearing_deg=(bearing + 180) % 360,
            distance_m=total_extent / 2,
        )
        origin_lon, origin_lat = GeoCalculator.destination(
            lon=origin_lon,
            lat=origin_lat,
            bearing_deg=(bearing - 90) % 360,
            distance_m=total_extent / 2,
        )

        # Build grid arrays
        elevations = []
        lons = []
        lats = []

        for row in range(n_cells):
            elev_row = []
            lon_row = []
            lat_row = []

            for col in range(n_cells):
                # Calculate position
                lon, lat = GeoCalculator.destination(
                    lon=origin_lon,
                    lat=origin_lat,
                    bearing_deg=(bearing + 90) % 360,
                    distance_m=col * GeometricTuningConfig.GRID_RESOLUTION_M,
                )
                lon, lat = GeoCalculator.destination(
                    lon=lon,
                    lat=lat,
                    bearing_deg=bearing,
                    distance_m=row * GeometricTuningConfig.GRID_RESOLUTION_M,
                )

                elev = self.dem.get_elevation(lon=lon, lat=lat)
                if elev is None:
                    raise RuntimeError(
                        f"DEM returned None for grid point at row={row}, col={col} "
                        f"(lon={lon}, lat={lat}), cannot build grid with missing elevation data"
                    )

                elev_row.append(elev)
                lon_row.append(lon)
                lat_row.append(lat)

            elevations.append(elev_row)
            lons.append(lon_row)
            lats.append(lat_row)

        # Find start and target nodes
        start_node = self._find_nearest_node(target_lon=start_lon, target_lat=start_lat, lons=lons, lats=lats)
        target_node = self._find_nearest_node(target_lon=target_lon, target_lat=target_lat, lons=lons, lats=lats)

        if start_node is None or target_node is None:
            return None

        return elevations, lons, lats, start_node, target_node

    def _find_nearest_node(
        self,
        target_lon: float,
        target_lat: float,
        lons: list[list[float]],
        lats: list[list[float]],
    ) -> GridNode | None:
        """Find grid node nearest to target coordinates."""
        best_dist = float("inf")
        best_node = None

        for row in range(len(lons)):
            for col in range(len(lons[0])):
                dist = GeoCalculator.haversine_distance_m(
                    lat1=lats[row][col],
                    lon1=lons[row][col],
                    lat2=target_lat,
                    lon2=target_lon,
                )
                if dist < best_dist:
                    best_dist = dist
                    best_node = GridNode(row=row, col=col)

        return best_node

    def _graph_dijkstra(
        self,
        elevations: list[list[float]],
        start: GridNode,
        target: GridNode,
        target_grade_pct: float,
        lons: list[list[float]],
        lats: list[list[float]],
        gradient_mode: GradientMode = GradientMode.DOWNHILL,
    ) -> tuple[list[GridNode] | None, int, int]:
        """Least-cost path using SciPy's C-optimized Dijkstra.

        Builds a sparse graph from the elevation grid and uses
        scipy.sparse.csgraph.shortest_path for efficient pathfinding.
        """
        n_rows = len(elevations)
        n_cols = len(elevations[0])
        N = n_rows * n_cols

        # Build sparse graph (row, col, data) for CSR matrix
        row_list: list[int] = []
        col_list: list[int] = []
        data_list: list[float] = []

        for r in range(n_rows):
            for c in range(n_cols):
                from_elev = elevations[r][c]
                if math.isnan(from_elev):
                    continue

                from_lon = lons[r][c]
                from_lat = lats[r][c]
                from_id = r * n_cols + c

                for dr, dc in PlannerConfig.NEIGHBORS_8:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                        continue

                    to_elev = elevations[nr][nc]
                    if math.isnan(to_elev):
                        continue

                    to_lon = lons[nr][nc]
                    to_lat = lats[nr][nc]

                    edge_cost = self._calc_edge_cost(
                        from_elev=from_elev,
                        to_elev=to_elev,
                        from_lat=from_lat,
                        to_lat=to_lat,
                        from_lon=from_lon,
                        to_lon=to_lon,
                        target_grade_pct=target_grade_pct,
                        gradient_mode=gradient_mode,
                    )

                    if edge_cost < float("inf"):
                        row_list.append(from_id)
                        col_list.append(nr * n_cols + nc)
                        data_list.append(edge_cost)

        if not row_list:
            return None, 0, 0

        csgraph = csr_matrix(
            (data_list, (row_list, col_list)),
            shape=(N, N),
            dtype=np.float64,
        )

        start_id = start.row * n_cols + start.col
        target_id = target.row * n_cols + target.col

        dist, pred = shortest_path(
            csgraph=csgraph,
            method="auto",  # chooses fastest (Dijkstra for positive weights)
            directed=True,
            indices=start_id,
            return_predecessors=True,
        )

        if np.isinf(dist[target_id]):
            return None, 0, 0

        # Reconstruct path
        path_ids: list[int] = []
        current = target_id
        while True:
            path_ids.append(current)
            if current == start_id:
                break
            current = pred[current]
            if current == -9999:
                return None, 0, 0

        path_ids.reverse()

        path_nodes = [GridNode(row=pid // n_cols, col=pid % n_cols) for pid in path_ids]

        # Return same tuple shape for drop-in compatibility
        return path_nodes, len(path_nodes), N

    def _calc_edge_cost(
        self,
        from_elev: float,
        to_elev: float,
        from_lon: float,
        from_lat: float,
        to_lon: float,
        to_lat: float,
        target_grade_pct: float,
        gradient_mode: GradientMode = GradientMode.DOWNHILL,
    ) -> float:
        """Edge cost = distance × exp(|actual_grade − target_grade| / σ) × against-mode.

        The exponential attractor pulls the path toward target_grade_pct; on steep
        ground a gentle target can't be held straight, so the path serpentines. An edge
        that runs AGAINST the gradient_mode (climbing in DOWNHILL mode, or descending in
        UPHILL mode) is exponentially penalized — that one-way monotonicity is what stops
        the path from looping. It's a soft preference; any hard cap is enforced by the caller.
        """
        # Horizontal distance
        horiz_dist = GeoCalculator.haversine_distance_m(lat1=from_lat, lon1=from_lon, lat2=to_lat, lon2=to_lon)

        if horiz_dist < 0.1:
            return float("inf")

        # Actual grade (positive = downhill, negative = uphill)
        drop = from_elev - to_elev
        actual_grade = (drop / horiz_dist) * 100

        # Attractor: exponential penalty for deviating from the target grade.
        grade_cost = exp(abs(actual_grade - target_grade_pct) / GeometricTuningConfig.COST_SIGMA)

        # Penalize running against the segment's direction: DOWNHILL penalizes climbing
        # (actual_grade < 0), UPHILL penalizes descending (actual_grade > 0). This
        # one-way monotonicity is what makes loops impossible.
        against_penalty = 1.0
        wrong_way = actual_grade < 0 if enum_eq(a=gradient_mode, b=GradientMode.DOWNHILL) else actual_grade > 0
        if wrong_way:
            against_penalty = exp(abs(actual_grade) / GeometricTuningConfig.COST_SIGMA)

        return horiz_dist * grade_cost * against_penalty

    def _path_to_points(
        self,
        path_nodes: list[GridNode],
        elevations: list[list[float]],
        lons: list[list[float]],
        lats: list[list[float]],
    ) -> list[PathPoint]:
        """Convert grid path to PathPoints."""
        points = []
        for node in path_nodes:
            points.append(
                PathPoint(
                    lon=lons[node.row][node.col],
                    lat=lats[node.row][node.col],
                    elevation=elevations[node.row][node.col],
                )
            )
        return points

    def _smooth_path_spline(
        self,
        points: list[PathPoint],
        target_grade_pct: float,
        step_m: float = 7.0,
    ) -> list[PathPoint]:
        """Smooth a grid path with a cubic spline, then re-query the DEM for elevations.

        The grid-based Dijkstra produces staircase paths (8-directional movement). This
        fits a smoothing spline and resamples at step_m, then replaces each point's
        elevation with the DEM value (the planner path must follow the ground).

        Args:
            points: Raw grid path points
            target_grade_pct: Target grade — gentler targets get more aggressive smoothing
            step_m: Output point spacing in meters (default 7m)
        """
        # Gentler difficulties smooth more aggressively (green 4.0 → red/black 2.0).
        difficulty = TerrainAnalyzer.classify_difficulty(slope_pct=target_grade_pct)
        smoothing_factor = {"green": 4.0, "blue": 3.0}.get(difficulty, 2.0)

        smoothed = resample_cubic_spline(points=points, smoothing_factor=smoothing_factor, step_m=step_m)
        if smoothed is points:
            return points

        # Planner paths follow the ground: re-query the DEM at each smoothed position.
        return [
            PathPoint(lon=p.lon, lat=p.lat, elevation=self.dem.get_elevation(lon=p.lon, lat=p.lat) or p.elevation)
            for p in smoothed
        ]
