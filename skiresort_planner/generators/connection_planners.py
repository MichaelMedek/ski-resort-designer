"""Connection Path Planner - Grid-based Dijkstra for generating paths to specific target points.

This module provides an algorithm for generating ski paths that connect
a start point to a user-specified target point. Unlike fan generation which
radiates paths in natural directions, connection paths must reach a specific
destination while maintaining a target effective slope.

Uses SciPy's optimized sparse graph Dijkstra for performance, followed by
cubic spline smoothing to eliminate grid artifacts.

Reference: DETAILS.md Section 7 for algorithm details.
"""

import logging
import math
from dataclasses import dataclass
from math import exp
from typing import Optional

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from skiresort_planner.constants import PathConfig, PlannerConfig
from skiresort_planner.core.dem_service import DEMService
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.proposed_path import ProposedSlopeSegment

logger = logging.getLogger(__name__)


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
        cost(edge) = distance × exp(|actual_slope - target_slope| / σ) × uphill_penalty

    Where:
    - σ (COST_SIGMA) controls sensitivity (smaller = stricter slope matching)
    - uphill_penalty = 1.0 if downhill, exp(|slope|/σ) if uphill

    This exponential cost heavily penalizes slope deviations, causing the
    algorithm to prefer longer traverses over steep shortcuts.

    Configuration: See PlannerConfig in constants.py for tunable parameters.
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
        target_slope_pct: float,
        side: str,
    ) -> Optional[ProposedSlopeSegment]:
        """Plan path using grid-based Dijkstra search.

        Args:
            start_lon, start_lat, start_elevation: Starting point
            target_lon, target_lat, target_elevation: Target point
            target_slope_pct: Target effective slope percentage
            side: "left" or "right" - preferred side of direct line

        Returns:
            ProposedSlopeSegment if path found, None otherwise.
        """
        # Validate inputs
        drop_m = start_elevation - target_elevation
        if drop_m <= 0:
            return None

        direct_distance_m = GeoCalculator.haversine_distance_m(
            lat1=start_lat, lon1=start_lon, lat2=target_lat, lon2=target_lon
        )

        if direct_distance_m < PathConfig.STEP_SIZE_M:
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
            target_slope_pct=target_slope_pct,
            side=side,
            lons=lons,
            lats=lats,
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
            target_slope_pct=target_slope_pct,
        )

        return ProposedSlopeSegment(
            points=points,
            target_slope_pct=target_slope_pct,
            is_connector=True,
        )

    def _build_grid(
        self,
        start_lon: float,
        start_lat: float,
        target_lon: float,
        target_lat: float,
        direct_distance: float,
    ) -> Optional[tuple[list[list[float]], list[list[float]], list[list[float]], GridNode, GridNode]]:
        """Build elevation grid covering the search area."""
        # Calculate grid bounds with buffer
        buffer_m = direct_distance * PlannerConfig.GRID_BUFFER_FACTOR
        total_extent = direct_distance + 2 * buffer_m

        # Grid dimensions
        n_cells = int(total_extent / PlannerConfig.GRID_RESOLUTION_M) + 1
        n_cells = min(n_cells, PlannerConfig.MAX_GRID_SIZE)  # Cap grid size for performance

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
                    distance_m=col * PlannerConfig.GRID_RESOLUTION_M,
                )
                lon, lat = GeoCalculator.destination(
                    lon=lon,
                    lat=lat,
                    bearing_deg=bearing,
                    distance_m=row * PlannerConfig.GRID_RESOLUTION_M,
                )

                elev = self.dem.get_elevation(lon=lon, lat=lat)
                if elev is None:
                    raise RuntimeError(
                        f"DEM returned None for grid point at row={row}, col={col} (lon={lon}, lat={lat}), cannot build grid with missing elevation data"
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
    ) -> Optional[GridNode]:
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
        target_slope_pct: float,
        side: str,
        lons: list[list[float]],
        lats: list[list[float]],
    ) -> tuple[Optional[list[GridNode]], int, int]:
        """Least-cost path using SciPy's C-optimized Dijkstra.

        Builds a sparse graph from the elevation grid and uses
        scipy.sparse.csgraph.shortest_path for efficient pathfinding.
        """
        n_rows = len(elevations)
        n_cols = len(elevations[0])
        N = n_rows * n_cols

        # Target coords (used for side preference in edge cost)
        t_lon = lons[target.row][target.col]
        t_lat = lats[target.row][target.col]

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
                        from_lon=from_lon,
                        from_lat=from_lat,
                        to_lon=to_lon,
                        to_lat=to_lat,
                        target_slope_pct=target_slope_pct,
                        side=side,
                        target_lon=t_lon,
                        target_lat=t_lat,
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
        target_slope_pct: float,
        side: str,
        target_lon: float,
        target_lat: float,
    ) -> float:
        """Minimal cost function: distance weighted by slope deviation and uphill penalty.

        Two soft constraints:
        1. Exponential penalty for slope deviation from target
        2. Exponential uphill penalty (zero for downhill, grows for uphill)

        No hard cutoffs - allows occasional uphill due to DEM noise.
        """
        # Horizontal distance
        horiz_dist = GeoCalculator.haversine_distance_m(lat1=from_lat, lon1=from_lon, lat2=to_lat, lon2=to_lon)

        if horiz_dist < 0.1:
            return float("inf")

        # Actual slope (positive = downhill, negative = uphill)
        drop = from_elev - to_elev
        actual_slope = (drop / horiz_dist) * 100

        # Slope deviation penalty
        slope_diff = abs(actual_slope - target_slope_pct)
        slope_cost = exp(slope_diff / PlannerConfig.COST_SIGMA)

        # Uphill penalty: zero for downhill, exponential for uphill
        # Uses same sigma as slope deviation for consistency
        uphill_penalty = 1.0
        if actual_slope < 0:
            uphill_penalty = exp(abs(actual_slope) / PlannerConfig.COST_SIGMA)

        return horiz_dist * slope_cost * uphill_penalty

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
        target_slope_pct: float,
        step_m: float = 7.0,
    ) -> list[PathPoint]:
        """Smooth grid path using cubic spline interpolation and resample at fixed intervals.

        The grid-based Dijkstra produces staircase paths due to 8-directional movement.
        This method fits a smooth cubic spline through the points and resamples at
        regular intervals, eliminating grid artifacts while preserving the overall shape.

        Args:
            points: Raw grid path points
            target_slope_pct: Target slope - controls smoothing aggressiveness
            step_m: Output point spacing in meters (default 7m)

        Returns:
            Smoothed path with regular point spacing and DEM-sampled elevations.
        """
        if len(points) < 4:
            return points

        lons = np.array([p.lon for p in points])
        lats = np.array([p.lat for p in points])
        elevs = np.array([p.elevation for p in points])

        # Cumulative horizontal distance (so spline respects real path length)
        cumdist = np.zeros(len(points))
        for i in range(1, len(points)):
            cumdist[i] = cumdist[i - 1] + GeoCalculator.haversine_distance_m(
                lat1=lats[i - 1], lon1=lons[i - 1], lat2=lats[i], lon2=lons[i]
            )

        total_length = cumdist[-1]
        if total_length < step_m * 2:
            return points

        # Smoothing factor: higher = more aggressive smoothing
        # Green: 4.0 for flowing traverses
        # Blue: 3.0 for moderate smoothing
        # Red/Black: 2.0 for nearly straight paths
        difficulty = TerrainAnalyzer.classify_difficulty(slope_pct=target_slope_pct)
        if difficulty == "green":
            smoothing_factor = 4.0
        elif difficulty == "blue":
            smoothing_factor = 3.0
        else:
            smoothing_factor = 2.0

        try:
            # Fit cubic smoothing spline
            # splprep returns a complex tuple that Mypy can't unpack into tck, u
            tck, _ = splprep(
                [lons, lats, elevs],
                u=cumdist,
                s=smoothing_factor * len(points),
                k=3,
            )

            # Resample evenly along the path
            new_dists = np.arange(0, total_length + step_m / 2, step_m)
            # splev has multiple overloads; Mypy can't determine which applies
            new_lon, new_lat, new_elev_approx = splev(new_dists, tck)  # type: ignore[call-overload]

            # Re-query DEM for accurate elevations at smoothed positions
            final_points = []
            for i in range(len(new_lon)):
                real_elev = self.dem.get_elevation(lon=float(new_lon[i]), lat=float(new_lat[i]))
                if real_elev is None:
                    real_elev = float(new_elev_approx[i])
                final_points.append(
                    PathPoint(
                        lon=float(new_lon[i]),
                        lat=float(new_lat[i]),
                        elevation=real_elev,
                    )
                )

            return final_points

        except Exception as e:
            logger.error(f"Spline smoothing failed: {e}, returning raw points")
            return points
