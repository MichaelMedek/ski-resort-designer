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
from math import exp, radians, sin
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
from skiresort_planner.model.proposed_path import ProposedPathSegment

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
        cost(edge) = distance × exp(|actual_grade - target_grade| / σ) × uphill_penalty

    Where:
    - σ (COST_SIGMA) controls sensitivity (smaller = stricter grade matching)
    - uphill_penalty = 1.0 if downhill, exp(|grade|/σ) if uphill (only when
      allow_uphill is False)

    This exponential cost heavily penalizes grade deviations, causing the algorithm to
    prefer longer traverses over steep shortcuts — so a gentle target grade serpentines
    on steep ground instead of exceeding it.

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
        target_grade_pct: float,
        side: str,
        allow_uphill: bool = False,
        incoming_bearing: Optional[float] = None,
        earthwork_tolerance_m: float = 0.0,
    ) -> Optional[ProposedPathSegment]:
        """Plan a least-cost path that holds a target grade from start to target.

        Args:
            start_lon/lat/elevation, target_lon/lat/elevation: the two endpoints.
            target_grade_pct: signed grade the path aims to hold (+ = descending).
            side: "left" or "right" preference (currently a no-op, see _calc_edge_cost).
            allow_uphill: if False the path must net-descend and climbing is penalized
                (skiable pistes); if True it may climb or descend freely (vehicle roads).
            incoming_bearing: heading (deg) inherited from a prior segment; adds a
                decaying turn penalty so the path leaves in-line (no kink). None → off.
            earthwork_tolerance_m: max cut/fill (m) the interior may use to gentle the
                grade; 0 keeps every point on the ground. Endpoints always stay on ground.

        Returns:
            ProposedPathSegment if path found, None otherwise.
        """
        # Descent-only paths need a net drop; climb-capable paths may go either way.
        if (not allow_uphill) and (start_elevation - target_elevation <= 0):
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
            target_grade_pct=target_grade_pct,
            side=side,
            lons=lons,
            lats=lats,
            allow_uphill=allow_uphill,
            incoming_bearing=incoming_bearing,
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
            incoming_bearing=incoming_bearing,
        )

        # Let the interior cut/fill within tolerance to gentle the grade (endpoints stay
        # on the ground). No-op when tolerance is 0.
        points = self._apply_earthwork_allowance(points=points, tolerance_m=earthwork_tolerance_m)

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
    ) -> Optional[tuple[list[list[float]], list[list[float]], list[list[float]], GridNode, GridNode]]:
        """Build elevation grid covering the search area."""
        # Calculate grid bounds with buffer
        # TODO(serpentine): this buffer (0.5×direct) is too tight for switchbacks — a
        # zig-zag needs much more lateral room than a near-straight line. When a gentle
        # target grade forces serpentining on steep ground, widen this buffer so those
        # switchbacks physically fit in the search grid instead of being clipped.
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
        target_grade_pct: float,
        side: str,
        lons: list[list[float]],
        lats: list[list[float]],
        allow_uphill: bool = False,
        incoming_bearing: Optional[float] = None,
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

        # Start coords: momentum's turn penalty decays with distance from here.
        s_lon = lons[start.row][start.col]
        s_lat = lats[start.row][start.col]

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
                        target_grade_pct=target_grade_pct,
                        side=side,
                        target_lon=t_lon,
                        target_lat=t_lat,
                        allow_uphill=allow_uphill,
                        incoming_bearing=incoming_bearing,
                        start_lon=s_lon,
                        start_lat=s_lat,
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
        side: str,
        target_lon: float,
        target_lat: float,
        allow_uphill: bool = False,
        incoming_bearing: Optional[float] = None,
        start_lon: Optional[float] = None,
        start_lat: Optional[float] = None,
    ) -> float:
        """Edge cost = distance × exp(|actual_grade − target_grade| / σ) × uphill.

        The exponential attractor pulls the path toward target_grade_pct; on steep
        ground a gentle target can't be held straight, so the path serpentines. When
        allow_uphill is False an extra penalty discourages climbing (descent-only
        pistes); when True climbing is free (the signed target already sets direction).
        It's a soft preference — any hard cap is enforced by the caller.

        Momentum (incoming_bearing set): a decaying turn penalty near the start node
        keeps the path leaving in-line (no kink), fading to nothing by MOMENTUM_DECAY_M.

        FIXME(side-bias): `side`, `target_lon`, `target_lat` are DEAD — cost is
        grade/position only, so left and right trace the same route. Kept for a future
        side-aware (cross-track) or heading-aware planner; until then callers emit one side.
        """
        # Horizontal distance
        horiz_dist = GeoCalculator.haversine_distance_m(lat1=from_lat, lon1=from_lon, lat2=to_lat, lon2=to_lon)

        if horiz_dist < 0.1:
            return float("inf")

        # Actual grade (positive = downhill, negative = uphill)
        drop = from_elev - to_elev
        actual_grade = (drop / horiz_dist) * 100

        # Attractor: exponential penalty for deviating from the target grade.
        grade_cost = exp(abs(actual_grade - target_grade_pct) / PlannerConfig.COST_SIGMA)

        # Penalize climbing unless the caller allows it (target grade already carries
        # the intended direction when uphill is permitted).
        uphill_penalty = 1.0
        if not allow_uphill and actual_grade < 0:
            uphill_penalty = exp(abs(actual_grade) / PlannerConfig.COST_SIGMA)

        base_cost = horiz_dist * grade_cost * uphill_penalty

        return base_cost * self._momentum_multiplier(
            from_lon=from_lon,
            from_lat=from_lat,
            to_lon=to_lon,
            to_lat=to_lat,
            incoming_bearing=incoming_bearing,
            start_lon=start_lon,
            start_lat=start_lat,
        )

    def _momentum_multiplier(
        self,
        from_lon: float,
        from_lat: float,
        to_lon: float,
        to_lat: float,
        incoming_bearing: Optional[float],
        start_lon: Optional[float],
        start_lat: Optional[float],
    ) -> float:
        """Distance-decaying momentum penalty for a clean departure from a node.

        Two stacked terms, both fading to 1.0 (no effect) past their range so
        mid-segment routing is untouched:

        - TURN: an edge whose heading deviates from `incoming_bearing` costs more,
          fading over MOMENTUM_DECAY_M. Keeps the path leaving at the right heading.
        - POSITION: the edge's endpoint drifting laterally off the incoming line
          (cross-track offset) costs more — stronger weight, MUCH faster fade
          (MOMENTUM_POS_DECAY_M). Pins WHERE the path leaves so it can't jump
          sideways off the node, then releases quickly to terrain-following.

        Returns 1.0 when there is no incoming heading / start position.
        """
        if incoming_bearing is None or start_lon is None or start_lat is None:
            return 1.0

        dist_from_start = GeoCalculator.haversine_distance_m(
            lat1=start_lat, lon1=start_lon, lat2=from_lat, lon2=from_lon
        )

        # TURN term (heading continuity), fading over the long decay.
        turn_penalty = 0.0
        if dist_from_start < PlannerConfig.MOMENTUM_DECAY_M:
            edge_bearing = GeoCalculator.initial_bearing_deg(lon1=from_lon, lat1=from_lat, lon2=to_lon, lat2=to_lat)
            turn = abs(edge_bearing - incoming_bearing) % 360
            if turn > 180:
                turn = 360 - turn  # normalize to [0, 180]
            decay = 1.0 - dist_from_start / PlannerConfig.MOMENTUM_DECAY_M
            turn_penalty = PlannerConfig.MOMENTUM_TURN_WEIGHT * decay * (turn / 90.0)

        # POSITION term (cross-track pin), stronger but fading over the short decay.
        # Cross-track = perpendicular distance of the edge endpoint from the incoming
        # line through the start node; that is what a sideways jump increases.
        pos_penalty = 0.0
        to_dist_from_start = GeoCalculator.haversine_distance_m(
            lat1=start_lat, lon1=start_lon, lat2=to_lat, lon2=to_lon
        )
        if to_dist_from_start < PlannerConfig.MOMENTUM_POS_DECAY_M:
            to_bearing = GeoCalculator.initial_bearing_deg(lon1=start_lon, lat1=start_lat, lon2=to_lon, lat2=to_lat)
            angle_off = abs(to_bearing - incoming_bearing) % 360
            if angle_off > 180:
                angle_off = 360 - angle_off
            cross_track_m = to_dist_from_start * abs(sin(radians(angle_off)))
            pos_decay = 1.0 - to_dist_from_start / PlannerConfig.MOMENTUM_POS_DECAY_M
            lateral_units = cross_track_m / PlannerConfig.MOMENTUM_POS_SCALE_M
            pos_penalty = PlannerConfig.MOMENTUM_POS_WEIGHT * pos_decay * lateral_units

        if turn_penalty == 0.0 and pos_penalty == 0.0:
            return 1.0
        return exp(turn_penalty + pos_penalty)

    def _apply_earthwork_allowance(self, points: list[PathPoint], tolerance_m: float) -> list[PathPoint]:
        """Let the path's INTERIOR cut below / fill above the ground to gentle its grade.

        The traced elevations sit on the natural ground, so the grade is whatever the
        terrain does. We pull each interior point toward the STRAIGHT line between the
        two endpoints (the gentlest possible profile) but never move it more than the
        earthwork budget from the real ground — the classic cut-and-fill trade. The
        budget is ``min(tolerance_m, dist_from_nearest_end / EARTHWORK_TAPER_RATIO)`` so
        the START and END stay exactly on the ground. Iterative Laplacian smoothing.

        Only elevation is touched; the horizontal route (lon/lat) is preserved.
        """
        # tolerance_m == 0 → returned unchanged
        if tolerance_m <= 0.0 or len(points) < 3:
            return points

        # Cumulative along-path distance at each point (drives the straight-line lerp + taper).
        cum = [0.0]
        for i in range(1, len(points)):
            cum.append(cum[-1] + points[i - 1].distance_to(other=points[i]))
        total = cum[-1]
        if total <= 0.0:
            return points

        start_elev, end_elev = points[0].elevation, points[-1].elevation
        adjusted = [points[0]]  # first point kept exactly (endpoint stays on ground)
        for i in range(1, len(points) - 1):
            ground = points[i].elevation
            straight = start_elev + (end_elev - start_elev) * (cum[i] / total)  # gentlest profile
            # Budget tapers to 0 at both ends (min-distance-to-either-end / ratio), capped at tolerance.
            budget = min(tolerance_m, min(cum[i], total - cum[i]) / PlannerConfig.EARTHWORK_TAPER_RATIO)
            elev = min(ground + budget, max(ground - budget, straight))  # move toward line, clamp to ±budget
            adjusted.append(PathPoint(lon=points[i].lon, lat=points[i].lat, elevation=elev))
        adjusted.append(points[-1])  # last point kept exactly (endpoint stays on ground)
        return adjusted

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

    def _nudge_join_toward_bearing(self, points: list[PathPoint], incoming_bearing: Optional[float]) -> list[PathPoint]:
        """Gently rotate the first join-region points toward the incoming heading.

        Keeps the anchor (points[0]) fixed and, for each early point within
        MOMENTUM_DECAY_M of it, blends the point's bearing-from-anchor toward
        `incoming_bearing` by a factor that decays to zero across the region. The
        per-point rotation is clamped to MAX_TURN_PER_STEP_DEG so the join can never
        swing wildly (no cliff dives / reversed turns). Distance-from-anchor and
        elevation are preserved; the caller re-samples elevation from the DEM after.
        Returns points unchanged when there is no incoming heading.
        """
        if incoming_bearing is None or len(points) < 3:
            return points

        anchor = points[0]
        nudged: list[PathPoint] = [anchor]
        for pt in points[1:]:
            dist = GeoCalculator.haversine_distance_m(lat1=anchor.lat, lon1=anchor.lon, lat2=pt.lat, lon2=pt.lon)
            if dist >= PlannerConfig.MOMENTUM_DECAY_M or dist < 0.1:
                nudged.append(pt)
                continue

            bearing = GeoCalculator.initial_bearing_deg(lon1=anchor.lon, lat1=anchor.lat, lon2=pt.lon, lat2=pt.lat)
            diff = (incoming_bearing - bearing + 180) % 360 - 180  # signed shortest turn to incoming
            weight = 1.0 - dist / PlannerConfig.MOMENTUM_DECAY_M  # full at anchor → 0 at decay distance
            rotate = max(-PathConfig.MAX_TURN_PER_STEP_DEG, min(PathConfig.MAX_TURN_PER_STEP_DEG, diff * weight))
            new_lon, new_lat = GeoCalculator.destination(
                lon=anchor.lon, lat=anchor.lat, bearing_deg=(bearing + rotate) % 360, distance_m=dist
            )
            nudged.append(PathPoint(lon=new_lon, lat=new_lat, elevation=pt.elevation))
        return nudged

    def _smooth_path_spline(
        self,
        points: list[PathPoint],
        target_grade_pct: float,
        step_m: float = 7.0,
        incoming_bearing: Optional[float] = None,
    ) -> list[PathPoint]:
        """Smooth grid path using cubic spline interpolation and resample at fixed intervals.

        The grid-based Dijkstra produces staircase paths due to 8-directional movement.
        This method fits a smooth cubic spline through the points and resamples at
        regular intervals, eliminating grid artifacts while preserving the overall shape.

        Args:
            points: Raw grid path points
            target_grade_pct: Target grade - gentler targets get more aggressive smoothing
            step_m: Output point spacing in meters (default 7m)
            incoming_bearing: Optional heading (deg) the path arrives with. When set,
                the join region (first MOMENTUM_DECAY_M) is gently nudged toward that
                heading BEFORE the spline pass, so the path leaves the node in-line
                instead of kinking. The nudge is bounded (MAX_TURN_PER_STEP_DEG),
                decays to zero over the region, and never moves the anchor point;
                elevations are still DEM-sourced below, so it cannot invent terrain.

        Returns:
            Smoothed path with regular point spacing and DEM-sampled elevations.
        """
        if len(points) < 4:
            return points

        points = self._nudge_join_toward_bearing(points=points, incoming_bearing=incoming_bearing)

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
        difficulty = TerrainAnalyzer.classify_difficulty(slope_pct=target_grade_pct)
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
            new_lon, new_lat, new_elev_approx = splev(new_dists, tck)

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
