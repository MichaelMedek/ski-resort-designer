"""Connection Path Planner - Grid-based Dijkstra for generating paths to specific target points.

Domain-agnostic: connects a start to a target while holding a caller-supplied grade (knows nothing
of pistes/roads). SciPy sparse Dijkstra + cubic-spline smoothing. Reference: DETAILS.md Section 7.
"""

import logging
from dataclasses import dataclass
from enum import StrEnum
from math import exp

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from shapely.geometry import LineString

from skiresort_planner.constants import GeometricTuningConfig, PlannerConfig
from skiresort_planner.core.dem_service import DEMService
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_smoothing import smooth_proposal_points
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

    Steps:
    1. Grid sized from the REQUIRED grade-holding length L=100·drop/g (so serpentines fit)
    2. Sparse graph over (node × lateral heading), radius-R coprime neighborhood, edges weighted by
       slope-deviation cost + a switchback-reversal penalty (lateral momentum)
    3. SciPy Dijkstra for the minimum-cost path
    4. Cubic-spline smooth (light factor) and resample at regular DEM-sampled intervals

    Cost = distance × exp(|actual−target|/σ) × against_penalty (+ reversal). σ=COST_SIGMA (lower =
    stricter); against_penalty penalizes running the wrong way (no loops); reversal favours few large
    switchbacks. See GeometricTuningConfig for tunables.
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
        smoothing_factor: float,
        gradient_mode: GradientMode = GradientMode.DOWNHILL,
    ) -> ProposedPathSegment | None:
        """Plan a least-cost path that holds a target grade from start to target.

        Args:
            start_lon/lat/elevation, target_lon/lat/elevation: the two endpoints.
            target_grade_pct: signed grade the path aims to hold (+ = descending).
            smoothing_factor: spline smoothing budget — the caller passes the kind's finish factor
                (slope/road) so the proposal previews the finished shape.
            gradient_mode: DOWNHILL (default) forces net-descent and penalizes climbing
                (skiable pistes); UPHILL forces net-ascent and penalizes descending. The
                monotonicity is what prevents the path from looping.

        Returns:
            ProposedPathSegment if path found, None otherwise.
        """
        # The segment must actually run in its mode's direction (net drop / net climb).
        net_drop = start_elevation - target_elevation
        if gradient_mode == GradientMode.DOWNHILL and net_drop <= 0:
            logger.debug(
                f"plan: no path — DOWNHILL mode but net_drop={net_drop:.1f}m <= 0 "
                f"(start_elev={start_elevation:.0f}m, target_elev={target_elevation:.0f}m)"
            )
            return None
        if gradient_mode == GradientMode.UPHILL and net_drop >= 0:
            logger.debug(
                f"plan: no path — UPHILL mode but net_drop={net_drop:.1f}m >= 0 "
                f"(start_elev={start_elevation:.0f}m, target_elev={target_elevation:.0f}m)"
            )
            return None

        direct_distance_m = GeoCalculator.haversine_distance_m(
            lat1=start_lat, lon1=start_lon, lat2=target_lat, lon2=target_lon
        )

        if direct_distance_m < GeometricTuningConfig.STEP_SIZE_M:
            logger.debug(
                f"plan: no path — direct_distance={direct_distance_m:.1f}m "
                f"< STEP_SIZE_M={GeometricTuningConfig.STEP_SIZE_M:.1f}m "
                f"from ({start_lon:.5f}, {start_lat:.5f}) to ({target_lon:.5f}, {target_lat:.5f})"
            )
            return None

        # Build the search grid, sized from the required grade-holding path length (not the straight
        # start→target distance) so gentle-grade serpentines on steep ground physically fit.
        elevations, lons, lats, start_node, target_node, grid_res_m = self._build_grid(
            start_lon=start_lon,
            start_lat=start_lat,
            target_lon=target_lon,
            target_lat=target_lat,
            direct_distance=direct_distance_m,
            net_drop=net_drop,
            target_grade_pct=target_grade_pct,
        )

        # Search the (node × heading) graph for the least-cost grade-holding route.
        path_nodes, _, _ = self._graph_dijkstra(
            elevations=elevations,
            start=start_node,
            target=target_node,
            target_grade_pct=target_grade_pct,
            gradient_mode=gradient_mode,
            grid_res_m=grid_res_m,
        )

        if path_nodes is None:
            logger.debug(
                f"plan: no path — Dijkstra found no route holding target_grade={target_grade_pct:.1f}% "
                f"({gradient_mode}) from ({start_lon:.5f}, {start_lat:.5f}) to ({target_lon:.5f}, {target_lat:.5f})"
            )
            return None

        # Convert grid path to PathPoints
        raw_points = self._path_to_points(
            path_nodes=path_nodes,
            elevations=elevations,
            lons=lons,
            lats=lats,
        )

        # Smooth + DEM-requery. Capped at PLANNER_SMOOTHING_FACTOR: a heavy finish factor over the switchback
        # apexes over-rounds (shortens the path) and overshoots vertically across the gaps (dips below ground).
        points = smooth_proposal_points(
            points=raw_points,
            smoothing_factor=min(smoothing_factor, GeometricTuningConfig.PLANNER_SMOOTHING_FACTOR),
            step_m=GeometricTuningConfig.RESAMPLE_STEP_M,
            elevation_fn=self.dem.get_elevation,
        )

        # Quality gate: a self-crossing polyline is a degenerate route (over-tight switchback), so refuse
        # it rather than propose a tangled path — the caller falls back to a straighter alternative.
        if self._self_intersects(points):
            logger.debug(
                f"plan: no path — smoothed route self-intersects (target_grade={target_grade_pct:.1f}%, "
                f"{gradient_mode}) from ({start_lon:.5f}, {start_lat:.5f}) to ({target_lon:.5f}, {target_lat:.5f})"
            )
            return None

        return ProposedPathSegment(
            points=points,
            target_slope_pct=target_grade_pct,
            is_connector=True,
        )

    @staticmethod
    def _self_intersects(points: list[PathPoint]) -> bool:
        """True if the horizontal polyline crosses itself (projected to a local metre frame)."""
        if len(points) < 4:
            return False
        lon0, lat0 = points[0].lon, points[0].lat
        m_per_deg_lon, m_per_deg_lat = GeoCalculator.meters_per_degree(lat=lat0)
        xy = [((p.lon - lon0) * m_per_deg_lon, (p.lat - lat0) * m_per_deg_lat) for p in points]
        return not LineString(xy).is_simple

    def _build_grid(
        self,
        start_lon: float,
        start_lat: float,
        target_lon: float,
        target_lat: float,
        direct_distance: float,
        net_drop: float,
        target_grade_pct: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], GridNode, GridNode, float]:
        """Build the elevation grid (metres lattice) over the search area, sized by `_grid_extents`.

        Vectorized: cell lon/lats via a two-step `destination` (col along bearing+90, row along bearing),
        one batched `get_elevations`. Raises if any cell is off-DEM (no-missing-data invariant).
        """
        along_m, across_m, res = self._grid_extents(
            direct_distance=direct_distance, net_drop=net_drop, target_grade_pct=target_grade_pct
        )
        n_rows = int(along_m / res) + 1
        n_cols = int(across_m / res) + 1

        # Center point
        center_lon = (start_lon + target_lon) / 2
        center_lat = (start_lat + target_lat) / 2

        # Bearing from center to target for grid orientation (the `along` axis follows the chord)
        bearing = GeoCalculator.initial_bearing_deg(lon1=center_lon, lat1=center_lat, lon2=target_lon, lat2=target_lat)

        # Origin = back off half the ACTUAL laid extent per axis, so the chord midpoint stays centred.
        origin_lon, origin_lat = GeoCalculator.destination(
            lon=center_lon,
            lat=center_lat,
            bearing_deg=(bearing + 180) % 360,
            distance_m=(n_rows - 1) * res / 2,
        )
        origin_lon, origin_lat = GeoCalculator.destination(
            lon=origin_lon,
            lat=origin_lat,
            bearing_deg=(bearing - 90) % 360,
            distance_m=(n_cols - 1) * res / 2,
        )

        # Step 1: one start per column (col·res along bearing+90); step 2: row·res along bearing.
        col_dist = np.arange(n_cols, dtype=np.float64) * res
        row_dist = np.arange(n_rows, dtype=np.float64) * res
        col_lon, col_lat = GeoCalculator.destination_vec(origin_lon, origin_lat, (bearing + 90) % 360, col_dist)
        lons, lats = GeoCalculator.destination_vec(col_lon[None, :], col_lat[None, :], bearing, row_dist[:, None])

        elevations = self.dem.get_elevations(lons.ravel(), lats.ravel()).reshape(n_rows, n_cols)
        if np.isnan(elevations).any():
            r, c = (int(i) for i in np.argwhere(np.isnan(elevations))[0])
            raise RuntimeError(
                f"DEM returned None for grid point at row={r}, col={c} "
                f"(lon={lons[r, c]}, lat={lats[r, c]}), cannot build grid with missing elevation data"
            )

        # Find start and target nodes
        start_node = self._find_nearest_node(target_lon=start_lon, target_lat=start_lat, lons=lons, lats=lats)
        target_node = self._find_nearest_node(target_lon=target_lon, target_lat=target_lat, lons=lons, lats=lats)

        return elevations, lons, lats, start_node, target_node, res

    @staticmethod
    def _grid_extents(direct_distance: float, net_drop: float, target_grade_pct: float) -> tuple[float, float, float]:
        """Grid (along_m, across_m, resolution_m) sized to hold a grade-`target_grade_pct` path.

        L=100·|drop|/|g| sheds the drop at grade g; over a shorter chord it bows by sqrt((L/2)²−(chord/2)²),
        so across=2·bow. Below MIN_GRADE_PCT_FOR_LENGTH the formula diverges → fall back to the chord.
        Resolution = L/DIVISOR (floored), coarsened so neither axis exceeds MAX_GRID_SIZE cells.
        """
        g = abs(target_grade_pct)
        if g >= GeometricTuningConfig.MIN_GRADE_PCT_FOR_LENGTH:
            required_len = 100.0 * abs(net_drop) / g
        else:
            required_len = direct_distance
        half_chord = direct_distance / 2.0
        bow = float(np.sqrt(max((required_len / 2.0) ** 2 - half_chord**2, 0.0)))

        along_m = direct_distance * GeometricTuningConfig.GRID_ALONG_MARGIN + GeometricTuningConfig.GRID_PADDING_M
        across_m = min(
            max(2.0 * bow + GeometricTuningConfig.GRID_PADDING_M, GeometricTuningConfig.GRID_ACROSS_MIN_M),
            GeometricTuningConfig.GRID_ACROSS_MAX_M,
        )

        res = max(required_len / GeometricTuningConfig.GRID_RES_DIVISOR, GeometricTuningConfig.GRID_RES_MIN_M)
        res = max(res, max(along_m, across_m) / (GeometricTuningConfig.MAX_GRID_SIZE - 1))  # cap cells
        assert res > 0, f"grid resolution must be positive, got {res} (along={along_m}, across={across_m})"
        return along_m, across_m, res

    def _find_nearest_node(
        self,
        target_lon: float,
        target_lat: float,
        lons: npt.NDArray[np.float64] | list[list[float]],
        lats: npt.NDArray[np.float64] | list[list[float]],
    ) -> GridNode:
        """Grid node nearest to the target — vectorized haversine + argmin (row-major first-min, the
        same tie-break as the strict-`<` scan). Accepts numpy arrays or nested lists.
        """
        lons_a = np.asarray(lons, dtype=np.float64)
        lats_a = np.asarray(lats, dtype=np.float64)
        assert lons_a.size, (
            f"_find_nearest_node: empty grid for target ({target_lon}, {target_lat}) — must be non-empty by construction"
        )
        dist = GeoCalculator.haversine_vec(lats_a, lons_a, target_lat, target_lon)
        row, col = (int(i) for i in np.unravel_index(int(np.argmin(dist)), dist.shape))
        return GridNode(row=row, col=col)

    def _graph_dijkstra(
        self,
        elevations: npt.NDArray[np.float64] | list[list[float]],
        start: GridNode,
        target: GridNode,
        target_grade_pct: float,
        gradient_mode: GradientMode = GradientMode.DOWNHILL,
        grid_res_m: float = GeometricTuningConfig.GRID_RES_MIN_M,
    ) -> tuple[list[GridNode] | None, int, int]:
        """Least-cost path via SciPy Dijkstra with lateral momentum.

        State = node × heading {left, straight, right}; a heading REVERSAL (switchback) costs
        SWITCHBACK_REVERSAL_PENALTY×cell, favouring few large switchbacks over a sawtooth. Uniform
        lattice → an offset's horizontal distance is the scalar res·√(dr²+dc²), no per-cell geodesy.
        """
        elev = np.asarray(elevations, dtype=np.float64)
        n_rows, n_cols = elev.shape
        assert n_cols > 0, f"elevations[0] is empty; grid has {n_rows} rows but 0 columns"
        N = n_rows * n_cols
        sigma = GeometricTuningConfig.COST_SIGMA
        reversal_cost = GeometricTuningConfig.SWITCHBACK_REVERSAL_PENALTY * grid_res_m
        n_states = N * 3  # state = node_id*3 + heading (0=left dc<0, 1=straight dc==0, 2=right dc>0)
        ids = np.arange(N).reshape(n_rows, n_cols)
        prev_h = np.array([0, 1, 2])  # the 3 possible incoming headings

        row_parts: list[npt.NDArray[np.int64]] = []
        col_parts: list[npt.NDArray[np.int64]] = []
        data_parts: list[npt.NDArray[np.float64]] = []
        # One masked pass per neighbor offset; the slice windows replace an in-bounds check.
        for dr, dc in PlannerConfig.NEIGHBORS:
            r0, r1 = max(0, -dr), n_rows - max(0, dr)
            c0, c1 = max(0, -dc), n_cols - max(0, dc)
            if r1 <= r0 or c1 <= c0:
                continue
            from_elev = elev[r0:r1, c0:c1]
            to_elev = elev[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
            horiz = grid_res_m * float(np.hypot(dr, dc))  # uniform lattice → scalar hop distance
            mask = ~np.isnan(from_elev) & ~np.isnan(to_elev)
            # A wildly off-grade edge's exp() overflows to +inf — a forbidden edge (scipy reads inf
            # as no edge), not a bug; ignore the overflow.
            with np.errstate(over="ignore"):
                actual = (from_elev - to_elev) / horiz * 100  # positive = downhill
                grade_cost = np.exp(np.abs(actual - target_grade_pct) / sigma)
                wrong_way = actual < 0 if gradient_mode == GradientMode.DOWNHILL else actual > 0
                against = np.where(wrong_way, np.exp(np.abs(actual) / sigma), 1.0)
                base_cost = (horiz * grade_cost * against)[mask]
            from_ids = ids[r0:r1, c0:c1][mask]
            to_ids = ids[r0 + dr : r1 + dr, c0 + dc : c1 + dc][mask]
            new_heading = 1 if dc == 0 else (2 if dc > 0 else 0)
            # Expand the 3 incoming headings at once: reversal is left(0)↔right(2).
            is_reversal = ((prev_h == 0) & (new_heading == 2)) | ((prev_h == 2) & (new_heading == 0))
            row_parts.append((from_ids[None, :] * 3 + prev_h[:, None]).ravel())
            col_parts.append(np.broadcast_to(to_ids * 3 + new_heading, (3, to_ids.size)).ravel())
            data_parts.append((base_cost[None, :] + np.where(is_reversal, reversal_cost, 0.0)[:, None]).ravel())

        row_arr = np.concatenate(row_parts) if row_parts else np.empty(0, dtype=np.int64)
        if row_arr.size == 0:
            return None, 0, 0
        col_arr = np.concatenate(col_parts)
        data_arr = np.concatenate(data_parts)

        # Assemble the directed state-graph as a sparse CSR matrix for scipy.
        csgraph = csr_matrix(
            (data_arr, (row_arr, col_arr)),
            shape=(n_states, n_states),
            dtype=np.float64,
        )

        # Start straight (heading 1); the target is reachable in any heading, so search from the one
        # start state and pick the cheapest of the target's three sub-states.
        start_state = (start.row * n_cols + start.col) * 3 + 1
        target_node_id = target.row * n_cols + target.col
        target_states = [target_node_id * 3 + h for h in (0, 1, 2)]

        dist, pred = shortest_path(
            csgraph=csgraph,
            method="D",  # Dijkstra (positive weights)
            directed=True,
            indices=start_state,
            return_predecessors=True,
        )

        best_target = min(target_states, key=lambda s: dist[s])
        if np.isinf(dist[best_target]):
            return None, 0, 0

        # Reconstruct path (state → node_id via // 3)
        path_ids: list[int] = []
        current = best_target
        while True:
            path_ids.append(current // 3)
            if current == start_state:
                break
            current = pred[current]
            if current == -9999:
                return None, 0, 0

        path_ids.reverse()

        path_nodes = [GridNode(row=pid // n_cols, col=pid % n_cols) for pid in path_ids]
        assert len(path_nodes) >= 2, (
            f"path_nodes has {len(path_nodes)} nodes; start→target path must have at least 2 endpoints"
        )

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
        """Scalar edge cost mirrored by the vectorized `_graph_dijkstra` (kept for unit tests).

        cost = distance × exp(|actual−target|/σ) × against-mode; against-mode penalizes the wrong way
        (climbing in DOWNHILL / descending in UPHILL), forbidding loops.
        """
        # Horizontal distance
        horiz_dist = GeoCalculator.haversine_distance_m(lat1=from_lat, lon1=from_lon, lat2=to_lat, lon2=to_lon)

        if horiz_dist < 0.1:
            return float("inf")

        actual_grade = (from_elev - to_elev) / horiz_dist * 100  # positive = downhill
        grade_cost = exp(abs(actual_grade - target_grade_pct) / GeometricTuningConfig.COST_SIGMA)

        # Against-mode penalty: wrong-way edges cost exp(|grade|/σ), forbidding loops.
        against_penalty = 1.0
        wrong_way = actual_grade < 0 if gradient_mode == GradientMode.DOWNHILL else actual_grade > 0
        if wrong_way:
            against_penalty = exp(abs(actual_grade) / GeometricTuningConfig.COST_SIGMA)

        return horiz_dist * grade_cost * against_penalty

    def _path_to_points(
        self,
        path_nodes: list[GridNode],
        elevations: npt.NDArray[np.float64] | list[list[float]],
        lons: npt.NDArray[np.float64] | list[list[float]],
        lats: npt.NDArray[np.float64] | list[list[float]],
    ) -> list[PathPoint]:
        """Convert grid path to PathPoints (float()-wrapping so numpy scalars don't leak into PathPoint)."""
        points = []
        for node in path_nodes:
            points.append(
                PathPoint(
                    lon=float(lons[node.row][node.col]),
                    lat=float(lats[node.row][node.col]),
                    elevation=float(elevations[node.row][node.col]),
                )
            )
        return points
