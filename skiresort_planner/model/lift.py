"""Lift - Ski lift connecting two nodes.

A Lift provides uphill transport between two nodes.
Multiple lift types supported: surface_lift, chairlift, gondola, aerial_tram.

Pylon positions are calculated using 3-phase catenary simulation:
- Phase 1: Place pylons where cable clearance < min_clearance_m
- Phase 2: Enforce max_spacing_m by adding midpoint pylons
- Phase 3: Re-check clearance after spacing pylons are added

Reference: DETAILS.md
"""

import logging
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from skiresort_planner.constants import EntityPrefixes, GeometricTuningConfig, LiftConfig, LiftType, NameConfig
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.model.node_connected import NodeConnected
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_smoothing import simplify_path_points_vertical
from skiresort_planner.model.pylon import Pylon

if TYPE_CHECKING:
    from skiresort_planner.core.dem_service import DEMService
    from skiresort_planner.model.node import Node

logger = logging.getLogger(__name__)


@dataclass
class Lift(NodeConnected):
    """A ski lift connecting two nodes.

    Lifts store only IDs of start/end nodes. Derived properties
    (vertical_rise, length, avg_slope) are computed on demand.
    Pylon positions are calculated via catenary simulation and stored
    for persistence.

    Attributes:
        id: Unique identifier (e.g., "L1", "L2", ...)
        name: Display name with number prefix
        start_node_id: ID of the bottom station node
        end_node_id: ID of the top station node
        lift_type: Type of lift (surface_lift, chairlift, gondola, aerial_tram)
        pylons: List of Pylon objects representing support structures

    Example:
        lift = Lift(
            id="L1",
            name="1 (Alpine Express)",
            start_node_id="N1",
            end_node_id="N5",
            lift_type="chairlift",
        )
    """

    id: str
    name: str
    start_node_id: str
    end_node_id: str
    lift_type: str
    terrain_points: list[PathPoint]
    pylons: list[Pylon]
    cable_points: list[PathPoint]
    source: str | None = None  # provenance tag (e.g. EntitySource.OSM)

    @property
    def number(self) -> int:
        """Lift number derived from ID."""
        return Lift.number_from_id(lift_id=self.id)

    @staticmethod
    def number_from_id(lift_id: str) -> int:
        """Extract lift number from lift ID.

        Args:
            lift_id: Lift ID (e.g., "L1", "L5")

        Returns:
            Numeric part of the ID.
        """
        return int(lift_id[len(EntityPrefixes.LIFT) :])

    @staticmethod
    def sample_terrain(
        start_node: "Node",
        end_node: "Node",
        dem: "DEMService",
    ) -> list[PathPoint]:
        """Sample terrain points along the lift path.

        Args:
            start_node: Bottom station node
            end_node: Top station node
            dem: DEM service for elevation lookup

        Returns:
            List of PathPoint with terrain elevations along the lift.
        """
        total_dist = GeoCalculator.haversine_distance_m(
            lat1=start_node.lat,
            lon1=start_node.lon,
            lat2=end_node.lat,
            lon2=end_node.lon,
        )

        # If the lift is very short, just return start and end points
        if total_dist < LiftConfig.TERRAIN_SAMPLE_STEP_M:
            return [
                PathPoint(lon=start_node.lon, lat=start_node.lat, elevation=start_node.elevation),
                PathPoint(lon=end_node.lon, lat=end_node.lat, elevation=end_node.elevation),
            ]

        n_steps = max(2, int(total_dist / LiftConfig.TERRAIN_SAMPLE_STEP_M))
        brng = GeoCalculator.initial_bearing_deg(
            lon1=start_node.lon,
            lat1=start_node.lat,
            lon2=end_node.lon,
            lat2=end_node.lat,
        )

        points = []
        for i in range(n_steps + 1):
            frac = i / n_steps
            dist = total_dist * frac

            if i == 0:
                lon, lat = start_node.lon, start_node.lat
                elev = start_node.elevation
            elif i == n_steps:
                lon, lat = end_node.lon, end_node.lat
                elev = end_node.elevation
            else:
                lon, lat = GeoCalculator.destination(
                    lon=start_node.lon,
                    lat=start_node.lat,
                    bearing_deg=brng,
                    distance_m=dist,
                )
                # Get elevation from DEM (may be None if outside bounds)
                maybe_elev = dem.get_elevation(lon=lon, lat=lat)
                if maybe_elev is None:
                    # Interpolate if point is outside DEM coverage
                    elev = start_node.elevation + frac * (end_node.elevation - start_node.elevation)
                else:
                    elev = maybe_elev

            points.append(PathPoint(lon=lon, lat=lat, elevation=elev))

        return points

    @staticmethod
    def cable_elevation(
        t: float,
        start_elev: float,
        end_elev: float,
        span_m: float,
        sag_factor: float,
    ) -> float:
        """Calculate cable elevation at fraction t along a span using parabolic sag.

        The cable sags below the straight line connecting anchor points.
        Formula: z(t) = (1-t)*z0 + t*z1 - 4*s*t*(1-t)
        where s = sag_factor * span_m is the maximum sag at midpoint.

        Args:
            t: Fraction along span (0.0 = start, 1.0 = end)
            start_elev: Elevation at start anchor (z0)
            end_elev: Elevation at end anchor (z1)
            span_m: Horizontal distance of span in meters
            sag_factor: Sag factor (typically 0.01-0.02 from LiftConfig)

        Returns:
            Cable elevation at position t.
        """
        max_sag = sag_factor * span_m
        linear_elev = (1 - t) * start_elev + t * end_elev
        sag_at_point = 4 * max_sag * t * (1 - t)
        return linear_elev - sag_at_point

    @staticmethod
    def generate_name(
        lift_type: str,
        lift_id: str,
        length_m: float,
        vertical_rise_m: float,
        avg_bearing: float,
    ) -> str:
        """Generate a creative lift name.

        Args:
            lift_type: Type of lift (surface_lift, chairlift, gondola, aerial_tram)
            lift_id: Lift ID (e.g., "L1")
            length_m: Horizontal length in meters
            vertical_rise_m: Vertical rise in meters
            avg_bearing: Average bearing in degrees

        Returns:
            Creative lift name like "1 (Gams Nord Sesselbahn)"
        """
        lift_number = Lift.number_from_id(lift_id=lift_id)
        prefixes = NameConfig.LIFT_PREFIXES[lift_type]
        prefix = random.choice(prefixes)

        direction = NameConfig.get_compass_direction(bearing_deg=avg_bearing) + " "

        suffixes = NameConfig.LIFT_SUFFIXES[lift_type]
        suffix = random.choice(suffixes)

        length_desc = ""
        if length_m < NameConfig.LENGTH_SHORT_MAX_M:
            length_desc = random.choice(NameConfig.LENGTH_DESCRIPTORS["short"]) + " "
        elif length_m > NameConfig.LENGTH_LONG_MIN_M:
            length_desc = random.choice(NameConfig.LENGTH_DESCRIPTORS["long"]) + " "

        name = f"{length_desc}{prefix} {direction}{suffix}"

        if vertical_rise_m > NameConfig.SUMMIT_RISE_M:
            name = f"{prefix} {direction}Gipfel {suffix}"

        return f"{lift_number} ({name.strip()})"

    def __post_init__(self) -> None:
        """Validate lift type and required data."""
        if self.lift_type not in LiftConfig.TYPES:
            raise ValueError(f"Invalid lift_type '{self.lift_type}'. Must be one of: {LiftConfig.TYPES}")
        if len(self.terrain_points) < 2:
            raise ValueError(f"Lift {self.id} must have at least 2 terrain_points, got {len(self.terrain_points)}")
        if len(self.cable_points) < 2:
            raise ValueError(f"Lift {self.id} must have at least 2 cable_points, got {len(self.cable_points)}")

    @staticmethod
    def finalize_geometry(
        terrain_points: list[PathPoint], lift_type: str
    ) -> tuple[list[PathPoint], list["Pylon"], list[PathPoint]]:
        """Single source of truth for a lift's geometry: vertical-DP the terrain (distance,elevation)
        profile, then recompute pylons + cable. Pure (no DEM/nodes) and idempotent — build == JSON load.

        Returns:
            Tuple of (thinned_terrain_points, pylons, cable_points).
        """
        thinned = simplify_path_points_vertical(
            points=terrain_points, tolerance_m=GeometricTuningConfig.TERRAIN_SIMPLIFY_TOLERANCE_M
        )
        assert len(thinned) >= 2, f"vertical-DP collapsed terrain below 2 points: {len(thinned)}"
        total_m = PathPoint.total_length_m(points=thinned)
        assert total_m > 0, f"finalize_geometry: thinned terrain has non-positive length {total_m}"
        pylons = Lift.calculate_pylons(terrain_points=thinned, lift_type=lift_type, total_distance_m=total_m)
        cable_points = Lift.calculate_cable_points(
            terrain_points=thinned,
            pylons=pylons,
            start_elevation=thinned[0].elevation,
            end_elevation=thinned[-1].elevation,
            lift_type=lift_type,
            total_distance_m=total_m,
        )
        assert len(cable_points) >= 2, f"finalize_geometry produced <2 cable points: {len(cable_points)}"
        return thinned, pylons, cable_points

    @staticmethod
    def _compute_type_dependent_data(
        terrain_points: list[PathPoint],
        start_node: "Node",
        end_node: "Node",
        lift_type: str,
        lift_id: str,
    ) -> tuple[str, list[PathPoint], list["Pylon"], list[PathPoint], float]:
        """Single source of truth for name + finalized geometry (terrain thinning, pylons, cable). Used by
        create/update_type/rebuild; geometry is delegated to finalize_geometry so build == JSON load.

        Args:
            terrain_points: Pre-sampled terrain along lift path
            start_node: Bottom station node
            end_node: Top station node
            lift_type: Type of lift
            lift_id: Lift ID for naming (e.g., "L1")

        Returns:
            Tuple of (name, thinned_terrain_points, pylons, cable_points, length_m).
        """
        # Calculate metrics
        length_m = GeoCalculator.haversine_distance_m(
            lat1=start_node.lat,
            lon1=start_node.lon,
            lat2=end_node.lat,
            lon2=end_node.lon,
        )
        vertical_rise_m = end_node.elevation - start_node.elevation

        # Bearing from TOP to BOTTOM (slope-facing direction)
        avg_bearing = GeoCalculator.initial_bearing_deg(
            lon1=end_node.lon,
            lat1=end_node.lat,
            lon2=start_node.lon,
            lat2=start_node.lat,
        )

        # Generate name
        name = Lift.generate_name(
            lift_type=lift_type,
            lift_id=lift_id,
            length_m=length_m,
            vertical_rise_m=vertical_rise_m,
            avg_bearing=avg_bearing,
        )

        # Thin terrain + recompute pylons/cable via the shared geometry finalizer.
        thinned, pylons, cable_points = Lift.finalize_geometry(terrain_points=terrain_points, lift_type=lift_type)

        return name, thinned, pylons, cable_points, length_m

    @classmethod
    def create(
        cls,
        start_node: "Node",
        end_node: "Node",
        dem: "DEMService",
        lift_type: str,
        lift_id: str,
    ) -> "Lift":
        """Factory method to create a complete Lift with all computed data.

        Samples terrain, calculates pylons via catenary simulation, and
        pre-computes cable points for rendering/export.

        Args:
            start_node: Bottom station node
            end_node: Top station node
            dem: DEM service for terrain sampling
            lift_type: Type of lift (surface_lift, chairlift, gondola, aerial_tram)
            lift_id: Unique identifier (e.g., "L1")

        Returns:
            Fully initialized Lift object.
        """
        # Sample terrain along lift path
        terrain_points = cls.sample_terrain(
            start_node=start_node,
            end_node=end_node,
            dem=dem,
        )

        # Compute all type-dependent data via shared helper (thins terrain to final density)
        name, terrain_points, pylons, cable_points, length_m = cls._compute_type_dependent_data(
            terrain_points=terrain_points,
            start_node=start_node,
            end_node=end_node,
            lift_type=lift_type,
            lift_id=lift_id,
        )

        vertical_rise_m = end_node.elevation - start_node.elevation
        logger.info(f"Creating lift: {name}, type={lift_type}, length={length_m:.0f}m, rise={vertical_rise_m:.0f}m")

        return cls(
            id=lift_id,
            name=name,
            start_node_id=start_node.id,
            end_node_id=end_node.id,
            lift_type=lift_type,
            terrain_points=terrain_points,
            pylons=pylons,
            cable_points=cable_points,
        )

    def _resolve_nodes(self, nodes: dict[str, "Node"]) -> tuple["Node", "Node"]:
        """The two station nodes; raises if either id is absent from the graph."""
        return nodes[self.start_node_id], nodes[self.end_node_id]

    def get_vertical_rise(self, nodes: dict[str, "Node"]) -> float:
        """Calculate elevation gain from start to end node.

        Args:
            nodes: Dict of node_id -> Node

        Returns:
            Vertical rise in meters.
        """
        start, end = self._resolve_nodes(nodes)
        return end.elevation - start.elevation

    def get_length_m(self, nodes: dict[str, "Node"]) -> float:
        """Calculate horizontal distance between nodes.

        Args:
            nodes: Dict of node_id -> Node

        Returns:
            Length in meters.
        """
        start, end = self._resolve_nodes(nodes)
        return GeoCalculator.haversine_distance_m(
            lat1=start.lat,
            lon1=start.lon,
            lat2=end.lat,
            lon2=end.lon,
        )

    def center(self, nodes: dict[str, "Node"]) -> tuple[float, float]:
        """(lon, lat) midpoint between the two station nodes.

        Args:
            nodes: Dict of node_id -> Node.

        Returns:
            (lon, lat) midpoint.
        """
        start, end = self._resolve_nodes(nodes)
        return ((start.lon + end.lon) / 2, (start.lat + end.lat) / 2)

    def update_type(self, new_type: str, start_node: "Node", end_node: "Node") -> None:
        """Change lift type and refresh type-dependent geometry.

        Uses _compute_type_dependent_data() to ensure consistency with create().
        Updates: lift_type, pylons, cable_points. The NAME is deliberately kept — a type change
        must not clobber a user's (or OSM's) name; rename it explicitly via the UI if desired.

        Args:
            new_type: New lift type (must be valid from LiftConfig.TYPES)
            start_node: Bottom station node
            end_node: Top station node
        """
        if new_type == self.lift_type:
            logger.debug(f"Lift {self.id} already has type {new_type}, no update needed")
            return  # No change needed

        if new_type not in LiftConfig.TYPES:
            raise ValueError(f"Invalid lift_type '{new_type}'. Must be one of: {LiftConfig.TYPES}")

        self.lift_type = new_type

        # Recompute type-dependent geometry via shared helper; keep the existing name.
        _name, self.terrain_points, self.pylons, self.cable_points, _length = self._compute_type_dependent_data(
            terrain_points=self.terrain_points,
            start_node=start_node,
            end_node=end_node,
            lift_type=new_type,
            lift_id=self.id,
        )

        logger.info(f"Updated lift {self.id} type to {new_type}")

    def rebuild(self, start_node: "Node", end_node: "Node", dem: "DEMService") -> None:
        """Recompute all geometry for moved station endpoints, keeping identity (id, name, type).

        Used when a station node moves (e.g. node-merge): re-samples terrain along the NEW station
        line, then recomputes pylons + cable_points via the same _compute_type_dependent_data helper
        that create()/update_type() use. Unlike update_type (which changes the type and reuses stale
        terrain_points), this refreshes terrain_points from the moved endpoints. The regenerated name
        is discarded — a rebuild must not clobber the existing (user/OSM) name.

        Args:
            start_node: Bottom station node (new position).
            end_node: Top station node (new position).
            dem: DEM service for terrain sampling.
        """
        self.terrain_points = self.sample_terrain(start_node=start_node, end_node=end_node, dem=dem)
        _name, self.terrain_points, self.pylons, self.cable_points, _length = self._compute_type_dependent_data(
            terrain_points=self.terrain_points,
            start_node=start_node,
            end_node=end_node,
            lift_type=self.lift_type,
            lift_id=self.id,
        )
        logger.info(f"Rebuilt lift {self.id} geometry for moved endpoints")

    @staticmethod
    def calculate_pylons(
        terrain_points: list[PathPoint],
        lift_type: str,
        total_distance_m: float,
    ) -> list[Pylon]:
        """Calculate pylon positions via 3-phase catenary sim (place on clearance<min, enforce max-spacing,
        re-check) in DISTANCE-space — spans measured in metres, so uniform (build) or DP-thinned (load) both work.

        Args:
            terrain_points: List of PathPoint sampled along lift path
            lift_type: Type of lift (determines pylon parameters from LiftConfig)
            total_distance_m: Total horizontal distance of lift (validation only; spans use terrain arc)

        Returns:
            List of Pylon objects with calculated positions.
        """
        if len(terrain_points) < 2:
            raise ValueError(f"terrain_points must have at least 2 points, got {len(terrain_points)}")
        if total_distance_m <= 0:
            raise ValueError(f"total_distance_m must be positive, got {total_distance_m}")

        config = cast(dict[str, int | float], LiftConfig.PYLON_CONFIG[LiftType(lift_type)])
        # Run the physics on a fine uniform internal grid so pylon placement resolution is independent
        # of how coarse the stored (vertical-DP thinned) terrain is. The grid is transient.
        grid = Lift._resample_uniform(terrain_points=terrain_points, step_m=LiftConfig.TERRAIN_SAMPLE_STEP_M)
        n = len(grid)
        assert n >= 2, f"resampled grid must have >=2 points, got {n}"
        dists = PathPoint.cumulative_distances(points=grid)  # metres along the terrain polyline
        assert dists[-1] > 0, f"resampled grid has non-positive arc length {dists[-1]}"

        terrain_elevs = [p.elevation for p in grid]
        pylon_height = cast(int, config["pylon_height_m"])
        station_height = cast(int, config["station_height_m"])
        min_spacing_m = cast(int, config["min_spacing_m"])
        min_clearance = cast(int, config["min_clearance_m"])
        sag_factor = cast(float, config["sag_factor"])
        max_spacing_m = cast(int, config["max_spacing_m"])

        # Station cable elevations
        start_cable_elev = terrain_elevs[0] + station_height
        end_cable_elev = terrain_elevs[-1] + station_height

        # Phase 1: Fix all clearance violations
        pylon_indices = Lift._find_clearance_violations(
            start_idx=0,
            end_idx=n - 1,
            start_elev=start_cable_elev,
            end_elev=end_cable_elev,
            pylon_set=set(),
            terrain_elevs=terrain_elevs,
            dists=dists,
            min_spacing_m=min_spacing_m,
            min_clearance=min_clearance,
            pylon_height=pylon_height,
            sag_factor=sag_factor,
        )
        pylon_indices = sorted(set(pylon_indices))

        # Phase 2: Enforce maximum spacing
        pylon_indices = Lift._enforce_max_spacing(
            pylon_indices=pylon_indices,
            n=n,
            dists=dists,
            max_spacing_m=max_spacing_m,
        )

        # Phase 3: Re-check clearance after spacing pylons
        pylon_indices = Lift._recheck_clearance(
            pylon_indices=pylon_indices,
            n=n,
            start_cable_elev=start_cable_elev,
            end_cable_elev=end_cable_elev,
            terrain_elevs=terrain_elevs,
            dists=dists,
            min_spacing_m=min_spacing_m,
            min_clearance=min_clearance,
            pylon_height=pylon_height,
            sag_factor=sag_factor,
        )

        # Convert indices to Pylon objects (distance is the terrain arc length at that grid vertex)
        pylons = []
        for idx in pylon_indices:
            assert 0 <= idx < n, f"pylon index {idx} out of range [0, {n})"
            point = grid[idx]
            pylons.append(
                Pylon(
                    distance_m=dists[idx],
                    lat=point.lat,
                    lon=point.lon,
                    ground_elevation_m=point.elevation,
                    height_m=pylon_height,
                )
            )

        # Guard the min-spacing invariant: consecutive PYLONS never sit closer than min_spacing (the grid
        # step floors it). Station↔station on a lift shorter than min_spacing is fine — that's no pylons.
        pylon_dists = [dists[i] for i in pylon_indices]
        adjacent_gaps = [pylon_dists[i + 1] - pylon_dists[i] for i in range(len(pylon_dists) - 1)]
        assert all(g >= min(min_spacing_m, LiftConfig.TERRAIN_SAMPLE_STEP_M) - 1e-6 for g in adjacent_gaps), (
            f"adjacent pylons closer than min_spacing {min_spacing_m}m ({lift_type}): gaps {adjacent_gaps}"
        )
        return pylons

    @staticmethod
    def _cable_elev_at_idx(
        start_idx: int,
        end_idx: int,
        z0: float,
        z1: float,
        idx: int,
        dists: list[float],
        sag_factor: float,
    ) -> float:
        """Cable elevation at a terrain index using Lift.cable_elevation, in distance-space."""
        span_m = dists[end_idx] - dists[start_idx]
        if span_m <= 0:
            return z0
        t = (dists[idx] - dists[start_idx]) / span_m
        assert 0.0 <= t <= 1.0, f"_cable_elev_at_idx: t={t} out of [0,1] (idx={idx} in [{start_idx},{end_idx}])"
        return Lift.cable_elevation(t=t, start_elev=z0, end_elev=z1, span_m=span_m, sag_factor=sag_factor)

    @staticmethod
    def _find_clearance_violations(
        start_idx: int,
        end_idx: int,
        start_elev: float,
        end_elev: float,
        pylon_set: set[int],
        terrain_elevs: list[float],
        dists: list[float],
        min_spacing_m: float,
        min_clearance: int,
        pylon_height: int,
        sag_factor: float,
    ) -> list[int]:
        """Recursively find terrain indices where cable clearance is below minimum (distance-space)."""
        if dists[end_idx] - dists[start_idx] < min_spacing_m * 2:
            return []

        worst_violation: float = 0.0
        worst_idx = -1

        for i in range(start_idx + 1, end_idx):
            # Cable at pylons can not be a violation
            if i in pylon_set:
                continue
            # Honour min spacing to either anchor in metres, not index count.
            if dists[i] - dists[start_idx] < min_spacing_m or dists[end_idx] - dists[i] < min_spacing_m:
                continue
            cable_elev = Lift._cable_elev_at_idx(
                start_idx=start_idx,
                end_idx=end_idx,
                z0=start_elev,
                z1=end_elev,
                idx=i,
                dists=dists,
                sag_factor=sag_factor,
            )
            clearance = cable_elev - terrain_elevs[i]
            violation = min_clearance - clearance

            if violation > worst_violation:
                worst_violation = violation
                worst_idx = i

        if worst_violation <= 0 or worst_idx < 0:
            return []

        pylon_top_elev = terrain_elevs[worst_idx] + pylon_height
        new_pylon_set = pylon_set | {worst_idx}

        left_pylons = Lift._find_clearance_violations(
            start_idx=start_idx,
            end_idx=worst_idx,
            start_elev=start_elev,
            end_elev=pylon_top_elev,
            pylon_set=new_pylon_set,
            terrain_elevs=terrain_elevs,
            dists=dists,
            min_spacing_m=min_spacing_m,
            min_clearance=min_clearance,
            pylon_height=pylon_height,
            sag_factor=sag_factor,
        )
        right_pylons = Lift._find_clearance_violations(
            start_idx=worst_idx,
            end_idx=end_idx,
            start_elev=pylon_top_elev,
            end_elev=end_elev,
            pylon_set=new_pylon_set,
            terrain_elevs=terrain_elevs,
            dists=dists,
            min_spacing_m=min_spacing_m,
            min_clearance=min_clearance,
            pylon_height=pylon_height,
            sag_factor=sag_factor,
        )

        return left_pylons + [worst_idx] + right_pylons

    @staticmethod
    def _nearest_index_to_distance(dists: list[float], target_m: float, lo: int, hi: int) -> int:
        """Terrain index in the OPEN interval (lo, hi) whose arc distance is nearest target_m, or -1
        if the interval has no interior index (can happen on coarse DP-thinned terrain).
        """
        best_idx = -1
        best_gap = float("inf")
        for i in range(lo + 1, hi):
            gap = abs(dists[i] - target_m)
            if gap < best_gap:
                best_gap = gap
                best_idx = i
        return best_idx

    @staticmethod
    def _enforce_max_spacing(
        pylon_indices: list[int],
        n: int,
        dists: list[float],
        max_spacing_m: int,
    ) -> list[int]:
        """Phase 2: insert midpoint pylons so no span exceeds max_spacing (distance-space).

        The midpoint is the terrain vertex nearest the span's distance-midpoint. On coarse terrain a
        span may have no interior vertex to host one — then it is left as-is (bounded, non-fatal).
        """
        for _ in range(20):  # Safety limit
            anchors = [0] + sorted(pylon_indices) + [n - 1]
            new_spacing_pylons = []

            for seg_idx in range(len(anchors) - 1):
                seg_start = anchors[seg_idx]
                seg_end = anchors[seg_idx + 1]

                if dists[seg_end] - dists[seg_start] > max_spacing_m:
                    mid_target = (dists[seg_start] + dists[seg_end]) / 2
                    mid_idx = Lift._nearest_index_to_distance(
                        dists=dists, target_m=mid_target, lo=seg_start, hi=seg_end
                    )
                    if mid_idx >= 0 and mid_idx not in pylon_indices:
                        assert seg_start < mid_idx < seg_end, (
                            f"midpoint pylon {mid_idx} not strictly inside span ({seg_start}, {seg_end})"
                        )
                        new_spacing_pylons.append(mid_idx)

            if not new_spacing_pylons:
                break
            pylon_indices = sorted(set(pylon_indices + new_spacing_pylons))

        return pylon_indices

    @staticmethod
    def _recheck_clearance(
        pylon_indices: list[int],
        n: int,
        start_cable_elev: float,
        end_cable_elev: float,
        terrain_elevs: list[float],
        dists: list[float],
        min_spacing_m: float,
        min_clearance: int,
        pylon_height: int,
        sag_factor: float,
    ) -> list[int]:
        """Phase 3: re-check clearance per span after spacing pylons added (distance-space)."""
        pylon_set = set(pylon_indices)
        anchors = [0] + sorted(pylon_indices) + [n - 1]
        anchor_elevs = [start_cable_elev]
        for idx in sorted(pylon_indices):
            anchor_elevs.append(terrain_elevs[idx] + pylon_height)
        anchor_elevs.append(end_cable_elev)

        new_clearance_pylons = []
        for seg_idx in range(len(anchors) - 1):
            seg_start = anchors[seg_idx]
            seg_end = anchors[seg_idx + 1]
            seg_start_elev = anchor_elevs[seg_idx]
            seg_end_elev = anchor_elevs[seg_idx + 1]

            additional = Lift._find_clearance_violations(
                start_idx=seg_start,
                end_idx=seg_end,
                start_elev=seg_start_elev,
                end_elev=seg_end_elev,
                pylon_set=pylon_set,
                terrain_elevs=terrain_elevs,
                dists=dists,
                min_spacing_m=min_spacing_m,
                min_clearance=min_clearance,
                pylon_height=pylon_height,
                sag_factor=sag_factor,
            )
            new_clearance_pylons.extend(additional)

        if new_clearance_pylons:
            pylon_indices = sorted(set(pylon_indices + new_clearance_pylons))

        return pylon_indices

    @staticmethod
    def calculate_cable_points(
        terrain_points: list[PathPoint],
        pylons: list[Pylon],
        start_elevation: float,
        end_elevation: float,
        lift_type: str,
        total_distance_m: float,
    ) -> list[PathPoint]:
        """Calculate cable points along the lift path with parabolic sag between anchors (stations +
        pylons). Pre-computes cable positions for efficient rendering and GPX export.

        Args:
            terrain_points: Terrain points along the lift path (for lat/lon interpolation)
            pylons: Calculated pylon positions
            start_elevation: Bottom station ground elevation
            end_elevation: Top station ground elevation
            lift_type: Type of lift for config lookup
            total_distance_m: Total horizontal distance of lift

        Returns:
            List of PathPoint representing cable positions.
        """
        if len(terrain_points) < 2:
            raise ValueError(f"terrain_points must have at least 2 points, got {len(terrain_points)}")
        if total_distance_m <= 0:
            raise ValueError(f"total_distance_m must be positive, got {total_distance_m}")

        config = cast(
            dict[str, int | float | None],
            LiftConfig.PYLON_CONFIG[LiftType(lift_type)],  # strict: lift_type is validated
        )
        station_height = cast(int, config["station_height_m"])
        sag_factor = cast(float, config["sag_factor"])

        # Build anchor points: [stations + all pylons]
        anchor_x = [0.0]  # Bottom station
        anchor_y = [start_elevation + station_height]

        for pylon in pylons:
            assert 0 < pylon.distance_m < total_distance_m, (
                f"pylon distance {pylon.distance_m} outside lift (0, {total_distance_m})"
            )
            anchor_x.append(pylon.distance_m)
            anchor_y.append(pylon.top_elevation_m)

        anchor_x.append(total_distance_m)  # Top station
        anchor_y.append(end_elevation + station_height)

        # Sort anchor points by distance
        anchor_sorted = sorted(zip(anchor_x, anchor_y, strict=True), key=lambda p: p[0])
        anchor_x = [p[0] for p in anchor_sorted]
        anchor_y = [p[1] for p in anchor_sorted]

        # Generate cable curve with sag for each segment
        cable_points = []
        terrain_dists = PathPoint.cumulative_distances(terrain_points)
        assert terrain_dists[-1] > 0, f"terrain polyline has non-positive length {terrain_dists[-1]}"

        for seg_idx in range(len(anchor_x) - 1):
            start_x = anchor_x[seg_idx]
            end_x = anchor_x[seg_idx + 1]
            start_y = anchor_y[seg_idx]
            end_y = anchor_y[seg_idx + 1]
            span = end_x - start_x

            if span <= 0:
                continue

            # Curvature-adaptive sampling: the cable is a parabola with max sag = sag_factor*span, so an
            # n-segment chord deviates by ~sag/n². Pick n to hold that under CABLE_SAG_TOLERANCE_M — short
            # low-sag spans get 1-2 points, long deep-sag spans get more. Anchors (i=0 / i=n) stay pinned.
            max_sag = sag_factor * span
            n_seg_points = max(1, math.ceil(math.sqrt(max_sag / LiftConfig.CABLE_SAG_TOLERANCE_M)))
            assert n_seg_points >= 1, f"n_seg_points must be >=1, got {n_seg_points} (span={span})"
            for i in range(n_seg_points + 1):
                # Skip duplicate at segment boundaries (except first segment start)
                if seg_idx > 0 and i == 0:
                    continue

                x = start_x + (end_x - start_x) * i / n_seg_points
                frac = i / n_seg_points
                cable_elev = Lift.cable_elevation(
                    t=frac,
                    start_elev=start_y,
                    end_elev=end_y,
                    span_m=span,
                    sag_factor=sag_factor,
                )
                pt = PathPoint.interpolate_at_distance(points=terrain_points, distances=terrain_dists, target_m=x)
                cable_points.append(PathPoint(lon=pt.lon, lat=pt.lat, elevation=cable_elev))

        return cable_points

    @staticmethod
    def _resample_uniform(terrain_points: list[PathPoint], step_m: float) -> list[PathPoint]:
        """Resample terrain onto a uniform step_m grid (endpoints included) by distance interpolation, so
        the pylon physics runs at a fixed resolution independent of the coarse stored terrain. Transient.
        """
        assert len(terrain_points) >= 2, f"_resample_uniform needs >=2 terrain points, got {len(terrain_points)}"
        assert step_m > 0, f"_resample_uniform step_m must be positive, got {step_m}"
        dists = PathPoint.cumulative_distances(terrain_points)
        total = dists[-1]
        n_steps = max(1, round(total / step_m))
        grid = [
            PathPoint.interpolate_at_distance(points=terrain_points, distances=dists, target_m=total * i / n_steps)
            for i in range(n_steps + 1)
        ]
        assert grid[0].elevation == terrain_points[0].elevation, "resample must preserve the start station"
        return grid

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Lift":
        """Create Lift from dictionary.

        All fields are required - raises KeyError if missing.
        """
        return cls(
            id=cast(str, data["id"]),
            name=cast(str, data["name"]),
            start_node_id=cast(str, data["start_node_id"]),
            end_node_id=cast(str, data["end_node_id"]),
            lift_type=cast(str, data["lift_type"]),
            terrain_points=[PathPoint(**p) for p in cast(list[dict[str, float]], data["terrain_points"])],
            pylons=[
                Pylon(
                    distance_m=float(p["distance_m"]),
                    lat=float(p["lat"]),
                    lon=float(p["lon"]),
                    ground_elevation_m=float(p["ground_elevation_m"]),
                    height_m=float(p["height_m"]),
                )
                for p in cast(list[dict[str, float]], data["pylons"])
            ],
            cable_points=[PathPoint(**p) for p in cast(list[dict[str, float]], data["cable_points"])],
            source=cast("str | None", data.get("source")),
        )

    def __repr__(self) -> str:
        return f"Lift({self.id}, {self.lift_type})"
