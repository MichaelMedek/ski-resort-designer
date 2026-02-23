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
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from skiresort_planner.constants import EntityPrefixes, LiftConfig, NameConfig
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.pylon import Pylon

if TYPE_CHECKING:
    from skiresort_planner.core.dem_service import DEMService
    from skiresort_planner.model.node import Node

logger = logging.getLogger(__name__)


@dataclass
class Lift:
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
        return int(lift_id[len(EntityPrefixes.LIFT):])

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
                elev = dem.get_elevation(lon=lon, lat=lat)
                if elev is None:
                    # Interpolate if point is outside DEM coverage
                    elev = start_node.elevation + frac * (end_node.elevation - start_node.elevation)

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
            Creative lift name like "1 (Alpine Ridge Express)"
        """
        lift_number = Lift.number_from_id(lift_id=lift_id)
        prefixes = NameConfig.LIFT_PREFIXES[lift_type]
        prefix = random.choice(prefixes)

        direction = NameConfig.get_compass_direction(bearing_deg=avg_bearing) + " "

        suffixes = NameConfig.LIFT_SUFFIXES[lift_type]
        suffix = random.choice(suffixes)

        length_desc = ""
        if length_m < 500:
            length_desc = random.choice(NameConfig.LENGTH_DESCRIPTORS["short"]) + " "
        elif length_m > 1500:
            length_desc = random.choice(NameConfig.LENGTH_DESCRIPTORS["long"]) + " "

        name = f"{length_desc}{prefix} {direction}{suffix}"

        if vertical_rise_m > 500:
            name = f"{prefix} {direction}Summit {suffix}"

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
    def _compute_type_dependent_data(
        terrain_points: list[PathPoint],
        start_node: "Node",
        end_node: "Node",
        lift_type: str,
        lift_id: str,
    ) -> tuple[str, list["Pylon"], list[PathPoint], float]:
        """Compute all type-dependent lift data.

        Single source of truth for name, pylons, cable_points calculation.
        Used by both create() and update_type() to ensure consistency.

        Args:
            terrain_points: Pre-sampled terrain along lift path
            start_node: Bottom station node
            end_node: Top station node
            lift_type: Type of lift
            lift_id: Lift ID for naming (e.g., "L1")

        Returns:
            Tuple of (name, pylons, cable_points, length_m)
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

        # Calculate pylons via catenary simulation
        pylons = Lift.calculate_pylons(
            terrain_points=terrain_points,
            lift_type=lift_type,
            total_distance_m=length_m,
        )

        # Calculate cable points (depend on pylons)
        cable_points = Lift.calculate_cable_points(
            terrain_points=terrain_points,
            pylons=pylons,
            start_elevation=start_node.elevation,
            end_elevation=end_node.elevation,
            lift_type=lift_type,
            total_distance_m=length_m,
        )

        return name, pylons, cable_points, length_m

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

        # Compute all type-dependent data via shared helper
        name, pylons, cable_points, length_m = cls._compute_type_dependent_data(
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

    def get_vertical_rise(self, nodes: dict[str, "Node"]) -> float:
        """Calculate elevation gain from start to end node.

        Args:
            nodes: Dict of node_id -> Node

        Returns:
            Vertical rise in meters.
        """
        start = nodes.get(self.start_node_id)
        end = nodes.get(self.end_node_id)
        if not start or not end:
            return 0.0
        return end.elevation - start.elevation

    def get_length_m(self, nodes: dict[str, "Node"]) -> float:
        """Calculate horizontal distance between nodes.

        Args:
            nodes: Dict of node_id -> Node

        Returns:
            Length in meters.
        """
        start = nodes.get(self.start_node_id)
        end = nodes.get(self.end_node_id)
        if not start or not end:
            raise ValueError(f"Start or end node not found for lift {self.id}")
        return GeoCalculator.haversine_distance_m(
            lat1=start.lat,
            lon1=start.lon,
            lat2=end.lat,
            lon2=end.lon,
        )

    def update_type(self, new_type: str, start_node: "Node", end_node: "Node") -> None:
        """Change lift type and update all dependent fields.

        Uses _compute_type_dependent_data() to ensure consistency with create().
        Updates: lift_type, name, pylons, cable_points.

        Args:
            new_type: New lift type (must be valid from LiftConfig.TYPES)
            start_node: Bottom station node
            end_node: Top station node
        """
        if new_type == self.lift_type:
            logger.info(f"Lift {self.id} already has type {new_type}, no update needed")
            return  # No change needed

        if new_type not in LiftConfig.TYPES:
            raise ValueError(f"Invalid lift_type '{new_type}'. Must be one of: {LiftConfig.TYPES}")

        self.lift_type = new_type

        # Recompute all type-dependent data via shared helper
        self.name, self.pylons, self.cable_points, _ = self._compute_type_dependent_data(
            terrain_points=self.terrain_points,
            start_node=start_node,
            end_node=end_node,
            lift_type=new_type,
            lift_id=self.id,
        )

        logger.info(f"Updated lift {self.id} type to {new_type}")

    @staticmethod
    def calculate_pylons(
        terrain_points: list[PathPoint],
        lift_type: str,
        total_distance_m: float,
    ) -> list[Pylon]:
        """Calculate pylon positions using 3-phase catenary simulation.

        All lift types use catenary simulation for realistic cable sag and pylon placement:
            Phase 1: Place pylons where cable clearance < min_clearance_m
            Phase 2: Enforce max_spacing_m by adding midpoint pylons where spans are too long
            Phase 3: Re-check clearance violations (spacing pylons may pull cable down)

        Cable sag formula: z(t) = (1-t)*z₀ + t*z₁ - 4*s*t*(1-t)
        where t=x/L and s=sag_factor*L

        Args:
            terrain_points: List of PathPoint sampled along lift path
            lift_type: Type of lift (determines pylon parameters from LiftConfig)
            total_distance_m: Total horizontal distance of lift

        Returns:
            List of Pylon objects with calculated positions.
        """
        if len(terrain_points) < 2:
            raise ValueError(f"terrain_points must have at least 2 points, got {len(terrain_points)}")
        if total_distance_m <= 0:
            raise ValueError(f"total_distance_m must be positive, got {total_distance_m}")
        if len(terrain_points) < 3:
            logger.warning(
                f"Not enough terrain points for pylon calculation (got {len(terrain_points)}), skipping pylons"
            )
            return []

        config = LiftConfig.PYLON_CONFIG[lift_type]
        n = len(terrain_points)
        dist_per_step = total_distance_m / (n - 1) if n > 1 else 0

        terrain_elevs = [p.elevation for p in terrain_points]
        pylon_height = config["pylon_height_m"]
        station_height = config["station_height_m"]
        min_spacing_m = config["min_spacing_m"]
        min_clearance = config["min_clearance_m"]
        sag_factor = config["sag_factor"]
        max_spacing_m = config["max_spacing_m"]

        # Minimum spacing in indices
        min_spacing_idx = max(2, int(min_spacing_m / dist_per_step)) if dist_per_step > 0 else 2

        def cable_elev_at_idx(start_idx: int, end_idx: int, z0: float, z1: float, idx: int) -> float:
            """Cable elevation at index using Lift.cable_elevation."""
            if end_idx <= start_idx:
                return z0
            span_idx = end_idx - start_idx
            t = (idx - start_idx) / span_idx
            span_m = span_idx * dist_per_step
            return Lift.cable_elevation(t=t, start_elev=z0, end_elev=z1, span_m=span_m, sag_factor=sag_factor)

        def find_clearance_violations(
            start_idx: int, end_idx: int, start_elev: float, end_elev: float, pylon_set: set[int]
        ) -> list[int]:
            """Recursively find where cable clearance is below minimum."""
            if end_idx - start_idx < min_spacing_idx * 2:
                return []

            worst_violation = 0
            worst_idx = -1

            for i in range(start_idx + min_spacing_idx, end_idx - min_spacing_idx + 1):
                if i in pylon_set:
                    continue
                cable_elev = cable_elev_at_idx(start_idx=start_idx, end_idx=end_idx, z0=start_elev, z1=end_elev, idx=i)
                clearance = cable_elev - terrain_elevs[i]
                violation = min_clearance - clearance

                if violation > worst_violation:
                    worst_violation = violation
                    worst_idx = i

            if worst_violation <= 0 or worst_idx < 0:
                return []

            pylon_top_elev = terrain_elevs[worst_idx] + pylon_height
            new_pylon_set = pylon_set | {worst_idx}

            left_pylons = find_clearance_violations(
                start_idx=start_idx,
                end_idx=worst_idx,
                start_elev=start_elev,
                end_elev=pylon_top_elev,
                pylon_set=new_pylon_set,
            )
            right_pylons = find_clearance_violations(
                start_idx=worst_idx,
                end_idx=end_idx,
                start_elev=pylon_top_elev,
                end_elev=end_elev,
                pylon_set=new_pylon_set,
            )

            return left_pylons + [worst_idx] + right_pylons

        # Station cable elevations
        start_cable_elev = terrain_elevs[0] + station_height
        end_cable_elev = terrain_elevs[-1] + station_height

        # Phase 1: Fix all clearance violations
        pylon_indices = find_clearance_violations(
            start_idx=0,
            end_idx=n - 1,
            start_elev=start_cable_elev,
            end_elev=end_cable_elev,
            pylon_set=set(),
        )
        pylon_indices = sorted(set(pylon_indices))

        # Phase 2: Enforce maximum spacing
        if max_spacing_m is not None and dist_per_step > 0:
            max_spacing_idx = int(max_spacing_m / dist_per_step)

            for _ in range(20):  # Safety limit
                anchors = [0] + sorted(pylon_indices) + [n - 1]
                new_spacing_pylons = []

                for seg_idx in range(len(anchors) - 1):
                    seg_start = anchors[seg_idx]
                    seg_end = anchors[seg_idx + 1]
                    span_idx = seg_end - seg_start

                    if span_idx > max_spacing_idx:
                        mid_idx = (seg_start + seg_end) // 2
                        if mid_idx not in pylon_indices and 0 < mid_idx < n - 1:
                            new_spacing_pylons.append(mid_idx)

                if not new_spacing_pylons:
                    break
                pylon_indices = sorted(set(pylon_indices + new_spacing_pylons))

        # Phase 3: Re-check clearance after spacing pylons
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

            additional = find_clearance_violations(
                start_idx=seg_start,
                end_idx=seg_end,
                start_elev=seg_start_elev,
                end_elev=seg_end_elev,
                pylon_set=pylon_set,
            )
            new_clearance_pylons.extend(additional)

        if new_clearance_pylons:
            pylon_indices = sorted(set(pylon_indices + new_clearance_pylons))

        # Convert indices to Pylon objects
        pylons = []
        for idx in pylon_indices:
            point = terrain_points[idx]
            pylons.append(
                Pylon(
                    index=idx,
                    distance_m=idx * dist_per_step,
                    lat=point.lat,
                    lon=point.lon,
                    ground_elevation_m=point.elevation,
                    height_m=pylon_height,
                )
            )

        return pylons

    @staticmethod
    def calculate_cable_points(
        terrain_points: list[PathPoint],
        pylons: list[Pylon],
        start_elevation: float,
        end_elevation: float,
        lift_type: str,
        total_distance_m: float,
    ) -> list[PathPoint]:
        """Calculate cable points along the lift path with sag.

        The cable follows parabolic sag between anchor points (stations and pylons).
        This pre-computes cable positions for efficient rendering and GPX export.

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

        config = LiftConfig.PYLON_CONFIG.get(lift_type, LiftConfig.PYLON_CONFIG["chairlift"])
        station_height = config["station_height_m"]
        sag_factor = config["sag_factor"]

        # Build anchor points: [stations + all pylons]
        anchor_x = [0.0]  # Bottom station
        anchor_y = [start_elevation + station_height]

        for pylon in pylons:
            anchor_x.append(pylon.distance_m)
            anchor_y.append(pylon.top_elevation_m)

        anchor_x.append(total_distance_m)  # Top station
        anchor_y.append(end_elevation + station_height)

        # Sort anchor points by distance
        anchor_sorted = sorted(zip(anchor_x, anchor_y), key=lambda p: p[0])
        anchor_x = [p[0] for p in anchor_sorted]
        anchor_y = [p[1] for p in anchor_sorted]

        # Generate cable curve with sag for each segment
        cable_points = []
        n_terrain = len(terrain_points)

        for seg_idx in range(len(anchor_x) - 1):
            start_x = anchor_x[seg_idx]
            end_x = anchor_x[seg_idx + 1]
            start_y = anchor_y[seg_idx]
            end_y = anchor_y[seg_idx + 1]
            span = end_x - start_x

            if span <= 0:
                continue

            # Generate points along segment with parabolic sag
            n_seg_points = max(10, int(span / 20))
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

                # Interpolate lat/lon from terrain points
                terrain_frac = x / total_distance_m
                terrain_idx = terrain_frac * (n_terrain - 1)
                idx_low = int(terrain_idx)
                idx_high = min(idx_low + 1, n_terrain - 1)
                interp_frac = terrain_idx - idx_low

                pt_low = terrain_points[idx_low]
                pt_high = terrain_points[idx_high]
                lon = pt_low.lon + (pt_high.lon - pt_low.lon) * interp_frac
                lat = pt_low.lat + (pt_high.lat - pt_low.lat) * interp_frac

                cable_points.append(PathPoint(lon=lon, lat=lat, elevation=cable_elev))

        return cable_points

    @classmethod
    def from_dict(cls, data: dict) -> "Lift":
        """Create Lift from dictionary.

        All fields are required - raises KeyError if missing.
        """
        return cls(
            id=data["id"],
            name=data["name"],
            start_node_id=data["start_node_id"],
            end_node_id=data["end_node_id"],
            lift_type=data["lift_type"],
            terrain_points=[PathPoint(**p) for p in data["terrain_points"]],
            pylons=[Pylon(**p) for p in data["pylons"]],
            cable_points=[PathPoint(**p) for p in data["cable_points"]],
        )

    def __repr__(self) -> str:
        return f"Lift({self.id}, {self.lift_type})"
