"""ResortGraph - Central manager for ski resort entities.

Owns and manages all nodes, segments, slopes, and lifts.
Provides operations for:
- Committing proposed paths to segments
- Finishing slopes (grouping segments)
- Adding/removing lifts
- Undo/redo operations
- Serialization/deserialization
- Graph cleanup (isolated nodes, auto-backup)

Reference: DETAILS.md
"""

import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Optional, cast

from skiresort_planner.constants import EntityPrefixes, GeometricTuningConfig, UndoConfig
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.core.terrain_analyzer import TerrainAnalyzer
from skiresort_planner.enum_utils import enum_eq
from skiresort_planner.model.lift import Lift
from skiresort_planner.model.node import Node
from skiresort_planner.model.path_point import PathPoint
from skiresort_planner.model.path_segment import PathSegment, SegmentKind
from skiresort_planner.model.path_smoothing import smooth_joined_path
from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.road import Road
from skiresort_planner.model.slope import Slope

if TYPE_CHECKING:
    from skiresort_planner.core.dem_service import DEMService

logger = logging.getLogger(__name__)


# =============================================================================
# Undo Action Types
# =============================================================================


class ActionType(Enum):
    """Enum identifying the type of (undo) action for reliable dispatch."""

    ADD_SEGMENTS = auto()
    FINISH_SLOPE = auto()
    ADD_LIFT = auto()
    FINISH_ROAD = auto()
    DELETE_SLOPE = auto()
    DELETE_LIFT = auto()
    DELETE_ROAD = auto()


@dataclass(frozen=True)
class AddSegmentsAction:
    """Undo action for committed path segments."""

    segment_ids: tuple[str, ...]
    node_ids: tuple[str, ...]

    @property
    def action_type(self) -> ActionType:
        """Return the enum type for dispatch."""
        return ActionType.ADD_SEGMENTS


@dataclass(frozen=True)
class FinishSlopeAction:
    """Undo action for finishing a slope."""

    slope_id: str
    segment_ids: tuple[str, ...]
    slope_name: str
    start_node_id: str | None

    @property
    def action_type(self) -> ActionType:
        """Return the enum type for dispatch."""
        return ActionType.FINISH_SLOPE


@dataclass(frozen=True)
class AddLiftAction:
    """Undo action for creating a lift."""

    lift_id: str

    @property
    def action_type(self) -> ActionType:
        """Return the enum type for dispatch."""
        return ActionType.ADD_LIFT


@dataclass(frozen=True)
class FinishRoadAction:
    """Undo action for finishing a road (mirrors FinishSlopeAction).

    Ungroups the Road but keeps its segments, so undo returns to road building
    with the segments intact; further undos peel each segment (AddSegmentsAction).
    """

    road_id: str
    segment_ids: tuple[str, ...]
    road_name: str
    start_node_id: str | None

    @property
    def action_type(self) -> ActionType:
        """Return the enum type for dispatch."""
        return ActionType.FINISH_ROAD


@dataclass(frozen=True)
class DeleteSlopeAction:
    """Undo action for deleting a slope (stores data for restore)."""

    slope_id: str
    deleted_slope: "Slope"
    deleted_segments: tuple["PathSegment", ...]
    deleted_nodes: tuple["Node", ...] = ()  # Nodes orphaned by segment removal

    @property
    def action_type(self) -> ActionType:
        """Return the enum type for dispatch."""
        return ActionType.DELETE_SLOPE


@dataclass(frozen=True)
class DeleteLiftAction:
    """Undo action for deleting a lift (stores data for restore)."""

    lift_id: str
    deleted_lift: "Lift"
    deleted_nodes: tuple["Node", ...] = ()  # Nodes orphaned by lift removal

    @property
    def action_type(self) -> ActionType:
        """Return the enum type for dispatch."""
        return ActionType.DELETE_LIFT


@dataclass(frozen=True)
class DeleteRoadAction:
    """Undo action for deleting a road (stores data for restore)."""

    road_id: str
    deleted_road: "Road"
    deleted_segments: tuple["PathSegment", ...]
    deleted_nodes: tuple["Node", ...] = ()  # Nodes orphaned by segment removal

    @property
    def action_type(self) -> ActionType:
        """Return the enum type for dispatch."""
        return ActionType.DELETE_ROAD


UndoAction = (
    AddSegmentsAction
    | FinishSlopeAction
    | AddLiftAction
    | FinishRoadAction
    | DeleteSlopeAction
    | DeleteLiftAction
    | DeleteRoadAction
)


class ResortGraph:
    """Graph representing the ski resort.

    Central manager owning all nodes, segments, slopes, and lifts.
    Provides operations for building and modifying the resort.

    Example:
        graph = ResortGraph()
        graph.commit_paths(paths=[proposed_path])
        graph.finish_slope(segment_ids=["S1", "S2"])
    """

    def __init__(self) -> None:
        """Initialize empty resort graph."""
        self.nodes: dict[str, Node] = {}
        self.segments: dict[str, PathSegment] = {}
        self.slopes: dict[str, Slope] = {}
        self.lifts: dict[str, Lift] = {}
        self.roads: dict[str, Road] = {}
        self.undo_stack: list[UndoAction] = []

        self._node_counter = 0
        self._segment_counter = 0
        self._slope_counter = 0
        self._lift_counter = 0
        self._road_counter = 0

    def _next_node_id(self) -> str:
        self._node_counter += 1
        return f"{EntityPrefixes.NODE}{self._node_counter}"

    def _next_segment_id(self) -> str:
        self._segment_counter += 1
        return f"{EntityPrefixes.SEGMENT}{self._segment_counter}"

    def _next_slope_id(self) -> str:
        self._slope_counter += 1
        return f"{EntityPrefixes.SLOPE}{self._slope_counter}"

    def _next_lift_id(self) -> str:
        self._lift_counter += 1
        return f"{EntityPrefixes.LIFT}{self._lift_counter}"

    def _next_road_id(self) -> str:
        self._road_counter += 1
        return f"{EntityPrefixes.ROAD}{self._road_counter}"

    def _push_undo(self, action: UndoAction) -> None:
        """Push action to undo stack with size limiting.

        Discards oldest actions when stack exceeds MAX_UNDO_STACK_SIZE.
        """
        self.undo_stack.append(action)
        # Trim oldest actions if stack is too large
        while len(self.undo_stack) > UndoConfig.MAX_UNDO_STACK_SIZE:
            self.undo_stack.pop(0)

    # =========================================================================
    # Node Operations
    # =========================================================================

    def find_nearest_node(
        self,
        lon: float,
        lat: float,
        threshold_m: float = GeometricTuningConfig.STEP_SIZE_M,
    ) -> Optional[Node]:
        """Find nearest node within threshold distance.

        Args:
            lon, lat: Target coordinates
            threshold_m: Maximum distance in meters

        Returns:
            Nearest Node or None if none within threshold.
        """
        best_dist = threshold_m
        best_node = None

        for node in self.nodes.values():
            dist = node.distance_to(lon=lon, lat=lat)
            if dist < best_dist:
                best_dist = dist
                best_node = node

        return best_node

    def get_or_create_node(
        self,
        lon: float,
        lat: float,
        elevation: float,
    ) -> tuple[Node, bool]:
        """Get existing node or create new one.

        Args:
            lon, lat: Node coordinates
            elevation: Node elevation

        Returns:
            Tuple of (node, was_created)
        """
        existing = self.find_nearest_node(lon=lon, lat=lat)
        if existing:
            return existing, False

        node = Node(
            id=self._next_node_id(),
            location=PathPoint(lon=lon, lat=lat, elevation=elevation),
        )
        self.nodes[node.id] = node
        return node, True

    def get_connection_count(self, node_id: str) -> int:
        """Count connections (segments + lifts) for a node.

        Args:
            node_id: ID of the node to check

        Returns:
            Total number of segments and lifts connected to this node.
        """
        segment_count = sum(1 for s in self.segments.values() if s.start_node_id == node_id or s.end_node_id == node_id)
        lift_count = sum(
            1 for lift in self.lifts.values() if lift.start_node_id == node_id or lift.end_node_id == node_id
        )
        return segment_count + lift_count

    # =========================================================================
    # Commit Operations
    # =========================================================================

    def commit_paths(
        self,
        paths: list[ProposedPathSegment],
        record_undo: bool = True,
    ) -> list[str]:
        """Commit proposed paths to the graph.

        Simple workflow:
        1. Get or create node at path start (snaps to existing node if nearby)
        2. Get or create node at path end (snaps to existing node if nearby)
        3. Create segment connecting them

        Note: No auto-snap to segment lines. Endpoints only snap to existing
        nodes (via get_or_create_node's find_nearest_node check).

        Args:
            paths: List of ProposedPathSegment to commit
            record_undo: When True, push a segment-level AddSegmentsAction. Both
                slopes and roads pass True so undo peels one segment at a time;
                finishing then records a FinishSlopeAction / FinishRoadAction on top.

        Returns:
            List of end node IDs for continuation.
        """
        new_segment_ids = []
        new_node_ids = []
        end_node_ids = []

        for path in paths:
            if not path.points:
                continue

            # Get start node.
            # If the proposal extends from an existing node, reuse it EXACTLY.
            # Spline smoothing could drift the traced start point.
            # Same fix the end uses via target_node_id.
            start_pt = path.start
            assert start_pt is not None  # Guaranteed by `if not path.points: continue` check
            if path.start_node_id and path.start_node_id in self.nodes:
                start_node = self.nodes[path.start_node_id]
                start_created = False
                # Snap path geometry to the exact node coordinates.
                path.points[0] = PathPoint(
                    lon=start_node.lon,
                    lat=start_node.lat,
                    elevation=start_node.elevation,
                )
            else:
                start_node, start_created = self.get_or_create_node(
                    lon=start_pt.lon,
                    lat=start_pt.lat,
                    elevation=start_pt.elevation,
                )
            if start_created:
                new_node_ids.append(start_node.id)

            # Get or create end node
            # If this is a connector path with target_node_id, use that node directly
            # to avoid creating a duplicate node slightly off from the target
            if path.target_node_id and path.target_node_id in self.nodes:
                end_node = self.nodes[path.target_node_id]
                end_created = False
                # Snap path geometry to exact node coordinates (avoids visual kinks in 3D)
                if path.points:
                    path.points[-1] = PathPoint(
                        lon=end_node.lon,
                        lat=end_node.lat,
                        elevation=end_node.elevation,
                    )
            else:
                end_pt = path.end
                assert end_pt is not None  # Guaranteed by `if not path.points: continue` check
                end_node, end_created = self.get_or_create_node(
                    lon=end_pt.lon,
                    lat=end_pt.lat,
                    elevation=end_pt.elevation,
                )
            if end_created:
                new_node_ids.append(end_node.id)

            # Calculate side slope (requires terrain analysis, stored in segment)
            if len(path.points) < 2:
                raise ValueError(
                    f"Path must have at least 2 points to compute side slope, got {len(path.points)}: {path}"
                )
            side_info = TerrainAnalyzer.compute_side_slope(
                start_lon=path.points[0].lon,
                start_lat=path.points[0].lat,
                end_lon=path.points[1].lon,
                end_lat=path.points[1].lat,
            )
            side_slope_pct = side_info.slope_pct
            side_slope_dir = side_info.direction

            # Create segment (metrics computed as properties from points)
            segment_id = self._next_segment_id()
            segment = PathSegment(
                id=segment_id,
                name=f"Segment {self._segment_counter}",
                points=path.points,
                start_node_id=start_node.id,
                end_node_id=end_node.id,
                side_slope_pct=side_slope_pct,
                side_slope_dir=side_slope_dir,
                kind=path.kind,  # slope vs road identity carried from the proposal
            )
            self.segments[segment_id] = segment
            new_segment_ids.append(segment_id)
            end_node_ids.append(end_node.id)

        # Record for undo
        if record_undo and new_segment_ids:
            self._push_undo(
                AddSegmentsAction(
                    segment_ids=tuple(new_segment_ids),
                    node_ids=tuple(new_node_ids),
                )
            )

        return end_node_ids

    def _resolve_finish_endpoints(
        self, segment_ids: list[str]
    ) -> tuple[PathSegment, PathSegment, Node, Node, float] | None:
        """Validate a finish request and return (first_seg, last_seg, start_node, end_node, avg_bearing).

        Returns None if the segment list is empty or any segment/endpoint node is missing.
        Shared by finish_slope / finish_road (validation + bearing only).
        """
        if not segment_ids:
            return None

        first_seg = self.segments.get(segment_ids[0])
        last_seg = self.segments.get(segment_ids[-1])
        if not first_seg or not last_seg:
            return None

        start_node = self.nodes.get(first_seg.start_node_id)
        end_node = self.nodes.get(last_seg.end_node_id)
        if not start_node or not end_node:
            return None

        avg_bearing = GeoCalculator.initial_bearing_deg(
            lon1=start_node.lon, lat1=start_node.lat, lon2=end_node.lon, lat2=end_node.lat
        )
        return first_seg, last_seg, start_node, end_node, avg_bearing

    def _smooth_finished_path(self, segment_ids: list[str], smoothing_factor: float) -> None:
        """Whole-path smooth a finished entity across its junctions, in place.

        No-op for a single segment. EVERY node on the path (outer endpoints + every junction)
        stays pinned exactly on the ribbon, so markers sit on the path and any node can be a
        branch point; only the shape between nodes rounds. Never rejects — a road may drift
        over the ±15% build cap here (bridge/cut/fill), which is intentional; not re-applied.

        smoothing_factor: higher = smoother (roads); lower hugs terrain (slopes).
        """
        if len(segment_ids) < 2:
            return  # single-segment path has no junction to smooth
        segments = [self.segments[sid] for sid in segment_ids]
        # Boundary nodes: start of the first segment, then each segment's end node.
        boundary_node_ids = [segments[0].start_node_id, *(seg.end_node_id for seg in segments)]

        before = max(seg.max_slope_pct for seg in segments)
        smoothed = smooth_joined_path(
            segment_point_lists=[seg.points for seg in segments],
            node_anchors=[self.nodes[nid].location for nid in boundary_node_ids],
            step_m=GeometricTuningConfig.RESAMPLE_STEP_M,
            smoothing_factor=smoothing_factor,
            node_weight=GeometricTuningConfig.NODE_WEIGHT,
            corridor_weight=GeometricTuningConfig.CORRIDOR_WEIGHT,
        )
        for seg, pts in zip(segments, smoothed):
            seg.points = pts
        after = max(seg.max_slope_pct for seg in segments)
        logger.info(f"Smoothed finished path {segment_ids}: max_slope_pct {before:.1f}% -> {after:.1f}%")

    # =========================================================================
    # Slope Operations
    # =========================================================================

    def finish_slope(
        self,
        segment_ids: list[str],
        name: Optional[str] = None,
    ) -> Optional[Slope]:
        """Finish a slope by grouping segments.

        Args:
            segment_ids: List of segment IDs to group
            name: Optional custom name (generates creative name if None)

        Returns:
            Created Slope or None if invalid.
        """
        self._smooth_finished_path(
            segment_ids=segment_ids, smoothing_factor=GeometricTuningConfig.SLOPE_SMOOTHING_FACTOR
        )
        resolved = self._resolve_finish_endpoints(segment_ids=segment_ids)
        if resolved is None:
            return None
        first_seg, last_seg, start_node, end_node, avg_bearing = resolved

        slope_id = self._next_slope_id()
        # Difficulty from the steepest section (max_slope_pct over rolling windows).
        max_slope = max(self.segments[sid].max_slope_pct for sid in segment_ids if sid in self.segments)
        difficulty = TerrainAnalyzer.classify_difficulty(slope_pct=max_slope)
        if name is None:
            name = Slope.generate_name(
                difficulty=difficulty,
                slope_id=slope_id,
                start_elevation=start_node.elevation,
                end_elevation=end_node.elevation,
                avg_bearing=avg_bearing,
            )

        slope = Slope(
            id=slope_id,
            name=name,
            segment_ids=segment_ids,
            start_node_id=first_seg.start_node_id,
            end_node_id=last_seg.end_node_id,
        )
        self.slopes[slope_id] = slope
        for seg_id in segment_ids:
            self.segments[seg_id].name = name
        logger.info(f"Slope finished: {name}, {len(segment_ids)} segments, difficulty={difficulty}")
        self._push_undo(
            FinishSlopeAction(
                slope_id=slope_id,
                segment_ids=tuple(segment_ids),
                slope_name=name,
                start_node_id=first_seg.start_node_id,
            )
        )
        return slope

    # =========================================================================
    # Road Operations
    # =========================================================================

    def finish_road(
        self,
        segment_ids: list[str],
        name: Optional[str] = None,
    ) -> Optional[Road]:
        """Group committed segments into a vehicle Road.

        Records a FinishRoadAction (mirrors finish_slope): undo ungroups the road
        but keeps its segments, which carry their own AddSegmentsAction entries.

        Args:
            segment_ids: Segment IDs the road is made of.
            name: Optional custom name (generates a compass name if None).

        Returns:
            Created Road or None if invalid.
        """
        self._smooth_finished_path(
            segment_ids=segment_ids, smoothing_factor=GeometricTuningConfig.ROAD_SMOOTHING_FACTOR
        )
        resolved = self._resolve_finish_endpoints(segment_ids=segment_ids)
        if resolved is None:
            return None
        first_seg, last_seg, _start_node, _end_node, avg_bearing = resolved

        road_id = self._next_road_id()
        if name is None:
            name = Road.generate_name(road_id=road_id, avg_bearing=avg_bearing)

        road = Road(
            id=road_id,
            name=name,
            segment_ids=segment_ids,
            start_node_id=first_seg.start_node_id,
            end_node_id=last_seg.end_node_id,
        )
        self.roads[road_id] = road
        for seg_id in segment_ids:
            self.segments[seg_id].name = name
        logger.info(f"Road finished: {name}, {len(segment_ids)} segments")
        self._push_undo(
            FinishRoadAction(
                road_id=road_id,
                segment_ids=tuple(segment_ids),
                road_name=name,
                start_node_id=first_seg.start_node_id,
            )
        )
        return road

    def delete_road(self, road_id: str) -> bool:
        """Delete a road and its segments.

        Args:
            road_id: ID of road to delete.

        Returns:
            True if deleted, False if not found.
        """
        road = self.roads.get(road_id)
        if not road:
            return False

        deleted_segments = [self.segments[seg_id] for seg_id in road.segment_ids if seg_id in self.segments]
        for seg_id in road.segment_ids:
            self.segments.pop(seg_id, None)
        del self.roads[road_id]

        # Nodes orphaned by segment removal (connection_count == 0).
        orphaned_nodes = [self.nodes[nid] for nid in self.nodes if self.get_connection_count(node_id=nid) == 0]
        self._push_undo(
            DeleteRoadAction(
                road_id=road_id,
                deleted_road=road,
                deleted_segments=tuple(deleted_segments),
                deleted_nodes=tuple(orphaned_nodes),
            )
        )
        self.cleanup_isolated_nodes()
        logger.info(f"Deleted road {road.name} with {len(road.segment_ids)} segments")
        return True

    # =========================================================================
    # Lift Operations
    # =========================================================================

    def add_lift(
        self,
        start_node_id: str,
        end_node_id: str,
        lift_type: str,
        dem: "DEMService",
    ) -> Lift:
        """Add a lift between two nodes.

        Args:
            start_node_id: ID of bottom station
            end_node_id: ID of top station
            lift_type: Type of lift
            dem: DEM service for terrain sampling

        Returns:
            Created Lift.

        Raises:
            KeyError: If start or end node not found.
        """
        start = self.nodes[start_node_id]
        end = self.nodes[end_node_id]

        lift_id = self._next_lift_id()

        lift = Lift.create(
            start_node=start,
            end_node=end,
            dem=dem,
            lift_type=lift_type,
            lift_id=lift_id,
        )

        self.lifts[lift_id] = lift
        self._push_undo(AddLiftAction(lift_id=lift_id))

        return lift

    # =========================================================================
    # Undo Operations
    # =========================================================================

    def undo_last(self) -> UndoAction:
        """Undo the last action.

        Returns:
            The undone action.

        Raises:
            RuntimeError: If undo stack is empty (caller should check first).
        """
        if not self.undo_stack:
            raise RuntimeError("undo_last called with empty undo_stack")

        action = self.undo_stack.pop()

        # Dispatch via enum_eq (reload-safe): ActionType is a plain Enum and the undo_stack
        # lives in st.session_state holding OLD-class references after a Streamlit reload,
        # so plain-enum `is`/`==` would fail. enum_eq compares the stable string form.
        action_type = action.action_type

        if enum_eq(action_type, ActionType.ADD_SEGMENTS):
            add_seg = cast(AddSegmentsAction, action)
            for seg_id in add_seg.segment_ids:
                self.segments.pop(seg_id, None)
            self.cleanup_isolated_nodes()  # Remove orphaned nodes from segment removal

        elif enum_eq(action_type, ActionType.ADD_LIFT):
            add_lift = cast(AddLiftAction, action)
            self.lifts.pop(add_lift.lift_id, None)
            self.cleanup_isolated_nodes()  # Remove orphaned station nodes

        elif enum_eq(action_type, ActionType.FINISH_SLOPE):
            # Ungroup the slope but keep its segments; their own AddSegmentsAction
            # entries handle per-segment removal on further undo.
            finish = cast(FinishSlopeAction, action)
            self.slopes.pop(finish.slope_id, None)
            for seg_id in finish.segment_ids:
                self.segments[seg_id].name = f"Segment {seg_id[1:]}"

        elif enum_eq(action_type, ActionType.FINISH_ROAD):
            # Mirror FINISH_SLOPE: ungroup the road, keep its segments.
            finish_road = cast(FinishRoadAction, action)
            self.roads.pop(finish_road.road_id, None)
            for seg_id in finish_road.segment_ids:
                self.segments[seg_id].name = f"Segment {seg_id[1:]}"

        elif enum_eq(action_type, ActionType.DELETE_SLOPE):
            del_slope = cast(DeleteSlopeAction, action)
            # Restore orphaned nodes first (they're needed by segments)
            for node in del_slope.deleted_nodes:
                self.nodes[node.id] = node
            # Restore deleted slope and its segments
            self.slopes[del_slope.slope_id] = del_slope.deleted_slope
            for seg in del_slope.deleted_segments:
                self.segments[seg.id] = seg
            logger.info(
                f"Restored slope {del_slope.slope_id} with {len(del_slope.deleted_segments)} segments "
                f"and {len(del_slope.deleted_nodes)} nodes"
            )

        elif enum_eq(action_type, ActionType.DELETE_LIFT):
            del_lift = cast(DeleteLiftAction, action)
            # Restore orphaned nodes first (they're needed by lift)
            for node in del_lift.deleted_nodes:
                self.nodes[node.id] = node
            # Restore deleted lift
            self.lifts[del_lift.lift_id] = del_lift.deleted_lift
            logger.info(f"Restored lift {del_lift.lift_id} with {len(del_lift.deleted_nodes)} nodes")

        elif enum_eq(action_type, ActionType.DELETE_ROAD):
            del_road = cast(DeleteRoadAction, action)
            # Restore orphaned nodes first (they're needed by segments)
            for node in del_road.deleted_nodes:
                self.nodes[node.id] = node
            self.roads[del_road.road_id] = del_road.deleted_road
            for seg in del_road.deleted_segments:
                self.segments[seg.id] = seg
            logger.info(
                f"Restored road {del_road.road_id} with {len(del_road.deleted_segments)} segments "
                f"and {len(del_road.deleted_nodes)} nodes"
            )

        else:
            raise RuntimeError(f"Unknown action type in undo_last: {action_type}")

        return action

    def delete_slope(self, slope_id: str) -> bool:
        """Delete a slope and its segments.

        Args:
            slope_id: ID of slope to delete

        Returns:
            True if deleted, False if not found.
        """
        slope = self.slopes.get(slope_id)
        if not slope:
            return False

        deleted_segments = [self.segments[seg_id] for seg_id in slope.segment_ids if seg_id in self.segments]
        for seg_id in slope.segment_ids:
            self.segments.pop(seg_id, None)
        del self.slopes[slope_id]

        # Nodes orphaned by segment removal (connection_count == 0).
        orphaned_nodes = [self.nodes[nid] for nid in self.nodes if self.get_connection_count(node_id=nid) == 0]
        self._push_undo(
            DeleteSlopeAction(
                slope_id=slope_id,
                deleted_slope=slope,
                deleted_segments=tuple(deleted_segments),
                deleted_nodes=tuple(orphaned_nodes),
            )
        )
        self.cleanup_isolated_nodes()
        logger.info(f"Deleted slope {slope.name} with {len(slope.segment_ids)} segments")
        return True

    def delete_lift(self, lift_id: str) -> bool:
        """Delete a lift.

        Args:
            lift_id: ID of lift to delete

        Returns:
            True if deleted, False if not found.
        """
        lift = self.lifts.get(lift_id)
        if not lift:
            return False

        # Remove the lift
        del self.lifts[lift_id]

        # Identify nodes that will be orphaned (connection_count == 0 after removal)
        orphaned_nodes = [self.nodes[nid] for nid in self.nodes if self.get_connection_count(node_id=nid) == 0]

        # Push to undo stack with full data for restore (including orphaned nodes)
        self._push_undo(
            DeleteLiftAction(
                lift_id=lift_id,
                deleted_lift=lift,
                deleted_nodes=tuple(orphaned_nodes),
            )
        )

        self.cleanup_isolated_nodes()  # Remove orphaned station nodes

        logger.info(f"Deleted lift {lift.name}")
        return True

    # =========================================================================
    # Query Operations
    # =========================================================================

    def get_slope_by_segment_id(self, segment_id: str) -> Optional[Slope]:
        """Find the slope containing a given segment.
        Not applicabale for roads, as they have no segements.

        Args:
            segment_id: ID of segment to find

        Returns:
            Slope containing the segment, or None if segment is not in any slope.
        """
        for slope in self.slopes.values():
            if segment_id in slope.segment_ids:
                return slope
        return None

    def get_segment_stats(self, segment_ids: list[str]) -> dict[str, Any]:
        """Get statistics for specific segments (used for running stats during building).

        Args:
            segment_ids: List of segment IDs to calculate stats for

        Returns:
            Dict with: total_drop, total_length, avg_gradient, max_gradient, difficulty, start_elev, current_elev
            All numeric values are guaranteed non-None (defaults to 0.0 if segments not found).
        """
        default_stats = {
            "total_drop": 0.0,
            "total_length": 0.0,
            "avg_gradient": 0.0,
            "max_gradient": 0.0,
            "difficulty": "green",
            "start_elev": 0.0,
            "current_elev": 0.0,
        }

        if not segment_ids:
            return default_stats

        first_seg = self.segments.get(segment_ids[0])
        last_seg = self.segments.get(segment_ids[-1])

        if not first_seg or not last_seg:
            logger.warning(
                f"get_segment_stats: missing segments - first={segment_ids[0]} exists={first_seg is not None}, "
                f"last={segment_ids[-1]} exists={last_seg is not None}"
            )
            return default_stats

        assert first_seg.start is not None  # Segments always have points
        assert last_seg.end is not None  # Segments always have points
        start_elev = first_seg.start.elevation
        current_elev = last_seg.end.elevation

        total_length = sum(seg.length_m for seg_id in segment_ids if (seg := self.segments.get(seg_id)))

        total_drop = start_elev - current_elev
        avg_gradient = (total_drop / total_length * 100) if total_length > 0 else 0.0

        # Difficulty based on steepest section in any segment (max_slope_pct uses rolling window)
        max_slope = max(self.segments[sid].max_slope_pct for sid in segment_ids if sid in self.segments)
        difficulty = TerrainAnalyzer.classify_difficulty(slope_pct=max_slope)

        return {
            "total_drop": total_drop,
            "total_length": total_length,
            "avg_gradient": avg_gradient,
            "max_gradient": max_slope,
            "difficulty": difficulty,
            "start_elev": start_elev,
            "current_elev": current_elev,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get resort statistics."""
        if not self.segments:
            return {
                "total_slopes": 0,
                "total_segments": 0,
                "total_vertical_m": 0,
                "total_length_m": 0,
                "longest_run_m": 0,
                "total_lifts": len(self.lifts),
                "total_roads": len(self.roads),
                "total_road_length_m": 0,
            }

        total_vertical = sum(s.total_drop_m for s in self.segments.values())
        total_length = sum(s.length_m for s in self.segments.values())

        longest = 0.0
        for slope in self.slopes.values():
            slope_length = slope.get_total_length(segments=self.segments)
            if slope_length > longest:
                longest = slope_length
        for seg in self.segments.values():
            if seg.length_m > longest:
                longest = seg.length_m

        total_road_length = sum(road.get_total_length(segments=self.segments) for road in self.roads.values())

        return {
            "total_slopes": len(self.slopes),
            "total_segments": len(self.segments),
            "total_vertical_m": total_vertical,
            "total_length_m": total_length,
            "longest_run_m": longest,
            "total_lifts": len(self.lifts),
            "total_roads": len(self.roads),
            "total_road_length_m": total_road_length,
        }

    def get_elevation_range(self) -> tuple[float, float] | None:
        """Return (min, max) elevation across all nodes, or None if empty."""
        if not self.nodes:
            return None
        elevations = [n.elevation for n in self.nodes.values()]
        return min(elevations), max(elevations)

    def get_center(self) -> tuple[float, float] | None:
        """Return (lon, lat) mean of all node coordinates, or None if empty."""
        if not self.nodes:
            return None
        lons = [n.lon for n in self.nodes.values()]
        lats = [n.lat for n in self.nodes.values()]
        return sum(lons) / len(lons), sum(lats) / len(lats)

    def get_parking_nodes(self) -> list[Node]:
        """Nodes where a road meets a slope or lift — computed parking places.

        A parking place is not a stored entity: it exists wherever a road's
        segment shares a node with a slope segment or a lift station. Computed
        fresh so it always tracks the current roads (appears/disappears as
        roads are added or removed).
        """
        road_segment_ids = {sid for road in self.roads.values() for sid in road.segment_ids}
        if not road_segment_ids:
            return []

        # Nodes touched by road segments.
        road_nodes: set[str] = set()
        for sid in road_segment_ids:
            seg = self.segments.get(sid)
            if seg:
                road_nodes.add(seg.start_node_id)
                road_nodes.add(seg.end_node_id)

        # Nodes touched by slopes (their segments) or lift stations.
        slope_segment_ids = {sid for slope in self.slopes.values() for sid in slope.segment_ids}
        ski_nodes: set[str] = set()
        for sid in slope_segment_ids:
            seg = self.segments.get(sid)
            if seg:
                ski_nodes.add(seg.start_node_id)
                ski_nodes.add(seg.end_node_id)
        for lift in self.lifts.values():
            ski_nodes.add(lift.start_node_id)
            ski_nodes.add(lift.end_node_id)

        shared = road_nodes & ski_nodes
        return [self.nodes[nid] for nid in shared if nid in self.nodes]

    def change_token(self) -> tuple[int, int, int, int, int, int]:
        """Return a cheap snapshot that changes on any graph mutation.

        The entity counters only grow; undo_stack length moves on
        commit/cancel/undo. Comparing this token to the last-saved one tells
        the autosave hook whether a write is needed — no full serialization.
        """
        return (
            self._node_counter,
            self._segment_counter,
            self._slope_counter,
            self._lift_counter,
            self._road_counter,
            len(self.undo_stack),
        )

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Serialize entire graph to JSON-compatible dict."""
        return {
            "version": "2.0",
            "nodes": {nid: asdict(node) for nid, node in self.nodes.items()},
            "segments": {sid: asdict(seg) for sid, seg in self.segments.items()},
            "slopes": {slid: asdict(slope) for slid, slope in self.slopes.items()},
            "lifts": {lid: asdict(lift) for lid, lift in self.lifts.items()},
            "roads": {rid: asdict(road) for rid, road in self.roads.items()},
            "counters": {
                "node": self._node_counter,
                "segment": self._segment_counter,
                "slope": self._slope_counter,
                "lift": self._lift_counter,
                "road": self._road_counter,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResortGraph":
        """Deserialize graph from dict."""
        graph = cls()

        for nid, node_data in data["nodes"].items():
            graph.nodes[nid] = Node.from_dict(data=node_data)

        for sid, seg_data in data["segments"].items():
            graph.segments[sid] = PathSegment.from_dict(data=seg_data)

        for slid, slope_data in data["slopes"].items():
            graph.slopes[slid] = Slope.from_dict(data=slope_data)

        for lid, lift_data in data["lifts"].items():
            graph.lifts[lid] = Lift.from_dict(data=lift_data)

        # Roads were added after the first backups shipped, so a pre-roads
        # backup has no "roads" key — default to empty
        for rid, road_data in data.get("roads", {}).items():
            graph.roads[rid] = Road.from_dict(data=road_data)

        # A road-owned segment MUST be kind=ROAD; fail loudly rather than mis-render it as a slope.
        for road in graph.roads.values():
            for seg_id in road.segment_ids:
                seg = graph.segments.get(seg_id)
                assert seg is not None and enum_eq(seg.kind, SegmentKind.ROAD), (
                    f"road {road.id} owns segment {seg_id} with kind "
                    f"{seg.kind if seg else 'MISSING'} — expected ROAD (corrupt/stale save)"
                )

        # Discard orphan segments: any segment owned by no slope or road.
        # Drop them (and any nodes they orphan) rather than keep undeletable data in the graph.
        owned_segment_ids = {sid for slope in graph.slopes.values() for sid in slope.segment_ids}
        owned_segment_ids |= {sid for road in graph.roads.values() for sid in road.segment_ids}
        orphan_segment_ids = [sid for sid in graph.segments if sid not in owned_segment_ids]
        if orphan_segment_ids:
            logger.warning(
                f"Discarding {len(orphan_segment_ids)} orphan segment(s) owned by no slope/road "
                f"(interrupted-build leftovers): {orphan_segment_ids}"
            )
            for sid in orphan_segment_ids:
                del graph.segments[sid]
            graph.cleanup_isolated_nodes()

        counters = data["counters"]
        graph._node_counter = counters["node"]
        graph._segment_counter = counters["segment"]
        graph._slope_counter = counters["slope"]
        graph._lift_counter = counters["lift"]
        graph._road_counter = counters.get("road", 0)

        return graph

    def to_gpx(self) -> str:
        """Export resort to GPX format."""
        import xml.etree.ElementTree as ET

        gpx_ns = "http://www.topografix.com/GPX/1/1"
        ET.register_namespace("", gpx_ns)

        gpx = ET.Element("gpx", xmlns=gpx_ns, version="1.1", creator="Ski Resort Planner")

        metadata = ET.SubElement(gpx, "metadata")
        ET.SubElement(metadata, "name").text = "Ski Resort Planner Export"
        ET.SubElement(metadata, "time").text = datetime.now().isoformat()

        # Track finished slope segments
        finished_segment_ids = set()
        for slope in self.slopes.values():
            finished_segment_ids.update(slope.segment_ids)

        # Export finished slopes
        for slope in self.slopes.values():
            all_points = slope.get_all_points(segments=self.segments)
            difficulty = slope.get_difficulty(segments=self.segments)
            total_length = slope.get_total_length(segments=self.segments)
            total_drop = slope.get_total_drop(segments=self.segments)

            trk = ET.SubElement(gpx, "trk")
            ET.SubElement(trk, "name").text = slope.name
            ET.SubElement(
                trk, "desc"
            ).text = f"{difficulty.capitalize()} - Drop {total_drop:.0f}m - Length {total_length:.0f}m"
            ET.SubElement(trk, "type").text = f"slope_{difficulty}"

            trkseg = ET.SubElement(trk, "trkseg")
            for pt in all_points:
                trkpt = ET.SubElement(trkseg, "trkpt", lat=str(pt.lat), lon=str(pt.lon))
                ET.SubElement(trkpt, "ele").text = f"{pt.elevation:.1f}"

        # Export lifts (using terrain_points for accurate terrain following)
        for lift in self.lifts.values():
            start_node = self.nodes.get(lift.start_node_id)
            end_node = self.nodes.get(lift.end_node_id)
            if not start_node or not end_node:
                raise ValueError(f"Lift {lift.name} has invalid start or end node")

            trk = ET.SubElement(gpx, "trk")
            ET.SubElement(trk, "name").text = f"{lift.name} - {lift.lift_type}"
            vertical = lift.get_vertical_rise(nodes=self.nodes)
            length = lift.get_length_m(nodes=self.nodes)
            ET.SubElement(trk, "desc").text = f"Rise {vertical:.0f}m - Length {length:.0f}m"
            ET.SubElement(trk, "type").text = f"lift_{lift.lift_type}"

            trkseg = ET.SubElement(trk, "trkseg")

            # Use cable_points for 3D visualization (follows cable line with sag)
            if not lift.cable_points:
                raise ValueError(f"Lift {lift.name} must have cable_points for GPX export")
            for pt in lift.cable_points:
                trkpt = ET.SubElement(trkseg, "trkpt", lat=str(pt.lat), lon=str(pt.lon))
                ET.SubElement(trkpt, "ele").text = f"{pt.elevation:.1f}"

        # Export roads (Road is a SegmentPath like Slope; no difficulty)
        for road in self.roads.values():
            all_points = road.get_all_points(segments=self.segments)
            total_length = road.get_total_length(segments=self.segments)
            total_drop = road.get_total_drop(segments=self.segments)

            trk = ET.SubElement(gpx, "trk")
            ET.SubElement(trk, "name").text = road.name
            ET.SubElement(
                trk, "desc"
            ).text = f"Road - Elevation change {-total_drop:+.0f}m - Length {total_length:.0f}m"
            ET.SubElement(trk, "type").text = "road"

            trkseg = ET.SubElement(trk, "trkseg")
            for pt in all_points:
                trkpt = ET.SubElement(trkseg, "trkpt", lat=str(pt.lat), lon=str(pt.lon))
                ET.SubElement(trkpt, "ele").text = f"{pt.elevation:.1f}"

        return ET.tostring(gpx, encoding="unicode", method="xml")

    # =========================================================================
    # Cleanup and Maintenance
    # =========================================================================

    def cleanup_isolated_nodes(self) -> int:
        """Remove nodes not connected to any segment or lift.

        Returns:
            Number of nodes removed.
        """
        isolated_node_ids = [node_id for node_id in self.nodes if self.get_connection_count(node_id=node_id) == 0]

        for node_id in isolated_node_ids:
            del self.nodes[node_id]

        return len(isolated_node_ids)
