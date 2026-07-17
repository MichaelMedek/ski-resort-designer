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

import copy
import logging
import statistics
from dataclasses import asdict
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, NamedTuple, TypedDict, cast

from skiresort_planner.constants import EntityPrefixes, GeometricTuningConfig, MergeConfig, OSMConfig, UndoConfig
from skiresort_planner.core.geo_calculator import GeoCalculator
from skiresort_planner.core.terrain_analyzer import SideDirection, TerrainAnalyzer
from skiresort_planner.model.actions import (
    AddLiftAction,
    AddSegmentsAction,
    DeleteLiftAction,
    DeleteNodesAction,
    DeleteRoadAction,
    DeleteSlopeAction,
    FinishRoadAction,
    FinishSlopeAction,
    ImportOSMAction,
    InsertNodeAction,
    MergeNodesAction,
    UndoAction,
)
from skiresort_planner.model.lift import Lift
from skiresort_planner.model.node import Node
from skiresort_planner.model.path_point import PathPoint, endpoints_match
from skiresort_planner.model.path_segment import PathSegment, SegmentKind
from skiresort_planner.model.path_smoothing import smooth_joined_path
from skiresort_planner.model.proposed_path import ProposedPathSegment
from skiresort_planner.model.road import Road
from skiresort_planner.model.segment_path import SegmentPath
from skiresort_planner.model.slope import Slope
from skiresort_planner.model.undo_handlers import UNDO_HANDLERS

if TYPE_CHECKING:
    from skiresort_planner.core.dem_service import DEMService
    from skiresort_planner.generators.osm_importer import ImportResult

logger = logging.getLogger(__name__)


class NodeDeletability(StrEnum):
    """Why a node can or can't be deleted by the merge-mode Delete tool (reload-safe StrEnum)."""

    DELETABLE_INTERIOR = "deletable_interior"  # pure interior of one chain -> fuse its two segments
    DELETABLE_END = "deletable_end"  # clean endpoint of a >1-segment path -> trim its boundary segment
    IS_PATH_ENDPOINT = "is_path_endpoint"  # shared/branch boundary -> delete that path first
    IS_LIFT_STATION = "is_lift_station"  # a lift station -> delete the lift first
    LAST_SEGMENT = "last_segment"  # sole segment of its path -> delete the whole path instead
    NOT_INTERIOR = "not_interior"  # none of the deletable shapes


# Human sentence per non-deletable reason (one place, so the toast text can't drift from the enum).
_DELETABILITY_REASONS: dict[NodeDeletability, str] = {
    NodeDeletability.IS_PATH_ENDPOINT: "it is a junction of another path — delete that path first",
    NodeDeletability.IS_LIFT_STATION: "it is a lift station — delete the lift first",
    NodeDeletability.LAST_SEGMENT: "it is the only segment of its path — delete the path instead",
    NodeDeletability.NOT_INTERIOR: "it is not an interior or end node of a single path",
}


def deletability_reason(node_id: str, reason: NodeDeletability) -> str:
    """Human sentence for why a node can't be deleted (used by the UnableToDeleteMessage toast)."""
    return f"{node_id} {_DELETABILITY_REASONS[reason]}"


# Add-node-on-path rejection sentences (one place, mirroring _DELETABILITY_REASONS).
_INSERT_REJECT_NOT_FINISHED = "click a finished path to add a node"
_INSERT_REJECT_TOO_CLOSE = "too close to an existing node (within {gap:.0f}m)"


def _chain_node_sequence(segments: list[PathSegment]) -> list[str]:
    """The ordered node ids a segment chain visits: first segment's start, then each segment's end."""
    return [segments[0].start_node_id, *(s.end_node_id for s in segments)]


class SegmentStats(TypedDict):
    """Running stats for a set of segments (numeric fields default to 0.0 when absent)."""

    total_drop: float
    total_length: float
    avg_gradient: float
    max_gradient: float
    difficulty: str
    start_elev: float
    current_elev: float


class ResortStats(TypedDict):
    """Whole-resort summary counts and totals."""

    total_slopes: int
    total_segments: int
    total_vertical_m: float
    total_length_m: float
    longest_run_m: float
    total_lifts: int
    total_roads: int
    total_road_length_m: float


class OSMImportResult(NamedTuple):
    """Counts from one import_osm call (named so callers don't guess tuple positions)."""

    slopes_added: int
    lifts_added: int
    duplicates_skipped: int  # entities skipped because the graph already has that endpoint fingerprint


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

    def drop_undo_actions_for_removed_segments(self) -> None:
        """Drop undo entries left dangling after their segments were removed.

        Keyed on the ``segment_ids`` attribute, so it needs no per-type list: actions without it
        (delete/merge, which snapshot their own state) are self-contained and always kept. Called by
        every segment-removal path — delete_slope/road, merge collapse, and the cancel-build discard.
        """

        def keep(action: UndoAction) -> bool:
            segment_ids: tuple[str, ...] = getattr(action, "segment_ids", ())
            return not segment_ids or any(sid in self.segments for sid in segment_ids)

        self.undo_stack = [action for action in self.undo_stack if keep(action)]

    # =========================================================================
    # Node Operations
    # =========================================================================

    def find_nearest_node(
        self,
        lon: float,
        lat: float,
        threshold_m: float = GeometricTuningConfig.STEP_SIZE_M,
    ) -> Node | None:
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
        lift_count = sum(1 for lift in self.lifts.values() if lift.touches(node_id))
        return len(self._segments_touching(node_id)) + lift_count

    def _segments_touching(self, node_id: str) -> list[PathSegment]:
        """Every segment with node_id as an endpoint."""
        return [s for s in self.segments.values() if node_id in (s.start_node_id, s.end_node_id)]

    def _side_slope_for_points(self, points: list[PathPoint]) -> tuple[float, SideDirection]:
        """Side slope (pct, direction) for a segment from its first two points — the single call site.

        Extracted from commit_paths so delete/insert can't drift from how the builder computes it.
        """
        info = TerrainAnalyzer.compute_side_slope(
            start_lon=points[0].lon, start_lat=points[0].lat, end_lon=points[1].lon, end_lat=points[1].lat
        )
        return info.slope_pct, info.direction

    def _build_segment(
        self, points: list[PathPoint], start_node_id: str, end_node_id: str, kind: SegmentKind
    ) -> PathSegment:
        """Create + register a PathSegment between two nodes (id/name from the counter, side slope
        computed from points). The single PathSegment assembly site — shared by commit and insert.
        """
        side_slope_pct, side_slope_dir = self._side_slope_for_points(points)
        segment = PathSegment(
            id=self._next_segment_id(),
            name=f"Segment {self._segment_counter}",
            points=points,
            start_node_id=start_node_id,
            end_node_id=end_node_id,
            side_slope_pct=side_slope_pct,
            side_slope_dir=side_slope_dir,
            kind=kind,
        )
        self.segments[segment.id] = segment
        return segment

    def node_deletability(self, node_id: str) -> NodeDeletability:
        """Why a node can/can't be deleted. Single source used by the UI button + the delete op.

        Deletable when the node is a PURE INTERIOR node of one finished chain (its two segments
        fuse into one) OR a CLEAN ENDPOINT of a >1-segment finished path (its lone boundary segment
        trims off). Never for lift stations or shared/branch junctions.
        """
        # A lift station is never deletable (delete the lift first) — checked before anything else.
        if any(lift.touches(node_id) for lift in self.lifts.values()):
            return NodeDeletability.IS_LIFT_STATION

        touching = self._segments_touching(node_id)
        owners = {owner.id: owner for s in touching if (owner := self.get_entity_by_segment_id(s.id)) is not None}
        sole_owner = next(iter(owners.values())) if len(owners) == 1 else None
        is_boundary_anywhere = any(p.touches(node_id) for p in self.segment_path_entities)

        # Exactly one classification per node — each an explicit positive condition, no silent default.
        if len(touching) == 2 and sole_owner is not None and not is_boundary_anywhere:
            # Interior of one finished chain (a boundary-of-another node is a branch → endpoint below).
            return NodeDeletability.DELETABLE_INTERIOR
        if len(touching) == 1 and sole_owner is not None and sole_owner.touches(node_id):
            # Clean endpoint of one path; a sole-segment path would be emptied → delete the path.
            if len(sole_owner.segment_ids) <= 1:
                return NodeDeletability.LAST_SEGMENT
            return NodeDeletability.DELETABLE_END
        if is_boundary_anywhere:
            # A boundary of some path but shared/branch (or interior of one + boundary of another).
            return NodeDeletability.IS_PATH_ENDPOINT
        # Anything else (isolated, unfinished-only, or a multi-path crossing) is simply not deletable.
        return NodeDeletability.NOT_INTERIOR

    def delete_nodes_rejection(self, node_ids: list[str]) -> str | None:
        """Why the selected nodes can't be deleted together, or None if they can.

        Single source for the delete preconditions, mirroring insert_node_rejection: every node must
        be individually deletable AND the deletions together must leave each affected path with ≥2
        nodes (an end node + the sole interior node of a 2-segment slope would empty it). The UI
        pre-checks this for a friendly toast; delete_nodes calls it too as a fail-fast backstop.
        """
        # Check 1 (per node): every selected node must be individually deletable on its own shape.
        # The first one that isn't (lift station, branch junction, sole segment, …) names the reason.
        for nid in node_ids:
            reason = self.node_deletability(nid)
            if reason not in (NodeDeletability.DELETABLE_INTERIOR, NodeDeletability.DELETABLE_END):
                return deletability_reason(node_id=nid, reason=reason)

        # Check 2 (per path): even all-individually-deletable nodes can, TOGETHER, empty a path (e.g.
        # deleting both nodes of a 2-segment slope). Refuse any owning path left with < 2 surviving nodes.
        drop = set(node_ids)
        for path in self._paths_owning_nodes(node_ids):
            node_seq = _chain_node_sequence([self.segments[sid] for sid in path.segment_ids])
            if sum(1 for n in node_seq if n not in drop) < 2:
                return f"this would delete the whole path {path.name} — delete the path instead"
        return None  # both checks passed → the selection is safe to delete

    def _owning_path(self, node_id: str) -> "SegmentPath | None":
        """The finished slope/road that owns a segment touching node_id, or None."""
        touching = self._segments_touching(node_id)
        return self.get_entity_by_segment_id(touching[0].id) if touching else None

    def _paths_owning_nodes(self, node_ids: list[str]) -> list["SegmentPath"]:
        """The distinct finished paths owning the given nodes — the single node→path grouping used by
        both the delete precondition check and the delete executor, so they can't drift.
        """
        by_id = {owner.id: owner for nid in node_ids if (owner := self._owning_path(nid)) is not None}
        return list(by_id.values())

    # =========================================================================
    # Commit Operations
    # =========================================================================

    def commit_paths(
        self,
        paths: list[ProposedPathSegment],
        *,
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

            # Side slope needs ≥2 points; the segment factory computes it from the first two.
            if len(path.points) < 2:
                raise ValueError(
                    f"Path must have at least 2 points to compute side slope, got {len(path.points)}: {path}"
                )

            # Create segment (metrics computed as properties from points)
            segment = self._build_segment(
                points=path.points, start_node_id=start_node.id, end_node_id=end_node.id, kind=path.kind
            )
            new_segment_ids.append(segment.id)
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

    def _resolve_finish_endpoints(self, segment_ids: list[str]) -> tuple[PathSegment, PathSegment, Node, Node, float]:
        """Validate a finish request and return (first_seg, last_seg, start_node, end_node, avg_bearing).

        Raises ValueError if the segment list is empty or any segment/endpoint node is missing.
        Shared by finish_slope / finish_road (validation + bearing only).
        """
        if not segment_ids:
            raise ValueError("cannot finish: empty segment_ids")

        first_seg = self.segments.get(segment_ids[0])
        last_seg = self.segments.get(segment_ids[-1])
        if not first_seg or not last_seg:
            raise ValueError(
                f"cannot finish: missing segment(s) - first={segment_ids[0]} exists={first_seg is not None}, "
                f"last={segment_ids[-1]} exists={last_seg is not None}"
            )

        start_node = self.nodes.get(first_seg.start_node_id)
        end_node = self.nodes.get(last_seg.end_node_id)
        if not start_node or not end_node:
            raise ValueError(
                f"cannot finish: missing endpoint node(s) - start={first_seg.start_node_id} "
                f"exists={start_node is not None}, end={last_seg.end_node_id} exists={end_node is not None}"
            )

        avg_bearing = GeoCalculator.initial_bearing_deg(
            lon1=start_node.lon, lat1=start_node.lat, lon2=end_node.lon, lat2=end_node.lat
        )
        return first_seg, last_seg, start_node, end_node, avg_bearing

    def _smooth_finished_path(self, segment_ids: list[str], smoothing_factor: float) -> None:
        """Whole-path smooth a finished entity across its junctions, in place.

        EVERY node on the path (outer endpoints + every junction) stays pinned exactly on the
        ribbon, so markers sit on the path and any node can be a branch point; only the shape
        between nodes rounds. A single segment is smoothed too. Never rejects — a road may
        drift over the ±15% build cap here (bridge/cut/fill), which is intentional.

        smoothing_factor: higher = smoother (roads); lower hugs terrain (slopes).
        """
        segments = [self.segments[sid] for sid in segment_ids]
        assert len(segments) > 0, f"_smooth_finished_path: segment_ids={segment_ids} resolved to empty segments list"
        # Boundary nodes: start of the first segment, then each segment's end node.
        boundary_node_ids = _chain_node_sequence(segments)

        before = max(seg.max_slope_pct for seg in segments)
        smoothed = smooth_joined_path(
            segment_point_lists=[seg.points for seg in segments],
            node_anchors=[self.nodes[nid].location for nid in boundary_node_ids],
            step_m=GeometricTuningConfig.RESAMPLE_STEP_M,
            smoothing_factor=smoothing_factor,
            node_weight=GeometricTuningConfig.NODE_WEIGHT,
            corridor_weight=GeometricTuningConfig.CORRIDOR_WEIGHT,
        )
        for seg, pts in zip(segments, smoothed, strict=True):
            seg.points = pts
        after = max(seg.max_slope_pct for seg in segments)
        logger.info(f"Smoothed finished path {segment_ids}: max_slope_pct {before:.1f}% -> {after:.1f}%")

    # =========================================================================
    # Slope Operations
    # =========================================================================

    def finish_slope(
        self,
        segment_ids: list[str],
        name: str | None = None,
        *,
        record_undo: bool = True,
    ) -> Slope:
        """Finish a slope by grouping segments.

        Args:
            segment_ids: List of segment IDs to group
            name: Optional custom name (generates creative name if None)
            record_undo: Push a FinishSlopeAction. False when a batch (OSM import) owns the undo.

        Returns:
            Created Slope.
        """
        first_seg, last_seg, start_node, end_node, avg_bearing = self._resolve_finish_endpoints(segment_ids=segment_ids)
        assert all(sid in self.segments for sid in segment_ids), (
            f"finish_slope: segment_ids contain missing segments {[s for s in segment_ids if s not in self.segments]}"
        )
        self._smooth_finished_path(
            segment_ids=segment_ids, smoothing_factor=GeometricTuningConfig.SLOPE_SMOOTHING_FACTOR
        )

        slope_id = self._next_slope_id()
        # Difficulty from the steepest section (max_slope_pct over rolling windows).
        max_slope = max(self.segments[sid].max_slope_pct for sid in segment_ids)
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
        if record_undo:
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
        name: str | None = None,
    ) -> Road:
        """Group committed segments into a vehicle Road.

        Records a FinishRoadAction (mirrors finish_slope): undo ungroups the road
        but keeps its segments, which carry their own AddSegmentsAction entries.

        Args:
            segment_ids: Segment IDs the road is made of.
            name: Optional custom name (generates a compass name if None).

        Returns:
            Created Road.
        """
        first_seg, last_seg, _start_node, _end_node, avg_bearing = self._resolve_finish_endpoints(
            segment_ids=segment_ids
        )
        assert all(sid in self.segments for sid in segment_ids), (
            f"finish_road: segment_ids contain missing segments {[s for s in segment_ids if s not in self.segments]}"
        )
        self._smooth_finished_path(
            segment_ids=segment_ids, smoothing_factor=GeometricTuningConfig.ROAD_SMOOTHING_FACTOR
        )

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

        deleted_segments = [self.segments[seg_id] for seg_id in road.segment_ids]
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
        self.drop_undo_actions_for_removed_segments()
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
        name: str | None = None,
        *,
        record_undo: bool = True,
    ) -> Lift:
        """Add a lift between two nodes.

        Args:
            start_node_id: ID of bottom station
            end_node_id: ID of top station
            lift_type: Type of lift
            dem: DEM service for terrain sampling
            name: Optional custom name (Lift.create generates a creative name if None)
            record_undo: Push an AddLiftAction. False when a batch (OSM import) owns the undo.

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
        if name is not None:
            lift.name = name

        self.lifts[lift_id] = lift
        if record_undo:
            self._push_undo(AddLiftAction(lift_id=lift_id))

        return lift

    def import_osm(self, result: "ImportResult", *, dem: "DEMService") -> OSMImportResult:
        """Add an OSM import (lifts + slope chains) as ONE undoable unit, tagging every entity.

        Each slope chain (its ordered segment point-lists + optional name) is committed segment by
        segment and finished as one slope; each lift (bottom, top, type, name) becomes a lift with
        regenerated pylons. Every created slope, segment, and lift is stamped with `result.source`
        (e.g. EntitySource.OSM). All individual undo entries are suppressed and replaced by a single
        ImportOSMAction, so one undo removes the whole import; newly-created nodes are tracked and
        removed on undo, reused (shared) nodes are left alone.

        Re-import is conservative: an incoming slope/lift is skipped (counted as a duplicate) if the
        graph already holds an entity with a matching endpoint fingerprint (within OSM_DEDUP_TOL_M)
        OR the same source and a matching non-empty name — we err toward NOT re-importing.

        Args:
            result: The importer output — lifts, slope_chains, and the provenance source.
            dem: DEM service for lift terrain sampling / pylon placement.

        Returns:
            OSMImportResult(slopes_added, lifts_added, duplicates_skipped).
        """
        source = result.source
        nodes_before = set(self.nodes)
        slope_ids: list[str] = []
        lift_ids: list[str] = []
        segment_ids: list[str] = []
        duplicates = 0

        for chain, name in result.slope_chains:
            head, tail = chain[0][0], chain[-1][-1]
            if self.has_endpoint_duplicate(a=head, b=tail, tol_m=OSMConfig.OSM_DEDUP_TOL_M) or self._has_source_name(
                name=name, source=source
            ):
                logger.debug(f"import_osm: skipping duplicate slope '{name}' at endpoints {head} -> {tail}")
                duplicates += 1
                continue
            chain_seg_ids: list[str] = []
            for points in chain:  # commit each segment top→bottom so finish_slope's chain stitch is valid
                segments_before = set(self.segments)
                self.commit_paths(paths=[ProposedPathSegment(points=points, kind=SegmentKind.SLOPE)], record_undo=False)
                (seg_id,) = (sid for sid in self.segments if sid not in segments_before)
                chain_seg_ids.append(seg_id)
            slope = self.finish_slope(segment_ids=chain_seg_ids, name=name, record_undo=False)
            slope.source = source
            for seg_id in chain_seg_ids:  # stamp AFTER finish_slope (it rewrites name, never source)
                self.segments[seg_id].source = source
            segment_ids.extend(chain_seg_ids)
            slope_ids.append(slope.id)

        for bottom, top, lift_type, lift_name in result.lifts:
            if self.has_endpoint_duplicate(a=bottom, b=top, tol_m=OSMConfig.OSM_DEDUP_TOL_M) or self._has_source_name(
                name=lift_name, source=source
            ):
                logger.debug(
                    f"import_osm: skipping duplicate lift '{lift_name}' (type={lift_type}) at endpoints "
                    f"{bottom} -> {top}"
                )
                duplicates += 1
                continue
            start, _ = self.get_or_create_node(lon=bottom.lon, lat=bottom.lat, elevation=bottom.elevation)
            end, _ = self.get_or_create_node(lon=top.lon, lat=top.lat, elevation=top.elevation)
            lift = self.add_lift(
                start_node_id=start.id,
                end_node_id=end.id,
                lift_type=lift_type,
                dem=dem,
                name=lift_name,
                record_undo=False,
            )
            lift.source = source
            lift_ids.append(lift.id)

        created_node_ids = tuple(nid for nid in self.nodes if nid not in nodes_before)
        self._push_undo(
            ImportOSMAction(
                slope_ids=tuple(slope_ids),
                lift_ids=tuple(lift_ids),
                segment_ids=tuple(segment_ids),
                node_ids=created_node_ids,
            )
        )
        logger.info(
            f"OSM import: {len(slope_ids)} slopes, {len(lift_ids)} lifts, "
            f"{len(created_node_ids)} new nodes, {duplicates} duplicates skipped"
        )
        return OSMImportResult(slopes_added=len(slope_ids), lifts_added=len(lift_ids), duplicates_skipped=duplicates)

    def _has_source_name(self, name: str | None, source: str) -> bool:
        """True if a slope or lift with this same `source` and a matching non-empty `name` exists —
        a named entity already imported from that source (skip re-importing). Empty/None never matches.
        """
        if not name:
            return False
        slope_hit = any(s.source == source and s.name == name for s in self.slopes.values())
        lift_hit = any(lf.source == source and lf.name == name for lf in self.lifts.values())
        return slope_hit or lift_hit

    def has_endpoint_duplicate(
        self, a: PathPoint, b: PathPoint, *, tol_m: float = GeometricTuningConfig.STEP_SIZE_M
    ) -> bool:
        """True if an existing slope or lift already spans endpoints `a` and `b` (within `tol_m`).

        Direct geometric comparison against each stored entity's endpoints (endpoints_match, either
        orientation). Default `tol_m` is the import snap distance; the OSM re-import passes a coarser
        OSM_DEDUP_TOL_M. No coordinate rounding or shared-node lookup, so it stays correct where many
        runs cluster around one junction.
        """
        pair = (a, b)
        slope_match = any(
            endpoints_match(pair_a=pair, pair_b=slope.endpoints(nodes=self.nodes), tol_m=tol_m)
            for slope in self.slopes.values()
        )
        lift_match = any(
            endpoints_match(pair_a=pair, pair_b=lift.endpoints(nodes=self.nodes), tol_m=tol_m)
            for lift in self.lifts.values()
        )
        return slope_match or lift_match

    def max_node_span_m(self, node_ids: list[str]) -> float:
        """Largest pairwise distance (m) among the given nodes; 0 for fewer than two."""
        pts = [self.nodes[nid] for nid in node_ids]
        return max(
            (a.distance_to(lon=b.lon, lat=b.lat) for i, a in enumerate(pts) for b in pts[i + 1 :]),
            default=0.0,
        )

    def merge_nodes(self, node_ids: list[str], dem: "DEMService") -> None:
        """Collapse several nodes into one, at their median lat/lon (elevation re-sampled from DEM).

        The survivor (first id) moves to the median point; every segment/lift endpoint on the other
        nodes is repointed onto it, then the others are deleted. ONE undoable MergeNodesAction.

        Raises:
            ValueError: fewer than two nodes, or any pair farther apart than MergeConfig.MAX_SPAN_M
                (the caller should pre-check via max_node_span_m to show a friendly message).
        """
        if len(node_ids) < 2:
            raise ValueError(f"merge_nodes needs at least two nodes, got {len(node_ids)}")
        if self.max_node_span_m(node_ids) > MergeConfig.MAX_SPAN_M:
            raise ValueError(f"nodes span more than {MergeConfig.MAX_SPAN_M:.0f}m apart — refusing to merge")

        survivor_id, merged_ids = node_ids[0], node_ids[1:]
        survivor = self.nodes[survivor_id]
        survivor_before = Node(id=survivor.id, location=survivor.location)
        merged = set(merged_ids)
        # The survivor moves to the median too, so a builder touching ANY of the selected nodes
        # (survivor included) needs re-stitching — not just those on the merged-away nodes.
        touched = set(node_ids)

        # Which builders touch the selected nodes — snapshot them BEFORE any mutation so undo can
        # restore their exact pre-merge state (all mutated in place below).
        affected_segments = [
            s for s in self.segments.values() if s.start_node_id in touched or s.end_node_id in touched
        ]
        affected_lifts = [ln for ln in self.lifts.values() if ln.start_node_id in touched or ln.end_node_id in touched]
        # Slopes/roads store their own boundary node ids (mirroring their first/last segment), so a
        # merge that repoints those nodes must repoint the entity boundary too. Include any path that
        # OWNS an affected segment, not just those whose boundary is touched.
        affected_paths_by_id: dict[str, SegmentPath] = {
            p.id: p for p in self.segment_path_entities if p.start_node_id in touched or p.end_node_id in touched
        }
        for seg in affected_segments:
            owner = self.get_entity_by_segment_id(seg.id)
            if owner is not None:
                affected_paths_by_id[owner.id] = owner
        affected_paths: list[SegmentPath] = list(affected_paths_by_id.values())
        segments_before = tuple(copy.deepcopy(s) for s in affected_segments)
        lifts_before = tuple(copy.deepcopy(ln) for ln in affected_lifts)
        paths_before = tuple(copy.deepcopy(p) for p in affected_paths)

        # Repoint every segment / lift / slope / road endpoint on a merged-away node onto the survivor
        # (one uniform rule for every id-holder — the same convention the whole model uses).
        for seg in self.segments.values():
            if seg.start_node_id in merged:
                seg.start_node_id = survivor_id
            if seg.end_node_id in merged:
                seg.end_node_id = survivor_id
        for lift in self.lifts.values():
            if lift.start_node_id in merged:
                lift.start_node_id = survivor_id
            if lift.end_node_id in merged:
                lift.end_node_id = survivor_id
        for path in self.segment_path_entities:
            if path.start_node_id in merged:
                path.start_node_id = survivor_id
            if path.end_node_id in merged:
                path.end_node_id = survivor_id

        # Move the survivor to the median point; drop the merged-away nodes.
        deleted_nodes = tuple(self.nodes[nid] for nid in merged_ids)
        lats = [self.nodes[nid].lat for nid in node_ids]
        lons = [self.nodes[nid].lon for nid in node_ids]
        med_lat, med_lon = statistics.median(lats), statistics.median(lons)
        med_elev = dem.get_elevation(lon=med_lon, lat=med_lat)
        if med_elev is None:
            raise ValueError(f"median point ({med_lat:.5f}, {med_lon:.5f}) has no DEM elevation")
        survivor.location = PathPoint(lon=med_lon, lat=med_lat, elevation=med_elev)
        for nid in merged_ids:
            del self.nodes[nid]
        self.cleanup_isolated_nodes()

        # A merge collapses an entity onto one node (both boundaries → survivor: zero length) — delete
        # it here, inside the same MergeNodesAction, tracking the segments it drops.
        removed_segment_ids: set[str] = set()
        collapsed_ids = {p.id for p in affected_paths if p.start_node_id == p.end_node_id}
        for path in affected_paths:
            if path.id in collapsed_ids:
                removed_segment_ids.update(self._remove_collapsed_path(path))

        # A surviving multi-segment path can still hold a MIDDLE segment whose own endpoints both
        # became the survivor (a zero-length "curl") — splice it out so the path stays continuous.
        for path in affected_paths:
            if path.id not in collapsed_ids:
                removed_segment_ids.update(self._drop_collapsed_segments_in_chain(path))

        # Re-stitch every surviving affected builder fresh from the moved endpoints (each model owns
        # its recompute; a road is just segments with kind=ROAD, so no per-kind branch is needed).
        # A segment we just dropped with its collapsed parent or as a middle curl is skipped.
        for seg in affected_segments:
            if seg.id in removed_segment_ids:
                continue
            seg.restitch(start_node=self.nodes[seg.start_node_id], end_node=self.nodes[seg.end_node_id], dem=dem)
        for lift in affected_lifts:
            if lift.start_node_id == lift.end_node_id:
                self._remove_collapsed_lift(lift)
                continue
            lift.rebuild(start_node=self.nodes[lift.start_node_id], end_node=self.nodes[lift.end_node_id], dem=dem)

        self._push_undo(
            MergeNodesAction(
                survivor_id=survivor_id,
                survivor_before=survivor_before,
                deleted_nodes=deleted_nodes,
                segments_before=segments_before,
                lifts_before=lifts_before,
                paths_before=paths_before,
            )
        )
        # A merge can collapse a zero-length slope/road and delete its segments (_remove_collapsed_path);
        # drop any now-stale AddSegmentsAction that referenced them.
        self.drop_undo_actions_for_removed_segments()
        logger.info(f"Merged {len(node_ids)} nodes into {survivor_id} at ({med_lat:.5f}, {med_lon:.5f})")

    def _remove_collapsed_path(self, path: "SegmentPath") -> list[str]:
        """Remove a collapsed slope/road (+ its segments) during a merge; return the removed segment ids.

        No own undo action / no cleanup: the enclosing merge owns the single MergeNodesAction and
        its snapshot already carries this entity and its segments for restore.
        """
        for seg_id in path.segment_ids:
            self.segments.pop(seg_id, None)
        self.entity_dict_for_kind(path.kind).pop(path.id, None)
        logger.info(f"Merge collapsed {path.name} to zero length — deleted it and its {len(path.segment_ids)} segments")
        return path.segment_ids

    def _remove_collapsed_lift(self, lift: Lift) -> None:
        """Remove a collapsed lift (both stations merged onto one node) during a merge.

        No own undo action / no cleanup: the enclosing merge owns the single MergeNodesAction and
        its snapshot already carries this lift for restore.
        """
        self.lifts.pop(lift.id, None)
        logger.info(f"Merge collapsed lift {lift.name} to zero length — deleted it")

    def _drop_collapsed_segments_in_chain(self, path: "SegmentPath") -> list[str]:
        """Drop any zero-length (start==end) segment from a surviving path's chain; return their ids.

        Used after a merge repoints endpoints: a segment whose own two endpoints collapsed onto the
        survivor is a curl. Its neighbours already meet at the survivor, so removing it from the
        chain (and re-deriving the boundary ids) keeps the path continuous.
        """
        collapsed = [
            sid for sid in path.segment_ids if self.segments[sid].start_node_id == self.segments[sid].end_node_id
        ]
        if not collapsed:
            return []
        path.segment_ids = [sid for sid in path.segment_ids if sid not in collapsed]
        for sid in collapsed:
            del self.segments[sid]
        # Re-derive boundaries from the surviving chain (the first/last segment's outer node).
        path.start_node_id = self.segments[path.segment_ids[0]].start_node_id
        path.end_node_id = self.segments[path.segment_ids[-1]].end_node_id
        logger.info(f"Merge dropped {len(collapsed)} zero-length segment(s) from {path.name}")
        return collapsed

    # =========================================================================
    # Node delete / insert (merge-mode editing tools)
    # =========================================================================

    def delete_nodes(self, node_ids: list[str], dem: "DEMService") -> None:
        """Delete path nodes, keeping the rest of each path (interior fusion / clean-endpoint trim).

        An interior node fuses its two segments into one; a clean endpoint trims its lone boundary
        segment. The caller pre-checks delete_nodes_rejection; this re-checks and raises ValueError
        as a fail-fast backstop (never a user path). ONE undoable DeleteNodesAction.
        """
        rejection = self.delete_nodes_rejection(node_ids)
        if rejection is not None:
            raise ValueError(f"delete_nodes: {rejection}")

        to_delete = set(node_ids)
        # Each deletable node belongs to exactly one finished path (guaranteed by the rejection check).
        affected_paths = self._paths_owning_nodes(node_ids)

        deleted_nodes = tuple(self.nodes[nid] for nid in node_ids)
        paths_before = tuple(copy.deepcopy(p) for p in affected_paths)
        segments_before = tuple(copy.deepcopy(self.segments[sid]) for p in affected_paths for sid in p.segment_ids)

        for path in affected_paths:
            self._rebuild_chain_without_nodes(path=path, drop_nodes=to_delete, dem=dem)

        # Only remove a selected node once nothing references it.
        self.cleanup_isolated_nodes()

        # Post-condition: the graph must stay referentially intact — no segment may point at a deleted
        # node. Fail here (at the source) rather than let a later click crash on the dangling id.
        for seg in self.segments.values():
            assert seg.start_node_id in self.nodes and seg.end_node_id in self.nodes, (
                f"delete_nodes left segment {seg.id} referencing a missing node "
                f"({seg.start_node_id}->{seg.end_node_id})"
            )

        self._push_undo(
            DeleteNodesAction(deleted_nodes=deleted_nodes, paths_before=paths_before, segments_before=segments_before)
        )
        self.drop_undo_actions_for_removed_segments()
        logger.info(f"Deleted {len(node_ids)} node(s) across {len(affected_paths)} path(s)")

    def _rebuild_chain_without_nodes(self, path: "SegmentPath", drop_nodes: set[str], dem: "DEMService") -> None:
        """Rewrite path.segment_ids with drop_nodes removed, via a single node-sequence walk.

        The chain is the node sequence [n0, n1, …, nN] (segment i spans node i→i+1). Keep the
        surviving nodes; between each consecutive surviving pair, fuse the spanned segments into one
        (points concatenated, first segment's id reused). Segments before the first / after the last
        surviving node are trimmed. One pass, so a deleted end node adjacent to another deleted node
        can't leave a dangling boundary.
        """
        seg_ids = list(path.segment_ids)
        segs = [self.segments[sid] for sid in seg_ids]
        node_seq = _chain_node_sequence(segs)
        kept_idx = [i for i, n in enumerate(node_seq) if n not in drop_nodes]
        assert len(kept_idx) >= 2, f"delete would leave <2 nodes on {path.id} (caller must pre-check)"

        # Fuse original segments [a, b) into one new segment per surviving-node pair (a, b).
        new_ids: list[str] = []
        for a, b in zip(kept_idx, kept_idx[1:], strict=False):
            head = segs[a]
            points = list(head.points)
            for s in segs[a + 1 : b]:
                points = points + s.points[1:]  # drop the duplicate junction point
            head.points = points
            head.start_node_id = node_seq[a]
            head.end_node_id = node_seq[b]
            new_ids.append(head.id)

        # Drop every original segment not reused as a fused head (consumed mid-run, or a trimmed end).
        for sid in seg_ids:
            if sid not in new_ids:
                del self.segments[sid]

        # Recompute side slope + re-drape each fused head (endpoints unchanged, so restitch keeps it
        # on-terrain — mirrors merge/commit).
        for sid in new_ids:
            seg = self.segments[sid]
            seg.side_slope_pct, seg.side_slope_dir = self._side_slope_for_points(seg.points)
            seg.restitch(start_node=self.nodes[seg.start_node_id], end_node=self.nodes[seg.end_node_id], dem=dem)

        path.segment_ids = new_ids
        path.start_node_id = self.segments[new_ids[0]].start_node_id
        path.end_node_id = self.segments[new_ids[-1]].end_node_id

    def _nearest_vertex_index(self, seg: PathSegment, lon: float, lat: float) -> int:
        """Index of the segment vertex closest to (lon, lat). Segments are dense (~7m), so the click
        effectively lands ON a vertex — snapping to it needs no projection or DEM (it has elevation).
        """
        return min(range(len(seg.points)), key=lambda i: seg.points[i].distance_to(other=PathPoint(lon, lat, 0.0)))

    def insert_node_rejection(self, segment_id: str, lon: float, lat: float) -> str | None:
        """Why a node can't be inserted on this segment at (lon, lat), or None if it can.

        Single source for the add-node preconditions (unknown/unfinished segment, or the nearest
        vertex too close to an endpoint node to make a real interior split). The UI pre-checks this
        for a friendly toast; insert_node_on_path calls it too as a fail-fast backstop.
        """
        # segment_id is an external map-click id (may be stale after a rerun/delete) — a tolerated
        # fallback, surfaced as a reason rather than crashing.
        seg = self.segments.get(segment_id)
        if seg is None or self.get_entity_by_segment_id(segment_id) is None:
            return _INSERT_REJECT_NOT_FINISHED
        vertex = seg.points[self._nearest_vertex_index(seg, lon, lat)]
        gap = GeometricTuningConfig.STEP_SIZE_M
        start_node, end_node = self.nodes[seg.start_node_id], self.nodes[seg.end_node_id]
        if (
            start_node.distance_to(lon=vertex.lon, lat=vertex.lat) < gap
            or end_node.distance_to(lon=vertex.lon, lat=vertex.lat) < gap
        ):
            return _INSERT_REJECT_TOO_CLOSE.format(gap=gap)
        return None

    def insert_node_on_path(self, segment_id: str, lon: float, lat: float) -> str:
        """Insert a node on a segment at the vertex nearest (lon, lat), splitting it into two and
        updating the owning path's segment_ids [seg] -> [A', B']. Returns the new node id.

        The new node IS the nearest existing vertex (no new geometry, no DEM). The caller pre-checks
        insert_node_rejection; this re-checks and raises ValueError as a fail-fast backstop.
        """
        rejection = self.insert_node_rejection(segment_id=segment_id, lon=lon, lat=lat)
        if rejection is not None:
            raise ValueError(f"insert_node_on_path: {rejection}")
        seg = self.segments[segment_id]
        owner = self.get_entity_by_segment_id(segment_id)
        assert owner is not None  # guaranteed by insert_node_rejection

        path_before = copy.deepcopy(owner)
        segment_before = copy.deepcopy(seg)

        # Split at the nearest vertex; it becomes the new node and is shared by both halves (the
        # junction point A' ends on and B' starts on, exactly as adjacent segments already share nodes).
        i = self._nearest_vertex_index(seg, lon, lat)
        split_pt = seg.points[i]
        node = Node(id=self._next_node_id(), location=split_pt)
        self.nodes[node.id] = node

        new_ids: list[str] = []
        for pts, s_node, e_node in (
            (seg.points[: i + 1], seg.start_node_id, node.id),
            (seg.points[i:], node.id, seg.end_node_id),
        ):
            new_seg = self._build_segment(points=list(pts), start_node_id=s_node, end_node_id=e_node, kind=seg.kind)
            new_ids.append(new_seg.id)

        idx = owner.segment_ids.index(segment_id)
        owner.segment_ids[idx : idx + 1] = new_ids  # boundary ids unchanged (interior insert)
        del self.segments[segment_id]

        self._push_undo(
            InsertNodeAction(
                created_node_id=node.id,
                created_segment_ids=tuple(new_ids),
                path_before=path_before,
                segment_before=segment_before,
            )
        )
        self.drop_undo_actions_for_removed_segments()
        logger.info(f"Inserted node {node.id} on {owner.name}, split {segment_id} into {new_ids}")
        return node.id

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
        # Dispatch via the UNDO_HANDLERS registry keyed by ActionType.name.
        UNDO_HANDLERS[action.action_type.name].apply_undo(self, action)
        return action

    def rename(self, entity_id: str, new_name: str) -> None:
        """Rename a slope, lift, or road by id (and its segments, for segment-path entities).

        Ids are uniquely prefixed (SL/L/R), so no kind is needed. Slopes and roads also rename their
        segments — finish_slope/finish_road set segment names, and the elevation profile shows them.
        """
        # Find the segment-path entity by id across every SegmentKind.
        segment_path: SegmentPath | None = None
        for kind in SegmentKind:
            found = self.entity_dict_for_kind(kind).get(entity_id)
            if found is not None:
                segment_path = found
                break
        entity = segment_path or self.lifts.get(entity_id)
        if entity is None:
            raise KeyError(f"No slope/lift/road with id {entity_id}")
        entity.name = new_name
        # A SegmentPath (slope/road) also renames its segments; a Lift has none.
        if segment_path is not None:
            for seg_id in segment_path.segment_ids:
                self.segments[seg_id].name = new_name
        logger.info(f"Renamed {entity_id} to '{new_name}'")

    def delete_slope(self, slope_id: str) -> bool:
        """Delete a slope and its segments.

        Args:
            slope_id: ID of slope to delete

        Returns:
            True if deleted, False if not found.
        """
        slope = self.slopes.get(slope_id)
        if not slope:
            logger.debug(f"delete_slope: slope {slope_id} not found, nothing to delete")
            return False

        deleted_segments = [self.segments[seg_id] for seg_id in slope.segment_ids]
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
        self.drop_undo_actions_for_removed_segments()
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
            logger.debug(f"delete_lift: lift {lift_id} not found, nothing to delete")
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

    @property
    def segment_path_entities(self) -> list[SegmentPath]:
        """Every finished SegmentPath-owning entity (slopes + roads), in one place.

        The single source for "iterate all segment-group entities" so a new SegmentPath kind is
        added HERE once, not at each call site (merge repoint, boundary snapshot, segment lookup).
        Guarded by test_completeness_guards: every buildable SegmentKind must show up here.
        """
        return [*self.slopes.values(), *self.roads.values()]

    def entity_dict_for_kind(self, kind: SegmentKind) -> dict[str, SegmentPath]:
        """The storage dict (slopes or roads) that owns entities of the given kind.

        Single source for kind→container dispatch so a new SegmentKind is wired HERE once, not
        re-branched at each merge/collapse/undo site. Guarded by test_completeness_guards.
        """
        by_kind: dict[SegmentKind, dict[str, SegmentPath]] = {
            SegmentKind.SLOPE: cast("dict[str, SegmentPath]", self.slopes),
            SegmentKind.ROAD: cast("dict[str, SegmentPath]", self.roads),
        }
        return by_kind[kind]

    def get_entity_by_segment_id(self, segment_id: str) -> SegmentPath | None:
        """Find the finished entity (slope OR road) that owns a segment, or None.

        For a one-frame race: a SEGMENT marker can carry a finished entity's
        segment id before the map re-tags it as its Slope/Road (SEGMENT clicks normally resolve to
        the parent entity at render time). An orphan (parent already deleted) legitimately returns
        None — the caller ignores it — so this must NOT raise. Kind-generic via segment_path_entities.
        """
        for entity in self.segment_path_entities:
            if segment_id in entity.segment_ids:
                return entity
        return None

    def get_segment_stats(self, segment_ids: list[str]) -> SegmentStats:
        """Get statistics for specific segments (used for running stats during building).

        Args:
            segment_ids: List of segment IDs to calculate stats for

        Returns:
            Dict with: total_drop, total_length, avg_gradient, max_gradient, difficulty, start_elev, current_elev
            All numeric values are guaranteed non-None. Empty segment_ids returns zeroed defaults;
            every id in a non-empty list must exist (raises KeyError otherwise).
        """
        default_stats: SegmentStats = {
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

        first_seg = self.segments[segment_ids[0]]
        last_seg = self.segments[segment_ids[-1]]

        assert first_seg.start is not None  # Segments always have points
        assert last_seg.end is not None  # Segments always have points
        start_elev = first_seg.start.elevation
        current_elev = last_seg.end.elevation

        total_length = sum(self.segments[seg_id].length_m for seg_id in segment_ids)

        total_drop = start_elev - current_elev
        avg_gradient = (total_drop / total_length * 100) if total_length > 0 else 0.0

        # Difficulty based on steepest section in any segment (max_slope_pct uses rolling window)
        max_slope = max(self.segments[sid].max_slope_pct for sid in segment_ids)
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

    def get_stats(self) -> ResortStats:
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
            longest = max(slope_length, longest)
        for seg in self.segments.values():
            longest = max(seg.length_m, longest)

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
        """Return (lon, lat) per-coordinate median of all nodes, or None if empty."""
        if not self.nodes:
            return None
        lons = [n.lon for n in self.nodes.values()]
        lats = [n.lat for n in self.nodes.values()]
        return statistics.median(lons), statistics.median(lats)

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
            seg = self.segments[sid]
            road_nodes.add(seg.start_node_id)
            road_nodes.add(seg.end_node_id)

        # Nodes touched by slopes (their segments) or lift stations.
        slope_segment_ids = {sid for slope in self.slopes.values() for sid in slope.segment_ids}
        ski_nodes: set[str] = set()
        for sid in slope_segment_ids:
            seg = self.segments[sid]
            ski_nodes.add(seg.start_node_id)
            ski_nodes.add(seg.end_node_id)
        for lift in self.lifts.values():
            ski_nodes.add(lift.start_node_id)
            ski_nodes.add(lift.end_node_id)

        shared = road_nodes & ski_nodes
        return [self.nodes[nid] for nid in shared]

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

    def to_dict(self) -> dict[str, object]:
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
    def from_dict(cls, data: dict[str, object]) -> "ResortGraph":
        """Deserialize graph from dict."""
        graph = cls()

        nodes = cast(dict[str, dict[str, object]], data["nodes"])
        for nid, node_data in nodes.items():
            graph.nodes[nid] = Node.from_dict(data=node_data)

        segments = cast(dict[str, dict[str, object]], data["segments"])
        for sid, seg_data in segments.items():
            graph.segments[sid] = PathSegment.from_dict(data=seg_data)

        slopes = cast(dict[str, dict[str, object]], data["slopes"])
        for slid, slope_data in slopes.items():
            graph.slopes[slid] = Slope.from_dict(data=slope_data)

        lifts = cast(dict[str, dict[str, object]], data["lifts"])
        for lid, lift_data in lifts.items():
            graph.lifts[lid] = Lift.from_dict(data=lift_data)

        # Roads were added after the first backups shipped, so a pre-roads
        # backup has no "roads" key — default to empty
        roads = cast(dict[str, dict[str, object]], data.get("roads", {}))
        for rid, road_data in roads.items():
            graph.roads[rid] = Road.from_dict(data=road_data)

        # A road-owned segment MUST be kind=ROAD; fail loudly rather than mis-render it as a slope.
        for road in graph.roads.values():
            for seg_id in road.segment_ids:
                seg = graph.segments.get(seg_id)
                assert seg is not None and seg.kind == SegmentKind.ROAD, (
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

        counters = cast(dict[str, int], data["counters"])
        graph._node_counter = counters["node"]
        graph._segment_counter = counters["segment"]
        graph._slope_counter = counters["slope"]
        graph._lift_counter = counters["lift"]
        graph._road_counter = counters.get("road", 0)

        logger.info(
            f"Loaded resort: {len(graph.nodes)} nodes, {len(graph.segments)} segments, "
            f"{len(graph.slopes)} slopes, {len(graph.roads)} roads, {len(graph.lifts)} lifts"
        )
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
            ).text = f"Road - Elevation change {abs(total_drop):.0f}m - Length {total_length:.0f}m"
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

        if isolated_node_ids:
            logger.debug(f"Cleaned up {len(isolated_node_ids)} isolated node(s): {isolated_node_ids}")
        return len(isolated_node_ids)
